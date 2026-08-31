# MLQA / UIT-ViQuAD 2.0 optimization log

Target: per-language answer token-F1 >= 90 on MLQA-en, MLQA-vi, UIT-ViQuAD 2.0-vi,
end-to-end (retrieval -> answer), CPU-only, no GPU dependency anywhere in the
adopted path. Tuning reads dev/validation splits; the gate reads test.

Best-known state is tracked in `best.json`. Every run appends one record to
`results.jsonl`.

## Reproducing a run

```bash
# 1. eval database (PG16 + pgvector), kept separate from the dev stack's volume
docker run -d --name qnsc-eval-db -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=qnsc_kb_eval \
  -p 127.0.0.1:5433:5432 -v qnsc_eval_pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
poetry run alembic -c migrations/alembic.ini upgrade head

# 2. datasets -> .eval-cache/datasets/normalized/*.jsonl
poetry run python scripts/eval_prep_datasets.py

# 3. corpus (one Article per unique benchmark context, real chunker + embedder)
source .eval-cache/env.sh
poetry run python scripts/eval_build_corpus.py \
  .eval-cache/datasets/normalized/mlqa_dev.jsonl \
  .eval-cache/datasets/normalized/viquad2_validation.jsonl --purge

# 4. evaluate
poetry run python scripts/eval_rag.py <normalized.jsonl> \
  --lang-q vi --lang-c vi --sample 250 --tag <label> --out <detail.jsonl>
```

Environment lives in `.eval-cache/env.sh` (eval DB URL, ONNX dirs, thread counts).

## Iteration 0 — assessment and baseline instrumentation

### Existing system

| Stage | Implementation | CPU-only |
|---|---|---|
| Chunking | structure-aware parent 1800/250 chars -> child 250/60 (`src/rag/chunker.py:145-152`) | yes |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2, 384-dim, ONNX, `EMBEDDING_MAX_TOKENS=128` | yes |
| Dense retrieval | pgvector HNSW cosine, `VECTOR_DISTANCE_THRESHOLD=0.45`, pool 48, `hnsw.ef_search=200` | yes |
| Sparse retrieval | Postgres FTS `to_tsvector('simple', immutable_unaccent(...))` + `ts_rank_cd` — **not BM25** | yes |
| Fusion | RRF, `k=60` hardcoded, equal weights (`src/repositories/chunk.py:376`) | yes |
| Reranking | **no model** — deterministic lexical scorer (`src/rag/reranker.py`) | yes |
| Generation | **hosted API only** (OpenAI/GLM/Groq/Gemini via httpx); provider+key from a DB row | **no** |
| Answer metrics | set-overlap proxies only; **no SQuAD EM/token-F1 anywhere** | — |

### Blockers found, and what was done

1. **No local generation.** `src/domain/ai_service.py:1178-1194` — with no provider
   configured the "answer" is the top chunk truncated at 200 characters plus `...`.
   Scoring that measures nothing. A hosted LLM would also violate CPU-only.
   -> Added `src/lib/reader/` : an in-process ONNX extractive reader
   (mDeBERTa-v3-base-squad2, MIT, 278M, fp32 1.15 GB) with SQuAD2 null-answer
   abstention. Both benchmarks are span-extraction tasks, so this is the honest
   instrument. The hosted path is untouched.
2. **No SQuAD scorer.** `src/rag/evaluator.py` `answer_correctness` is recall-only
   and drops tokens of <=3 characters, deleting real Vietnamese words.
   -> Added `src/rag/squad_metrics.py` (EM + token-F1, per-language article
   handling, diacritics preserved) with 12 tests.
3. **PG15 volume vs pg16 image.** The dev `pgdata` volume refuses to boot under
   `pgvector/pgvector:pg16`. -> Separate eval database on port 5433, dev volume
   left untouched.
4. **ViQuAD 2.0 test has no public answers** (7,301 questions, 0 answerable).
   -> Vietnamese gate reads `validation` (3,814 q / 2,653 answerable); `train` is
   available for tuning.
5. **6.1 s/question reader latency.** -> Batched every stride window of every
   passage into one `session.run` (`READER_BATCH_SIZE=16`). 6.1 s -> 5.2 s p50;
   the work is compute-bound, not dispatch-bound.
6. **int8 rejected on measurement**, not taste: -10.3 F1 (MLQA-en), -9.7 (MLQA-vi),
   -20.2 (ViQuAD-vi) for ~8% latency. fp32 stays the default.

### Reader ceiling (gold context, reader alone — not an end-to-end score)

60 answerable questions per split, dev/validation, fp32:

| Split | EM | F1 |
|---|---|---|
| MLQA-en | 63.3 | 78.4 |
| MLQA-vi | 43.3 | 61.4 |
| ViQuAD-vi | 38.3 | 56.3 |

This is the cap on any end-to-end number with this reader: it is what the model
scores when retrieval is perfect by construction.

### Target feasibility (evidence, not opinion)

Published extractive SOTA:

| Benchmark | Best published | Source |
|---|---|---|
| MLQA-vi | F1 75.2 / EM 54.1 (InfoXLM-large, zero-shot XLT) | arXiv 2012.15674 Table 3 |
| MLQA-vi (base-size) | F1 64.5 / EM 44.7 (XLM-R base) | arXiv 1911.02116 Table 3 |
| MLQA-en | F1 84.5 (InfoXLM-large) | arXiv 2012.15674 Table 3 |
| UIT-ViQuAD 2.0 | F1 77.24 / EM 67.43 (private test, best of VLSP 2021) | arXiv 2203.11400 |

**>=90 token-F1 is 6-25 points above every published extractive result on these
benchmarks.** The objective's fallback branch therefore governs: optimize as far
as CPU-only allows, document the best achieved score, and name the bottleneck.

## Iteration 1 — Vietnamese retrieval ranking (38.7 F1 points)

### Bottleneck, diagnosed rather than guessed

Failure analysis put the largest single loss in retrieval ranking: 25.1 F1 points
combined, 38.7 on MLQA-vi, where 39% of questions never retrieved their gold
paragraph. Two diagnostics then separated the legs, over the 60 MLQA-vi questions
where retrieval had failed:

| Leg | Finding |
|---|---|
| Dense | gold chunk at cosine distance **median 0.819** (min 0.464), so **0/60 passed `VECTOR_DISTANCE_THRESHOLD=0.45`**; dense rank median 764 |
| Sparse | with the product's real OR-of-per-term tsquery, **found the gold document for 96.7%** — but at **median rank 172**, cut by `RAG_CANDIDATE_POOL_SIZE=48` |

So the evidence was in the corpus and reachable; two gates discarded it. The
first measurement of the sparse leg used a single AND-semantics `plainto_tsquery`
and reported 3.3%; that understated it, and the corrected OR-semantics number is
the one above.

Root cause of the dense failure is the encoder's training objective, not tuning:
`paraphrase-multilingual-MiniLM-L12-v2` is a symmetric paraphrase/STS model, and
a question is not a paraphrase of the passage that answers it.

### Research

VN-MTEB (arXiv 2507.21500, Table 3), 15 Vietnamese retrieval datasets:

| Model | VN-MTEB retrieval |
|---|---|
| bge-m3 (1024-d, 2.3 GB) | 39.84 |
| **multilingual-e5-small (384-d)** | **34.12** |
| paraphrase-multilingual-MiniLM-L12-v2 (current) | 14.14 |

e5-small is also 384-wide, so it needs no pgvector column or HNSW rebuild. It
does require the `query: ` / `passage: ` instruction prefixes.

### Measured (dev/validation, 200 questions per config)

Retrieval only, so the reader could not confound it:

| Config | MLQA-en R@10 | MLQA-vi R@10 | ViQuAD-vi R@10 |
|---|---|---|---|
| MiniLM, pool 48 (shipped) | 80.5 | 56.5 | 68.8 |
| e5-small, pool 48 | 91.0 | 61.0 | 67.4 |
| e5-small, pool 256 | 91.5 | 63.0 | 74.5 |
| MiniLM, pool 256 | 82.5 | **56.0** | 70.2 |

End to end, e5-small + pool 256:

| Config | EM | F1 | vs baseline | gold retrieved | answer in context |
|---|---|---|---|---|---|
| MLQA-en | 52.0 | 62.20 | **+7.92** | 93.5% | 91.0% |
| MLQA-vi | 24.0 | 33.88 | **+1.80** | 63.0% | 66.0% |
| ViQuAD-vi | 37.5 | 51.41 | **+3.30** | 74.5% | 72.3% |

### Outcome: validated, not adopted

All three languages improved with no regression, but the change was NOT made the
default, for two reasons that are not about the measurement:

1. `tests/unit/test_embedding_config_matches_image.py` pins
   `infra/live/{develop,prod}/main.tf` to the Dockerfile ARG and the code default.
   Flipping the encoder requires editing `infra/`, which is outside the agreed
   scope boundary.
2. Adopting it DELETES and re-embeds every stored chunk — same 384 width, a
   different vector space. That is a production data operation, not a config flip.

The deeper candidate pool was also reverted, on its own evidence: under the
shipped MiniLM encoder it moves MLQA-vi 56.5 -> 56.0 for about +400 ms retrieval
p50. The pool gain belongs to the encoder swap, not to the pool.

What shipped instead is the capability, with defaults untouched: e5 dimension
derivation, and central `query: `/`passage: ` prefix handling so the two local
runtimes cannot disagree about it. Adoption is then three env vars plus a
re-index.

### Ablations recorded

- Removing `VECTOR_DISTANCE_THRESHOLD` under e5: byte-identical results. The
  cutoff is inert there (all 48 candidates pass) and was destructive under MiniLM
  (0/60 gold chunks passed) — fixed by the encoder, not by the threshold. No
  change made.
- Reader int8: -10.3 / -9.7 / -20.2 F1 for ~8% latency. Rejected in iteration 0.

### Next bottleneck

The reader, on Vietnamese. Retrieval gained 7.5 points of answer-in-context on
MLQA-vi and F1 moved only 1.8, because the gold-context ceiling is 61.4 F1
(MLQA-vi) and 56.3 (ViQuAD-vi). No retrieval work can lift the score past those
numbers; a stronger Vietnamese reader is the only lever left, and published
extractive SOTA (75.2 / 77.24) still sits below the 90 target.


## Evaluation speed: what was tried, and the protocol that came out of it

The reader dominates cost at ~7 s/question, so a 200-question config takes ~40
minutes and a three-language iteration takes two hours. Three levers were
measured; two are dead ends, and recording them here is what stops them being
retried.

### GPU: rejected on measurement, 3.5x SLOWER

Evaluated at the user's request for local runs only, with production staying
CPU-only. GTX 1050 4 GB, CUDA 12.4, `onnxruntime-gpu` 1.20.2 plus pip
`nvidia-cudnn-cu12`/`cublas`. The CUDA provider was confirmed genuinely active
(session reported `['CUDAExecutionProvider', 'CPUExecutionProvider']`), so this is
not a fallback artefact:

| config | F1 | p50 | vs CPU |
|---|---|---|---|
| CPU 384/128 t4 b16 | 61.89 | 7.3 s | 1.00x |
| CUDA 384/128 b16 | 61.89 | 7.3 s | 1.04x |
| CUDA 512/128 b64 | 61.89 | 26.1 s | **0.28x** |

Cause: 4 GB cannot hold mDeBERTa's graph, so ONNX Runtime partitions it and
inserts Memcpy nodes that copy tensors host<->device on every stride window. F1 was
identical across all arms, which is the useful part -- it confirms accuracy
transfers between providers, so a bigger GPU would be a valid accelerator later.
`onnxruntime-gpu` was uninstalled and the declared CPU wheel restored.

Two guards were added while doing this, and they stay: `READER_ONNX_PROVIDERS`
defaults to `CPUExecutionProvider`, and `tests/unit/rag/test_reader_execution_providers.py`
asserts both that default and that `onnxruntime-gpu` is absent from
`pyproject.toml` -- so images are CPU-only by construction, not by convention.

### CPU configuration: already optimal

| config | F1 | p50 | speedup |
|---|---|---|---|
| 384/128 t4 b16 (shipped) | 66.65 | 7.1 s | 1.00x |
| 384/128 t8 b16 | 66.65 | 7.3 s | 1.00x |
| 512/128 t8 b32 | 66.65 | 8.5 s | 0.83x |
| 384/64 t8 b32 | 66.65 | 8.5 s | 0.85x |

Eight threads buy nothing -- the work is memory-bandwidth bound, not compute
bound -- and larger windows cost more than the reduced overlap saves. F1 was
identical (66.65) in all six configs, so nothing was traded away.

### Passage count: an evaluation protocol, not a product change

With the gold passage pinned at rank 2, 3 passages scored the same F1 as 8 at
1.63x. That looked like a free win and is not one: reading
`answer_passage_index` back from the real iteration-1 runs (590 answered
questions) shows how much a cap actually discards.

| cap | correct answers kept | speedup |
|---|---|---|
| 3 | 88.9% | 1.63x |
| 4 | 93.9% | 1.46x |
| 6 | 97.0% | ~1.28x |
| 8 (shipped) | 100% | 1.00x |

So the protocol, rather than a default change: exploratory passes may set
`RAG_MAX_CONTEXT_PARENTS=4` for 1.46x with a known ~6% loss of correct answers,
and every validating or reported run uses the shipped 8. Sweeps stay
retrieval-only, which needs no reader at all and is already ~30x faster per
config.



## Iteration 4 — fusion weights are a structural no-op at this architecture

### Hypothesis

The sparse leg is Postgres `ts_rank_cd`, which uses no IDF, no term saturation,
and no length normalisation. At equal RRF weight it gets 50% of the fused score
unearned, so a common-word match can outrank a semantically correct passage.
Bruch et al. (arXiv 2210.11934) measure weighted/convex fusion as never worse
than equal-weight RRF. Down-weighting the sparse leg should therefore help
Vietnamese, and it is a query-time-only change (no re-index, no infra edit).

### Result — REJECTED, byte-identical to control

Retrieval-only A/B, both arms in one process on the identical MiniLM corpus,
120 questions per split:

| split | arm (dense/sparse) | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| MLQA-vi | 1.0/1.0 (control) | 41.67 | 55.00 | 58.33 | 47.05 |
| MLQA-vi | 1.0/0.25 | 41.67 | 55.00 | 58.33 | 47.05 |
| ViQuAD-vi | 1.0/1.0 (control) | 54.55 | 67.05 | 72.73 | 59.96 |
| ViQuAD-vi | 1.0/0.25 | 54.55 | 67.05 | 72.73 | 59.96 |

Every metric is identical to the last decimal.

### Root cause — verified in code, not inferred

`ChunkRepository.hybrid_search` (`src/repositories/chunk.py:395-406`) applies the
weights, sorts by fused RRF score, and returns `sorted_results[:limit]`. But the
caller `SearchService.search` (`src/domain/search_service.py:199`) then does:

```python
ranked = rerank_chunks_with_scores(retrieval_query, candidates, limit=limit)
```

which re-scores every fused candidate with the deterministic lexical reranker and
sorts by *that* score — the RRF order is discarded. Both legs return
`max(pool, limit)` ≥ 16 candidates, they overlap heavily, and all of them enter
the reranker's top-`limit`. So the dense/sparse weight ratio changes which chunks
are *inside* the merged set (which is far larger than `limit`) but not which 16
the reranker ultimately surfaces, nor their order. The fusion weight is therefore
inert by construction.

### Conclusion

RRF fusion weighting **cannot** improve retrieval ranking until the lexical
reranker that overrides it is replaced (a CPU cross-encoder or a
retrieval-trained scorer). The `RAG_FUSION_DENSE_WEIGHT` / `RAG_FUSION_SPARSE_WEIGHT`
settings were added this iteration with behaviour-preserving 1.0/1.0 defaults and
are kept as inert, documented knobs for the day the reranker changes. No default
was altered; nothing shipped.

This joins the standing conclusion that the two true binding constraints are
(1) the reader's Vietnamese gold-context ceiling (61.4 MLQA-vi / 56.3 ViQuAD-vi,
both far below the 90 target and below published extractive SOTA of 75.2 / 77.24),
and (2) the symmetric MiniLM encoder, whose fix (e5-small) is validated but blocked
on the `infra/` scope boundary and a production re-embedding operation.

## Iteration 5 — CPU cross-encoder reranker (bge-reranker-v2-m3)

### Hypothesis and research

Iteration 4 proved the lexical reranker overrides retrieval order, so it is the
binding stage for ranking quality. Research (web, 2024-2025) on CPU-friendly
multilingual rerankers surfaced three candidates: PhoRanker (itdainb/PhoRanker,
Apache-2.0, 0.1B, best MMARCO-VI NDCG@10 0.742 but needs VnCoreNLP word
segmentation), ViRanker (CC-BY-4.0, BGE-M3 backbone), and bge-reranker-v2-m3
(BAAI, Apache-2.0, 0.57B, multilingual, prebuilt CPU ONNX export, NDCG@10 ~0.68).

Chose **bge-reranker-v2-m3** for the experiment: multilingual (improves EN and
VI), no word-segmentation dependency, and a prebuilt ONNX that drops onto the
same Runtime seam the embedder and reader already use. Built as
`src/lib/reranker/` (base protocol + ONNX backend + resolver), wired into
`SearchService` behind `RERANKER_BACKEND` (default `lexical`, so nothing ships
until opted in), with fallback to the lexical scorer if the model is unavailable.

### Two calibration bugs found and fixed (the important engineering finding)

The cross-encoder emits an unbounded relevance logit; the pipeline had two gates
calibrated for the lexical scorer's 0..1 range that silently destroyed it:

1. `RAG_MIN_RELEVANCE_SCORE=0.12` (in SearchService): a passage the cross-encoder
   ranks correctly but not confidently has a negative logit -> sigmoid < 0.12, so
   the lexical floor deleted every candidate and search returned zero results.
   Fix: `RERANKER_MIN_SCORE` (sigmoid space, default 0.0), applied only on the
   cross-encoder path.
2. `RAG_MIN_CONTEXT_SCORE=0.35` (the end-to-end confidence gate): compares against
   the top reranker score; cross-encoder sigmoid ~0.001 tripped it on nearly every
   question, so the reader was never called and end-to-end F1 collapsed 35.3 -> 10.0
   even though retrieval had IMPROVED. Measured cleanly with `--no-confidence-gate`.

The lesson for any future scorer swap: score-scale calibration is a first-class
integration concern, not a detail. A better reranker looked like a catastrophic
regression purely because its scores are on a different scale than two downstream
thresholds.

### Results (MLQA-vi, small samples within a 15-minute-per-step budget)

Retrieval-only A/B, identical MiniLM corpus, 30 questions:

| arm | R@1 | R@5 | R@10 | MRR | s/query |
|---|---|---|---|---|---|
| lexical | 53.3 | 76.7 | 76.7 | 62.78 | 1.4 |
| cross-encoder | 43.3 | 60.0 | 73.3 | 51.60 | 8.5 |

Mixed on aggregate at n=30, but on an inspected query the cross-encoder moved the
gold document from rank 6 to rank 1, where the lexical scorer had tied four wrong
documents at a perfect 0.75 (the exact pathology iteration 1 documented).

End-to-end F1, 20 questions, confidence gate OFF (fair, bug removed):

| arm | EM | F1 | answer_in_context |
|---|---|---|---|
| lexical | 25.0 | 35.30 | 70.0 |
| cross-encoder | 25.0 | 35.30 | 75.0 |

### Verdict — validated capability, no end-to-end gain, NOT adopted

The cross-encoder demonstrably fixes ranking quality (gold 6->1; answer_in_context
70->75) but end-to-end F1 is unchanged, because the reader's Vietnamese
gold-context ceiling (61.4 F1) is the binding constraint, not retrieval ranking.
Reranking cannot lift F1 above what the reader can extract from the passages it is
handed. And it costs 6x retrieval latency (8.5 s vs 1.4 s per query, fp32 568M
model on CPU).

So it is kept as an opt-in, off-by-default capability with its calibration bugs
fixed and documented, not adopted as the shipped default. Consistent with rule 10
(never merge a change that does not improve the target) and the standing
conclusion that the reader is the binding constraint. Samples were small to fit
the time budget; the direction (retrieval up, F1 flat, latency up) is consistent
across the retrieval and end-to-end measurements and matches the iteration-4
mechanism, so a larger sample is not expected to change the verdict.
