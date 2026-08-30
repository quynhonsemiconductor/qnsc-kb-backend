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
