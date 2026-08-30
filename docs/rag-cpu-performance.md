# Improving RAG accuracy and latency, CPU-only

Companion to `rag-accuracy-at-scale.md`. That document established *that* accuracy decays
with corpus growth and that index tuning cannot fix crowding. This one answers what we can
actually do about it on 0.5-2 vCPU with no GPU — the question the earlier research ran out
of budget before reaching.

Five parallel investigations, each required to give CPU latency on named hardware and a URL
per number. Where an agent's claim was checkable against this repo or reproducible locally,
I checked it rather than relaying it. Two claims I corrected; both corrections are recorded.

## Bottom line

| Finding | Status | Effort |
|---|---|---|
| `hnsw.ef_search` (40) is below `RAG_CANDIDATE_POOL_SIZE` (48) — structural recall loss | **verified, live defect** | config |
| Reranker ignores Vietnamese typed without diacritics — 6.3x score collapse | **reproduced, live defect** | ~1 line |
| Cross-encoder reranking in-request | **NO-GO, 22x over budget** | — |
| Swap to `multilingual-e5-small` (same 384 dims) | **contested — do not act yet** | re-embed |
| Golden set cannot detect an EDA regression (6 travel-policy cases) | verified | data work |

The two defects are config-level, cost no request latency, and are independent of every
strategic question below. They are the whole of what I'd do first.

## 1. Verified defect: the vector index cannot supply the candidates we ask for

`hnsw.ef_search` is never set, so it is **40**. `RAG_CANDIDATE_POOL_SIZE` is **48**, and
`src/repositories/chunk.py:239` issues `LIMIT max(48, limit)`. A single HNSW pass cannot
return more candidates than `ef_search`, so we ask for 48 and the index can offer at most
40 — *before* the permission bitmask and published-status filters in `where_clauses` cut
into that 40.

Verified directly against pgvector's own README rather than the agent's summary:

> "With approximate indexes, filtering is applied *after* the index is scanned. If a
> condition matches 10% of rows, with HNSW and the default `hnsw.ef_search` of 40, only 4
> rows will match on average."

— https://github.com/pgvector/pgvector#filtering

Defaults confirmed from the same source: `ef_search` 40, `m` 16, `ef_construction` 64.

Two fixes, both config:

- `SET LOCAL hnsw.ef_search = 200` per query. Measured recall on a 128-dim/1M benchmark:
  `40 → 95.4%`, `200 → 99.8%`, at 1.19ms → 4.60ms p99
  ([Katz](https://jkatz05.com/post/postgres/pgvector-scalar-binary-quantization/)).
- `SET LOCAL hnsw.iterative_scan = relaxed_order`, which makes the index keep scanning
  until enough rows survive the filter. Needs **pgvector 0.8.0+**.

Version reality check: RDS PG16 ships 0.8.2, so production can do this. But
`docker-compose.yml` pins `ankane/pgvector:v0.5.1`, which **cannot** — so local dev cannot
currently reproduce production retrieval behaviour at all. That mismatch is worth fixing on
its own.

Caveat I'd flag on the recall numbers: they come from an `r7gd.16xlarge` with 64 vCPU. Treat
the recall column as transferable and the latency column as a floor, not a prediction for
Fargate. No published 384-dim CPU-only curve exists.

One trap found in the same code, worth recording before anyone refactors it: the
`cosine_distance <= 0.45` predicate in the `WHERE` clause is what lets Postgres prove
`embedding IS NOT NULL` and therefore use the *partial* index. `<=>` is strict, so
`predtest.c` derives the null-check from it. Moving that filter into a CTE without adding an
explicit `embedding IS NOT NULL` would silently lose the index.

## 2. Reproduced defect: Vietnamese without diacritics scores 6.3x lower

`src/rag/reranker.py` has a Vietnamese-aware `STOPWORDS` list, but only the *accented*
forms. Vietnamese users routinely type without diacritics. I reproduced this directly:

```
'CTS la gi'   score=0.333   is_definition_query=False
'CTS là gì'   score=2.100   is_definition_query=True
'what is CTS'  score=2.100  is_definition_query=True
```

```
accented in STOPWORDS:   ['là', 'gì', 'của', 'về']
unaccented in STOPWORDS: []
```

Two independent failures from one cause. `la` and `gi` are not recognised as stopwords, so
they count as content terms and dilute the coverage score. And `is_definition_query` returns
`False`, losing the definition bonus. Result: 2.100 → 0.333.

That matters because `RAG_MIN_RELEVANCE_SCORE = 0.12` is an absolute floor. A 6.3x
penalty pushes borderline-but-correct passages under it, and the system then *refuses to
answer* rather than returning a weak hit. The same agent measured pure-Vietnamese queries
scoring 0.000 on the current embedding model — already below the floor.

Fix: fold diacritics before stopword and marker matching, and add the unaccented forms.
The repo already has the folding helper — `src/repositories/chunk.py:256` defines `fold()`
using `unicodedata` NFD, and migration 58 wired `immutable_unaccent` into both the FTS index
and query. So accent-insensitive matching is an established convention here; the reranker
just never got it.

## 3. NO-GO: cross-encoder reranking cannot run in-request

This was the most-wanted answer and it is a clean no. Measured on an i5-8300H (AVX2, no
VNNI), onnxruntime 1.28, `intra_op=2`, int8, 48 pairs at ~400 tokens, best-of-5:

| model | 48 cand | 16 cand | vs 300ms |
|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 6.79 s | 2.09 s | 22.6x over |
| ms-marco-MiniLM-L-2-v2 | 2.20 s | 0.84 s | 7.3x over |
| bge-reranker-base (278M, multilingual) | 38.2 s | 10.7 s | 127x over |
| L-2-v2 @128 tokens, top-16 | — | **0.15 s** | only sub-300ms config |
| **current deterministic reranker** | **0.0023 s** | — | in budget |

The one configuration that fits the budget is English-only and was measured failing
Vietnamese completely. The constraints do not intersect.

The Vietnamese measurement is the decisive part, and it is more interesting than the
latency: English-trained ms-marco cross-encoders do not rank Vietnamese at all. All four
candidate scores collapsed into a **0.16-logit band** — noise — versus **18.76 logits** of
separation on the same question in English. Zero `[UNK]` tokens, so this is training
coverage, not tokenization. `bge-reranker-base` (XLM-R vocab) ranked correctly *with*
diacritics (spread 3.18) and produced a spread of exactly **0.00** without them.

Two honest caveats from the agent, both against its own conclusion being too soft:
production will be *slower*, because 2 threads on this laptop are 2 physical cores while
Fargate 2 vCPU is 2 hyperthreads on one core; at 1 thread it measured 31.7 s. And int8
bought only **1.08-1.13x** here because this CPU lacks VNNI — even a generous 3x on
VNNI-capable Fargate leaves L-6 at ~2.3 s.

Arithmetic corroboration that the harness wasn't the bottleneck: L-6 at 512 tokens is 13.3
GFLOPs/pair, 638 GFLOPs for 48; 2-core AVX2 fp32 peak ~74 GFLOP/s predicts 24.8 s at 35%
efficiency against 28.1 s measured.

Conclusion: **keep the deterministic reranker.** It is 3000x cheaper than the cheapest
cross-encoder that can handle Vietnamese. If a real reranker is ever wanted it has to be an
out-of-band service on >=4 vCPU over top-16, which is infrastructure work, not a config
change. Fixing the stopword defect in §2 improves the reranker we already have, for free.

## 4. Contested: the embedding swap. Do not act on this yet

Two agents investigated `intfloat/multilingual-e5-small` independently and reached opposite
conclusions. That disagreement is the finding, so I am not resolving it by picking the
answer I prefer.

What both agree on, and what I consider settled:

- It is **384-dimensional**, so the `vector(384)` column and `ix_article_chunks_embedding_hnsw`
  are untouched. Effort is a re-embed, **not** a schema migration. That is the reason this
  option is attractive at all — the repo already carries three dimension-realignment
  migrations.
- It is *the same architecture* as our current model: both `BertModel`, hidden 384, 12
  layers, 12 heads, vocab 250037, both 117.65M params. E5-small is a retrained
  MiniLM-L12-H384. So CPU cost is near-parity — measured 7.6/10.7 ms vs 7.3/9.1 ms per query.
- It is trained **for retrieval** (contrastive pretraining plus supervised fine-tuning with
  mined hard negatives and cross-encoder distillation, [arXiv 2402.05672](https://arxiv.org/abs/2402.05672));
  ours is a paraphrase/STS model. This is exactly the variable the prior research isolated
  as deciding whether dense retrieval survives corpus growth.
- It requires `query: ` / `passage: ` prefixes, asymmetrically, on both the ingest and query
  paths. Omitting them is a silent quality regression.
- Its window is 512 tokens vs our 128 — but 512-token chunks measured **5.8x** the cost per
  chunk, so the window is not free.

BEIR nDCG@10 strongly favours e5-small on 8 of 9 tasks — NQ 29.80 → 56.27, HotpotQA 30.01 →
65.09, SciFact 48.49 → 67.58 ([MTEB results repo](https://github.com/embeddings-benchmark/results)).
VN-MTEB retrieval: 14.14 → 34.12.

**Where they disagree, and why I am not acting:**

| evidence | direction |
|---|---|
| BEIR (8 of 9 tasks), VN-MTEB retrieval +141% | favours e5-small |
| Agent A's local probe: VN query → EN passage cosine 0.019 → 0.758 without diacritics | favours e5-small |
| **MLQA vie→eng 69.03 vs 53.64**, eng→vie 79.66 vs 75.54 | favours **current model** |
| **Agent B's 24-pair VN→EN semiconductor probe: recall@1 0.792 vs 0.625** | favours **current model** |

The two agents ran opposing local probes on the same question and got opposing answers. Both
probes are small (5 and 24 hand-written pairs), self-authored, and not benchmarks. MLQA —
a real benchmark — sides with the current model on precisely our hard case, Vietnamese query
against English content.

Agent B also reported, against its own recommendation, that its prefix ablation came out
counter-intuitive: no-prefix 0.708 beat correct-prefix 0.625. Correct prefixes did beat
swapped ones, so the asymmetry matters directionally, but the magnitude is unestablished and
the model card's "you will see a performance degradation" is a claim with no published
ablation behind it.

And a contested data point in the benchmark itself: VN-MTEB Table 3 lists "MiniLM-L12" at
33.4M params, but its appendix names our exact multilingual checkpoint, which measures
117.7M. 33.4M is the *English* `all-MiniLM-L12-v2`. So either the params column is wrong or
the wrong model was benchmarked. VN-MTEB is also machine-translated, and retained only
33-40% of samples on scientific sets — our domain.

**Verdict: run an offline A/B on our own corpus before committing the re-embed.** The
evidence is strong on general retrieval and genuinely contradictory on Vietnamese-to-English,
which is the case we care about. This is what §6 is for, and it is the reason §6 comes before
any embedding decision.

## 5. The FTS leg is degraded for Vietnamese, not broken

PostgreSQL ships 29 text-search dictionaries and none is Vietnamese; Snowball has no
Vietnamese stemmer either. Our `'simple'` config does no stemming and splits on whitespace,
so "máy tính" — one word, two syllables — indexes as two independent lexemes. Real precision
loss, magnitude unpublished.

But the premise that `unaccent` is a missing remedy is **wrong**, and I verified this:
migration `20260816_58_search_fts_alignment.py` already wraps both the index and the query
in `immutable_unaccent`, and the index expression matches the query expression exactly.
Diacritic-insensitive lexical matching works today. Stemming would buy nothing anyway —
Vietnamese is isolating, with no inflection. The problem is segmentation, which dictionaries
do not solve.

Not recommended: a Vietnamese tokenizer (`underthesea`/`pyvi`/VnCoreNLP) at ingest. High
effort, unproven for our corpus, and the §2 and §4 items dominate it.

Also ruled out, with reasons: Vietnamese-monolingual encoders (`bkai/vietnamese-bi-encoder`,
`dangvantuan/vietnamese-embedding`) need external word segmentation at both index and query
time, are 768-dim, and would hurt our English technical content. `bge-m3` and
`AITeamVN/Vietnamese_Embedding` are 1024-dim/568M — measured ~11x slower than a small model
on *stronger* hardware than ours.

## 6. What actually gates every decision above: we cannot measure any of this

`evaluation/golden.json` has **6 cases**, all generic travel-policy/SOP fixtures, with zero
semiconductor or EDA content. I verified this. The CI job named `RAG regression gate (live)`
is `if: github.event_name == 'workflow_dispatch'` — it has never run on a PR, which is why
it shows as `skipping` on every one of them. The offline step scores fixtures, not the
pipeline. `src/rag/evaluator.py` has `context_recall` / `lexical_faithfulness` /
`answer_correctness` but **no ranking metrics** — no Recall@k, nDCG, or MRR.

So: the gate we would use to judge the embedding swap in §4 would pass regardless of what
happened to EDA retrieval. That is the single most consequential finding in this document,
because it is what makes §4 unresolvable and would make any future tuning unfalsifiable.

Recommended shape, deterministic and CPU-only so it can block per-PR: `pytrec_eval` (NIST
binding, MIT) or `ranx`, scoring Recall@k / nDCG@k / MRR over frozen chunk IDs. No LLM
judge in the blocking tier — GPT-4-as-judge measured **65% order-swap consistency**
([arXiv 2306.05685](https://arxiv.org/abs/2306.05685)), which cannot gate CI. Keep
LLM-graded faithfulness in the existing `workflow_dispatch` tier.

Golden-set size: n=34 detects a medium effect at α.05/β.20, n=199 for a small effect
([Sakai](https://link.springer.com/article/10.1007/s10791-015-9273-z)). Target **>=50 EDA
cases including Vietnamese queries against English content** — that last category is exactly
what §4 is contested on and what §2's defect breaks.

A trap to avoid: the prior research confirmed a case where retrieval precision *improved*
while answer faithfulness *collapsed* (P@10 0.77→0.86, RAGAS faithfulness 0.61→0.35).
Retrieval-only and end-to-end metrics have to be tracked separately.

## 7. Free win, with a correction to the research

`src/domain/indexing.py:237` embeds `[item[0] for item in pending_children]` — the bare child
text. `parent_heading` is already in that same tuple, built at line 221 and carried at line
235, then discarded. Prepending the heading path before embedding is the free, deterministic
variant of contextual retrieval, and it attacks embedding-space crowding directly — the
mechanism the prior research confirmed as the actual cause of decay.

Anthropic measured LLM-generated context cutting retrieval failures 5.7% → 3.7%
([contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval)), though
on Gemini/Voyage embeddings, not ours. The free heading-only variant has no published number;
dsRAG ships it ungraded.

**Correction.** The agent reported 312-char children costing 71-89 tokens, concluding a
heading prepend is unconditionally free (89 + 23 <= 128). I measured it against the real
tokenizer and got **76-108 tokens**:

| text | 312 chars → tokens |
|---|---|
| EN tool reference | 76 |
| VI with diacritics | 100 |
| VI mixed with EN terms | 108 |

At 108 tokens, adding a 23-token heading path gives 131 and **exceeds the 128 cap** — silent
tail truncation at `src/lib/embeddings/local_onnx.py:82`. So heading-prepend is *not*
unconditionally free for Vietnamese-heavy chunks. It needs either a smaller child budget or
heading truncation. The direction is right; the "free" claim is not.

What both measurements agree on: `src/rag/chunker.py:138` assumes **2.5 chars/token**
"deliberately pessimistic" and predicts 125 tokens for a 312-char child. Real is 2.90-3.96
measured, so typical children use 76-108 of the 128 available. The window is underused, and
the docstring's stated reasoning is wrong in the direction that matters.

Also ruled out for chunking: semantic/cluster/LLM chunking — fixed-size measured
equal-or-better on realistic documents ([arXiv 2410.13070](https://arxiv.org/abs/2410.13070)),
and cluster methods need global corpus statistics plus re-chunking as data arrives, which is
fatal at tens of thousands of documents.

## Recommended order

1. **`ef_search` + `iterative_scan`** (§1) — config, no latency cost, fixes a verified
   recall loss. Bump the dev `pgvector` pin so dev matches prod.
2. **Reranker diacritic folding** (§2) — ~1 line, fixes a reproduced 6.3x defect on ordinary
   Vietnamese input.
3. **Golden set + deterministic ranking metrics in the skipping gate** (§6) — prerequisite
   for judging anything else.
4. **Then** A/B the embedding swap (§4) and heading-prepend (§7) against that gate.

Do not add a cross-encoder (§3). Do not add a Vietnamese tokenizer or change dimensions (§5).

## Stale comment worth fixing separately

`src/lib/embeddings/local_onnx.py:20-31` documents an `optimum-cli export onnx --model
BAAI/bge-m3` step and reasons about "~2.3 GB of fp32 weights". The Dockerfile bakes
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` using the model's own published
ONNX export, with no `optimum-cli` step — the file's own comment at line 155 says so. The
int8-vs-fp32 measurements in that docstring were taken against bge-m3 and no longer describe
what ships.
