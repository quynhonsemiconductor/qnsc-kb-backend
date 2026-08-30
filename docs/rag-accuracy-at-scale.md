# RAG accuracy as the corpus grows

Research output for: how to hold retrieval accuracy as this KB grows from hundreds to
tens of thousands of documents.

Method: 6 search angles, 27 sources, 135 claims extracted, 25 put to a 3-vote
adversarial panel. **7 confirmed, 14 refuted, 4 unverified.** Every claim below carries
its panel vote. A claim with no vote is inference and says so.

The refuted list matters more than the confirmed list here, because the most
quotable and most actionable-sounding numbers in the whole set are the ones that
died. Read that section before acting on anything.

## Bottom line

Ordered by evidence strength, not by appeal:

1. **Degradation is real and unavoidable.** Plan for it as a continuous measurement
   problem, not a one-time fix. [confirmed 2-1, 3-0]
2. **Keep the lexical leg.** Sparse retrieval covers a failure mode single-vector dense
   retrieval structurally cannot. Do not weaken it. But do not weight *toward* it
   either — see refuted. [confirmed 2-1]
3. **A reranker is the mechanism with the strongest theoretical case** — it is not bound
   by embedding dimension. The CPU-viable latency numbers we need are unverified. [confirmed 2-1]
4. **Embedding dimension is a real ceiling in principle, unquantified at 384.** The
   claim that our 384-dim model is the binding problem was killed 0-3. [confirmed 2-1 with caveat]

What the research does *not* establish, despite being asked: HNSW/pgvector tuning at
scale, cross-lingual Vietnamese retrieval, and refuse-threshold calibration. Zero
confirmed claims on all three. Those remain open questions and this document does not
answer them.

## Confirmed

### 1. Everything degrades with corpus growth, every paradigm [2-1]

28 nested tiers spanning ~450x (1,144 → 511,959 documents), questions held fixed:

| retriever | small | large |
|---|---|---|
| BM25 | 74.7 | 50.5 |
| DenseRAG | 58.1 | 29.9 |

Dense loses roughly half its accuracy; BM25 about a third.
Source: https://arxiv.org/html/2607.26497v2

### 2. Dense does not lose to BM25 — training objective decides that [3-0]

MS MARCO passage retrieval, MRR@10 across index sizes:

| retriever | 10k | 8.8M |
|---|---|---|
| dense 768d, hard negatives | 91.48 | 28.55 |
| BM25 | 79.93 | 17.56 |
| dense 768d, **no** hard negatives | — | 17.34 |

> "For small index sizes, we observe that dense approaches drastically reduce the error
> rate compared to BM25 retrieval. With increasing index sizes, the gap closes."

The gap closes; dense still wins at 880x the corpus — 28.55 vs 17.56. But dense trained
without mined hard negatives collapses to 17.34, below BM25.
Source: https://arxiv.org/abs/2012.14210

**[INFERENCE, not verified]** The variable that decides whether dense retrieval survives
scale is how the embedding model was trained, not its width.
`paraphrase-multilingual-MiniLM-L12-v2` is a paraphrase-similarity model, not a
retrieval model trained with mined hard negatives. That is a sharper concern about our
stack than "384 dimensions is too few", and it points at a different fix — swap for a
retrieval-trained multilingual model of the same size class rather than a wider one.
This is the single most useful thing the research produced and it is my inference from
one confirmed claim, not itself confirmed.

### 3. Sparse covers a failure mode dense structurally cannot [2-1]

LIMIT benchmark, 50k corpus: BM25 85.7 recall@2 / 93.6 recall@100, versus best dense
3.0 / 18.9.

> "Sparse models […] can be thought of as single vectors but with very high
> dimensionality. This dimensionality helps BM25 avoid the problems of the neural
> embedding models […] they can scale to many more combinations than their dense vector
> counterparts."

Mechanistic, not a tuning preference. Argues for keeping a keyword leg permanently.
Source: https://arxiv.org/pdf/2508.21038

### 4. Embedding dimension is a ceiling — with the caveat baked in [2-1]

Extrapolated "critical-n" corpus size, top-k=2: ~500k docs @512d, 1.7M @768d, 4M @1024d.

The panel confirmed this **only** with its limits attached: it is a best-case bound from
vectors optimised directly on the test qrel matrix, extrapolated by a degree-3
polynomial fit over small d. Applying it to a 384-dim production model is an
extrapolation the paper does not validate.

At our projected scale — tens of thousands of documents — even the pessimistic reading
of this bound is orders of magnitude away. Dimension is not our binding constraint.
Source: https://arxiv.org/pdf/2508.21038

### 5. Rerankers escape the dimension bound [2-1]

A long-context reranker given all 46 documents solved 100% of 1000 queries in one
forward pass, versus under 60% recall@2 for the best embedding model. Multi-vector late
interaction (GTE-ModernColBERT) beat every single-vector model — 97.6 recall@10 on
LIMIT-small vs ~68 for E5-Mistral 7B at 4096 dims, on a much smaller backbone.

Caveat the panel insisted on: the reranker tested is a frontier LLM, not a small
quantized cross-encoder, and the paper says cross-encoders are unsuitable for
first-stage retrieval at scale. So this confirms the *mechanism*, not that we can run it
on 0.5-2 vCPU.
Source: https://arxiv.org/pdf/2508.21038

### 6. Crowding is intrinsic, not an ANN artifact [3-0]

Controlled 100x sweep at 5,000 / 16,000 / 50,000 / 160,000 / 511,962 documents: Recall@10
falls and top-10 cosine similarity rises for both BM25 and vector search.

> "Note: vector search for this experiment uses exact nearest neighbors."

That detail is the whole value of the claim. The recall loss is embedding-space crowding
and distractor abundance — **not** HNSW recall loss. Tuning our index will not buy this
back. (Trend direction only; the paper gives no per-scale numeric table.)
Source: https://arxiv.org/pdf/2605.05253

### 7. Enterprise corpora are measurably more crowded than open-web [3-0]

Average cosine similarity to the 10 nearest neighbours: **0.83** for both a synthetic
enterprise benchmark and a real company corpus, versus **0.69** for open-web
BrowseComp-Plus. In-cluster 0.61 / 0.56 vs 0.29; cross-cluster 0.50 / 0.36 vs 0.20.

> "When a document's nearest neighbors lie close in embedding space, distractors are
> abundant, and the retriever cannot rely on a wide margin to separate the gold document
> from the rest of the corpus."

This is us. A single-vendor technical corpus — lecture slides and tool references over
one subject area — is the crowded case, so public-benchmark degradation curves are the
optimistic reading of what we should expect.
Source: https://arxiv.org/pdf/2605.05253

## Refuted — including things I would have recommended

14 of 25 claims were killed. These are the ones that would have changed what we build.

| claim | vote |
|---|---|
| Degradation rate is tied to embedding dimensionality — implicating our 384-dim model | 0-3 |
| Dense lost to BM25 at every corpus tier → weight the hybrid toward lexical | 0-3 |
| Technical corpora (tool names, part numbers, flags) favour exact-term matching over embeddings | 0-3 |
| Scaling 54 → 1,128 docs collapsed accuracy 75% → under 40% | 0-3 |
| Metadata scoping is the single highest-leverage fix (P@10 0.77 → 0.86) | 0-3 |
| Hybrid + cross-encoder reranking is the highest-yield single lever at this scale (Recall@5 0.816) | 0-3 |
| Query expansion is a poor lever; contextual retrieval outranks it | 0-3 |
| BM25 beat dense on a technical/tabular corpus → raise keyword weight | 0-3 |
| LLM query rewriting/decomposition does not help, hybrid and HyDE do | 0-3 |
| BM25 is fragile to vocabulary mismatch (drops >89% under synonym substitution) | 0-3 |
| LIMIT: corpus size directly drives dense failure; recall scales monotonically with dimension | 0-3 |
| There is a corpus-size tipping point past which BM25 overtakes dense | 1-2 |
| Improving retrieval precision cut RAGAS faithfulness 0.61 → 0.35 | 1-2 |
| On a ~500k-doc enterprise corpus BM25 beat dense on every headline metric | 1-2 |

Three things to take from that table.

**All three claims from the enterprise case-study source were killed** (0-3, 0-3, 1-2).
That source produced the most quotable, most actionable-sounding numbers in the entire
set — the "35-point collapse from scale alone" and "metadata filtering is the
highest-leverage fix" figures — and none survived. Had the unverified extraction reached
us, metadata scoping is what I would have told you to build first.

**Both directions of the BM25-vs-dense argument were killed.** "Weight toward lexical"
died 0-3 four separate times; "BM25 is fragile to vocabulary mismatch, so don't" also
died 0-3. The sources contradict each other and the panel refused both. The honest
position is that hybrid weighting is not resolved by this evidence, and our current
balance should be changed only against our own measurements.

**The reranker claim I most wanted died 0-3.** "Hybrid + cross-encoder is the
highest-yield single lever at this scale, Recall@5 0.816" was the cleanest justification
for adding a cross-encoder. What survives is only the *mechanistic* case (confirmed #5) —
that rerankers escape the dimension bound — demonstrated with a frontier LLM, not
something that runs on our CPU budget.

## Unverified — the session limit ran out here

Presented labelled, not as recommendations. These are precisely the four we most needed,
which is worth stating plainly: the research ran out of budget before verifying the
actionable CPU-cost numbers.

Reranker cost at fixed candidate depth (top-1000 BM25 passages, MS MARCO dev):

| reranker | s/query |
|---|---|
| TILDEv2 | ~0.02 |
| monoT5 (220M) | 4.5 |
| monoBERT (340M) | 15.8 |
| RankLLaMA (7B) | 82.4 |

Accuracy ascends in the same order except TILDEv2, which is fastest and weakest — and its
speed depends on precomputed indexing, so it evaporates for newly added passages. For a
KB that ingests continuously, that caveat is disqualifying if true.
Source (unverified): https://aclanthology.org/2024.emnlp-main.981/

HyDE: 0.318 avg at 1.45 s/query for plain hybrid versus 0.353 at 11.71 s/query with HyDE
— ~8x latency for ~11% relative accuracy. The authors' own "balanced efficiency" recipe
drops HyDE.
Source (unverified): https://aclanthology.org/2024.emnlp-main.981/

Cross-encoder lift, six Arabic QA datasets: bge-reranker-v2-m3 on BGE-M3 retrieval raised
average RAGAS 70.99 → 74.15 (+3.16), best single dataset +6.0. Notable because Arabic is
a non-English, morphologically rich language — the closest proxy in the set for our
Vietnamese case.
Source (unverified): https://arxiv.org/abs/2506.06339

Same study: the lift is conditional. One dataset dropped slightly despite improved
precision, another gained 0.73 — the authors attribute this to rerankers adding little
when first-stage retrieval is already strong. Measure per-corpus rather than assume.
Source (unverified): https://arxiv.org/abs/2506.06339

## What this means for our stack

Grounded in what is actually in the repo today: 384-dim
`paraphrase-multilingual-MiniLM-L12-v2` with a 128-token window, pgvector HNSW
(`ix_article_chunks_embedding_hnsw`, cosine), hybrid vector + keyword, the deterministic
token-coverage reranker in `src/rag/reranker.py`, and an absolute refuse threshold
(`RAG_MIN_RELEVANCE_SCORE = 0.12`, `RAG_MIN_CONTEXT_SCORE = 0.35`).

**Do not change the embedding dimension.** The 384-specific indictment was killed 0-3,
and the dimension ceiling that *was* confirmed sits at ~500k documents for 512d. We are
orders of magnitude clear. A dimension change also costs a full re-embed and an HNSW
rebuild — `20260828_64_realign_embedding_dimension_384.py` is the third dimension
realignment migration in this repo's history. Not worth spending again on refuted
evidence.

**Do not re-weight the hybrid toward keyword search.** This is what I would have
recommended from the unverified extraction. Four separate 0-3 refutations.

**Do not build metadata scoping for accuracy reasons.** Killed 0-3. It may still be worth
building for permissions or UX, but not on this evidence.

**Keep the lexical leg exactly as it is.** Confirmed #3 is the one architectural claim
that survived with a mechanism attached.

**The measurement gap is the real finding.** Confirmed #1, #6 and #7 together say
accuracy will decay as we grow, that index tuning cannot recover it, and that our corpus
type decays worse than the benchmarks. We currently have no way to observe that
happening. A golden-set retrieval metric tracked over corpus growth is the prerequisite
for every other decision here — including whether a reranker is worth its latency, which
the research explicitly failed to answer.

**The strongest actionable lead is unconfirmed:** swapping to a multilingual model
*trained for retrieval with mined hard negatives*, at the same 384 dims and the same CPU
cost. Confirmed #2 isolates training objective as the variable; that it applies to our
specific model is my inference. It needs an A/B against a golden set before it is a plan
— which is the same prerequisite as everything else.

## Open questions this research did not answer

Asked, zero confirmed claims returned:

- HNSW tuning in pgvector at scale (`ef_search`, `m`, `ef_construction`), and when to
  move to IVFFlat. Confirmed #6 says index recall is not the degradation mechanism, which
  lowers the priority but does not answer the question.
- Vietnamese queries against mixed Vietnamese/English technical content. The only
  cross-lingual claim in the set was killed 0-3; the Arabic reranker study is the nearest
  proxy and is unverified.
- Calibrating the refuse-to-answer threshold as the corpus grows. Nothing returned.

## Provenance

6 angles · 27 sources · 135 claims extracted · 25 verified · 7 confirmed · 14 refuted ·
4 unverified. Verification is a 3-vote adversarial panel per claim; each verifier
re-fetches the source independently. The synthesis step never ran — the run exhausted
its budget after the verification phase, so this document is written directly from the
verified claim record rather than from the workflow's own synthesis.
