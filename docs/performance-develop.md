# Why develop search can look stuck: measurements and decisions

A partner reported that the API is slow, or looks stuck, "every time". **That report turned out to be
the partner's local docker-compose**: api and worker had `mem_limit: 2g` against a ~1.5 GiB in-process
model (OOM, exit 137), fixed by raising it to 4g in #196. Nothing below is the cause of that report.
The measurements were taken first, against develop's shape (the api task averages 0.5% CPU with peaks
to 99% on 2 vCPU shared with clamd, memory peaks at 40% of 8 GB, the model is
`multilingual-e5-large-instruct`, develop scales to zero and stops RDS outside working hours), and are
kept because they answer what the model, the reranker and the sizing actually cost.

This page records what was measured, what was decided, and, as importantly, **what was not measured**.

## How it was measured

`scripts/bench_inference.py` runs the repository's own code (`src.lib.embeddings`, `src.rag.reranker`,
`src.rag.cross_encoder`, `src.rag.chunker`) on the real model. `.github/workflows/bench-inference.yml`
runs it manually on a GitHub-hosted x86 runner (Intel Xeon Platinum 8370C, 2 cores x 2 threads,
AVX-512), one process per configuration, pinned with `taskset`. Reproduce with `workflow_dispatch`.

Read the numbers with these limits in mind:

- It is **not Fargate**. Same architecture and a vCPU is a hardware thread on both (assumption for
  Fargate: not verified against AWS documentation here), but Fargate Spot draws from a mixed fleet,
  and clamd's CPU use inside the task is not reproduced.
- Runner noise is real: the same configuration moved 224 -> 291 ms between two runs. Ranges are given
  where there are two runs.
- Model files had just been downloaded, so **page cache was warm**. "Model load" below is therefore a
  best case, not a cold start.
- The corpus is public Vietnamese Wikipedia introductions chunked by the real chunker (207-439
  chunks, ~220 characters each), not your documents. Queries are phrases from a chunk with the title
  words removed. This ranks configurations against each other. It is **not** a retrieval-quality
  score for your content, and it is easy: the top 10 already contains the answer 100% of the time.
- The database numbers use **synthetic random vectors** in a `pgvector/pgvector:pg16` container on the
  runner, not `db.t4g.small` and not your data.

## Results

Query embedding (one e5-large-instruct forward pass over a short query), milliseconds:

| CPUs available to the process | threads | p50 | p95 | note |
|---|---|---|---|---|
| 1 | 1 | 242 - 296 | 265 - 337 | |
| 2 = the two hyper-threads of **one** core (today's api) | 2 | 224 - 291 | 247 - 327 | no faster than 1 thread |
| 3 | 3 | 159 - 200 | 175 - 231 | |
| 4 (two cores), 2 threads | 2 | 130 - 156 | 143 - 180 | |
| 4 (two cores), 4 threads | 4 | 156 | 191 | no gain over 2 threads |

Everything else on the request path that this benchmark can see:

| Step | Result |
|---|---|
| Model load, warm page cache | 1.4 - 1.5 s, RSS 1.53 GB (matches the 1,569 MB quoted in `infra/live/develop/main.tf`) |
| Lexical rerank (the always-on reranker), pool 240 | 21 - 22 ms; pool 48: 4 ms; pool 120: 11 ms |
| pgvector `ORDER BY <=> LIMIT 240`, 30,000 synthetic rows, `ef_search` 40 / 200 / 1000 | 12 / 11 / 11 ms |
| Whole search without a database or an LLM, 2 vCPU | 175 - 315 ms (embedding is ~85 - 95% of it) |
| Cross-encoder `bge-reranker-v2-m3`, load | 5 - 23 s, +1.1 GB RSS |
| Cross-encoder, pool 10 / 30 on 2 vCPU | 5.2 - 8.5 s / 15.5 - 25 s |
| Cross-encoder, pool 10 / 30 on 4 vCPU | 4.7 - 5.3 s / 14.4 - 15.3 s |

### What the numbers say about "stuck"

- **Embedding the query is 0.13 - 0.29 s.** It is a tax, not a stall. It cannot by itself make a
  search look stuck.
- **The lexical rerank is 21 ms** at the full pool, so capping its pool buys nothing. Pools of 48, 80,
  120 and 240 ranked identically on the quality proxy, so there is also nothing to lose by keeping 80.
  `RAG_RERANK_LIMIT` and `RAG_CANDIDATE_POOL_SIZE` are left alone.
- **The cross-encoder is the one component that *can* make a search look stuck.** On CPU it costs
  ~0.5 s per passage at 2 vCPU: 15 - 25 s for the default pool of 30. It is off by default
  (`CROSS_ENCODER_RERANKER_ENABLED=False`), nothing in `infra/` sets it, and it fires only for
  comparison queries or queries with two or more question marks. **Whether it is on in develop is not
  verified** (no AWS access from this change; it could be set through the secrets bundle or a feature
  flag). If it is, that alone explains a stuck search on exactly those queries.
- **Not explained, because not measured:** cold start after scale-from-zero (pulling a ~4 GB image on
  Fargate, reading a 2.2 GB model from a cold disk), RDS start time and `db.t4g.small` CPU-credit
  exhaustion, the LLM call in `/ai/ask`, and clamd's own startup (`startPeriod` 300 s). These are the
  remaining suspects. `scripts/performance_probe.py` against develop, plus the RDS `CPUCreditBalance`
  metric, would settle most of them.

## Decisions in this change

1. **`/health/ready` waits for the model** (`src/core/warmup.py`, `src/api/main.py`). 503 with
   `{"embedding": {"state": "warming"}}` until the load ends; log lines on start, finish (with
   seconds), failure and timeout. `EMBEDDING_WARMUP_TIMEOUT_SECONDS` (240) bounds the wait. A failed or
   timed-out load stops holding the probe and reports `degraded: true`, because keyword search works
   without the model and holding it forever would fail every deploy. Liveness (`/health/live`,
   `/livez`) is untouched.
   - **This does not hold user traffic in develop.** Ingress is a Cloudflare Tunnel with no load
     balancer (`attach_alb = false`), so nothing routes on `/health/ready`; it is for orchestrators such
     as Kubernetes. A request that arrives during the load already waits on the `Lazy` lock for the rest
     of the load rather than failing; that is unchanged.
2. **Cross-encoder bounded** (`CROSS_ENCODER_TIMEOUT_SECONDS`, default 3 s): a search that does not
   get scores in time keeps the lexical order it already has, and only one scoring call runs at a time
   so an abandoned one cannot be stacked on. `CROSS_ENCODER_THREADS` sets torch's pool. With the
   measured costs the timeout will trip at every pool size tested, so on CPU the feature is
   effectively off; that is the correct outcome until it runs on a GPU. `CROSS_ENCODER_CANDIDATE_POOL`
   is unchanged at 30; if it is ever enabled on a CPU, 8 or fewer (unmeasured) is the starting point.
3. **The deploy polls `/health/deps`, not `/health/ready`.** `/health/deps` is the old readiness
   (database and Redis, nothing about the model). The shared `ci` deploy workflow polls for a fixed
   120 s, and a cold model load on Fargate has not been measured to fit in that, so a slow warm-up must
   not be able to fail a deploy. It is not liveness: it still fails when the database or Redis is gone.
   **Measure the cold load on the first develop deploy**: the api logs
   `Embedding model warm-up finished` with `seconds`. If that is comfortably under 120 s, the deploy can
   go back to `/health/ready`.

4. **Follow-ups to the above.**
   - A load that outlasts `EMBEDDING_WARMUP_TIMEOUT_SECONDS` is reported `timeout` (degraded, no longer
     holding readiness), and when it really ends it becomes `ready` or `failed`: the verdict follows the
     load instead of sticking. Only running out of patience is a `timeout`; an error raised by the load
     (a socket timeout while downloading) is a `failed`.
   - A CloudWatch metric filter and alarm (`<name>-api-embedding-model-unhealthy`) on the api log lines
     `Embedding model preload failed` and `Embedding model warm-up timed out`, publishing to the stack's
     alarm topic. Not tested against a live log group: after the first apply, check it with
     `aws logs test-metric-filter` or provoke a failure in develop.
   - The cross-encoder has a circuit breaker: after `CROSS_ENCODER_BREAKER_THRESHOLD` (3) consecutive
     timeouts it is skipped for `CROSS_ENCODER_BREAKER_COOLDOWN_SECONDS` (300), then one search probes it.
     When it is enabled, startup times a few pairs and warns if `CROSS_ENCODER_CANDIDATE_POOL` pairs
     cannot fit the timeout (the sample is short, so a warning is reliable and silence is not a promise).

### Considered and not done

- **api 2048 -> 4096 CPU** (with `EMBEDDING_ONNX_THREADS` = 2). Measured gain ~100 - 135 ms per
  search, because 2 vCPU is one core. Judged not worth the cost now. The measurements above are there
  if it is revisited.
- **Stop at 19:00 and wake Monday to Saturday.** The 19:00 stop was deliberately reverted on 2026-08-19.
  The schedule is unchanged.

### Is the cross-encoder enabled in develop?

From the infrastructure code: **no**. `infra/` sets no `CROSS_ENCODER_*` variable anywhere, the task
environment is a fixed list in `infra/modules/stack/main.tf` (`common_env`), and the injected secrets
are a fixed list of eight names, none of them a CROSS_ENCODER setting. The application default is
`CROSS_ENCODER_RERANKER_ENABLED=False`, and the feature flag `rag.cross_encoder_reranker` can only
narrow that further. This is a reading of the code, **not** of the deployed task definition (no AWS
credentials here); `aws ecs describe-task-definition` on the develop api would confirm it.

## Is e5-large on CPU viable on-prem, or is a GPU / hosted embedding the long-term answer?

Evidence: the on-prem k3s VM in `ONPREM-SERVER-SETUP-RUNBOOK.md` is 28 vCPU / 56 GB with an RTX
5060 Ti and a ~15 GB VRAM budget planned for Ollama and TEI.

- **Query embedding on CPU is viable.** 0.13 - 0.29 s for one query, 1.5 GB RAM, on 2 to 4 vCPU. A
  28-vCPU host has far more than that. This is not what to move to a GPU for.
- **Bulk indexing is where CPU hurts.** The 439-chunk run took 87 - 158 s to embed at 2 threads, i.e.
  0.2 - 0.36 s per chunk; 10,000 chunks is roughly 35 - 60 minutes. A GPU would be much faster, but that
  speed-up is **not measured here**.
- **The cross-encoder is not viable on CPU at any pool tested** (4.7 - 25 s). It needs a GPU or should
  stay off.
- **Recommendation:** keep local ONNX for query embedding. If a reranker or faster indexing is wanted,
  serve them from a GPU on-prem through TEI. That needs **a new provider**: `src/lib/embeddings/hosted.py`
  speaks only Google's `batchEmbedContents`, not TEI or the OpenAI-compatible shape, so "a hosted
  provider already exists" does not cover this.
- **The hosted provider that does exist is not a drop-in swap.** It sends document text to a third
  party, which is exactly what the local runtime was chosen to avoid. It also changes the vector space,
  so every chunk must be re-embedded, and its default 3072 dimensions do not match the 1024-wide
  column (`EMBEDDING_DIMENSION`). The int8 route was already rejected by the repo's own parity gate on
  bge-m3, and that gate has not been re-run for e5-large.

## Not verified

- Anything on real Fargate: absolute latencies, cold start, image pull, clamd contention, Spot variance.
- Whether the cross-encoder is enabled in develop (no AWS credentials were available).
- RDS: start time, CPU credits, and real query plans on real data.
- End-to-end `/api/v1/search` and `/ai/ask` latency, including the LLM.
- Retrieval quality on your documents. `evaluation/golden.json` has six scorer fixtures, not a labelled
  retrieval set, and there is no corpus in the repository. The Wikipedia proxy saturates (top-10 recall
  100%) and cannot separate the configurations; the cross-encoder rows cover only 10 queries.
- How long a cold model load takes on Fargate. The deploy no longer depends on it (it polls
  `/health/deps`); read it from the first develop deploy's `Embedding model warm-up finished` log line.
