# Why develop search can look stuck: measurements and decisions

A partner reported that the develop API is slow, or looks stuck, "every time". The evidence on the
table: the api task averages 0.5% CPU with peaks to 99% on 2 vCPU shared with clamd, memory peaks at
40% of 8 GB, the model is `multilingual-e5-large-instruct`, and develop scales to zero and stops RDS
outside working hours.

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
     balancer (`attach_alb = false`), so nothing routes on `/health/ready`. It is used by the deploy
     pipeline's post-deploy poll (`ci` `post-deploy-health-check`, default timeout 120 s) and by
     Kubernetes. A request that arrives during the load already waits on the `Lazy` lock for the rest
     of the load rather than failing; that is unchanged.
   - Deploy risk: if a cold load on Fargate exceeds the pipeline's 120 s poll, a deploy that used to
     pass will fail on readiness. The warm-cache load is 1.5 s; the cold one is unmeasured.
2. **api 2048 -> 4096 CPU, memory 8192 (unchanged), `EMBEDDING_ONNX_THREADS` and
   `CROSS_ENCODER_THREADS` = 2 for the api** (new module variable `api_inference_threads`; the worker
   keeps the application default). 2 vCPU is one core: the second thread bought nothing. Four vCPU
   gives ~100 - 135 ms per search back and leaves two vCPU for clamd, cloudflared and the event loop.
   Two threads rather than four because the second pair measured no faster. **This is a modest win,
   not a fix for a multi-second stall.** Memory stays 8192 because 4096 CPU requires at least that on
   Fargate, not because it is needed (observed peak ~40%).
3. **Cross-encoder bounded** (`CROSS_ENCODER_TIMEOUT_SECONDS`, default 3 s): a search that does not
   get scores in time keeps the lexical order it already has, and only one scoring call runs at a time
   so an abandoned one cannot be stacked on. With the measured costs the timeout will trip at every pool
   size tested, so on CPU the feature is effectively off; that is the correct outcome until it runs on
   a GPU. `CROSS_ENCODER_CANDIDATE_POOL` is unchanged at 30; if it is ever enabled on a CPU, 8 or fewer
   (unmeasured) is the starting point, not 30.
4. **Schedule**: stop passes `cron(0 19,22,2 * * ? *)`, wake `cron(0 8 ? * MON-SAT *)`, both
   Asia/Ho_Chi_Minh. The "api min 1 task during hours, 0 outside" is the scheduler's desired count; the
   service's `min_count` stays 0 and autoscaling stays off, because target tracking cannot scale from
   zero (see the validation in `infra/modules/stack/variables.tf`). The 19:00 stop is the one reverted
   on 2026-08-19 for cutting the evening short; there is still no one-click wake workflow, so after
   19:00 develop comes back only through a deploy. 22:00 and 02:00 are catch-up passes for a deploy that
   wakes it late.

### Cost delta (the schedule and the resize together)

Hours per week, from the schedules: before, 08:00 - 24:00 on 5 weekdays = **80 h**; after,
08:00 - 19:00 on 6 days = **66 h**. Late-deploy wake-ups are not counted in either figure.

| | before | after | delta |
|---|---|---|---|
| api (vCPU / GB) | 2 / 8 | 4 / 8 | |
| worker (vCPU / GB) | 4 / 8 | 4 / 8 | |
| vCPU-hours / week | (2+4) x 80 = 480 | (4+4) x 66 = 528 | +48 |
| GB-hours / week | (8+8) x 80 = 1,280 | (8+8) x 66 = 1,056 | -224 |
| RDS `db.t4g.small` hours / week | 80 | 66 | -14 |

Unit prices below are **assumptions to be replaced** (ap-southeast-1, Linux x86; check the AWS Pricing
Calculator, and note Fargate Spot prices vary): $0.0506 per vCPU-hour, $0.0055 per GB-hour on-demand,
Spot at 30% of that, RDS $0.037 per hour. Per week, on-demand: before 480 x 0.0506 + 1,280 x 0.0055 =
$31.3, after 528 x 0.0506 + 1,056 x 0.0055 = $32.5 (+$1.2). Spot: +$0.4. RDS: -14 x 0.037 = -$0.52.
Times 4.33 weeks, the net is about **+$3 / month on-demand and about -$0.7 / month on Spot**. In
other words roughly neutral: the shorter day pays for the larger api. Storage, data transfer, NAT and
the second develop environment are not included. Earlier repo comments put 112 -> 55 h/week at
$18.59 / month across two develop environments, which is a different baseline.

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
- Whether a cold model load fits inside the deploy pipeline's 120 s readiness poll.
