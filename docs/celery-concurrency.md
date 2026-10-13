# Celery worker: concurrency 1, what it costs, and what to change

Observed on develop after the worker was restored: one `ForkPoolWorker` (`--concurrency=1` in the `worker`
Dockerfile target), and a SharePoint sync takes about 141 s, so a backlog drains slowly and one slow sync blocks
everything else queued behind it, because the single process consumes every queue
(`document-preparation, celery, ingestion, connectors, permissions`).

## Why it is 1

The Dockerfile comment records the incident: with two children, six articles died with `signal 9 (SIGKILL)` and
stayed `index_status="processing"` forever, because each prefork child loads **its own** embedding session and the
task then had ~1.8 GB for the worker container. The worker task is 4 vCPU / 8 GB now (clamav holds 2048 MB and beat
256 MB as hard container limits, leaving about 5.9 GB for the worker container), so the question is whether N
sessions fit.

## Measured: memory per prefork child

`scripts/bench_worker_memory.py` forks N children from a parent that has not loaded the model (what billiard does),
each loads the real ONNX session through `src.lib.embeddings` and embeds one query. Linux container
(`python:3.13-slim`, `-m 6g`, cgroup v2, **arm64 on a Mac under Docker**, `EMBEDDING_ONNX_THREADS=2`),
multilingual-e5-large-instruct:

| children | RSS each | PSS each | cgroup `anon` after | load + first query |
|---|---|---|---|---|
| 1 | 1,531 MB | 1,527 MB | 1,493 MB | 2.9 s |
| 2 | 1,531 MB | 1,505 MB | 2,975 MB | 7.6 s |
| 3 | 1,511 MB | 1,492 MB | 4,458 MB | 6.4 s |

- **The weights are not shared.** PSS is almost equal to RSS and `anon` grows by ~1.49 GB per child: the external
  data is read into private memory, not memory-mapped. Memory is linear, N x 1.5 GB, with no discount for forking.
  (`memory.current` is larger still because it includes page cache; `anon` is the number that matters for an OOM
  kill.)
- The 1-child cold load of 2.9 s matches the 2.89 / 3.93 / 3.42 s measured on develop Fargate.

### What that means for the 5.9 GB

| `--concurrency` | embedding sessions | + interpreter etc. (~0.3 GB each, **estimate**) | left for OCR and everything else |
|---|---|---|---|
| 1 | 1.5 GB | 0.3 GB | ~4.1 GB |
| 2 | 3.0 GB | 0.6 GB | ~2.3 GB |
| 3 | 4.5 GB | 0.9 GB | ~0.5 GB |

Not measured, and the deciding term: **PaddleOCR**, loaded per scanned file inside the same child (the Dockerfile
comment sizes a spike for it), and the worker's real `MemoryUtilization` on develop (no AWS access when this was
written). Until those two numbers exist, **`--concurrency 2` is arithmetically possible and not shown to be safe,
and `3` is not**. The failure mode is the severe one above (SIGKILL, rows stuck `processing`), so it should not be
tried on a guess.

## What does not need a second model session

`sync_cloud_connector` (`src/domain/cloud_sync.py`) does not embed or index: it walks the drive, downloads, extracts
text, creates drafts and enqueues the follow-on work, which lands on `ingestion` (embeddings) and
`document-preparation` (LLM formatting). The embedding session is only ever needed by tasks on `ingestion`. So the
long sync (141 s of Graph I/O and extraction) is blocking the embedding queue without needing its memory.

## Proposal (not applied; it is an infrastructure change)

Split consumption by queue, so a sync cannot block embeddings or draft formatting and the model is loaded in one
process only:

- `worker-embed`: `-Q ingestion --concurrency=1` (unchanged cost: one session).
- `worker-io`: `-Q connectors,document-preparation,permissions,celery --concurrency=3` (or `--pool=threads`):
  I/O and LLM bound, no embedding session. Rate-limit waits (see `LLM_MAX_CONCURRENCY`) are sleeps in this
  process, which is the reason to give them their own consumers instead of the embedding one.

As two `celery worker` processes in the worker task (a second container with its own memory limit, carved out of the
~5.9 GB), or as a second ECS service if beat stays a singleton in the first. **Open question that must be answered
first:** does anything on `connectors`, `document-preparation` or `celery` call the embedder or OCR in-process? The
read of `cloud_sync.py` says the sync does not embed; `source_extraction.py` (OCR) is reachable from extraction and
was not traced. If it is, `worker-io` children load Paddle and need their own sizing, and concurrency 3 becomes 2.

Acceptance for whoever applies it: `MemoryUtilization` of the worker service under a real backlog, and no
`WorkerLostError` in the logs, before raising the concurrency of either process.

## Not measured

x86 Fargate memory (the numbers above are arm64 under Docker; the ONNX weights are the same size, the runtime's
arena behaviour may differ a little), OCR, the real queue depths, and how many of the 141 s are Graph latency
versus extraction.
