"""How much memory N Celery prefork children cost when each loads the embedding model.

Celery's prefork pool forks N children; each one loads its OWN ONNX session the first time it embeds
(`Lazy` in src/lib/embeddings/local_onnx.py), because the parent never does. The question for
`--concurrency N` is whether the task's memory covers N sessions. Counting RSS answers it wrongly if the
model's weights are memory-mapped: file-backed pages are shared through the page cache, so the container's
memory (cgroup `memory.current`) can grow by far less than N x RSS. So this reports both, and PSS.

Run on Linux (cgroup v2), in a container limited like the task:

    docker run --rm -m 6g -v $PWD:/app -v /tmp/e5onnx:/opt/embedding-onnx python:3.13-slim \
        bash -c 'pip install -q -r /tmp/req && PYTHONPATH=/app python /app/scripts/bench_worker_memory.py 1 2 3'

Measures the embedding session only. PaddleOCR (loaded per file by the worker) and the interpreter's own
baseline are NOT in these numbers, and that is the biggest unknown for a real `--concurrency 2`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def pss_rss_mb(pid: int) -> tuple[float, float]:
    pss = rss = 0
    for line in _read(f"/proc/{pid}/smaps_rollup").splitlines():
        if line.startswith("Pss:"):
            pss = int(line.split()[1])
        elif line.startswith("Rss:"):
            rss = int(line.split()[1])
    return round(pss / 1024, 1), round(rss / 1024, 1)


def cgroup_mb() -> dict[str, float]:
    out = {"memory.current": round(int(_read("/sys/fs/cgroup/memory.current") or 0) / 2**20, 1)}
    for line in _read("/sys/fs/cgroup/memory.stat").splitlines():
        key, _, value = line.partition(" ")
        if key in ("anon", "file", "file_mapped", "shmem"):
            out[key] = round(int(value) / 2**20, 1)
    return out


def child(ready, go) -> None:
    from src.lib.embeddings import get_bge_embedding, warm_up

    started = time.perf_counter()
    warm_up()
    get_bge_embedding("một câu hỏi ngắn để chạy thử mô hình")
    ready.put((os.getpid(), round(time.perf_counter() - started, 2)))
    go.wait()


def run(n: int) -> None:
    ctx = mp.get_context("fork")  # what billiard does: the parent has NOT loaded the model
    base = cgroup_mb()
    ready, go = ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=child, args=(ready, go)) for _ in range(n)]
    for p in procs:
        p.start()
    infos = [ready.get(timeout=600) for _ in procs]
    time.sleep(1)
    after = cgroup_mb()
    print(f"\n== {n} child(ren), each loaded the model and embedded one query")
    for pid, seconds in infos:
        pss, rss = pss_rss_mb(pid)
        print(f"   pid {pid}: load+first query {seconds}s  RSS {rss} MB  PSS {pss} MB")
    grew = round(after["memory.current"] - base["memory.current"], 1)
    print(f"   cgroup memory.current grew {grew} MB   (anon {after.get('anon')} MB, file_mapped {after.get('file_mapped')} MB)")
    print(f"   sum of RSS {round(sum(pss_rss_mb(pid)[1] for pid, _ in infos), 1)} MB, sum of PSS {round(sum(pss_rss_mb(pid)[0] for pid, _ in infos), 1)} MB")
    go.set()
    for p in procs:
        p.join()


if __name__ == "__main__":
    os.environ.setdefault("EMBEDDING_ONNX_DIR", "/opt/embedding-onnx")
    for count in (int(arg) for arg in sys.argv[1:] or ["1", "2"]):
        run(count)
