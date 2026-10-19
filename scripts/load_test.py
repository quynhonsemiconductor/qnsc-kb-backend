"""Small dependency-free async smoke/load test for a running API."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def run(base_url: str, concurrency: int, requests: int) -> dict[str, float | int]:
    semaphore = asyncio.Semaphore(concurrency)
    durations: list[float] = []
    failures = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:

        async def one() -> None:
            nonlocal failures
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.get("/health/live")
                    if response.status_code != 200:
                        failures += 1
                except httpx.HTTPError:
                    failures += 1
                durations.append((time.perf_counter() - started) * 1000)

        await asyncio.gather(*(one() for _ in range(requests)))

    ordered = sorted(durations)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)] if ordered else 0
    return {
        "requests": requests,
        "failures": failures,
        "p95_ms": round(p95, 2),
        "average_ms": round(statistics.fmean(durations), 2) if durations else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--max-p95-ms", type=float, default=1000)
    args = parser.parse_args()
    report = asyncio.run(run(args.base_url, args.concurrency, args.requests))
    print(report)
    return 1 if report["failures"] or report["p95_ms"] > args.max_p95_ms else 0


if __name__ == "__main__":
    raise SystemExit(main())
