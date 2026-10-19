"""Check deployment-visible security defaults on a running API."""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx


async def check(base_url: str) -> list[str]:
    failures: list[str] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        live = await client.get("/health/live")
        if live.status_code != 200:
            failures.append("live health endpoint is unavailable")
        docs = await client.get("/docs")
        openapi = await client.get("/api/v1/openapi.json")
        if docs.status_code == 200 or openapi.status_code == 200:
            failures.append("production API documentation is publicly enabled")
        metrics = await client.get("/metrics")
        if metrics.status_code != 200 or "qnsc_http_requests_total" not in metrics.text:
            failures.append("metrics endpoint is unavailable")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    failures = asyncio.run(check(args.base_url))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("security smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
