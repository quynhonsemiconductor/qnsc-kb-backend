"""Make the digest of every published document current, within a budget.

Digests (src/domain/digest.py) are made when a document is published or edited. Documents that
were published before the feature existed have none, and this script fills them in. It is meant
to be run by a person, once, and again whenever DIGEST_PROMPT_VERSION is raised.

DRY RUN BY DEFAULT. Without --apply it reads the collection, prints how many digests are
missing and what generating them would roughly cost, and makes no model call and no write:

    python scripts/backfill_digests.py
    python scripts/backfill_digests.py --company acme.test --limit 50

--apply generates them. It stops starting new work at --max-calls model calls or --max-tokens
tokens, and is safe to stop and run again: a digest is keyed by the hash of its document, so a
finished one is skipped at no cost. Run it where the application and its database credentials
are (the same one-off task pattern as scripts/create_admin.py):

    python scripts/backfill_digests.py --apply --max-calls 200 --max-tokens 1500000

--retry-failed retries documents that failed, ignoring the attempt limit and the backoff.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.core.config import settings
from src.domain.digest import (
    BackfillReport,
    plan_backfill,
    report_json,
    run_backfill,
)


def _print_plan(report: BackfillReport, *, applying: bool) -> None:
    print(f"scanned:             {report.scanned}")
    print(f"already current:     {report.current}")
    print(
        f"to generate:         {report.to_generate} (of which retrying failed: {report.retry_failed})"
    )
    print(f"embedding only:      {report.to_embed} (local, no model call)")
    print(
        f"skipped by policy:   {report.policy_skipped} (above DIGEST_MAX_SENSITIVITY)"
    )
    print(f"model calls needed:  {report.estimated_calls}")
    print(
        "tokens (rough):      "
        f"~{report.estimated_input_tokens:,} in, ~{report.estimated_output_tokens:,} out"
    )
    if not applying:
        print("\nDry run: nothing was called or written. Add --apply to generate.")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--apply", action="store_true", help="generate (default: dry run)"
    )
    parser.add_argument("--company", help="only this company_domain")
    parser.add_argument(
        "--limit", type=int, help="at most this many documents that need work"
    )
    parser.add_argument(
        "--max-calls", type=int, help="stop after this many model calls"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="stop after this many tokens used"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=settings.LLM_MAX_CONCURRENCY,
        help="documents in flight at once (default: LLM_MAX_CONCURRENCY)",
    )
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if not settings.DIGEST_ENABLED:
        print("DIGEST_ENABLED is false; nothing to do.", file=sys.stderr)
        return 1

    pending, report = await plan_backfill(
        company=args.company, limit=args.limit, retry_failed=args.retry_failed
    )
    _print_plan(report, applying=args.apply)
    if not args.apply or not pending:
        return 0

    done = 0

    async def progress(current: BackfillReport) -> None:
        nonlocal done
        done += 1
        if done % 20 == 0:
            print(
                f"  {done}/{len(pending)}  generated={current.generated} "
                f"failed={current.failed} tokens={current.tokens_used:,}",
                flush=True,
            )

    await run_backfill(
        pending,
        report,
        max_calls=args.max_calls,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        retry_failed=args.retry_failed,
        on_progress=progress,
    )
    print("\n" + report_json(report))
    if report.stopped_by_budget:
        print(
            "Stopped at the budget. Run again to continue; finished documents cost nothing."
        )
    return 0 if not report.failed else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
