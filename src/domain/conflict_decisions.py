"""A reviewer's choice of the right article in a conflict, applied when answering.

Two articles can state incompatible facts. A reviewer marks the conflict resolved and names
the article that is right (`ConflictRecord.kept_article_id`); from then on an answer must not
quote the other one. Without this a resolved conflict only closed a ticket: the next question
still retrieved both articles and still said "the knowledge base contradicts itself".

The loser is dropped only when the winner is in the same context. The winner being there is
what shows the question touches the conflicting fact; an article that lost one disagreement
may still be the only source for something unrelated.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.governance import ConflictRecord

logger = structlog.get_logger()

#: Resolved decisions read per answer. A tenant with more has bigger problems than this cap.
MAX_DECISIONS = 2000


def superseded_article_ids(
    decisions: Iterable[tuple[Iterable[str], str]], present: set[str]
) -> set[str]:
    """Articles to drop, given `(article_ids, kept_article_id)` decisions and what is present."""
    dropped: set[str] = set()
    for article_ids, kept in decisions:
        kept = str(kept)
        if kept not in present:
            continue
        for article_id in map(str, article_ids):
            if article_id != kept and article_id in present:
                dropped.add(article_id)
    return dropped


def drop_superseded(
    results: list[dict[str, Any]], dropped: set[str]
) -> list[dict[str, Any]]:
    if not dropped:
        return results
    return [item for item in results if str(item.get("article_id")) not in dropped]


async def load_decisions(
    db: AsyncSession, company_domain: str
) -> list[tuple[list[str], str]]:
    rows = await db.execute(
        select(ConflictRecord.article_ids, ConflictRecord.kept_article_id)
        .where(
            ConflictRecord.company_domain == company_domain,
            ConflictRecord.status == "resolved",
            ConflictRecord.kept_article_id.is_not(None),
        )
        .limit(MAX_DECISIONS)
    )
    return [(list(ids or []), str(kept)) for ids, kept in rows.all() if kept]


async def apply_conflict_decisions(
    db: AsyncSession, company_domain: str, results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Drop the passages of articles a reviewer decided are superseded.

    If the decisions cannot be read the results are returned unchanged: the answer then
    falls back to the open-conflict check, which flags the disagreement instead of quoting
    one side as settled.
    """
    present = {
        str(item.get("article_id")) for item in results if item.get("article_id")
    }
    if len(present) < 2:
        return results
    try:
        decisions = await load_decisions(db, company_domain)
    except Exception as exc:
        logger.warning("Conflict decisions could not be read", error=str(exc))
        return results
    dropped = superseded_article_ids(decisions, present)
    if dropped:
        logger.info("Conflict decisions applied", dropped_articles=len(dropped))
    return drop_superseded(results, dropped)
