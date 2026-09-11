"""Proactive contradiction detection, run at approval time rather than answer time.

`ai_service._detect_explicit_conflicts` already does the real detection work --
comparing clearly labelled facts ("effective date:", "deadline:", ...) across distinct
articles -- but it only ever runs over whatever pool a search happened to retrieve for
one user's question. A contradiction between two documents nobody has asked about
together yet sits undetected indefinitely.

This module runs the SAME detection at draft-approval time instead, against the
published articles most similar to what is being approved. Deliberately NON-BLOCKING: it
records a `ConflictRecord` for governance to review on the Coverage page, identical in
shape to the reactive path, but never refuses the approval itself. A hard block needs a
proven-safe way for a reviewer to acknowledge and override a false positive -- pattern
matching, not an LLM, decides what counts as a conflict here, and a pattern match on
formatting coincidence (two SOPs that both happen to use "Owner: ..." for unrelated
things) is a real failure mode a live approval queue cannot absorb without one. Recording
without blocking gets the same visibility with none of that risk.
"""
from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.similarity import find_similar_documents
from src.models.article import Article
from src.models.governance import ConflictRecord
from src.models.user import User

logger = structlog.get_logger()

#: How many of the most-similar published articles are worth an explicit-fact
#: comparison. Kept small: this runs synchronously right after a draft publishes, and
#: each candidate costs one more DB fetch plus regex work, not an LLM call, but the
#: fetches still add up.
CANDIDATE_LIMIT = 3

#: similarity.py's own MATCH_THRESHOLD (0.25) is tuned for near-duplicate detection, a
#: stricter question than this one -- two articles can share almost no wording and still
#: both carry a "Deadline: ..." line worth comparing -- so a lower bar is used here.
MIN_SIMILARITY = 0.12


async def detect_contradictions_for_draft(
    db: AsyncSession,
    user: User,
    draft_body_md: str,
    *,
    exclude_article_id: str | None = None,
) -> list[dict[str, Any]]:
    """Compare `draft_body_md` against similar published articles for conflicting facts.

    Returns the conflicts found (same shape as `ai_service._detect_explicit_conflicts`)
    and records each new one as an open `ConflictRecord`, deduplicated by fact text the
    same way the reactive path already is. Swallows its own failures and logs them --
    see the module docstring for why a broken check must not touch approval at all.
    """
    from src.domain.ai_service import _detect_explicit_conflicts

    try:
        matches = await find_similar_documents(db, user, draft_body_md)
    except Exception as exc:
        logger.warning(
            "Proactive contradiction check could not run similarity search",
            error=str(exc),
        )
        return []

    candidates = [
        match
        for match in matches
        if str(match.get("article_id")) != str(exclude_article_id)
        and (match.get("score") or 0) >= MIN_SIMILARITY
    ][:CANDIDATE_LIMIT]
    if not candidates:
        return []

    results: list[dict[str, Any]] = [
        {
            "article_id": exclude_article_id or "draft",
            "context_text": draft_body_md,
            "title": "(this draft)",
        }
    ]
    for match in candidates:
        article = await db.get(Article, match["article_id"])
        if article and article.body_md:
            results.append(
                {
                    "article_id": str(article.id),
                    "context_text": article.body_md,
                    "title": article.title,
                }
            )
    if len(results) < 2:
        return []

    conflicts = _detect_explicit_conflicts(results)
    if not conflicts:
        return []

    for conflict in conflicts:
        article_ids = sorted(
            {
                str(entry.get("article_id"))
                for entry in conflict.get("entries", [])
                if entry.get("article_id")
            }
        )
        if len(article_ids) < 2:
            continue
        existing = await db.scalar(
            select(ConflictRecord).where(
                ConflictRecord.company_domain == user.company_domain,
                ConflictRecord.fact == str(conflict["fact"])[:255],
                ConflictRecord.status == "open",
            )
        )
        if existing:
            continue
        db.add(
            ConflictRecord(
                company_domain=user.company_domain,
                fact=str(conflict["fact"])[:255],
                article_ids=article_ids,
                evidence=[
                    {
                        "article_id": str(entry.get("article_id")),
                        "title": (entry.get("source") or {}).get("title"),
                        "value": entry.get("value"),
                    }
                    for entry in conflict.get("entries", [])
                ],
            )
        )
        logger.info(
            "Proactive contradiction check recorded a conflict",
            fact=str(conflict["fact"])[:255],
            article_ids=article_ids,
        )
    return conflicts
