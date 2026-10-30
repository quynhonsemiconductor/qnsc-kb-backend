"""Put the articles that were in force on a given date first.

A question such as "what was the leave policy in March 2025?" is about one moment, and a
knowledge base usually holds several versions of the same rule. The planner (one call it
already makes) reports the date the question is about as `as_of`; this reorders the
retrieved passages so the articles whose validity window contains that date come first,
then those with no dates, then those that were not yet issued or had already expired.

Nothing is dropped: a reorder cannot hide the only article that answers the question, it
only changes which passages fit in the context first. With no date in the question, or no
article dates, the order is exactly what retrieval produced. The dates come from
`Article.structured_metadata` (`issue_date`, `expiry_date`), which the model fills in after a
publish only when the document states them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.article import Article

logger = structlog.get_logger()

IN_FORCE, UNDATED, OUT_OF_FORCE = 0, 1, 2


@dataclass(frozen=True)
class Validity:
    issue: date | None = None
    expiry: date | None = None


def _as_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def parse_validity(metadata: Any) -> Validity:
    if not isinstance(metadata, dict):
        return Validity()
    return Validity(
        issue=_as_date(metadata.get("issue_date")),
        expiry=_as_date(metadata.get("expiry_date")),
    )


def parse_as_of(value: Any) -> date | None:
    """The planner's date, or None when it is missing or not a plausible ISO date."""
    parsed = _as_date(value) if value else None
    if parsed is None or not (1990 <= parsed.year <= date.today().year + 1):
        return None
    return parsed


def bucket(validity: Validity, as_of: date) -> int:
    """0 in force on `as_of`, 1 nothing to judge by, 2 not in force then."""
    if validity.issue is None and validity.expiry is None:
        return UNDATED
    if validity.issue is not None and validity.issue > as_of:
        return OUT_OF_FORCE
    if validity.expiry is not None and validity.expiry < as_of:
        return OUT_OF_FORCE
    # With only an expiry date there is no proof it had been issued yet, so it is not
    # promoted over an article that states an issue date at or before `as_of`.
    return IN_FORCE if validity.issue is not None else UNDATED


def rank_by_time(
    results: list[dict[str, Any]], as_of: date, validity: dict[str, Validity]
) -> list[dict[str, Any]]:
    """Stable partition: in force, then undated, then not in force. Never drops a result."""

    def key(item: dict[str, Any]) -> int:
        return bucket(validity.get(str(item.get("article_id")), Validity()), as_of)

    return sorted(results, key=key)  # sorted() is stable: ties keep retrieval order


async def load_validity(db: AsyncSession, article_ids: set[str]) -> dict[str, Validity]:
    if not article_ids:
        return {}
    rows = await db.execute(
        select(Article.id, Article.structured_metadata).where(
            Article.id.in_(list(article_ids))
        )
    )
    return {str(row[0]): parse_validity(row[1]) for row in rows.all()}


async def apply_time_focus(
    db: AsyncSession | None, results: list[dict[str, Any]], as_of: date | None
) -> list[dict[str, Any]]:
    """Reorder `results` for `as_of`; any failure leaves the order as retrieval made it."""
    if db is None or as_of is None or len(results) < 2:
        return results
    try:
        validity = await load_validity(
            db, {str(r["article_id"]) for r in results if r.get("article_id")}
        )
    except Exception as exc:
        logger.warning("Time focus could not read article dates", error=str(exc))
        return results
    if not any(v.issue or v.expiry for v in validity.values()):
        return results
    return rank_by_time(results, as_of, validity)
