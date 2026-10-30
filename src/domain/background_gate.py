"""One check before the model-backed steps that follow a publish, and a record when skipped.

Structured-metadata and entity extraction run after a draft is approved and must never fail
it, so they swallow every failure. The cost of that was silence: with no model provider
configured they quietly did nothing and nobody could tell the graph was not being built.
This asks first, and when the answer is no it writes an audit row saying which step was
skipped and why, so an admin can see it instead of inferring it from an empty graph.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

REASON_NO_PROVIDER = "no_llm_provider"


def background_ai_skip_reason() -> str | None:
    """None when model-backed steps can run, else a short machine-readable reason."""
    from src.domain.llm_client import resolve_provider

    try:
        return None if resolve_provider() else REASON_NO_PROVIDER
    except Exception:
        return REASON_NO_PROVIDER


async def should_skip_background_step(db, user, article_id: str, step: str) -> bool:
    """True (and an audit row) when `step` cannot run. Never raises."""
    reason = background_ai_skip_reason()
    if reason is None:
        return False
    logger.info("Background AI step skipped", step=step, reason=reason)
    try:
        from src.repositories.audit import AuditRepository

        await AuditRepository(db).record(
            user.id,
            "background_ai_skipped",
            "article",
            str(article_id),
            detail={"step": step, "reason": reason},
        )
        await db.commit()
    except Exception as exc:
        logger.warning("Could not record a skipped background step", error=str(exc))
        if hasattr(db, "rollback"):
            try:
                await db.rollback()
            except Exception:
                pass
    return True
