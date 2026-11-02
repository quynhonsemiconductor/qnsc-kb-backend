"""Articles whose indexing was interrupted, and what to do about them.

`index_article` sets `index_status = "processing"`, does its work, and sets "ready" or "failed". A
failure it can see is recorded. One it cannot is not: when the OOM killer ends a prefork child
(`WorkerLostError: ... signal 9`) nothing in Python runs afterwards, so no `except` fires and the row
stays "processing". Celery then acknowledges the message anyway (`task_acks_late` does not requeue a
lost worker unless `task_reject_on_worker_lost` is set), so nothing will ever look at it again.

That is not cosmetic. Search only returns articles whose `index_status` is "ready", so an article
stuck in "processing" is gone from search, and the only existing recovery is the sweep at API start-up
(src/api/main.py), which happens when someone happens to restart the API. A worker redeploy has the
same effect on a long run: ECS stops the container after at most 120 s, and Redis hands the message
back only after VISIBILITY_TIMEOUT.

This sweeps for them. An article that has been "processing" for longer than any run can legitimately
take is queued again, at most `INDEX_RECOVERY_MAX_ATTEMPTS` times; after that it is marked "failed",
which the dashboard lists and the reprocess endpoint can retry. Bounded on purpose: the alternative,
`task_reject_on_worker_lost`, requeues without limit, so an article that kills the worker every time
would take it down in a loop and starve everything else behind it.

The count lives in `index_error`, in a fixed prefix (`INTERRUPTED_PREFIX`), because that field survives
the retry: `set_index_status` keeps a marker when it moves to "processing" and a successful run clears
it. No new column, so no migration.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta

import structlog
from sqlalchemy import select, update

from src.api.deps import SessionLocal, set_database_context
from src.models.article import Article
from src.workers.celery_app import VISIBILITY_TIMEOUT

logger = structlog.get_logger()

#: A run is killed at TASK_HARD_TIME_LIMIT, and a message that was never acknowledged comes back after
#: VISIBILITY_TIMEOUT. Waiting past both, plus a margin, means this never competes with a run that is
#: still going or about to be redelivered on its own.
INDEX_STUCK_MINUTES = VISIBILITY_TIMEOUT // 60 + 10

#: Times an interrupted article is queued again before it is reported as failed.
INDEX_RECOVERY_MAX_ATTEMPTS = 2

#: At most this many per pass, so one sweep stays short even after a bad night.
BATCH_LIMIT = 20

INTERRUPTED_PREFIX = "Interrupted while indexing"
_ATTEMPT = re.compile(r"attempt (\d+) of")

FAILED_MESSAGE = (
    "Indexing was interrupted {attempts} times (the worker restarted or ran out of memory) and did not "
    "complete. Check the worker, then reprocess this article."
)


def interrupted_attempts(index_error: str | None) -> int:
    """How many automatic re-queues are already recorded in `index_error`; 0 if it is any other text."""
    text = index_error or ""
    if not text.startswith(INTERRUPTED_PREFIX):
        return 0
    match = _ATTEMPT.search(text)
    return int(match.group(1)) if match else 0


def next_step(index_error: str | None) -> tuple[str, str]:
    """`("requeue", marker)` or `("fail", message)` for an article that is stuck in "processing"."""
    attempt = interrupted_attempts(index_error) + 1
    if attempt > INDEX_RECOVERY_MAX_ATTEMPTS:
        return "fail", FAILED_MESSAGE.format(attempts=INDEX_RECOVERY_MAX_ATTEMPTS + 1)
    return (
        "requeue",
        f"{INTERRUPTED_PREFIX} (attempt {attempt} of {INDEX_RECOVERY_MAX_ATTEMPTS}); queued again automatically.",
    )


async def recover_stuck_indexing(
    *, now: datetime | None = None, stuck_minutes: int = INDEX_STUCK_MINUTES
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """Claim stuck articles. Returns `(to_requeue, failed)`; the caller dispatches `to_requeue`.

    A requeued article stays "processing" with a fresh `updated_at`, rather than moving to "pending":
    if the dispatch is lost, or the new run dies the same way, it is simply found again after another
    `stuck_minutes`, with the count one higher. Dispatch happens after the commit that makes the claim
    durable, for the reason `replay_outbox_task` gives: a task started before its claim is visible can
    finish against the row as it was.
    """
    moment = now or datetime.utcnow()
    cutoff = moment - timedelta(minutes=stuck_minutes)
    requeue: list[uuid.UUID] = []
    failed: list[uuid.UUID] = []
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        rows = (
            await db.execute(
                # Columns, not the entity: loading an Article drags its departments and permissions along.
                select(Article.id, Article.index_error)
                .where(
                    Article.status == "published",
                    Article.lifecycle_status == "active",
                    Article.index_status == "processing",
                    Article.updated_at < cutoff,
                )
                .order_by(Article.updated_at)
                .limit(BATCH_LIMIT)
                .with_for_update(skip_locked=True)
            )
        ).all()
        for article_id, index_error in rows:
            action, text = next_step(index_error)
            if action == "fail":
                failed.append(article_id)
                values = {"index_status": "failed", "index_error": text}
            else:
                requeue.append(article_id)
                values = {"index_status": "processing", "index_error": text}
            await db.execute(
                update(Article)
                .where(Article.id == article_id)
                .values(**values, updated_at=moment)
            )
        await db.commit()
    if requeue or failed:
        logger.warning(
            "Recovered articles stuck in indexing",
            requeued=len(requeue),
            failed=len(failed),
            stuck_minutes=stuck_minutes,
        )
    return requeue, failed
