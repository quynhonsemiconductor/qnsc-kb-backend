"""Durable connector work queue helpers.

Webhook requests are intentionally separated from provider work.  The queue is
coalesced per connector/scope, so a burst of Graph notifications wakes one
delta run instead of creating one job per file.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import uuid

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.connectors import SyncRequest
from src.models.ops import ConnectorJob


ACTIVE_REQUEST_STATES = ("queued", "dispatched", "running")

# Coalescing only onto a request that has NOT started yet. Folding a new notification
# into a dispatched or running one loses it: that run has already read its change list
# from the provider, so the change it was told about would not be picked up until the
# next reconciliation. One in flight plus one waiting is the most this can accumulate.
COALESCIBLE_REQUEST_STATES = ("queued",)


async def enqueue_connector_sync(
    db: AsyncSession,
    connector_id: uuid.UUID,
    *,
    scope_id: uuid.UUID | None = None,
    reason: str = "notification",
    requested_by: uuid.UUID | None = None,
    priority: int = 50,
) -> SyncRequest:
    """Create or reuse one active request for a connector scope."""

    scope_filter = (
        SyncRequest.scope_id.is_(None)
        if scope_id is None
        else SyncRequest.scope_id == scope_id
    )
    existing = (
        await db.execute(
            select(SyncRequest)
            .where(
                SyncRequest.connector_id == connector_id,
                scope_filter,
                SyncRequest.status.in_(COALESCIBLE_REQUEST_STATES),
            )
            .order_by(SyncRequest.priority.asc(), SyncRequest.created_at.asc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing:
        if priority < existing.priority:
            existing.priority = priority
        if reason and existing.reason == "polling":
            existing.reason = reason
        await db.flush()
        return existing

    job = ConnectorJob(
        connector_id=connector_id,
        requested_by=requested_by,
        status="queued",
        attempts=0,
    )
    db.add(job)
    await db.flush()
    request = SyncRequest(
        connector_id=connector_id,
        scope_id=scope_id,
        job_id=job.id,
        reason=reason,
        priority=priority,
        status="queued",
        available_at=datetime.utcnow(),
    )
    db.add(request)
    await db.flush()
    return request


async def claim_sync_request(db: AsyncSession, request_id: uuid.UUID) -> SyncRequest | None:
    """Claim a queued request with a row lock before dispatching work."""

    request = (
        await db.execute(
            select(SyncRequest)
            .where(
                SyncRequest.id == request_id,
                SyncRequest.status == "queued",
                SyncRequest.available_at <= datetime.utcnow(),
            )
            .with_for_update(skip_locked=True)
        )
    ).scalar_one_or_none()
    if not request:
        return None
    request.status = "dispatched"
    request.locked_at = datetime.utcnow()
    request.attempts += 1
    await db.commit()
    return request


async def mark_sync_request_running(db: AsyncSession, request_id: uuid.UUID) -> None:
    request = await db.get(SyncRequest, request_id)
    if request:
        request.status = "running"
        request.started_at = request.started_at or datetime.utcnow()
        await db.commit()


async def finish_sync_request(
    db: AsyncSession,
    request_id: uuid.UUID,
    *,
    success: bool,
    error: str | None = None,
    retryable: bool = True,
) -> None:
    request = await db.get(SyncRequest, request_id)
    if not request:
        return
    request.locked_at = None
    request.last_error = error[:2000] if error else None
    if success:
        request.status = "completed"
        request.completed_at = datetime.utcnow()
    elif retryable and request.attempts < 8:
        request.status = "queued"
        request.available_at = datetime.utcnow() + timedelta(
            seconds=min(1800, 2 ** min(request.attempts, 10))
        )
    else:
        request.status = "failed"
    await db.commit()


async def recover_stale_sync_requests(
    db: AsyncSession,
    *,
    stale_after_minutes: int = 15,
    running_stale_after_minutes: int = 180,
) -> int:
    """Return abandoned dispatched/running requests to the durable queue.

    Two windows, because the two states mean different things. A ``dispatched`` request
    was handed to a worker that never started it, so a quarter of an hour of silence is
    already a lost message. A ``running`` request is a provider walk in progress, and
    nothing heartbeats it — a first full reconciliation of a large drive runs for a long
    time — so requeueing it on the same short window did not recover anything, it started
    a SECOND concurrent sync of the same connector.
    """

    now = datetime.utcnow()
    dispatched_cutoff = now - timedelta(minutes=stale_after_minutes)
    running_cutoff = now - timedelta(minutes=running_stale_after_minutes)
    rows = (
        await db.execute(
            select(SyncRequest).where(
                or_(
                    and_(
                        SyncRequest.status == "dispatched",
                        # A NULL lock is unclaimable rather than fresh: nothing can
                        # advance it, so it belongs back in the queue.
                        or_(
                            SyncRequest.locked_at.is_(None),
                            SyncRequest.locked_at < dispatched_cutoff,
                        ),
                    ),
                    and_(
                        SyncRequest.status == "running",
                        SyncRequest.started_at < running_cutoff,
                    ),
                )
            )
        )
    ).scalars().all()
    for request in rows:
        request.status = "queued"
        request.available_at = datetime.utcnow()
        request.locked_at = None
        request.last_error = "Recovered after worker/process interruption"
    if rows:
        await db.commit()
    return len(rows)
