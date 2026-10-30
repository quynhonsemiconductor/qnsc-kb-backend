"""Tell a job that is waiting from a job that nothing will ever pick up.

A connector job sits `queued` until a Celery worker takes it. When no worker is running the
job never moves, and the page only ever shows "queued", which reads as "in progress". Past a
threshold the job is reported as stalled so the page can say there is nothing consuming it.
Pure and cheap on purpose: it runs for every row of a jobs list, so it asks no broker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def stall_info(
    status: str,
    created_at: datetime | None,
    *,
    threshold_minutes: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    if status != "queued" or created_at is None:
        return {"stalled": False, "queued_minutes": None}
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    minutes = max(0, int((current - created_at).total_seconds() // 60))
    return {"stalled": minutes >= threshold_minutes, "queued_minutes": minutes}
