"""Read-only endpoints behind the Investigate area of the health dashboard.

Same gate as the other operational endpoints (`governance.read` at global scope), and for the
same reason: they describe the whole system, across companies, not one tenant. Nothing here
writes. See `src/domain/investigate.py` for what is computed and why it is windowed.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db, require_permission
from src.core.config import settings
from src.domain import investigate
from src.domain.rbac import AuthorizationService
from src.models import User

router = APIRouter()


@router.get("/requests")
async def investigate_requests(
    window_hours: int = Query(24, ge=1, le=investigate.MAX_WINDOW_HOURS),
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """API traffic over a recent window, against the window before it.

    Totals and latency percentiles, a bucketed series for charting, failures grouped by route
    and exception, the slowest routes, and one overall verdict. Health probes are excluded.
    """
    window = investigate.make_window(window_hours)
    return await investigate.load_requests(db, window, settings.METRICS_RETENTION_DAYS)


@router.get("/ai")
async def investigate_ai(
    window_hours: int = Query(24, ge=1, le=investigate.MAX_WINDOW_HOURS),
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """AI answer volume, tokens and latency over a recent window, and which models served it."""
    return await investigate.load_ai(db, investigate.make_window(window_hours))


@router.get("/pipeline")
async def investigate_pipeline(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Background work that is stuck or failing: indexing, dead letters, outbox, mail, syncs."""
    return await investigate.load_pipeline(
        db,
        include_titles=AuthorizationService.has_global_article_access(current_user),
    )
