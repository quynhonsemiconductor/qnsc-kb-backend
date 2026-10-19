import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db
from src.models import User
from src.models.ops import NotificationQueue

router = APIRouter()


def _response(item: NotificationQueue) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "type": item.type,
        "payload": item.payload,
        "created_at": item.created_at,
        "read_at": item.read_at,
    }


@router.get("")
async def list_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    stmt = select(NotificationQueue).where(
        NotificationQueue.recipient_user_id == current_user.id,
        NotificationQueue.type == "in_app",
    )
    if unread_only:
        stmt = stmt.where(NotificationQueue.read_at.is_(None))
    result = await db.execute(
        stmt.order_by(NotificationQueue.created_at.desc()).limit(limit)
    )
    return [_response(item) for item in result.scalars().all()]


@router.post("/{notification_id}/read")
async def mark_notification_read(
    notification_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    item = (
        await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.id == notification_id,
                NotificationQueue.recipient_user_id == current_user.id,
                NotificationQueue.type == "in_app",
            )
        )
    ).scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Notification not found")
    if item.read_at is None:
        item.read_at = datetime.utcnow()
        await db.commit()
        await db.refresh(item)
    return _response(item)
