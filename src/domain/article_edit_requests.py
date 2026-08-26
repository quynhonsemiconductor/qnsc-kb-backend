import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.permissions import PermissionService
from src.models import User
from src.models.governance import ArticleEditRequest
from src.models.ops import NotificationQueue
from src.repositories.article import ArticleRepository
from src.repositories.audit import AuditRepository
from src.repositories.user import UserRepository


async def create_article_edit_request(
    db: AsyncSession,
    current_user: User,
    article_id: uuid.UUID,
    request_text: str,
    source: str = "manual",
) -> dict[str, Any]:
    """Create a correction request and notify users authorized for the article."""
    article = await ArticleRepository(db).get_by_id(article_id, user=current_user)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    if PermissionService.can_edit_article(current_user, article):
        raise HTTPException(
            status_code=409,
            detail="You have permission to edit this article directly. Use the edit page instead.",
        )

    normalized_text = request_text.strip()
    if len(normalized_text) < 5:
        raise HTTPException(
            status_code=422,
            detail="Please describe the correction you want made.",
        )

    existing = await db.scalar(
        select(ArticleEditRequest).where(
            ArticleEditRequest.article_id == article.id,
            ArticleEditRequest.requested_by == current_user.id,
            ArticleEditRequest.status == "open",
        )
    )
    if existing:
        existing.request_text = normalized_text
        edit_request = existing
        deduplicated = True
    else:
        edit_request = ArticleEditRequest(
            company_domain=article.company_domain,
            article_id=article.id,
            requested_by=current_user.id,
            request_text=normalized_text,
            status="open",
        )
        db.add(edit_request)
        deduplicated = False

    candidates = await UserRepository(db).list_users(
        limit=500,
        viewer=None,
        company_domain=article.company_domain,
        active=True,
    )
    editors = [
        candidate
        for candidate in candidates
        if candidate.id != current_user.id
        and PermissionService.can_edit_article(candidate, article)
    ]
    await db.flush()
    for editor in editors:
        db.add(
            NotificationQueue(
                recipient_user_id=editor.id,
                type="in_app",
                payload={
                    "event": "article_edit_request",
                    "request_id": str(edit_request.id),
                    "article_id": str(article.id),
                    "article_title": article.title,
                    "requested_by": current_user.name,
                    "request_text": normalized_text,
                    "action_url": f"/articles/{article.id}/edit",
                },
            )
        )
    await db.commit()
    await db.refresh(edit_request)
    await AuditRepository(db).record(
        current_user.id,
        "article_edit_request",
        "article",
        str(article.id),
        detail={
            "request_id": str(edit_request.id),
            "notified_editor_count": len(editors),
            "source": source,
        },
    )
    return {
        "id": str(edit_request.id),
        "article_id": str(article.id),
        "article_title": article.title,
        "status": edit_request.status,
        "request_text": edit_request.request_text,
        "notified_editor_count": len(editors),
        "deduplicated": deduplicated,
        "message": (
            "Your edit request was sent to authorized article editors."
            if editors
            else "Your edit request was recorded. No editor notification could be delivered automatically."
        ),
    }
