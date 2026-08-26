from typing import Any
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.api.deps import get_db, get_current_user
from src.models import User
from src.domain.meta import MetaService
from src.domain.rbac import AuthorizationService
from src.models.article import TagCatalog
from src.api.deps import require_permission

router = APIRouter()

class GroupResponse(BaseModel):
    id: uuid.UUID
    name: str
    bitmask_position: int

    model_config = ConfigDict(from_attributes=True)


class TagCatalogRequest(BaseModel):
    tag: str = Field(min_length=1, max_length=80)
    active: bool = True


def _normalize_tag(value: str) -> str:
    import re
    import unicodedata
    folded = "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", folded.strip().casefold())

@router.get("/tags")
async def get_tags(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    service = MetaService(db)
    return await service.get_all_tags(current_user)


@router.get("/tag-catalog")
async def list_tag_catalog(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (await db.execute(select(TagCatalog).where(TagCatalog.company_domain == current_user.company_domain).order_by(TagCatalog.normalized_tag))).scalars().all()
    return [{"id": str(item.id), "tag": item.tag, "normalized_tag": item.normalized_tag, "active": item.active, "deprecated_at": item.deprecated_at} for item in rows]


@router.post("/tag-catalog", status_code=201)
async def create_tag_catalog(request: TagCatalogRequest, current_user: User = Depends(require_permission("article.publish", scope="company")), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    normalized = _normalize_tag(request.tag)
    if not normalized:
        raise HTTPException(status_code=422, detail="Tag must contain letters or numbers")
    existing = await db.scalar(select(TagCatalog).where(TagCatalog.company_domain == current_user.company_domain, TagCatalog.normalized_tag == normalized))
    if existing:
        existing.tag = request.tag.strip()
        existing.active = request.active
        existing.deprecated_at = None if request.active else (existing.deprecated_at or datetime.utcnow())
        item = existing
    else:
        item = TagCatalog(company_domain=current_user.company_domain, tag=request.tag.strip(), normalized_tag=normalized, active=request.active, created_by=current_user.id)
        db.add(item)
    await db.commit()
    await db.refresh(item)
    return {"id": str(item.id), "tag": item.tag, "normalized_tag": item.normalized_tag, "active": item.active}


@router.delete("/tag-catalog/{tag_id}", status_code=204)
async def deprecate_tag_catalog(tag_id: uuid.UUID, current_user: User = Depends(require_permission("article.publish", scope="company")), db: AsyncSession = Depends(get_db)) -> None:
    item = await db.scalar(select(TagCatalog).where(TagCatalog.id == tag_id, TagCatalog.company_domain == current_user.company_domain))
    if not item:
        raise HTTPException(status_code=404, detail="Tag not found")
    item.active = False
    item.deprecated_at = datetime.utcnow()
    await db.commit()

@router.get("/glossary")
async def get_glossary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    service = MetaService(db)
    return await service.get_glossary()

@router.get("/groups", response_model=list[GroupResponse])
async def get_groups(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    from src.repositories.user import UserRepository
    user_repo = UserRepository(db)
    return await user_repo.get_all_groups(None if AuthorizationService.can_view_all_access_groups(current_user) else current_user.company_domain)
