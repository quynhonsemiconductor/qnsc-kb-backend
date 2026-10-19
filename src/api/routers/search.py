from typing import Any
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_db, get_current_user
from src.models import User
from src.repositories.chunk import ChunkRepository
from src.repositories.governance import GovernanceRepository
from src.domain.search_service import SearchService
from src.repositories.feature_flags import FeatureFlagRepository

router = APIRouter()


@router.get("")
async def search(
    q: str = Query("", max_length=4_000, description="Search query string"),
    dept: str | None = Query(None, max_length=100),
    sensitivity: str | None = Query(None, max_length=30),
    type_: str | None = Query(None, alias="type", max_length=50),
    status: str | None = Query(None, max_length=30),
    language: str | None = Query(None, max_length=20),
    tag: str | None = Query(None, max_length=50),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    chunk_repo = ChunkRepository(db)
    gov_repo = GovernanceRepository(db)
    search_service = SearchService(chunk_repo, gov_repo, FeatureFlagRepository(db))

    filters = {}
    if dept:
        filters["dept"] = dept
    if sensitivity:
        filters["sensitivity"] = sensitivity
    if type_:
        filters["type"] = type_
    if status:
        filters["status"] = status
    if language:
        filters["language"] = language
    if tag:
        filters["tag"] = tag.strip().lower()
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    return await search_service.search(
        user=current_user, query=q, filters=filters, limit=limit
    )
