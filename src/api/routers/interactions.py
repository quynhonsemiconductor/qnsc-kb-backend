import uuid
from datetime import datetime
from typing import Any, Literal
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_db, get_current_user
from src.models import User
from src.repositories.interaction import InteractionRepository
from src.repositories.article import ArticleRepository
from src.domain.interactions import InteractionsService

router = APIRouter()

class CommentCreate(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)

class VoteCast(BaseModel):
    value: Literal[-1, 0, 1]  # 1 for upvote, -1 for downvote, 0 to clear

class UserBrief(BaseModel):
    id: uuid.UUID
    name: str
    email: str
    model_config = ConfigDict(from_attributes=True)

class CommentResponse(BaseModel):
    id: uuid.UUID
    article_id: uuid.UUID
    user_id: uuid.UUID
    text: str
    created_at: datetime
    user: UserBrief

    model_config = ConfigDict(from_attributes=True)

@router.post("/articles/{id}/comments")
async def add_comment(
    id: uuid.UUID,
    comment_in: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    return await service.add_comment(current_user, id, comment_in.text)

@router.get("/articles/{id}/comments")
async def get_comments(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    return await service.get_comments(current_user, id)

@router.delete("/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comment(
    comment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> None:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    await service.delete_comment(current_user, comment_id)

@router.post("/articles/{id}/votes")
async def cast_vote(
    id: uuid.UUID,
    vote_in: VoteCast,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    return await service.cast_vote(current_user, id, vote_in.value)

@router.get("/articles/{id}/votes")
async def get_votes_summary(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    return await service.get_votes_summary(current_user, id)

@router.get("/articles/{id}/user-vote")
async def get_user_vote(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    val = await service.get_user_vote(current_user, id)
    return {"vote": val}

@router.post("/articles/{id}/bookmark")
async def bookmark_article(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    success = await service.add_bookmark(current_user, id)
    return {"success": success}

@router.delete("/articles/{id}/bookmark")
async def unbookmark_article(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    success = await service.remove_bookmark(current_user, id)
    return {"success": success}

@router.get("/bookmarks")
async def list_bookmarks(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    int_repo = InteractionRepository(db)
    art_repo = ArticleRepository(db)
    service = InteractionsService(int_repo, art_repo)
    return await service.list_bookmarks(current_user)
