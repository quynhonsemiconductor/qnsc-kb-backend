import uuid
from typing import Sequence
from sqlalchemy import select, delete, and_, exists, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models.interaction import Comment, Vote, Bookmark
from src.models.article import Article
from src.models.user import User
from src.repositories.article import ArticleRepository


class InteractionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _authorized_bookmark_filters(user: User, article_id: uuid.UUID):
        """Scope a bookmark mutation to an authorized Article in SQL."""
        return [
            Bookmark.user_id == user.id,
            Bookmark.article_id == article_id,
            exists(
                select(Article.id).where(
                    Article.id == Bookmark.article_id,
                    *ArticleRepository._authorized_article_filters(user),
                )
            ),
        ]

    @staticmethod
    def _authorized_vote_ids(user: User, article_id: uuid.UUID, user_id: uuid.UUID):
        """Select existing votes only through an authorized Article scope."""
        return (
            select(Vote.id)
            .join(Article, Article.id == Vote.article_id)
            .where(
                Vote.user_id == user_id,
                Vote.article_id == article_id,
                *ArticleRepository._authorized_article_filters(user),
            )
        )

    # Comments
    async def create_comment(self, comment: Comment) -> Comment:
        self.db.add(comment)
        await self.db.commit()
        await self.db.refresh(comment)
        # Load user info
        result = await self.db.execute(
            select(Comment)
            .where(Comment.id == comment.id)
            .options(selectinload(Comment.user))
        )
        return result.scalar_one()

    async def get_comments(
        self, article_id: uuid.UUID, user: User
    ) -> Sequence[Comment]:
        result = await self.db.execute(
            select(Comment)
            .join(Article, Article.id == Comment.article_id)
            .where(
                Comment.article_id == article_id,
                *ArticleRepository._authorized_article_filters(user),
            )
            .order_by(Comment.created_at.asc())
            .options(selectinload(Comment.user))
        )
        return result.scalars().all()

    async def delete_comment(
        self, comment_id: uuid.UUID, user_id: uuid.UUID, user: User
    ) -> bool:
        result = await self.db.execute(
            select(Comment)
            .join(Article, Article.id == Comment.article_id)
            .where(
                Comment.id == comment_id,
                Comment.user_id == user_id,
                *ArticleRepository._authorized_article_filters(user),
            )
        )
        comment = result.scalar_one_or_none()
        if comment is None:
            return False
        await self.db.delete(comment)
        await self.db.commit()
        return True

    # Votes
    async def cast_vote(self, vote: Vote, user: User) -> Vote:
        # Delete any existing vote by this user on this article
        await self.db.execute(
            delete(Vote).where(
                Vote.id.in_(
                    self._authorized_vote_ids(user, vote.article_id, vote.user_id)
                )
            )
        )
        self.db.add(vote)
        await self.db.commit()
        await self.db.refresh(vote)
        return vote

    async def get_votes_summary(
        self, article_id: uuid.UUID, user: User
    ) -> dict[str, int]:
        result = await self.db.execute(
            select(Vote.value, func.count(Vote.id))
            .join(Article, Article.id == Vote.article_id)
            .where(
                Vote.article_id == article_id,
                *ArticleRepository._authorized_article_filters(user),
            )
            .group_by(Vote.value)
        )
        summary = {"upvotes": 0, "downvotes": 0}
        for row in result.all():
            val, cnt = row
            if val == 1:
                summary["upvotes"] = cnt
            elif val == -1:
                summary["downvotes"] = cnt
        return summary

    async def get_user_vote(
        self, article_id: uuid.UUID, user_id: uuid.UUID, user: User
    ) -> int:
        result = await self.db.execute(
            select(Vote.value)
            .join(Article, Article.id == Vote.article_id)
            .where(
                Vote.article_id == article_id,
                Vote.user_id == user_id,
                *ArticleRepository._authorized_article_filters(user),
            )
        )
        return result.scalar_one_or_none() or 0

    # Bookmarks
    async def add_bookmark(self, bookmark: Bookmark, user: User) -> Bookmark:
        # Delete any duplicate first for idempotence
        await self.db.execute(
            delete(Bookmark).where(
                *self._authorized_bookmark_filters(user, bookmark.article_id)
            )
        )
        self.db.add(bookmark)
        await self.db.commit()
        return bookmark

    async def remove_bookmark(self, user: User, article_id: uuid.UUID) -> bool:
        result = await self.db.execute(
            delete(Bookmark).where(*self._authorized_bookmark_filters(user, article_id))
        )
        await self.db.commit()
        return result.rowcount > 0

    async def get_bookmarks(self, user: User) -> Sequence[Article]:
        result = await self.db.execute(
            select(Article)
            .join(Bookmark, Bookmark.article_id == Article.id)
            .where(
                and_(
                    Bookmark.user_id == user.id,
                    *ArticleRepository._authorized_article_filters(user),
                )
            )
            .options(
                selectinload(Article.tags),
                selectinload(Article.owner),
                selectinload(Article.departments),
                selectinload(Article.sources),
            )
        )
        return result.scalars().all()

    async def is_bookmarked(self, user: User, article_id: uuid.UUID) -> bool:
        result = await self.db.execute(
            select(Bookmark)
            .join(Article, Article.id == Bookmark.article_id)
            .where(
                Bookmark.user_id == user.id,
                Bookmark.article_id == article_id,
                *ArticleRepository._authorized_article_filters(user),
            )
        )
        return result.scalar_one_or_none() is not None
