import uuid
from sqlalchemy import ForeignKey, String, Text, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin

class Comment(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "comments"

    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    article: Mapped["Article"] = relationship("Article")
    user: Mapped["User"] = relationship("User")

class Vote(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "votes"
    __table_args__ = (
        UniqueConstraint("article_id", "user_id", name="uq_article_user_vote"),
    )

    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # +1 for upvote, -1 for downvote
    value: Mapped[int] = mapped_column(Integer, nullable=False)

    article: Mapped["Article"] = relationship("Article")
    user: Mapped["User"] = relationship("User")

class Bookmark(Base, TimestampMixin):
    __tablename__ = "bookmarks"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)


class ArticleFollower(Base, TimestampMixin):
    """A user following an article for version/source-change notifications."""

    __tablename__ = "article_followers"
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)

    article: Mapped["Article"] = relationship("Article")
    user: Mapped["User"] = relationship("User")
