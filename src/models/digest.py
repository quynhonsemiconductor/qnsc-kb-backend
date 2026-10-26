from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.config import settings
from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.chunk import Vector

if TYPE_CHECKING:
    from src.models.article import Article


class ArticleDigest(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A short summary of one document, kept so questions about MANY documents need not reread them.

    Authority stays with the parent `Article`: a digest is derived text, replaced when the
    article's content or the digest prompt changes, and removed with the article. It is never
    shown or searched without the parent's visibility rules (`ArticleRepository`), because it
    carries the document's content and therefore its sensitivity.

    One row per article. `content_hash` is the hash of the text the summary was made from, so
    "is this digest still current" is a comparison and not a model call.
    """

    __tablename__ = "article_digests"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ready', 'failed')", name="ck_article_digests_status"
        ),
    )

    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    digest_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ready")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    topics: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    entities: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    key_numbers: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    doc_language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # The input was sampled because the document is longer than DIGEST_INPUT_MAX_CHARS.
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Same width and model as `article_chunks.embedding`, so one query embedding ranks both.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION), nullable=True
    )
    embedding_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    article: Mapped["Article"] = relationship("Article")
