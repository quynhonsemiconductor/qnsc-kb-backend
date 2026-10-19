from typing import TYPE_CHECKING
import uuid
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from src.models.article import Article, DocumentSource
    from src.models.chunk import ArticleChunk, ParentChunk


class DocumentEvidence(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A bounded table or visual region whose authority remains its parent Article."""

    __tablename__ = "document_evidence"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('table', 'figure', 'page_visual')",
            name="ck_document_evidence_kind",
        ),
        Index("ix_document_evidence_article_kind", "article_id", "kind"),
        Index("ix_document_evidence_source_page", "document_source_id", "page_number"),
    )

    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False
    )
    document_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_sources.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    bounding_box: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    page_width: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_height: Mapped[float | None] = mapped_column(Float, nullable=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    structured_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    extraction_version: Mapped[str] = mapped_column(String(50), nullable=False)

    article: Mapped["Article"] = relationship("Article", back_populates="evidence")
    document_source: Mapped["DocumentSource"] = relationship(
        "DocumentSource", back_populates="evidence"
    )
    parent_chunks: Mapped[list["ParentChunk"]] = relationship(
        "ParentChunk", back_populates="evidence"
    )
    child_chunks: Mapped[list["ArticleChunk"]] = relationship(
        "ArticleChunk", back_populates="evidence"
    )
