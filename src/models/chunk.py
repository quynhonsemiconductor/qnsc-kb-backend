import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, String, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
try:
    from pgvector.sqlalchemy import Vector
except ModuleNotFoundError:  # Keep domain/unit tests runnable without the optional DB extension.
    from sqlalchemy.types import UserDefinedType

    class Vector(UserDefinedType):
        cache_ok = True

        def __init__(self, dimensions: int):
            self.dimensions = dimensions

        def get_col_spec(self, **kwargs):
            return f"VECTOR({self.dimensions})"
from src.core.config import settings
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin

class ParentChunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "parent_chunks"

    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)  # e.g., "Section 2.1"
    chunk_type: Mapped[str] = mapped_column(String(40), nullable=False, default="section")
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    article: Mapped["Article"] = relationship("Article")
    child_chunks: Mapped[list["ArticleChunk"]] = relationship(
        "ArticleChunk", back_populates="parent_chunk", cascade="all, delete-orphan",
        order_by="ArticleChunk.chunk_index",
    )

class ArticleChunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "article_chunks"

    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    parent_chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("parent_chunks.id", ondelete="CASCADE"), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Vector size: dynamically loaded from configuration settings
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Denormalized organizational metadata for retrieval filtering
    department_id: Mapped[str] = mapped_column(String(100), nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(50), nullable=False)
    visibility: Mapped[str] = mapped_column(String(50), nullable=False)
    
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(40), nullable=False, default="text")
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunking_version: Mapped[str] = mapped_column(String(80), nullable=False, default="v1-fixed-character")
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: sha256 of this chunk's own `chunk_text` (the raw child text, NOT the
    #: contextual-header-augmented string sent to the embedder -- see
    #: src/rag/contextual_header.py). Lets a re-index (indexing.py) recognise a child
    #: that has not actually changed and reuse its existing embedding instead of paying
    #: for another embed call. Nullable: a chunk written before this column existed simply
    #: never matches on re-index, degrading to "re-embed everything" -- the behavior
    #: every article already had, not a new failure mode.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    article: Mapped["Article"] = relationship("Article")
    parent_chunk: Mapped[ParentChunk] = relationship("ParentChunk", back_populates="child_chunks")
    metadata_fields: Mapped[list["ChunkMetadata"]] = relationship(
        "ChunkMetadata", back_populates="chunk", cascade="all, delete-orphan"
    )

class ChunkMetadata(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "chunk_metadata"

    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("article_chunks.id", ondelete="CASCADE"), nullable=False)
    key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)

    chunk: Mapped[ArticleChunk] = relationship("ArticleChunk", back_populates="metadata_fields")
