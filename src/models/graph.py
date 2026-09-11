"""The tenant knowledge graph: entities and relationships extracted from articles.

Unlike a `PendingDraft`'s per-document suggestions (tags, department routing), this
graph accumulates ACROSS every published article a tenant has -- an entity mentioned in
ten different documents is one row, not ten. See `domain/entity_extraction.py` for how
one document's extraction is produced and `domain/graph_service.py` for how it is merged
into the graph that already exists.
"""
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class GraphEntity(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One deduplicated node in the tenant graph.

    Deduplicated the same way `TagCatalog` dedupes tags: on
    `(company_domain, normalized_name, entity_type)`, using the same NFKD accent-fold
    (`entity_extraction.normalize_entity_name`) so "An toan lao dong" and "An toàn lao
    động" merge into one entity instead of two.
    """

    __tablename__ = "graph_entities"
    __table_args__ = (
        UniqueConstraint(
            "company_domain",
            "normalized_name",
            "entity_type",
            name="uq_graph_entity_company_normalized_type",
        ),
        Index("ix_graph_entities_company_type", "company_domain", "entity_type"),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Denormalized on write (see graph_service.apply_extraction), not computed on read:
    # sorting an entity list by "how much of the KB talks about this" is exactly the
    # query pattern GraphExplorerPage wants, and counting mentions per row on every list
    # request would join article_entity_mentions for no reason on a value that only
    # changes when a new article mentions this entity.
    mention_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

    mentions: Mapped[list["ArticleEntityMention"]] = relationship(
        "ArticleEntityMention", back_populates="entity", cascade="all, delete-orphan"
    )


class GraphRelationship(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One directed, typed edge between two entities of the same tenant.

    Deduped on `(company_domain, source_entity_id, target_entity_id, relation_type)`:
    the same fact asserted in several documents is one edge, re-pointed at whichever
    article most recently asserted it (`article_id`) rather than accumulating duplicates.
    """

    __tablename__ = "graph_relationships"
    __table_args__ = (
        UniqueConstraint(
            "company_domain",
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            name="uq_graph_relationship_edge",
        ),
        Index("ix_graph_relationships_company_source", "company_domain", "source_entity_id"),
        Index("ix_graph_relationships_company_target", "company_domain", "target_entity_id"),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # SET NULL, not CASCADE: an article can be archived or deleted without erasing the
    # relationship itself, which may still be evidenced elsewhere or simply worth keeping
    # as tenant knowledge -- only the "where did this come from" pointer goes stale.
    article_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("articles.id", ondelete="SET NULL"), nullable=True
    )

    source_entity: Mapped["GraphEntity"] = relationship(
        "GraphEntity", foreign_keys=[source_entity_id]
    )
    target_entity: Mapped["GraphEntity"] = relationship(
        "GraphEntity", foreign_keys=[target_entity_id]
    )


class ArticleEntityMention(Base, UUIDPrimaryKeyMixin):
    """Provenance: which published article contributed which entity to the graph."""

    __tablename__ = "article_entity_mentions"
    __table_args__ = (
        UniqueConstraint("article_id", "entity_id", name="uq_article_entity_mention"),
    )

    # Denormalized company_domain, same reasoning as every other tenant-scoped table
    # here: RLS policies compare this column directly, and deriving it via a join to
    # articles/graph_entities on every row check would be slower for no benefit.
    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    entity: Mapped["GraphEntity"] = relationship("GraphEntity", back_populates="mentions")
