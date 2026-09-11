"""Persistence and traversal for the tenant knowledge graph.

Entities and relationships accumulate ACROSS every published article a tenant has, so
`apply_extraction` is a merge into a graph that already exists, not a per-document write
-- the same distinction `TagCatalog` draws between a tenant's tag vocabulary and any one
article's tags.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entity_extraction import ExtractionResult, normalize_entity_name
from src.models.article import Article
from src.models.graph import ArticleEntityMention, GraphEntity, GraphRelationship

#: A neighborhood past this is a graph dump, not something a reviewer explores one click
#: at a time -- and each additional hop can multiply the frontier size.
MAX_NEIGHBOR_DEPTH = 3


async def _get_or_create_entity(
    db: AsyncSession, company_domain: str, name: str, entity_type: str, description: str
) -> GraphEntity:
    normalized = normalize_entity_name(name)
    existing = await db.scalar(
        select(GraphEntity).where(
            GraphEntity.company_domain == company_domain,
            GraphEntity.normalized_name == normalized,
            GraphEntity.entity_type == entity_type,
        )
    )
    if existing:
        # Fill a blank description opportunistically; never overwrite one that already
        # says something, so the first document to describe an entity wins rather than
        # whichever one happened to be processed last.
        if description and not existing.description:
            existing.description = description
        return existing
    entity = GraphEntity(
        company_domain=company_domain,
        name=name,
        normalized_name=normalized,
        entity_type=entity_type,
        description=description or None,
    )
    db.add(entity)
    await db.flush()
    return entity


async def apply_extraction(db: AsyncSession, article: Article, result: ExtractionResult) -> int:
    """Merge one document's extraction into the tenant graph. Returns entities touched.

    Callers wrap this in the same try/except/rollback shape `governance.py::approve_draft`
    already uses for the contradiction check and structured-metadata extraction next to
    it -- a bad merge here must not undo the publish it followed.
    """
    if not result.entities:
        return 0

    entities_by_key: dict[str, GraphEntity] = {}
    for extracted in result.entities:
        entity = await _get_or_create_entity(
            db, article.company_domain, extracted.name, extracted.type, extracted.description
        )
        entities_by_key[normalize_entity_name(extracted.name)] = entity

        mention_exists = await db.scalar(
            select(ArticleEntityMention.id).where(
                ArticleEntityMention.article_id == article.id,
                ArticleEntityMention.entity_id == entity.id,
            )
        )
        if not mention_exists:
            db.add(
                ArticleEntityMention(
                    company_domain=article.company_domain,
                    article_id=article.id,
                    entity_id=entity.id,
                )
            )
            entity.mention_count += 1

    for edge in result.relationships:
        source = entities_by_key.get(normalize_entity_name(edge.source))
        target = entities_by_key.get(normalize_entity_name(edge.target))
        if not source or not target:
            continue
        stmt = (
            pg_insert(GraphRelationship)
            .values(
                company_domain=article.company_domain,
                source_entity_id=source.id,
                target_entity_id=target.id,
                relation_type=edge.relation,
                description=edge.description or None,
                article_id=article.id,
            )
            .on_conflict_do_update(
                index_elements=[
                    "company_domain",
                    "source_entity_id",
                    "target_entity_id",
                    "relation_type",
                ],
                # The same fact re-asserted in a newer document should point provenance
                # at that document; the description is left as first-written, same
                # reasoning as _get_or_create_entity above.
                set_={"article_id": article.id},
            )
        )
        await db.execute(stmt)

    await db.flush()
    return len(entities_by_key)


def _serialize_entity(entity: GraphEntity) -> dict[str, Any]:
    return {
        "id": str(entity.id),
        "name": entity.name,
        "type": entity.entity_type,
        "description": entity.description,
        "mention_count": entity.mention_count,
    }


def _serialize_relationship(edge: GraphRelationship) -> dict[str, Any]:
    return {
        "id": str(edge.id),
        "source_entity_id": str(edge.source_entity_id),
        "target_entity_id": str(edge.target_entity_id),
        "relation": edge.relation_type,
        "description": edge.description,
        "article_id": str(edge.article_id) if edge.article_id else None,
    }


async def entity_exists(db: AsyncSession, company_domain: str, entity_id: uuid.UUID) -> bool:
    return (
        await db.scalar(
            select(GraphEntity.id).where(
                GraphEntity.id == entity_id, GraphEntity.company_domain == company_domain
            )
        )
    ) is not None


async def get_entity_detail(
    db: AsyncSession, company_domain: str, entity_id: uuid.UUID
) -> dict[str, Any] | None:
    """One entity plus its direct relationships and the articles that mention it."""
    entity = await db.scalar(
        select(GraphEntity).where(
            GraphEntity.id == entity_id, GraphEntity.company_domain == company_domain
        )
    )
    if not entity:
        return None

    edges = (
        (
            await db.execute(
                select(GraphRelationship).where(
                    GraphRelationship.company_domain == company_domain,
                    (GraphRelationship.source_entity_id == entity_id)
                    | (GraphRelationship.target_entity_id == entity_id),
                )
            )
        )
        .scalars()
        .all()
    )
    neighbor_ids = {
        edge.target_entity_id if edge.source_entity_id == entity_id else edge.source_entity_id
        for edge in edges
    }
    neighbors = {
        neighbor.id: neighbor
        for neighbor in (
            (
                await db.execute(
                    select(GraphEntity).where(GraphEntity.id.in_(neighbor_ids))
                )
            )
            .scalars()
            .all()
        )
    } if neighbor_ids else {}

    articles = (
        (
            await db.execute(
                select(Article.id, Article.title, Article.status)
                .join(ArticleEntityMention, ArticleEntityMention.article_id == Article.id)
                .where(ArticleEntityMention.entity_id == entity_id)
                .order_by(Article.title)
            )
        )
        .all()
    )

    return {
        **_serialize_entity(entity),
        "relationships": [
            {
                **_serialize_relationship(edge),
                "other_entity": _serialize_entity(neighbors[
                    edge.target_entity_id if edge.source_entity_id == entity_id else edge.source_entity_id
                ])
                if (edge.target_entity_id if edge.source_entity_id == entity_id else edge.source_entity_id)
                in neighbors
                else None,
                "direction": "outgoing" if edge.source_entity_id == entity_id else "incoming",
            }
            for edge in edges
        ],
        "articles": [
            {"id": str(article_id), "title": title, "status": status}
            for article_id, title, status in articles
        ],
    }


async def get_entity_neighbors(
    db: AsyncSession, company_domain: str, entity_id: uuid.UUID, depth: int = 1
) -> dict[str, Any]:
    """Breadth-first expansion around one entity, up to `depth` hops.

    An iterative BFS over a handful of queries rather than a single recursive SQL query:
    `depth` is capped at MAX_NEIGHBOR_DEPTH, so the query count is bounded and each step
    stays a plain, easily tenant-scoped SELECT rather than raw recursive SQL.
    """
    depth = max(1, min(depth, MAX_NEIGHBOR_DEPTH))
    visited_ids: set[uuid.UUID] = {entity_id}
    frontier: set[uuid.UUID] = {entity_id}
    all_edges: dict[uuid.UUID, GraphRelationship] = {}

    for _ in range(depth):
        if not frontier:
            break
        edges = (
            (
                await db.execute(
                    select(GraphRelationship).where(
                        GraphRelationship.company_domain == company_domain,
                        (GraphRelationship.source_entity_id.in_(frontier))
                        | (GraphRelationship.target_entity_id.in_(frontier)),
                    )
                )
            )
            .scalars()
            .all()
        )
        next_frontier: set[uuid.UUID] = set()
        for edge in edges:
            all_edges[edge.id] = edge
            for candidate in (edge.source_entity_id, edge.target_entity_id):
                if candidate not in visited_ids:
                    next_frontier.add(candidate)
        visited_ids |= next_frontier
        frontier = next_frontier

    entities = (
        (
            await db.execute(
                select(GraphEntity).where(GraphEntity.id.in_(visited_ids))
            )
        )
        .scalars()
        .all()
    )
    return {
        "entities": [_serialize_entity(entity) for entity in entities],
        "relationships": [_serialize_relationship(edge) for edge in all_edges.values()],
    }


async def list_entities(
    db: AsyncSession,
    company_domain: str,
    *,
    query: str | None = None,
    entity_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    stmt = select(GraphEntity).where(GraphEntity.company_domain == company_domain)
    if entity_type:
        stmt = stmt.where(GraphEntity.entity_type == entity_type)
    if query:
        stmt = stmt.where(GraphEntity.normalized_name.icontains(normalize_entity_name(query)))
    stmt = stmt.order_by(GraphEntity.mention_count.desc(), GraphEntity.name).limit(limit).offset(offset)
    rows = (await db.execute(stmt)).scalars().all()
    return [_serialize_entity(entity) for entity in rows]
