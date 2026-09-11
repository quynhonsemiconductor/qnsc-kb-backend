"""Read access to the tenant knowledge graph, plus an admin backfill trigger.

Reading is scoped like any other article-derived view (`article.read`): the graph is
built entirely from published articles a user could already read via search, so no new
authorization concept is introduced here. Backfilling existing articles that published
before this feature existed is an operator action, gated the same way
`/governance/index/reprocess` is.
"""
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db, require_permission
from src.domain.entity_extraction import extract_entities_and_relationships
from src.domain.graph_service import (
    apply_extraction,
    entity_exists,
    get_entity_detail,
    get_entity_neighbors,
    list_entities,
)
from src.models import User
from src.models.article import Article
from src.models.graph import ArticleEntityMention

router = APIRouter()

#: Bounds one reprocess call to roughly the same order of magnitude as
#: APPROVAL_AGENT_BATCH_LIMIT: a corpus backfill is a deliberate, repeatable operator
#: action, not a single call expected to walk an unbounded article table inline.
REPROCESS_BATCH_LIMIT = 50


@router.get("/entities")
async def get_entities(
    q: str | None = Query(default=None, max_length=200),
    type: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(require_permission("article.read")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    return await list_entities(
        db,
        current_user.company_domain,
        query=q,
        entity_type=type,
        limit=limit,
        offset=offset,
    )


@router.get("/entities/{entity_id}")
async def get_entity(
    entity_id: uuid.UUID,
    current_user: User = Depends(require_permission("article.read")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    detail = await get_entity_detail(db, current_user.company_domain, entity_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Entity not found")
    return detail


@router.get("/entities/{entity_id}/neighbors")
async def get_neighbors(
    entity_id: uuid.UUID,
    depth: int = Query(default=1, ge=1, le=3),
    current_user: User = Depends(require_permission("article.read")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if not await entity_exists(db, current_user.company_domain, entity_id):
        raise HTTPException(status_code=404, detail="Entity not found")
    return await get_entity_neighbors(db, current_user.company_domain, entity_id, depth=depth)


@router.post("/reprocess", status_code=202)
async def reprocess_graph(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Backfill entities/relationships for published articles the graph has not seen.

    Synchronous and capped at REPROCESS_BATCH_LIMIT, the same shape
    `approval-agent/run` uses for an operator-triggered batch: no new job-tracking model
    is worth building for an action that is rerun by calling this again until it reports
    zero processed.
    """
    article_ids = (
        (
            await db.execute(
                select(Article.id)
                .outerjoin(
                    ArticleEntityMention, ArticleEntityMention.article_id == Article.id
                )
                .where(
                    Article.company_domain == current_user.company_domain,
                    Article.status == "published",
                    ArticleEntityMention.id.is_(None),
                )
                .limit(REPROCESS_BATCH_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    processed = 0
    for article_id in article_ids:
        article = await db.get(Article, article_id)
        if not article:
            continue
        try:
            extraction = await extract_entities_and_relationships(article.title, article.body_md)
            await apply_extraction(db, article, extraction)
            await db.commit()
            processed += 1
        except Exception:
            await db.rollback()

    return {
        "processed": processed,
        "remaining_at_least": max(0, len(article_ids) - processed),
    }
