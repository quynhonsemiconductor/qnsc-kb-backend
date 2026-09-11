"""Topical cross-linking between published articles, computed from stored embeddings.

The near-duplicate detector in `similarity.py` answers "does this look like the same
TEXT as something already here" -- checked at ingest time, on wording. It cannot find
two independently-written articles that cover the same topic in different words, because
nothing about that pair looks alike as text. This module answers the different question
of whether two articles are close in MEANING, reusing the same chunk embeddings the
search index already stores -- no new embedding calls, no LLM call, just a pgvector
nearest-neighbor query against vectors that exist regardless of whether this ever runs.

Deliberately does not go through `SearchService`/`hybrid_search`: those are
permission-aware, per-user-request machinery, and this is a company-wide maintenance
job with no user to scope them to. Every query here is filtered by `company_domain` and
published status directly in SQL instead -- articles from another tenant are never even
candidates, and a reader of the resulting `related_article_ids` still goes through the
ordinary `PermissionService` check on the read path (`article_repo.list_related`), so a
bug here can suggest a bad link but cannot bypass access control on its own.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.models.article import Article
from src.models.chunk import ArticleChunk

logger = structlog.get_logger()

#: Independently-written articles rarely land as close as a chunk a real search would
#: have retrieved for a query -- so linking reuses that same, already-tuned distance
#: (VECTOR_DISTANCE_THRESHOLD) as its own bar, rather than inventing a new one.
TOPICAL_DISTANCE_THRESHOLD = settings.VECTOR_DISTANCE_THRESHOLD
MAX_RELATED_ARTICLES = 5


async def _centroid_embedding(db: AsyncSession, article_id: uuid.UUID) -> list[float] | None:
    """Average every current-model chunk embedding for one article into one vector.

    A centroid over the whole article, not just its first chunk, so one atypical
    paragraph (a references section, a boilerplate header) cannot dominate what the
    article is considered to be "about". Returns None when the article has no chunks on
    the current embedding version yet -- freshly uploaded, still indexing, or on an old
    model version pending re-embedding -- so the caller can skip it for this run rather
    than linking off a stale or absent vector.
    """
    rows = (
        await db.execute(
            select(ArticleChunk.embedding).where(
                ArticleChunk.article_id == article_id,
                ArticleChunk.embedding_version == settings.EMBEDDING_VERSION,
                ArticleChunk.embedding.is_not(None),
            )
        )
    ).scalars().all()
    vectors = [list(vector) for vector in rows if vector]
    if not vectors:
        return None
    width = len(vectors[0])
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(width)]


def collapse_to_top_articles(
    chunk_rows: list[tuple[Any, float]],
    *,
    threshold: float = TOPICAL_DISTANCE_THRESHOLD,
    limit: int = MAX_RELATED_ARTICLES,
) -> list[str]:
    """Reduce per-CHUNK distance rows to the closest distinct ARTICLEs, pure and DB-free.

    An article can contribute many rows (one per chunk); each is collapsed to its best
    (minimum-distance) chunk before ranking, so a long article is not favoured simply for
    having more chances to land a close chunk. Rows are read in whatever order `chunk_rows`
    already has -- the caller's `ORDER BY distance` -- but the min-tracking below does not
    depend on that order being correct, only on having seen every row once.
    """
    best_distance_by_article: dict[str, float] = {}
    for candidate_article_id, distance in chunk_rows:
        key = str(candidate_article_id)
        if key not in best_distance_by_article or distance < best_distance_by_article[key]:
            best_distance_by_article[key] = distance

    ranked = sorted(
        (
            (candidate_id, distance)
            for candidate_id, distance in best_distance_by_article.items()
            if distance <= threshold
        ),
        key=lambda item: item[1],
    )
    return [candidate_id for candidate_id, _distance in ranked[:limit]]


async def find_topical_matches(
    db: AsyncSession, article: Article, *, limit: int = MAX_RELATED_ARTICLES
) -> list[str]:
    """IDs of published articles closest in meaning to `article`, same tenant only.

    Never raises for a missing embedding or an empty corpus: an article still indexing,
    or a tenant with too few other documents, both just yield no suggestions this run
    rather than failing whatever caller is populating `related_article_ids`.
    """
    centroid = await _centroid_embedding(db, article.id)
    if centroid is None:
        return []

    # Several rows per candidate article (one per chunk), so the row limit is well above
    # `limit` to leave room for finding that many DISTINCT articles once collapsed below.
    rows = (
        await db.execute(
            select(
                ArticleChunk.article_id,
                ArticleChunk.embedding.cosine_distance(centroid).label("distance"),
            )
            .join(Article, Article.id == ArticleChunk.article_id)
            .where(
                Article.company_domain == article.company_domain,
                Article.status == "published",
                Article.lifecycle_status == "active",
                Article.id != article.id,
                ArticleChunk.embedding_version == settings.EMBEDDING_VERSION,
                ArticleChunk.embedding.is_not(None),
            )
            .order_by("distance")
            .limit(limit * 20)
        )
    ).all()
    return collapse_to_top_articles(list(rows), limit=limit)
