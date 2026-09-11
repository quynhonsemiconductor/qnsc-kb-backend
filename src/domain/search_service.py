import httpx
import structlog
import time
import hashlib
from typing import Any
from fastapi import HTTPException
from src.core.config import settings
from src.core.privacy import REDACTED_OPERATIONAL_CONTENT
from src.models.user import User
from src.models.governance import Gap
from src.models.ai import AiUsageLog
from src.repositories.chunk import ChunkRepository
from src.repositories.governance import GovernanceRepository
from src.domain.permissions import PermissionService
from src.domain.rbac import AuthorizationService
from src.rag.reranker import (
    chunk_passage,
    normalize_query,
    prepare_query_for_chunks,
    rerank_chunks_with_scores,
    retrieval_score,
)
from src.rag.cross_encoder import CrossEncoderUnavailable, reorder_by_cross_encoder
from src.rag.cross_encoder import score as cross_encoder_score
from src.rag.query_router import is_comparison_query, split_comparison_subjects
from src.models.ops import SearchLog
from src.repositories.feature_flags import FeatureFlagRepository

logger = structlog.get_logger()

import asyncio
from collections import OrderedDict
from src.lib.embeddings import get_bge_embedding, get_bge_embeddings

# Embedding one query is a full forward pass of the model — the most expensive single
# step in a search, and a pure function of (text, model version). Queries repeat heavily
# in a knowledge base: the same question from different people, a retry, the AI path
# searching twice in one turn. This keeps the last few hundred.
#
# Keyed by DIGEST, not by the query itself, so no user's question text is retained in
# process memory. The entry is a tuple, and a copy is handed out, so no caller can
# mutate what the next one reads.
_QUERY_EMBEDDING_CACHE: "OrderedDict[tuple[str, str], tuple[float, ...]]" = OrderedDict()
_QUERY_EMBEDDING_CACHE_MAX = 256


def _query_embedding_key(text: str) -> tuple[str, str]:
    return (
        settings.EMBEDDING_VERSION,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def reset_query_embedding_cache() -> None:
    """Drop the cache. For tests, and for a deliberate model/version switch."""
    _QUERY_EMBEDDING_CACHE.clear()


class VectorSearchUnavailable(RuntimeError):
    """The query could not be embedded, so only keyword retrieval ran.

    Raised by `embed_query`, and carried on `SearchService.vector_search_degraded` for
    the callers that would rather serve keyword hits than nothing. It exists because the
    two states this used to collapse into a bare `None` are not the same thing: "the
    knowledge base has nothing on this" is an answer, and "half of retrieval is broken"
    is an incident.
    """


async def embed_query(text: str) -> list[float]:
    """Embed a query, raising `VectorSearchUnavailable` when the model cannot.

    Failing loudly is the point. Returning None here meant every caller silently
    continued on lexical matching alone, including the RAG answer path, which then
    grounded answers in a pool that was never scored by similarity -- with nothing in
    the response, and nothing short of reading the logs, to say so.
    """
    key = _query_embedding_key(text)
    cached = _QUERY_EMBEDDING_CACHE.get(key)
    if cached is not None:
        _QUERY_EMBEDDING_CACHE.move_to_end(key)
        return list(cached)
    try:
        embedding = await asyncio.to_thread(get_bge_embedding, text)
    except Exception as exc:
        logger.error(
            "Query embedding failed; vector search is unavailable for this query",
            error=str(exc),
            embedding_model=settings.EMBEDDING_MODEL,
        )
        raise VectorSearchUnavailable(str(exc)) from exc
    if not embedding:
        # An empty vector is as unusable as an exception, and pgvector would reject it.
        logger.error(
            "Query embedding returned no vector; vector search is unavailable",
            embedding_model=settings.EMBEDDING_MODEL,
        )
        raise VectorSearchUnavailable("the embedding model returned no vector")
    _QUERY_EMBEDDING_CACHE[key] = tuple(embedding)
    while len(_QUERY_EMBEDDING_CACHE) > _QUERY_EMBEDDING_CACHE_MAX:
        _QUERY_EMBEDDING_CACHE.popitem(last=False)
    logger.info(
        "Search embedding generated",
        query_length=len(text),
        embedding_dimension=len(embedding),
        embedding_model=settings.EMBEDDING_MODEL,
    )
    return embedding


async def get_text_embeddings(texts: list[str]) -> list[list[float]] | None:
    try:
        return await asyncio.to_thread(get_bge_embeddings, texts)
    except Exception as exc:
        logger.error("Error generating local BGE embedding batch", error=str(exc), batch_size=len(texts))
        return None

class SearchService:
    def __init__(self, chunk_repo: ChunkRepository, gov_repo: GovernanceRepository, feature_flags: FeatureFlagRepository | None = None):
        self.chunk_repo = chunk_repo
        self.gov_repo = gov_repo
        self.feature_flags = feature_flags
        #: Whether the most recent `search` ran without vector retrieval because the
        #: query could not be embedded. A caller that grounds anything on the results —
        #: `AiService.ask` above all — must read this, because a keyword-only pool is a
        #: different pool, and "found nothing" and "could only look lexically" call for
        #: different answers. One service instance serves one request here, so this is
        #: per-request state, not shared.
        self.vector_search_degraded = False

    async def _record_gap(self, user: User, query: str) -> None:
        """Note that a search found nothing, without letting that end the search.

        Both call sites used to await log_gap directly. A gap row that could not be
        written therefore propagated out of search and out of AiService.ask, so the
        reader got a 500 on the answer stream instead of an empty result -- caused by a
        query 26 characters over the column width.

        Recording that we found nothing is bookkeeping. It must never be the reason
        nothing is returned.
        """
        try:
            await self.gov_repo.log_gap(
                query=query, company_domain=user.company_domain, dept=user.dept
            )
        except Exception:
            logger.warning(
                "Could not record the search gap",
                query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                exc_info=True,
            )
            # log_gap commits, so a failure leaves the session in a failed transaction
            # and every later statement in the same request raises too. Swallowing the
            # error without this would move the 500 rather than remove it.
            try:
                await self.gov_repo.db.rollback()
            except Exception:
                logger.warning("Could not reset the session after a gap write", exc_info=True)

    async def search(
        self,
        user: User,
        query: str,
        filters: dict | None = None,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        # Reset before any early return, so a caller never reads the previous query's
        # verdict for one that never reached the embedder.
        self.vector_search_degraded = False
        if not query.strip():
            return []
        if not any(
            AuthorizationService.has_permission(user, "article.read", requested_scope=scope)
            for scope in ("own", "department", "company", "global")
        ):
            return []

        started = time.perf_counter()

        effective_filters = dict(filters or {})
        if not AuthorizationService.has_permission(user, "article.read", requested_scope="global"):
            effective_filters["company_domain"] = user.company_domain
        if not AuthorizationService.has_full_company_article_access(user):
            effective_filters["departments"] = sorted(AuthorizationService.member_department_names(user))
        if not AuthorizationService.has_permission(user, "article.read", requested_scope="company"):
            if AuthorizationService.has_permission(user, "article.read", requested_scope="department"):
                effective_filters["departments"] = sorted(AuthorizationService.owned_department_names(user))
            elif AuthorizationService.has_permission(user, "article.read", requested_scope="own"):
                effective_filters["owner_id"] = user.id
        logger.info(
            "Search started",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            query_length=len(query),
            limit=limit,
            filters=effective_filters,
            user_id=str(user.id),
            user_role=user.role,
            user_department=user.dept,
            department_count=len(user.departments),
        )
        
        # Question words such as "what is" / "là gì" are not useful search
        # terms. Normalize them before both vector and keyword retrieval so
        # the exact subject (for example, CTS) is not drowned out by generic
        # language. Keep the original query for diagnostics and gap logging.
        retrieval_query = normalize_query(query)
        logger.info(
            "Search query normalized",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            retrieval_query_length=len(retrieval_query),
        )

        if not retrieval_query:
            await self._record_gap(user, query)
            return []

        # A comparison question ("so sánh SOP-114 và SOP-118") retrieves worse as one
        # query than as several: the single embedding sits between both topics instead
        # of close to either. Detected on the ORIGINAL query, not `retrieval_query` --
        # normalize_query already strips "và"/"vs" as stopwords, which are exactly the
        # markers a comparison needs to split on.
        comparison_subjects = (
            split_comparison_subjects(query) if is_comparison_query(query) else []
        )
        multi_subject_queries = (
            [normalize_query(subject) for subject in comparison_subjects]
            if len(comparison_subjects) > 1
            else []
        )
        multi_subject_queries = [item for item in multi_subject_queries if item]

        async def _embed_or_degrade(text: str) -> list[float] | None:
            try:
                return await embed_query(text)
            except VectorSearchUnavailable as exc:
                # Keyword retrieval still finds things, and abandoning the search
                # entirely would take the product down for a recoverable model fault.
                # But the caller is told, so an answer built on this pool can say what
                # it is built on.
                self.vector_search_degraded = True
                logger.warning(
                    "Search degraded to keyword-only retrieval",
                    query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    reason=str(exc),
                    embedding_model=settings.EMBEDDING_MODEL,
                )
                return None

        # 1. Get embedding asynchronously
        embedding = await _embed_or_degrade(retrieval_query)

        reranking_enabled = not self.feature_flags or await self.feature_flags.is_enabled("rag.reranker", user)
        # The cross-encoder REORDERS the pool the lexical scorer already ranked; it never
        # runs on its own. Off entirely when the base reranker is off, so disabling
        # "rag.reranker" still means "no reranking of any kind" as it always has.
        cross_encoder_enabled = (
            reranking_enabled
            and settings.CROSS_ENCODER_RERANKER_ENABLED
            and (
                not self.feature_flags
                or await self.feature_flags.is_enabled("rag.cross_encoder_reranker", user)
            )
        )
        # A cross-encoder needs a wider pool than `limit` to have anything to reorder --
        # hybrid_search itself is asked for `limit` rows otherwise, unchanged from before
        # this feature existed. Requesting the wider pool only when the flag is actually
        # on keeps every other search exactly as cheap as it was.
        pool_limit = settings.CROSS_ENCODER_CANDIDATE_POOL if cross_encoder_enabled else limit

        # 2. Query hybrid search -- once per comparison subject when the query was
        # decomposed above, merged and de-duplicated by chunk id; a single call
        # otherwise, exactly as before this feature existed.
        if multi_subject_queries:
            candidates = []
            seen_chunk_ids: set = set()
            for subject_query in multi_subject_queries:
                subject_embedding = (
                    embedding
                    if subject_query == retrieval_query
                    else await _embed_or_degrade(subject_query)
                )
                subject_candidates = await self.chunk_repo.hybrid_search(
                    query=subject_query,
                    query_embedding=subject_embedding,
                    user=user,
                    limit=pool_limit,
                    filters=effective_filters,
                )
                for chunk in subject_candidates:
                    if chunk.id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk.id)
                    candidates.append(chunk)
        else:
            candidates = await self.chunk_repo.hybrid_search(
                query=retrieval_query,
                query_embedding=embedding,
                user=user,
                limit=pool_limit,
                filters=effective_filters
            )
        # Scored ONCE, here. The threshold below and the score reported per result both
        # reuse these values rather than scoring the same passage again.
        if reranking_enabled:
            ranked = rerank_chunks_with_scores(retrieval_query, candidates, limit=pool_limit)
        else:
            # Prepared against the same pool the reranker would have used, so turning
            # the reranker off changes the ordering and not the calibration.
            prepared = prepare_query_for_chunks(retrieval_query, candidates)
            ranked = [
                (chunk, retrieval_score(retrieval_query, chunk, prepared))
                for chunk in candidates[:pool_limit]
            ]

        if cross_encoder_enabled and len(ranked) > 1:
            try:
                cross_scores = await asyncio.to_thread(
                    cross_encoder_score,
                    retrieval_query,
                    [chunk_passage(chunk) for chunk, _lexical_score in ranked],
                )
                # Order comes from the cross-encoder; the VALUE reported and thresholded
                # below stays the lexical score. RAG_MIN_RELEVANCE_SCORE and every caller
                # reading `score` were tuned against that scale, and mixing in a
                # cross-encoder's own (unbounded, model-specific) scale here would shift
                # every threshold decision with no warning.
                ranked = reorder_by_cross_encoder(ranked, cross_scores)
            except CrossEncoderUnavailable as exc:
                logger.warning(
                    "Cross-encoder reranking failed, falling back to lexical order",
                    error=str(exc),
                )
        ranked = ranked[:limit]
        # Vector similarity alone is not enough: short or vague inputs can be
        # close to an unrelated document in embedding space. Keep a result only
        # when the reranked passage has at least one meaningful lexical signal.
        relevance_threshold = settings.RAG_MIN_RELEVANCE_SCORE
        scored_chunks = [
            (chunk, score) for chunk, score in ranked
            if getattr(chunk, "article", None) is not None
            and PermissionService.can_view_article(user, chunk.article)
            and score >= relevance_threshold
        ]
        chunks = [chunk for chunk, _score in scored_chunks]
        logger.info(
            "Search repository completed",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            embedding_available=embedding is not None,
            vector_search_degraded=self.vector_search_degraded,
            candidate_count=len(candidates),
            reranking_enabled=reranking_enabled,
            result_count=len(chunks),
        )

        # 3. Log search query gap if no results found
        if not chunks:
            logger.info(
                "Search returned zero results, logging gap",
                query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                reason="no published, permission-matching chunks matched vector or keyword search",
                filters=effective_filters,
            )
            await self._record_gap(user, query)

        # 4. Format search results
        formatted_results = []
        for idx, (chunk, score) in enumerate(scored_chunks):
            parent = chunk.parent_chunk
            article = chunk.article
            AuthorizationService.restrict_article_metadata(user, article)

            formatted_results.append({
                "chunk_id": str(chunk.id),
                "parent_chunk_id": str(chunk.parent_chunk_id),
                "article_id": str(article.id),
                "title": article.title,
                "dept": article.dept,
                "domain": article.domain,
                "type": article.type,
                "sensitivity": article.sensitivity,
                "chunk_text": chunk.chunk_text,
                "parent_text": parent.text if parent else chunk.chunk_text,
                "child_texts": [child.chunk_text for child in parent.child_chunks] if parent else [chunk.chunk_text],
                "section_ref": parent.section_ref if parent else None,
                "heading": (parent.heading if parent else None) or (parent.section_ref if parent else None),
                "chunk_type": getattr(chunk, "chunk_type", "text"),
                "chunking_version": getattr(chunk, "chunking_version", None),
                "page_number": chunk.page_number if chunk.page_number is not None else (parent.page_number if parent else None),
                "source_url": f"/api/v1/articles/{article.id}/source" + (f"?page={chunk.page_number or parent.page_number}" if (chunk.page_number or (parent and parent.page_number)) else ""),
                "owner_email": getattr(getattr(article, "owner", None), "email", None),
                "last_reviewed": article.last_reviewed.isoformat() if article.last_reviewed else None,
                "score": score
            })

        try:
            self.chunk_repo.db.add(SearchLog(
                user_id=user.id,
                query=REDACTED_OPERATIONAL_CONTENT,
                result_count=len(formatted_results),
                latency_ms=int((time.perf_counter() - started) * 1000),
            ))
            await self.chunk_repo.db.commit()
        except Exception as exc:
            logger.warning("Search log persistence failed", error=str(exc))
        return formatted_results
