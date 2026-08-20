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
from src.rag.reranker import normalize_query, rerank_chunks, score_retrieval_text
from src.models.ops import SearchLog
from src.repositories.feature_flags import FeatureFlagRepository

logger = structlog.get_logger()

import asyncio
from src.lib.embeddings import get_bge_embedding, get_bge_embeddings

async def get_text_embedding(text: str) -> list[float] | None:
    try:
        embedding = await asyncio.to_thread(get_bge_embedding, text)
        logger.info(
            "Search embedding generated",
            query_length=len(text),
            embedding_dimension=len(embedding) if embedding else 0,
            embedding_model=settings.EMBEDDING_MODEL,
        )
        return embedding
    except Exception as e:
        logger.error(
            "Error generating local BGE embedding; continuing with keyword search",
            error=str(e),
            embedding_model=settings.EMBEDDING_MODEL,
        )
        return None


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

    async def search(
        self,
        user: User,
        query: str,
        filters: dict | None = None,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if not any(
            AuthorizationService.has_permission(user, "article.read", requested_scope=scope)
            for scope in ("own", "department", "company", "global")
        ):
            return []

        started = time.perf_counter()

        user_bitmask = PermissionService.calculate_user_bitmask(user)
        effective_filters = dict(filters or {})
        if not AuthorizationService.has_permission(user, "article.read", requested_scope="global"):
            effective_filters["company_domain"] = user.company_domain
        effective_filters["bypass_access_groups"] = AuthorizationService.has_full_company_article_access(user)
        if not AuthorizationService.has_full_company_article_access(user):
            effective_filters["departments"] = sorted(AuthorizationService.member_department_names(user))
        if not AuthorizationService.has_permission(user, "article.read", requested_scope="company"):
            if AuthorizationService.has_permission(user, "article.read", requested_scope="department"):
                effective_filters["departments"] = sorted(AuthorizationService.owned_department_names(user))
                effective_filters["bypass_access_groups"] = True
            elif AuthorizationService.has_permission(user, "article.read", requested_scope="own"):
                effective_filters["owner_id"] = user.id
                effective_filters["bypass_access_groups"] = True
        logger.info(
            "Search started",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            query_length=len(query),
            limit=limit,
            filters=effective_filters,
            user_id=str(user.id),
            user_role=user.role,
            user_department=user.dept,
            access_group_count=len(user.groups),
            user_access_bitmask=user_bitmask,
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
            await self.gov_repo.log_gap(query=query, company_domain=user.company_domain, dept=user.dept)
            return []

        # 1. Get embedding asynchronously
        embedding = await get_text_embedding(retrieval_query)
        
        # 2. Query hybrid search
        candidates = await self.chunk_repo.hybrid_search(
            query=retrieval_query,
            query_embedding=embedding,
            user_bitmask=user_bitmask,
            user=user,
            limit=limit,
            filters=effective_filters
        )
        reranking_enabled = not self.feature_flags or await self.feature_flags.is_enabled("rag.reranker", user)
        chunks = rerank_chunks(retrieval_query, candidates, limit=limit) if reranking_enabled else candidates[:limit]
        # Vector similarity alone is not enough: short or vague inputs can be
        # close to an unrelated document in embedding space. Keep a result only
        # when the reranked passage has at least one meaningful lexical signal.
        relevance_threshold = settings.RAG_MIN_RELEVANCE_SCORE
        chunks = [
            chunk for chunk in chunks
            if getattr(chunk, "article", None) is not None and PermissionService.can_view_article(user, chunk.article)
            if score_retrieval_text(
                retrieval_query,
                getattr(chunk, "chunk_text", "") or getattr(getattr(chunk, "parent_chunk", None), "text", ""),
                getattr(getattr(chunk, "article", None), "title", ""),
                getattr(getattr(chunk, "parent_chunk", None), "section_ref", ""),
            ) >= relevance_threshold
        ]
        logger.info(
            "Search repository completed",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            embedding_available=embedding is not None,
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
                user_access_bitmask=user_bitmask,
                filters=effective_filters,
            )
            await self.gov_repo.log_gap(query=query, company_domain=user.company_domain, dept=user.dept)

        # 4. Format search results
        formatted_results = []
        for idx, chunk in enumerate(chunks):
            parent = chunk.parent_chunk
            article = chunk.article
            AuthorizationService.restrict_article_metadata(user, article)
            
            score = score_retrieval_text(retrieval_query, chunk.chunk_text, article.title, parent.section_ref if parent else "")
            
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
