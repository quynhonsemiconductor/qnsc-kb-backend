import uuid
import re
import unicodedata
import hashlib
import structlog
from typing import Sequence
from sqlalchemy import select, delete, update, and_, or_, text, func, exists, not_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models.chunk import ParentChunk, ArticleChunk, ChunkMetadata
from src.models.article import Article, ArticleTag, ArticleUserPermission
from src.models.user import Department
from src.core.config import settings
from src.repositories.article import ArticleRepository

logger = structlog.get_logger()

class ChunkRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_parent_chunk(self, parent: ParentChunk) -> ParentChunk:
        self.db.add(parent)
        await self.db.commit()
        await self.db.refresh(parent)
        return parent

    async def create_child_chunks(self, chunks: list[ArticleChunk]) -> list[ArticleChunk]:
        for c in chunks:
            self.db.add(c)
        await self.db.commit()
        for c in chunks:
            await self.db.refresh(c)
        return chunks

    async def delete_by_article_id(self, article_id: uuid.UUID) -> None:
        await self.db.execute(delete(ParentChunk).where(ParentChunk.article_id == article_id))
        await self.db.execute(delete(ArticleChunk).where(ArticleChunk.article_id == article_id))
        await self.db.commit()

    async def get_by_article_id(self, article_id: uuid.UUID) -> Sequence[ArticleChunk]:
        result = await self.db.execute(
            select(ArticleChunk)
            .where(ArticleChunk.article_id == article_id)
            .order_by(ArticleChunk.chunk_index)
        )
        return result.scalars().all()

    async def get_parent_chunk(self, parent_id: uuid.UUID) -> ParentChunk | None:
        result = await self.db.execute(
            select(ParentChunk).where(ParentChunk.id == parent_id)
        )
        return result.scalar_one_or_none()

    async def authorized_chunk_ids(self, user: object, chunk_ids: list[uuid.UUID]) -> set[str]:
        """Return only citation chunks still visible to the current user."""
        if not chunk_ids:
            return set()
        from src.domain.rbac import AuthorizationService

        conditions = [ArticleChunk.id.in_(chunk_ids), Article.status == "published", *ArticleRepository._authorized_article_filters(user)]
        result = await self.db.execute(
            select(ArticleChunk)
            .join(Article, Article.id == ArticleChunk.article_id)
            .options(
                selectinload(ArticleChunk.article).selectinload(Article.sources),
                selectinload(ArticleChunk.article).selectinload(Article.owner),
                selectinload(ArticleChunk.article).selectinload(Article.access_groups),
                selectinload(ArticleChunk.article).selectinload(Article.departments),
                selectinload(ArticleChunk.article).selectinload(Article.user_permissions),
            )
            .where(*conditions)
        )
        return {str(chunk.id) for chunk in result.scalars().all()}

    async def update_permissions(self, article_id: uuid.UUID, bitmap: int, sensitivity: str, visibility: str, dept: str) -> None:
        await self.db.execute(
            update(ArticleChunk)
            .where(ArticleChunk.article_id == article_id)
            .values(
                access_group_bitmap=bitmap,
                sensitivity=sensitivity,
                visibility=visibility,
                department_id=dept
            )
        )
        await self.db.commit()

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float] | None,
        user_bitmask: int,
        user: object,
        limit: int = 5,
        filters: dict | None = None
    ) -> list[ArticleChunk]:
        """
        Executes a hybrid search:
        - If query_embedding is provided, calculates vector similarity.
        - Performs a full-text search on chunk_text using tsvector.
        - Merges the two lists using a reciprocal rank scoring mechanism.
        - Enforces access control natively: (access_group_bitmap & user_bitmask) != 0.
        """
        filters = filters or {}
        
        # Base filter: permissions bitwise AND
        # We also enforce that the article must be published (not draft or soft deleted)
        where_clauses = [
            # Only successfully indexed, active published articles are RAG candidates.
            Article.status == "published",
            Article.lifecycle_status == "active",
            Article.index_status == "ready",
        ]
        # Apply the complete Article authorization predicate in the retrieval
        # query. The bitmask remains the fast native ACL for ordinary content;
        # explicit-user visibility and explicit denies are relational policy
        # records and are included in the same SQL statement.
        where_clauses.extend(ArticleRepository._authorized_article_filters(user))
        explicit_allow = exists(select(ArticleUserPermission.id).where(
            ArticleUserPermission.article_id == Article.id,
            ArticleUserPermission.user_id == user.id,
            ArticleUserPermission.effect == "allow",
        ))
        explicit_deny = exists(select(ArticleUserPermission.id).where(
            ArticleUserPermission.article_id == Article.id,
            ArticleUserPermission.user_id == user.id,
            ArticleUserPermission.effect == "deny",
        ))
        where_clauses.append(not_(explicit_deny))
        # Audience authorization is already present in the shared Article
        # predicate above.  Keeping a second bitmask gate here made search a
        # different permission algorithm and imposed a 62-group ceiling.

        if filters.get("company_domain"):
            where_clauses.append(Article.company_domain == filters["company_domain"])

        # SearchService narrows users with department- or owner-scoped read
        # permissions before retrieval. Keep those effective scopes in the
        # same SQL predicate as the Article authorization filters; otherwise
        # a department/own search could retrieve a broader candidate set and
        # rely on a later Python check. Public content remains searchable
        # alongside the user's effective narrow scope.
        scope_conditions = [Article.sensitivity == "public"]
        department_names = {
            str(item).strip()
            for item in filters.get("departments", []) or []
            if str(item).strip()
        }
        if department_names:
            scope_conditions.append(
                or_(
                    Article.dept.in_(department_names),
                    Article.departments.any(Department.name.in_(department_names)),
                )
            )
        if filters.get("owner_id"):
            scope_conditions.append(Article.owner_id == filters["owner_id"])
        if len(scope_conditions) > 1:
            where_clauses.append(or_(*scope_conditions))

        # A deactivated department is no longer a valid content scope.
        where_clauses.append(exists(select(Department.id).where(
            Department.company_domain == Article.company_domain,
            Department.name == Article.dept,
            Department.active.is_(True),
        )))

        if filters.get("dept"):
            where_clauses.append(or_(ArticleChunk.department_id == filters["dept"], Article.departments.any(Department.name == filters["dept"])))
        if filters.get("sensitivity"):
            where_clauses.append(ArticleChunk.sensitivity == filters["sensitivity"])
        if filters.get("type"):
            where_clauses.append(Article.type == filters["type"])
        if filters.get("status"):
            where_clauses.append(Article.status == filters["status"])
        if filters.get("language"):
            where_clauses.append(Article.language == filters["language"])
        if filters.get("tag"):
            where_clauses.append(Article.tags.any(ArticleTag.tag == filters["tag"]))
        if filters.get("date_from"):
            where_clauses.append(Article.created_at >= filters["date_from"])
        if filters.get("date_to"):
            where_clauses.append(Article.created_at <= filters["date_to"])

        # Diagnostic logging only. The previous implementation ran two extra
        # aggregate queries per search — one over the full filtered join and an
        # unfiltered per-status scan across ALL tenants' chunks — which scaled
        # with corpus size and leaked cross-tenant chunk counts into logs.
        logger.info(
            "Search candidate scope",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            query_length=len(query),
            user_access_bitmask=user_bitmask,
            filters=filters,
            embedding_available=query_embedding is not None,
        )

        # 1. Vector Search
        vector_results = []
        if query_embedding is not None:
            # cosine_distance: <=>
            # Query and document vectors must come from the same embedding
            # model/version: cross-model cosine distances are meaningless, so
            # chunks embedded with any other version are excluded instead of
            # silently polluting results (the index-wide re-embedding job is
            # the migration path, not mixed-version search).
            vector_stmt = (
                select(ArticleChunk)
                .join(Article, Article.id == ArticleChunk.article_id)
                .where(
                    and_(
                        *where_clauses,
                        ArticleChunk.embedding_version == settings.EMBEDDING_VERSION,
                        ArticleChunk.embedding.cosine_distance(query_embedding) <= settings.VECTOR_DISTANCE_THRESHOLD,
                    )
                )
                .order_by(ArticleChunk.embedding.cosine_distance(query_embedding))
                .limit(max(settings.RAG_CANDIDATE_POOL_SIZE, limit))
                .options(
                    selectinload(ArticleChunk.parent_chunk).selectinload(ParentChunk.child_chunks),
                    selectinload(ArticleChunk.article).selectinload(Article.sources),
                    selectinload(ArticleChunk.article).selectinload(Article.owner),
                    selectinload(ArticleChunk.article).selectinload(Article.access_groups),
                    selectinload(ArticleChunk.article).selectinload(Article.departments),
                    selectinload(ArticleChunk.article).selectinload(Article.user_permissions),
                )
            )
            vec_res = await self.db.execute(vector_stmt)
            vector_results = vec_res.scalars().all()
            logger.info(
                "Search vector candidates loaded",
                query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                vector_result_count=len(vector_results),
            )

        # 2. Full-Text Search (keyword). Keep the query expression aligned
        # with the immutable_unaccent GIN index created by migration 58.
        # The old leading-wildcard ILIKE path forced a scan of every chunk and
        # ranked against a different expression than the one it filtered.
        search_vector = func.to_tsvector("simple", func.immutable_unaccent(ArticleChunk.chunk_text))
        search_query = func.plainto_tsquery("simple", func.immutable_unaccent(query))
        def fold(value: str) -> str:
            return "".join(
                char for char in unicodedata.normalize("NFD", value)
                if unicodedata.category(char) != "Mn"
            ).lower()

        folded_query = fold(query)
        keyword_terms = [term for term in re.findall(r"[\w'-]+", folded_query) if len(term) > 1]
        keyword_conditions = [search_vector.op("@@")(search_query)]
        keyword_conditions.extend(func.immutable_unaccent(Article.title).ilike(f"%{term}%") for term in keyword_terms)
        keyword_stmt = (
            select(ArticleChunk)
            .join(Article, Article.id == ArticleChunk.article_id)
            .where(
                and_(
                    *where_clauses,
                    or_(*keyword_conditions) if keyword_conditions else search_vector.op("@@")(search_query)
                )
            )
            .order_by(
                func.ts_rank_cd(
                    search_vector,
                    search_query,
                ).desc()
            )
            .limit(max(settings.RAG_CANDIDATE_POOL_SIZE, limit))
            .options(
                selectinload(ArticleChunk.parent_chunk).selectinload(ParentChunk.child_chunks),
                selectinload(ArticleChunk.article).selectinload(Article.sources),
                selectinload(ArticleChunk.article).selectinload(Article.owner),
                selectinload(ArticleChunk.article).selectinload(Article.access_groups),
                selectinload(ArticleChunk.article).selectinload(Article.departments),
                selectinload(ArticleChunk.article).selectinload(Article.user_permissions),
            )
        )
        key_res = await self.db.execute(keyword_stmt)
        keyword_results = key_res.scalars().all()
        logger.info(
            "Search keyword candidates loaded",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            keyword_result_count=len(keyword_results),
        )

        # 3. Merge results using Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        def add_rrf_scores(results_list):
            for rank, chunk in enumerate(results_list):
                # RRF formula: score = 1 / (60 + rank)
                score = 1.0 / (60.0 + rank)
                if chunk.id not in rrf_scores:
                    rrf_scores[chunk.id] = {"chunk": chunk, "score": 0.0}
                rrf_scores[chunk.id]["score"] += score

        add_rrf_scores(vector_results)
        add_rrf_scores(keyword_results)

        # Sort by score descending
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        logger.info(
            "Search candidates merged",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            merged_result_count=len(sorted_results),
            returned_result_count=min(len(sorted_results), limit),
        )
        return [item["chunk"] for item in sorted_results[:limit]]
