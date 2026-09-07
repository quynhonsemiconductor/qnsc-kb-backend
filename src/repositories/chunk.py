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

#: Everything a retrieved chunk is asked for AFTER the query returns. The async session
#: cannot lazy-load, so anything missing here is not a slow path — it is a MissingGreenlet
#: at request time. Each entry names its consumer so it is clear what removing one breaks:
#:
#:   parent_chunk.child_chunks  the parent passage and `child_texts` in the response
#:   article.owner              `owner_email` in the response
#:   article.departments        AuthorizationService.can_access_article_departments
#:   article.user_permissions   PermissionService._explicit_user_effect
#:   article.sources            PermissionService._source_acl_allows
#:
#: `sources` is the trap: permissions reads it as getattr(article, "sources", []), so it
#: does not appear in a search for `.sources` and looks unused.
RETRIEVAL_LOAD_OPTIONS = (
    selectinload(ArticleChunk.parent_chunk).selectinload(ParentChunk.child_chunks),
    selectinload(ArticleChunk.article).selectinload(Article.owner),
    selectinload(ArticleChunk.article).selectinload(Article.departments),
    selectinload(ArticleChunk.article).selectinload(Article.user_permissions),
    selectinload(ArticleChunk.article).selectinload(Article.sources),
)


class ChunkRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_parent_chunk(self, parent: ParentChunk) -> ParentChunk:
        self.db.add(parent)
        # FLUSH, not commit. The parent's generated id is needed immediately (the children
        # reference it), and a flush produces it without ending the transaction. Committing
        # here is what made a reindex non-atomic: one commit per parent, on top of the
        # committed wipe below, so any failure mid-rebuild left the article published with
        # a partial chunk set and no way to tell it apart from a complete one.
        await self.db.flush()
        return parent

    async def create_child_chunks(self, chunks: list[ArticleChunk]) -> list[ArticleChunk]:
        self.db.add_all(chunks)
        await self.db.flush()
        return chunks

    async def delete_by_article_id(self, article_id: uuid.UUID) -> None:
        """Remove an article's chunks WITHOUT committing.

        The caller owns the commit, because for a reindex the wipe and the replacement
        chunks have to land together. Committing the wipe first meant an article stayed
        published and searchable with zero chunks for the whole rebuild, and permanently if
        the rebuild failed -- retrieval returned nothing for it while index_status still
        read "ready" from the previous run.
        """
        await self.db.execute(delete(ParentChunk).where(ParentChunk.article_id == article_id))
        await self.db.execute(delete(ArticleChunk).where(ArticleChunk.article_id == article_id))

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
        """Return only citation chunks still visible to the current user.

        Selects the id COLUMN, not the entity. Authorization is decided entirely by
        _authorized_article_filters in SQL, so the five eager loads this used to carry
        (sources, owner, departments, user_permissions) issued four extra
        round trips and materialised whole object graphs per citation check, and every
        one of them was discarded — the method only ever returned ids.
        """
        if not chunk_ids:
            return set()

        conditions = [ArticleChunk.id.in_(chunk_ids), Article.status == "published", *ArticleRepository._authorized_article_filters(user)]
        result = await self.db.execute(
            select(ArticleChunk.id)
            .join(Article, Article.id == ArticleChunk.article_id)
            .where(*conditions)
        )
        return {str(chunk_id) for chunk_id in result.scalars().all()}

    async def update_permissions(self, article_id: uuid.UUID, sensitivity: str, visibility: str, dept: str) -> None:
        await self.db.execute(
            update(ArticleChunk)
            .where(ArticleChunk.article_id == article_id)
            .values(
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
        user: object,
        limit: int = 5,
        filters: dict | None = None
    ) -> list[ArticleChunk]:
        """
        Executes a hybrid search:
        - If query_embedding is provided, calculates vector similarity.
        - Performs a full-text search on chunk_text using tsvector.
        - Merges the two lists using a reciprocal rank scoring mechanism.
        - Enforces access control natively via the shared Article predicate.
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
        # query. Explicit-user visibility and explicit denies are relational
        # policy records and are included in the same SQL statement.
        where_clauses.extend(ArticleRepository._authorized_article_filters(user))
        # An explicit ALLOW is already part of the shared Article predicate above; only
        # the DENY needs adding here. This used to build an unused `explicit_allow`
        # EXISTS on every search.
        explicit_deny = exists(select(ArticleUserPermission.id).where(
            ArticleUserPermission.article_id == Article.id,
            ArticleUserPermission.user_id == user.id,
            ArticleUserPermission.effect == "deny",
        ))
        where_clauses.append(not_(explicit_deny))
        # Audience authorization is already present in the shared Article
        # predicate above.

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
            filters=filters,
            embedding_available=query_embedding is not None,
        )

        # Widen the HNSW candidate list BEFORE the vector query, or the pool we ask for
        # cannot arrive. hnsw.ef_search defaults to 40 and caps how many candidates one
        # index pass yields, but the query below asks for RAG_CANDIDATE_POOL_SIZE (48) —
        # so 8 were unobtainable even before filtering. And pgvector applies filtering
        # AFTER the index scan, so the permission, published-status and
        # embedding_version predicates all cut into that 40: "If a condition matches 10%
        # of rows, with HNSW and the default hnsw.ef_search of 40, only 4 rows will match
        # on average" (pgvector README, Filtering). Every filtered search was silently
        # short of candidates, with no error and no log line — just weaker answers.
        #
        # Measured on pgvector 0.8.6 with 10% of rows passing the filter: a LIMIT 48
        # returned 23 rows at the default, and 48 with these two settings applied.
        #
        # set_config(..., true) rather than SET LOCAL, for two reasons. The `true` makes
        # it TRANSACTION-local exactly like the tenant context in api/deps.py, so it
        # cannot ride a pooled connection into the next request; and it takes bind
        # parameters, so the value never reaches the server as interpolated SQL.
        #
        # iterative_scan lets the index keep scanning until enough rows survive the
        # filters instead of giving up at the first ef_search candidates. It needs
        # pgvector 0.8.0+; relaxed_order is safe here because the reranker re-sorts
        # everything it is given, so exact distance ordering out of the index buys us
        # nothing. Applied best-effort: an older pgvector raises on the unknown GUC, and
        # a wider ef_search alone is still strictly better than the default.
        if query_embedding is not None:
            try:
                await self.db.execute(
                    text("SELECT set_config('hnsw.ef_search', :ef_search, true)"),
                    {"ef_search": str(int(settings.HNSW_EF_SEARCH))},
                )
            except Exception as exc:  # pragma: no cover - depends on server build
                logger.warning("Could not widen hnsw.ef_search", error=str(exc))
            try:
                await self.db.execute(
                    text("SELECT set_config('hnsw.iterative_scan', :mode, true)"),
                    {"mode": "relaxed_order"},
                )
            except Exception as exc:
                logger.info(
                    "hnsw.iterative_scan unavailable; needs pgvector 0.8.0+",
                    error=str(exc),
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
                .options(*RETRIEVAL_LOAD_OPTIONS)
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

        def fold(value: str) -> str:
            return "".join(
                char for char in unicodedata.normalize("NFD", value)
                if unicodedata.category(char) != "Mn"
            ).lower()

        folded_query = fold(query)
        keyword_terms = [term for term in re.findall(r"[\w'-]+", folded_query) if len(term) > 1]

        # OR the terms, do NOT and them. plainto_tsquery joins every lexeme with `&`, so a
        # question longer than its own subject could not match the passage containing that
        # subject: the passage had to contain EVERY word of the question. Measured against
        # the live corpus, where the chunk holding "Innovus CCOpt" is found by the subject
        # and lost the moment the question is phrased:
        #
        #     "CCOpt"                                    2 hits
        #     "Innovus CCOpt"                            2 hits
        #     "Innovus CCOpt hoat dong"                  0 hits   <- conjunction dies here
        #     "Innovus CCOpt hoat dong nhu the nao ..."  0 hits
        #
        # An English technical passage can never contain the Vietnamese syllables of the
        # question asking about it, so the keyword leg was dead for every natural-language
        # Vietnamese question — leaving only the vector leg, and a refusal when that also
        # missed.
        #
        # Each term goes through its own plainto_tsquery as a BIND PARAMETER and the
        # results are combined with the tsquery `||` (OR) operator, so no term text is ever
        # parsed as tsquery syntax. ts_rank_cd still ranks a passage matching many terms
        # above one matching a single term, and the reranker plus the relevance floor
        # discard the weak matches this admits.
        term_queries = [
            func.plainto_tsquery("simple", func.immutable_unaccent(term))
            for term in keyword_terms
        ]
        if term_queries:
            search_query = term_queries[0]
            for extra in term_queries[1:]:
                search_query = search_query.op("||")(extra)
        else:
            # No usable terms (all one-character, or an empty query): fall back to the
            # whole string rather than building an empty tsquery, which matches nothing.
            search_query = func.plainto_tsquery("simple", func.immutable_unaccent(query))

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
            .options(*RETRIEVAL_LOAD_OPTIONS)
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
        )
        # The WHOLE fused pool, not the top `limit`. `limit` is the size of the final
        # answer, and the cross-encoder is what decides which passages fill it; truncating
        # to `limit` here handed the reranker RRF's own top-16 and threw the rest of the
        # 48-candidate pool away, so the reranker could only ever reorder what a much
        # weaker signal had already selected. Both branches in SearchService apply the
        # final `limit` themselves.
        return [item["chunk"] for item in sorted_results]
