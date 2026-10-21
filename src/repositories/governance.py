import uuid
from datetime import datetime
from typing import Sequence
from sqlalchemy import case, select, delete, and_, or_, func, update, false
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer, selectinload
from src.models.governance import (
    PendingDraft,
    DraftTransition,
    DraftCandidate,
    ApproverRule,
    Gap,
    AuditLog,
)
from src.models.article import Article
from src.models.interaction import Vote
from src.models.ops import SearchLog, ApiRequestMetric
from src.models.ai import AiUsageLog
from src.models.user import User, Department
from src.domain.rbac import AuthorizationService

#: The LIKE escape character. `!`, not the conventional backslash: a backslash inside an SQL
#: string literal means different things depending on the server's
#: `standard_conforming_strings`, and SQLAlchemy renders it differently before and after it
#: has asked. `!` is the same everywhere, so this cannot depend on a server setting.
_LIKE_ESCAPE = "!"


def _escape_like(term: str) -> str:
    """Make user text safe inside a LIKE pattern that declares `_LIKE_ESCAPE` as its escape."""
    return (
        term.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", f"{_LIKE_ESCAPE}%")
        .replace("_", f"{_LIKE_ESCAPE}_")
    )


class GovernanceRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _draft_inbox_options():
        """Project only fields needed by the queue and lightweight status polling."""
        return (
            defer(PendingDraft.summary),
            defer(PendingDraft.restructured_body_md),
            defer(PendingDraft.restructure_candidate_md),
            defer(PendingDraft.page_texts),
            defer(PendingDraft.source_hash),
            defer(PendingDraft.storage_key),
            defer(PendingDraft.original_filename),
            defer(PendingDraft.mime_type),
            defer(PendingDraft.review_note),
            defer(PendingDraft.update_target_article_id),
            selectinload(PendingDraft.candidates).load_only(
                DraftCandidate.id,
                DraftCandidate.draft_id,
                DraftCandidate.status,
            ),
        )

    # Pending Drafts
    async def create_draft(self, draft: PendingDraft) -> PendingDraft:
        self.db.add(draft)
        if draft.status == "draft":
            await self.db.flush()
            self.db.add(
                DraftTransition(
                    draft_id=draft.id,
                    from_status=None,
                    to_status="draft",
                    actor_id=draft.created_by,
                    reason="Draft created",
                    outcome="applied",
                )
            )
        await self.db.commit()
        await self.db.refresh(draft)
        return draft

    async def get_draft(self, draft_id: uuid.UUID) -> PendingDraft | None:
        result = await self.db.execute(
            select(PendingDraft).where(PendingDraft.id == draft_id)
        )
        return result.scalar_one_or_none()

    async def get_draft_for_user(
        self,
        draft_id: uuid.UUID,
        user: User,
        *,
        for_update: bool = False,
        lightweight: bool = False,
    ) -> PendingDraft | None:
        """Load a draft only inside the actor's tenant/department scope."""
        stmt = select(PendingDraft).where(PendingDraft.id == draft_id)
        if lightweight:
            stmt = stmt.options(*self._draft_inbox_options())
        global_access = AuthorizationService.has_permission(
            user, "governance.read", requested_scope="global"
        ) or AuthorizationService.has_permission(
            user, "article.publish", requested_scope="global"
        )
        if not global_access:
            company_wide = user.role in {"Admin", "CEO"} or any(
                role.active is not False and role.name in {"Admin", "CEO"}
                for role in getattr(user, "roles", [])
            )
            stmt = stmt.where(PendingDraft.company_domain == user.company_domain)
            if not company_wide:
                departments = set(AuthorizationService.member_department_names(user))
                if user.dept:
                    departments.add(user.dept)
                stmt = stmt.where(
                    or_(
                        PendingDraft.assigned_approver_id == user.id,
                        PendingDraft.created_by == user.id,
                        PendingDraft.dept.in_(departments) if departments else false(),
                    )
                )
        if for_update:
            stmt = stmt.with_for_update()
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    #: The largest page the queue endpoint will serve, and also its default so an
    #: un-paginated caller sees exactly what it saw before paging existed.
    #:
    #: The review inbox needs queue metadata, not three complete copies of every
    #: document. At this row count those text columns made one poll return 18.7 MB, and
    #: a single stuck job caused the browser to repeat that request every four seconds.
    #: `_draft_inbox_options` fixed the payload; this bounds the row count, so one
    #: request can never be heavier than the response that caused that incident.
    DRAFT_PAGE_LIMIT = 500

    @staticmethod
    def _draft_scope_filters(
        status: str | None = None,
        company_domain: str | None = None,
        dept: str | None = None,
        depts: Sequence[str] | None = None,
        assigned_approver_id: uuid.UUID | None = None,
        search: str | None = None,
    ) -> list:
        """Build the queue's WHERE clauses once, for both the page and its total.

        Shared deliberately: a count computed from different predicates than the list
        would report rows the caller cannot see, which for a department-scoped reviewer
        means leaking the existence of another department's drafts. `search` is part of
        that: it narrows within the caller's scope and can never widen it.
        """
        filters = []
        if search and search.strip():
            # Matches title, source reference and department, ignoring case and accents:
            # a reviewer typing "quy trinh" must find "Quy trình". `immutable_unaccent` is
            # the wrapper from migration 58, which is what makes the expression usable in
            # an index later if this table ever needs one. The term is escaped because it
            # is user text going into a LIKE: "50%" must mean those three characters, not
            # "50 followed by anything".
            pattern = func.immutable_unaccent(f"%{_escape_like(search.strip())}%")
            filters.append(
                or_(
                    *(
                        func.immutable_unaccent(column).ilike(
                            pattern, escape=_LIKE_ESCAPE
                        )
                        for column in (
                            PendingDraft.title,
                            PendingDraft.source_ref,
                            PendingDraft.dept,
                        )
                    )
                )
            )
        if status:
            filters.append(PendingDraft.status == status)
        if company_domain:
            filters.append(PendingDraft.company_domain == company_domain)
        if depts:
            filters.append(PendingDraft.dept.in_(list(depts)))
        elif dept:
            filters.append(PendingDraft.dept == dept)
        if assigned_approver_id:
            assignment_scope = (PendingDraft.assigned_approver_id.is_(None)) | (
                PendingDraft.assigned_approver_id == assigned_approver_id
            )
            if depts:
                assignment_scope = assignment_scope | PendingDraft.dept.in_(list(depts))
            filters.append(assignment_scope)
        return filters

    async def list_drafts(
        self,
        status: str | None = None,
        company_domain: str | None = None,
        dept: str | None = None,
        depts: Sequence[str] | None = None,
        assigned_approver_id: uuid.UUID | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
        search: str | None = None,
    ) -> Sequence[PendingDraft]:
        """One page of the review queue, newest first.

        ORDERING IS PART OF THE CONTRACT. Paging with OFFSET is only correct under a
        TOTAL order, and `created_at` is not unique -- bulk uploads land many rows in the
        same transaction, and several here share a timestamp to the microsecond. Two rows
        the database is free to return in either order can then straddle a page boundary,
        so one is served twice and another never appears at all. `id` is the primary key,
        so appending it makes the order total and the paging exact.
        """
        stmt = select(PendingDraft).options(*self._draft_inbox_options())
        for condition in self._draft_scope_filters(
            status, company_domain, dept, depts, assigned_approver_id, search
        ):
            stmt = stmt.where(condition)
        page_size = self.DRAFT_PAGE_LIMIT if limit is None else limit
        result = await self.db.execute(
            stmt.order_by(PendingDraft.created_at.desc(), PendingDraft.id.desc())
            .limit(page_size)
            .offset(max(0, offset))
        )
        return result.scalars().all()

    async def count_drafts(
        self,
        status: str | None = None,
        company_domain: str | None = None,
        dept: str | None = None,
        depts: Sequence[str] | None = None,
        assigned_approver_id: uuid.UUID | None = None,
        *,
        search: str | None = None,
    ) -> int:
        """Total rows matching the same scope `list_drafts` would page through.

        A SQL count, not `len()` of a fetch: the whole point is to report a total larger
        than one page without transferring it. It takes the same `search`, so the total
        describes the rows being paged rather than the whole queue.
        """
        stmt = select(func.count(PendingDraft.id))
        for condition in self._draft_scope_filters(
            status, company_domain, dept, depts, assigned_approver_id, search
        ):
            stmt = stmt.where(condition)
        return int(await self.db.scalar(stmt) or 0)

    async def count_pending_for_user(self, user: User) -> int:
        """Count only review items that are actually awaiting this actor.

        The Home card must not reveal the existence of drafts assigned to a
        different approver. Reviewers see their own assignments plus unassigned
        drafts in member departments; company governance leads see the same
        tenant's unassigned/company queue; global governance users may see the
        global unassigned queue. This remains a SQL count rather than a broad
        fetch followed by application filtering.
        """
        can_review = any(
            AuthorizationService.has_permission(user, key, requested_scope=scope)
            for key in ("article.review", "article.publish")
            for scope in ("company", "global")
        )
        if not can_review:
            return 0

        stmt = select(func.count(PendingDraft.id)).where(
            PendingDraft.status == "pending"
        )
        global_access = any(
            AuthorizationService.has_permission(user, key, requested_scope="global")
            for key in ("governance.read", "article.review", "article.publish")
        )
        if global_access:
            stmt = stmt.where(
                or_(
                    PendingDraft.assigned_approver_id.is_(None),
                    PendingDraft.assigned_approver_id == user.id,
                )
            )
        else:
            stmt = stmt.where(PendingDraft.company_domain == user.company_domain)
            company_lead = user.role in {"Admin", "CEO"} or any(
                role.active is not False
                and role.name in {"Admin", "CEO"}
                and role.company_domain in {None, user.company_domain}
                for role in getattr(user, "roles", [])
            )
            if company_lead:
                assignment_scope = or_(
                    PendingDraft.assigned_approver_id.is_(None),
                    PendingDraft.assigned_approver_id == user.id,
                )
            else:
                departments = set(AuthorizationService.member_department_names(user))
                if user.dept:
                    departments.add(user.dept)
                unassigned_scope = and_(
                    PendingDraft.assigned_approver_id.is_(None),
                    (
                        PendingDraft.dept.in_(sorted(departments))
                        if departments
                        else false()
                    ),
                )
                assignment_scope = or_(
                    PendingDraft.assigned_approver_id == user.id, unassigned_scope
                )
            stmt = stmt.where(assignment_scope)
        return int(await self.db.scalar(stmt) or 0)

    async def update_draft(self, draft: PendingDraft) -> PendingDraft:
        self.db.add(draft)
        await self.db.commit()
        await self.db.refresh(draft)
        return draft

    async def get_approver_rule(
        self, company_domain: str, dept: str | None
    ) -> ApproverRule | None:
        if not dept:
            return None
        result = await self.db.execute(
            select(ApproverRule).where(
                ApproverRule.company_domain == company_domain,
                ApproverRule.dept == dept,
                ApproverRule.active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def list_approver_rules(
        self, company_domain: str | None = None
    ) -> Sequence[ApproverRule]:
        stmt = select(ApproverRule).where(ApproverRule.active.is_(True))
        if company_domain:
            stmt = stmt.where(ApproverRule.company_domain == company_domain)
        result = await self.db.execute(
            stmt.order_by(ApproverRule.company_domain, ApproverRule.dept)
        )
        return result.scalars().all()

    async def list_draft_transitions(
        self, draft_id: uuid.UUID, user: User
    ) -> Sequence[DraftTransition]:
        draft = await self.get_draft_for_user(draft_id, user)
        if not draft:
            return []
        result = await self.db.execute(
            select(DraftTransition)
            .where(DraftTransition.draft_id == draft.id)
            .order_by(DraftTransition.created_at.asc())
        )
        return result.scalars().all()

    async def list_candidates(
        self, draft_id: uuid.UUID, user: User
    ) -> Sequence[DraftCandidate]:
        draft = await self.get_draft_for_user(draft_id, user)
        if not draft:
            return []
        result = await self.db.execute(
            select(DraftCandidate)
            .where(DraftCandidate.draft_id == draft.id)
            .order_by(DraftCandidate.position.asc())
        )
        return result.scalars().all()

    # Gap Queue
    #: Taken from the column rather than written out, so the two cannot drift apart.
    # String types have .length; the TypeEngine stub does not declare it.
    _GAP_QUERY_CHARS = getattr(Gap.__table__.c.query.type, "length")

    async def log_gap(
        self, query: str, company_domain: str, dept: str | None = None
    ) -> Gap:
        """Record a query that found nothing.

        The query is truncated to the column width. It used to be written whole into a
        VARCHAR(255), so a longer query raised StringDataRightTruncationError -- and
        because the gap is recorded from inside search, that killed the search itself.
        A 281-character question returned a 500 instead of "no results".

        Truncating rather than widening the column is deliberate: `query` carries a
        unique index, and a btree entry has a hard size limit, so an unbounded value
        moves the failure rather than removing it. The cost is that two very long
        queries sharing a 255-character prefix count as one gap, which for a
        what-are-people-not-finding tally is an acceptable trade.
        """
        query = (query or "")[: self._GAP_QUERY_CHARS]
        # ONE statement. Select-then-increment lost counts under concurrency — two searches
        # reading count=4 both wrote 5 — and on a miss both inserted, so the loser hit
        # uq_gaps_company_query and raised out of the search that was recording it. The
        # upsert makes the read-modify-write atomic in the row lock the insert already takes.
        #
        # `dept` and `status` are set on insert only: a gap already triaged (assigned or
        # dismissed) must not be reopened by another miss, and an existing gap's department
        # is the one it was first seen in.
        # A query can legitimately be a gap in more than one tenant, hence the composite index.
        statement = (
            pg_insert(Gap)
            .values(
                id=uuid.uuid4(),
                query=query,
                company_domain=company_domain,
                count=1,
                dept=dept,
                status="open",
            )
            .on_conflict_do_update(
                index_elements=[Gap.company_domain, Gap.query],
                set_={
                    "count": Gap.__table__.c.count + 1,
                    "updated_at": datetime.utcnow(),
                },
            )
            .returning(Gap.id)
        )
        gap_id = (await self.db.execute(statement)).scalar_one()
        await self.db.commit()
        gap = await self.db.get(Gap, gap_id)
        assert gap is not None  # upserted and committed under this id just above
        return gap

    async def list_gaps(
        self, status: str | None = None, company_domain: str | None = None
    ) -> Sequence[Gap]:
        stmt = select(Gap)
        if status:
            stmt = stmt.where(Gap.status == status)
        if company_domain:
            stmt = stmt.where(Gap.company_domain == company_domain)
        result = await self.db.execute(stmt.order_by(Gap.count.desc()).limit(500))
        return result.scalars().all()

    async def get_gap(
        self, gap_id: uuid.UUID, company_domain: str | None = None
    ) -> Gap | None:
        stmt = select(Gap).where(Gap.id == gap_id)
        if company_domain:
            stmt = stmt.where(Gap.company_domain == company_domain)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_gap(self, gap: Gap) -> Gap:
        self.db.add(gap)
        await self.db.commit()
        await self.db.refresh(gap)
        return gap

    # Audit Logs
    async def log_audit(self, audit: AuditLog) -> AuditLog:
        self.db.add(audit)
        await self.db.commit()
        await self.db.refresh(audit)
        return audit

    async def list_audits(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        user_id: uuid.UUID | None = None,
        action: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> Sequence[AuditLog]:
        stmt = select(AuditLog)
        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        if start_time:
            stmt = stmt.where(AuditLog.created_at >= start_time)
        if end_time:
            stmt = stmt.where(AuditLog.created_at <= end_time)
        result = await self.db.execute(
            stmt.order_by(AuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
            .options(selectinload(AuditLog.user))
        )
        return result.scalars().all()

    # Health Dashboard Metrics
    async def get_health_metrics(self) -> dict:
        now = datetime.utcnow()

        # 1. Total active (published) articles
        total_stmt = select(func.count(Article.id)).where(Article.status == "published")
        total_res = await self.db.execute(total_stmt)
        total_articles = total_res.scalar_one() or 0

        # 2. Articles with owner
        owner_stmt = select(func.count(Article.id)).where(
            and_(Article.status == "published", Article.owner_id.isnot(None))
        )
        owner_res = await self.db.execute(owner_stmt)
        articles_with_owner = owner_res.scalar_one() or 0

        # 3. Overdue for review
        overdue_stmt = select(func.count(Article.id)).where(
            and_(Article.status == "published", Article.next_review < now)
        )
        overdue_res = await self.db.execute(overdue_stmt)
        overdue_articles = overdue_res.scalar_one() or 0

        # 4. Search Gaps Count (total open gaps)
        gaps_stmt = select(func.count(Gap.id)).where(Gap.status == "open")
        gaps_res = await self.db.execute(gaps_stmt)
        open_gaps = gaps_res.scalar_one() or 0

        # 5. Upvote/Downvote ratio (helpful rate)
        upvotes_stmt = select(func.count(Vote.id)).where(Vote.value == 1)
        upvotes_res = await self.db.execute(upvotes_stmt)
        upvotes = upvotes_res.scalar_one() or 0

        total_votes_stmt = select(func.count(Vote.id))
        total_votes_res = await self.db.execute(total_votes_stmt)
        total_votes = total_votes_res.scalar_one() or 0
        helpful_rate = (upvotes / total_votes * 100.0) if total_votes > 0 else 100.0

        search_total_res = await self.db.execute(select(func.count(SearchLog.id)))
        search_total = search_total_res.scalar_one() or 0
        search_miss_res = await self.db.execute(
            select(func.count(SearchLog.id)).where(SearchLog.result_count == 0)
        )
        search_misses = search_miss_res.scalar_one() or 0

        ai_total_res = await self.db.execute(select(func.count(AiUsageLog.id)))
        ai_total = ai_total_res.scalar_one() or 0
        ai_cache_res = await self.db.execute(
            select(func.count(AiUsageLog.id)).where(
                AiUsageLog.prompt_version == "cached"
            )
        )
        ai_cache_hits = ai_cache_res.scalar_one() or 0

        request_metrics = await self.db.execute(
            select(
                func.count(ApiRequestMetric.id),
                func.coalesce(
                    func.sum(case((ApiRequestMetric.status_code >= 500, 1), else_=0)), 0
                ),
                func.percentile_cont(0.95).within_group(ApiRequestMetric.duration_ms),
            )
        )
        request_count, error_requests, p95_latency = request_metrics.one()
        ai_usage_result = await self.db.execute(
            select(
                func.coalesce(func.sum(AiUsageLog.tokens_used), 0),
                func.coalesce(func.avg(AiUsageLog.latency_ms), 0),
            )
        )
        ai_tokens_total, ai_latency_avg = ai_usage_result.one()

        percent_with_owner = (
            (articles_with_owner / total_articles * 100.0)
            if total_articles > 0
            else 0.0
        )
        percent_overdue = (
            (overdue_articles / total_articles * 100.0) if total_articles > 0 else 0.0
        )

        return {
            "total_articles": total_articles,
            "percent_with_owner": percent_with_owner,
            "percent_overdue": percent_overdue,
            "open_gaps": open_gaps,
            "helpful_rate": helpful_rate,
            "search_miss_rate": (
                (search_misses / search_total * 100.0) if search_total else 0.0
            ),
            "ai_cache_hit_rate": (
                (ai_cache_hits / ai_total * 100.0) if ai_total else 0.0
            ),
            "api_request_count": int(request_count or 0),
            "api_error_rate": (
                (int(error_requests or 0) / int(request_count) * 100.0)
                if request_count
                else 0.0
            ),
            "api_p95_latency_ms": float(p95_latency or 0.0),
            "ai_requests": ai_total,
            "ai_tokens_total": int(ai_tokens_total or 0),
            "ai_average_latency_ms": float(ai_latency_avg or 0),
        }
