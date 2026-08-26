import uuid
from datetime import datetime
from typing import Sequence
from sqlalchemy import case, select, delete, and_, or_, func, update, false
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models.governance import PendingDraft, DraftTransition, DraftCandidate, ApproverRule, Gap, AuditLog
from src.models.article import Article
from src.models.interaction import Vote
from src.models.ops import SearchLog, ApiRequestMetric
from src.models.ai import AiUsageLog
from src.models.user import User, Department
from src.domain.rbac import AuthorizationService

class GovernanceRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # Pending Drafts
    async def create_draft(self, draft: PendingDraft) -> PendingDraft:
        self.db.add(draft)
        if draft.status == "draft":
            await self.db.flush()
            self.db.add(DraftTransition(
                draft_id=draft.id,
                from_status=None,
                to_status="draft",
                actor_id=draft.created_by,
                reason="Draft created",
                outcome="applied",
            ))
        await self.db.commit()
        await self.db.refresh(draft)
        return draft

    async def get_draft(self, draft_id: uuid.UUID) -> PendingDraft | None:
        result = await self.db.execute(
            select(PendingDraft).where(PendingDraft.id == draft_id)
        )
        return result.scalar_one_or_none()

    async def get_draft_for_user(self, draft_id: uuid.UUID, user: User, *, for_update: bool = False) -> PendingDraft | None:
        """Load a draft only inside the actor's tenant/department scope."""
        stmt = select(PendingDraft).where(PendingDraft.id == draft_id)
        global_access = (
            AuthorizationService.has_permission(user, "governance.read", requested_scope="global")
            or AuthorizationService.has_permission(user, "article.publish", requested_scope="global")
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
                stmt = stmt.where(or_(
                    PendingDraft.assigned_approver_id == user.id,
                    PendingDraft.created_by == user.id,
                    PendingDraft.dept.in_(departments) if departments else false(),
                ))
        if for_update:
            stmt = stmt.with_for_update()
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_drafts(self, status: str | None = None, company_domain: str | None = None, dept: str | None = None, depts: Sequence[str] | None = None, assigned_approver_id: uuid.UUID | None = None) -> Sequence[PendingDraft]:
        stmt = select(PendingDraft)
        if status:
            stmt = stmt.where(PendingDraft.status == status)
        if company_domain:
            stmt = stmt.where(PendingDraft.company_domain == company_domain)
        if depts:
            stmt = stmt.where(PendingDraft.dept.in_(list(depts)))
        elif dept:
            stmt = stmt.where(PendingDraft.dept == dept)
        if assigned_approver_id:
            assignment_scope = (PendingDraft.assigned_approver_id.is_(None)) | (PendingDraft.assigned_approver_id == assigned_approver_id)
            if depts:
                assignment_scope = assignment_scope | PendingDraft.dept.in_(list(depts))
            stmt = stmt.where(assignment_scope)
        result = await self.db.execute(
            stmt.order_by(PendingDraft.created_at.desc()).limit(500)
        )
        return result.scalars().all()

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

        stmt = select(func.count(PendingDraft.id)).where(PendingDraft.status == "pending")
        global_access = any(
            AuthorizationService.has_permission(user, key, requested_scope="global")
            for key in ("governance.read", "article.review", "article.publish")
        )
        if global_access:
            stmt = stmt.where(
                or_(PendingDraft.assigned_approver_id.is_(None), PendingDraft.assigned_approver_id == user.id)
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
                    PendingDraft.dept.in_(sorted(departments)) if departments else false(),
                )
                assignment_scope = or_(PendingDraft.assigned_approver_id == user.id, unassigned_scope)
            stmt = stmt.where(assignment_scope)
        return int(await self.db.scalar(stmt) or 0)

    async def update_draft(self, draft: PendingDraft) -> PendingDraft:
        self.db.add(draft)
        await self.db.commit()
        await self.db.refresh(draft)
        return draft

    async def get_approver_rule(self, company_domain: str, dept: str | None) -> ApproverRule | None:
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

    async def list_approver_rules(self, company_domain: str | None = None) -> Sequence[ApproverRule]:
        stmt = select(ApproverRule).where(ApproverRule.active.is_(True))
        if company_domain:
            stmt = stmt.where(ApproverRule.company_domain == company_domain)
        result = await self.db.execute(stmt.order_by(ApproverRule.company_domain, ApproverRule.dept))
        return result.scalars().all()

    async def list_draft_transitions(self, draft_id: uuid.UUID, user: User) -> Sequence[DraftTransition]:
        draft = await self.get_draft_for_user(draft_id, user)
        if not draft:
            return []
        result = await self.db.execute(
            select(DraftTransition)
            .where(DraftTransition.draft_id == draft.id)
            .order_by(DraftTransition.created_at.asc())
        )
        return result.scalars().all()

    async def list_candidates(self, draft_id: uuid.UUID, user: User) -> Sequence[DraftCandidate]:
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
    async def log_gap(self, query: str, company_domain: str, dept: str | None = None) -> Gap:
        # A query can legitimately be a gap in more than one tenant.
        result = await self.db.execute(
            select(Gap).where(Gap.query == query, Gap.company_domain == company_domain)
        )
        gap = result.scalar_one_or_none()
        if gap:
            gap.count += 1
            gap.updated_at = datetime.utcnow()
        else:
            gap = Gap(query=query, company_domain=company_domain, count=1, dept=dept, status="open")
            self.db.add(gap)
        await self.db.commit()
        await self.db.refresh(gap)
        return gap

    async def list_gaps(self, status: str | None = None, company_domain: str | None = None) -> Sequence[Gap]:
        stmt = select(Gap)
        if status:
            stmt = stmt.where(Gap.status == status)
        if company_domain:
            stmt = stmt.where(Gap.company_domain == company_domain)
        result = await self.db.execute(stmt.order_by(Gap.count.desc()).limit(500))
        return result.scalars().all()

    async def get_gap(self, gap_id: uuid.UUID, company_domain: str | None = None) -> Gap | None:
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
        owner_stmt = select(func.count(Article.id)).where(and_(Article.status == "published", Article.owner_id.isnot(None)))
        owner_res = await self.db.execute(owner_stmt)
        articles_with_owner = owner_res.scalar_one() or 0

        # 3. Overdue for review
        overdue_stmt = select(func.count(Article.id)).where(and_(Article.status == "published", Article.next_review < now))
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
        search_miss_res = await self.db.execute(select(func.count(SearchLog.id)).where(SearchLog.result_count == 0))
        search_misses = search_miss_res.scalar_one() or 0

        ai_total_res = await self.db.execute(select(func.count(AiUsageLog.id)))
        ai_total = ai_total_res.scalar_one() or 0
        ai_cache_res = await self.db.execute(
            select(func.count(AiUsageLog.id)).where(AiUsageLog.prompt_version == "cached")
        )
        ai_cache_hits = ai_cache_res.scalar_one() or 0

        request_metrics = await self.db.execute(select(
            func.count(ApiRequestMetric.id),
            func.coalesce(func.sum(case((ApiRequestMetric.status_code >= 500, 1), else_=0)), 0),
            func.percentile_cont(0.95).within_group(ApiRequestMetric.duration_ms),
        ))
        request_count, error_requests, p95_latency = request_metrics.one()
        ai_usage_result = await self.db.execute(
            select(func.coalesce(func.sum(AiUsageLog.tokens_used), 0), func.coalesce(func.avg(AiUsageLog.latency_ms), 0))
        )
        ai_tokens_total, ai_latency_avg = ai_usage_result.one()

        percent_with_owner = (articles_with_owner / total_articles * 100.0) if total_articles > 0 else 0.0
        percent_overdue = (overdue_articles / total_articles * 100.0) if total_articles > 0 else 0.0

        return {
            "total_articles": total_articles,
            "percent_with_owner": percent_with_owner,
            "percent_overdue": percent_overdue,
            "open_gaps": open_gaps,
            "helpful_rate": helpful_rate,
            "search_miss_rate": (search_misses / search_total * 100.0) if search_total else 0.0,
            "ai_cache_hit_rate": (ai_cache_hits / ai_total * 100.0) if ai_total else 0.0,
            "api_request_count": int(request_count or 0),
            "api_error_rate": (int(error_requests or 0) / int(request_count) * 100.0) if request_count else 0.0,
            "api_p95_latency_ms": float(p95_latency or 0.0),
            "ai_requests": ai_total,
            "ai_tokens_total": int(ai_tokens_total or 0),
            "ai_average_latency_ms": float(ai_latency_avg or 0),
        }
