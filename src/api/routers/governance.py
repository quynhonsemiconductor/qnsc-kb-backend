import uuid
import json
import time
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import Any
from fastapi import APIRouter, Depends, Query, status, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.api.deps import get_db, get_current_user, require_permission, set_database_context
from src.models import User
from src.models.article import Article
from src.models.user import Department
from src.models.governance import (
    AuditLog,
    ApprovalRule,
    ApproverRule,
    DraftTransition,
    DraftCandidate,
    PendingDraft,
)
from src.repositories.governance import GovernanceRepository
from src.repositories.article import ArticleRepository
from src.repositories.user import UserRepository
from src.repositories.ops import OpsRepository
from src.domain.governance import GovernanceService
from src.domain.ai_service import AIService
from src.domain.search_service import SearchService
from src.repositories.chunk import ChunkRepository
from src.repositories.ai import AIRepository
from src.models.ops import EvalQuestion, EvalSet, EvalRun, IndexReprocessJob, Connector, ApiRequestMetric
from src.rag.evaluator import answer_correctness, context_recall, lexical_faithfulness
from src.core.config import is_cloudflare_r2_endpoint, settings
from src.models.ops import FeatureFlag
from src.repositories.feature_flags import FeatureFlagRepository
from src.domain.review import ReviewService
from src.domain.rbac import AuthorizationService
from src.domain.departments import resolve_active_department
from src.domain.content_restructure import build_restructure_report, split_into_chunks
from src.domain.department_routing import suggest_departments
from src.domain.llm_client import resolve_provider
from src.domain.connector_providers import REMOTE_PROVIDERS
import structlog

logger = structlog.get_logger()

router = APIRouter()


def _permission_leakage_detected(retrieved: list[dict[str, Any]], citations: list[dict[str, Any]] | None) -> bool:
    """Detect citations that were not present in the authorized retrieval set."""
    authorized_chunk_ids = {
        str(item["chunk_id"]) for item in retrieved if item.get("chunk_id")
    }
    cited_chunk_ids = {
        str(item.get("chunk_id"))
        for item in (citations or [])
        if isinstance(item, dict) and item.get("chunk_id")
    }
    return bool(cited_chunk_ids - authorized_chunk_ids)


async def _ensure_candidate_routing(
    db: AsyncSession, company_domain: str, candidates: list[DraftCandidate]
) -> None:
    """Backfill recommendations for candidates created before async formatting finishes."""
    pending = [
        item
        for item in candidates
        if item.department_suggestions is None and item.proposed_department is None
    ]
    if not pending:
        return
    departments = list(
        (
            await db.execute(
                select(Department).where(
                    Department.company_domain == company_domain,
                    Department.active.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    for item in pending:
        ids, suggestions, proposed = suggest_departments(
            item.title, item.body_md, departments
        )
        item.department_ids = ids
        item.department_suggestions = suggestions
        item.proposed_department = proposed
    await db.commit()


def _r2_is_configured() -> bool:
    """Report the minimum R2 settings required for a usable client."""
    account_or_endpoint = (settings.R2_ACCOUNT_ID or "").strip()
    explicit_endpoint = (settings.S3_ENDPOINT_URL or "").strip()
    account_location_valid = bool(account_or_endpoint) and (
        not account_or_endpoint.lower().startswith(("http://", "https://"))
        or is_cloudflare_r2_endpoint(account_or_endpoint)
    )
    storage_location_valid = (
        is_cloudflare_r2_endpoint(explicit_endpoint)
        if explicit_endpoint
        else account_location_valid
    )
    return bool(
        (settings.SOURCE_STORAGE_BACKEND or "").strip().lower()
        in {"r2", "cloudflare_r2"}
        and (settings.SOURCE_STORAGE_BUCKET or "").strip()
        and (settings.R2_ACCESS_KEY_ID or "").strip()
        and (settings.R2_SECRET_ACCESS_KEY or "").strip()
        and storage_location_valid
    )


class ApproveRequest(BaseModel):
    dept: str | None = None
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    update_article_id: uuid.UUID | None = None
    treat_as_new: bool = False
    review_note: str | None = Field(default=None, max_length=2000)
    visibility: str | None = Field(default=None, pattern="^(public|department)$")
    explicit_user_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)
    denied_user_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)


class AssignRequest(BaseModel):
    dept: str


class AssignApproverRequest(BaseModel):
    approver_id: uuid.UUID | None = None
    use_rule: bool = False
    reason: str | None = Field(default=None, max_length=2000)


class ApproverRuleRequest(BaseModel):
    dept: str = Field(min_length=1, max_length=100)
    approver_id: uuid.UUID


class SubmitDraftRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class RejectRequest(BaseModel):
    review_note: str = Field(min_length=1, max_length=2000)


class RestructureDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(keep_ai|keep_lossless)$")


class CandidateOperationRequest(BaseModel):
    operation: str = Field(pattern="^(merge|split|rename|discard|set_departments)$")
    candidate_id: uuid.UUID
    other_candidate_id: uuid.UUID | None = None
    title: str | None = Field(default=None, max_length=255)
    split_at: int | None = Field(default=None, gt=0)
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    note: str | None = Field(default=None, max_length=2000)


class EvalQuestionCreate(BaseModel):
    question: str
    expected_answer: str
    expected_chunk_ids: list[str] = []
    category: str = "general"
    eval_set_id: uuid.UUID | None = None


class EvalSetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=50)
    environment: str = Field(default="uat", max_length=50)


class FeatureFlagUpdate(BaseModel):
    enabled: bool = True
    rollout_percent: int = 100
    role: str | None = None
    department: str | None = None


class IndexReprocessRequest(BaseModel):
    article_ids: list[uuid.UUID] = Field(default_factory=list, max_length=5000)


MANAGED_FEATURE_FLAGS = {
    "ai.document_restructure": {
        "label": "AI document reading view",
        "description": "Restructure uploaded content into a lossless Markdown reading view before indexing.",
        "default_enabled": settings.RESTRUCTURE_ENABLED,
    },
}


async def _run_inline_index_reprocess(job_id: uuid.UUID) -> None:
    """Run the async reprocess implementation on FastAPI's event loop."""
    from src.workers.tasks import run_reprocess_index_job

    await run_reprocess_index_job(str(job_id))


def _gap_response(gap: Any) -> dict[str, Any]:
    return {
        "id": gap.id,
        "query": gap.query,
        "count": gap.count,
        "dept": gap.dept,
        "status": gap.status,
        "created_at": gap.created_at,
        "updated_at": gap.updated_at,
    }


def _audit_response(audit: Any) -> dict[str, Any]:
    return {
        "id": audit.id,
        "user_id": audit.user_id,
        "action": audit.action,
        "target_type": audit.target_type,
        "target_id": audit.target_id,
        "outcome": audit.outcome,
        "detail": audit.detail_json,
        "created_at": audit.created_at,
        "user": (
            {"id": audit.user.id, "name": audit.user.name, "email": audit.user.email}
            if audit.user
            else None
        ),
    }


@router.post("/reviews/verify")
async def verify_review_deadlines(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Run the local review scan while the Celery worker is intentionally disabled."""
    overdue_ids = await ReviewService(ArticleRepository(db)).verify_review_deadlines()
    return {"overdue_article_ids": overdue_ids, "count": len(overdue_ids)}


@router.get("/pending-drafts")
async def list_pending_drafts(
    status: str | None = Query(None),
    current_user: User = Depends(require_permission("governance.read")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    drafts = await service.list_drafts(current_user, status)
    response = []
    for draft in drafts:
        can_view_content = (
            draft.assigned_approver_id == current_user.id
            or service._can_review_draft(current_user, draft)
        )
        visible_body = draft.restructured_body_md if can_view_content else None
        report_body = draft.restructure_candidate_md or visible_body
        report = (
            build_restructure_report(draft.summary or "", report_body)
            if can_view_content and report_body
            else None
        )
        response.append(
            {
                "id": str(draft.id),
                "title": draft.title,
                "company_domain": draft.company_domain,
                "dept": draft.dept,
                "source_ref": draft.source_ref,
                "source_hash": draft.source_hash,
                # Any reviewer/publisher authorized for this draft may use and
                # inspect the AI reading view and review unassigned drafts.
                "summary": draft.summary if can_view_content else None,
                "restructured_body_md": visible_body,
                "restructure_candidate_md": (
                    draft.restructure_candidate_md if can_view_content else None
                ),
                "restructure_decision": draft.restructure_decision,
                "restructure_status": draft.restructure_status,
                "restructure_model": draft.restructure_model,
                "restructure_error": draft.restructure_error,
                "restructure_report": asdict(report) if report else None,
                "restructure_chunk_count": (
                    len(split_into_chunks(report_body)) if report_body else 0
                ),
                "status": draft.status,
                "created_by": str(draft.created_by) if draft.created_by else None,
                "assigned_approver_id": (
                    str(draft.assigned_approver_id)
                    if draft.assigned_approver_id
                    else None
                ),
                "assigned_by": str(draft.assigned_by) if draft.assigned_by else None,
                "assigned_at": draft.assigned_at,
                "review_due_at": draft.assigned_at + timedelta(days=settings.REVIEW_SLA_DAYS) if draft.assigned_at else None,
                "review_overdue": bool(draft.assigned_at and draft.assigned_at < datetime.utcnow() - timedelta(days=settings.REVIEW_SLA_DAYS)),
                "reviewed_by": str(draft.reviewed_by) if draft.reviewed_by else None,
                "reviewed_at": draft.reviewed_at,
                "created_at": draft.created_at,
                "similarity_level": draft.similarity_level,
                "similarity_matches": draft.similarity_matches or [],
                "requires_update_confirmation": draft.requires_update_confirmation,
                "related_article_ids": draft.related_article_ids or [],
                "tags": draft.tags or [],
                "content_metadata": (
                    draft.content_metadata
                    if draft.assigned_approver_id == current_user.id
                    or service._can_review_draft(current_user, draft)
                    else None
                ),
                "external_document_id": (
                    str(draft.external_document_id)
                    if draft.external_document_id
                    else None
                ),
                "candidate_count": len(
                    [
                        item
                        for item in (getattr(draft, "candidates", []) or [])
                        if item.status == "candidate"
                    ]
                ),
            }
        )
    return response


@router.post("/pending-drafts/{id}/assign-approver")
async def assign_draft_approver(
    id: uuid.UUID,
    req: AssignApproverRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    draft = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).assign_approver(
        current_user, id, req.approver_id, req.use_rule, req.reason
    )
    return {
        "id": str(draft.id),
        "status": draft.status,
        "assigned_approver_id": str(draft.assigned_approver_id),
        "assigned_by": str(draft.assigned_by) if draft.assigned_by else None,
        "assigned_at": draft.assigned_at,
    }


@router.post("/pending-drafts/{id}/submit")
async def submit_draft(
    id: uuid.UUID,
    req: SubmitDraftRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    draft = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).submit_draft(
        current_user,
        id,
        req.reason if req else None,
    )
    return {
        "id": str(draft.id),
        "status": draft.status,
        "assigned_approver_id": (
            str(draft.assigned_approver_id) if draft.assigned_approver_id else None
        ),
        "message": "Draft submitted for independent approval.",
    }


@router.get("/pending-drafts/{id}/transitions")
async def list_draft_transitions(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    transitions = await GovernanceRepository(db).list_draft_transitions(
        id, current_user
    )
    if not transitions:
        draft = await GovernanceRepository(db).get_draft_for_user(id, current_user)
        if not draft:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Draft not found")
    return [
        {
            "id": str(item.id),
            "draft_id": str(item.draft_id),
            "from_status": item.from_status,
            "to_status": item.to_status,
            "actor_id": str(item.actor_id) if item.actor_id else None,
            "reason": item.reason,
            "outcome": item.outcome,
            "created_at": item.created_at,
        }
        for item in transitions
    ]


@router.get("/pending-drafts/{id}/candidates")
async def list_draft_candidates(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    service = GovernanceService(GovernanceRepository(db), ArticleRepository(db))
    # Authorize through the service before inspecting the draft status.
    candidates = list(await service.list_candidates(current_user, id))
    draft = await service._get_draft_for_user(id, current_user)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    # Candidates created by older uploads must not be exposed before their
    # reading view has completed either. New uploads do not create them yet.
    if draft.restructure_status in {"queued", "processing"}:
        return []
    await _ensure_candidate_routing(db, draft.company_domain, candidates)
    return [
        {
            "id": str(item.id),
            "draft_id": str(item.draft_id),
            "position": item.position,
            "title": item.title,
            "body_md": item.body_md,
            "source_start": item.source_start,
            "source_end": item.source_end,
            "source_position": getattr(item, "source_position", None),
            "heading": item.heading,
            "department_ids": item.department_ids or [],
            "department_suggestions": item.department_suggestions or [],
            "proposed_department": item.proposed_department,
            "status": item.status,
            "review_note": item.review_note,
        }
        for item in candidates
    ]


@router.post("/pending-drafts/{id}/candidates/operation")
async def review_draft_candidate(
    id: uuid.UUID,
    req: CandidateOperationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    candidates = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).review_candidate(
        current_user,
        id,
        req.operation,
        req.candidate_id,
        req.other_candidate_id,
        req.title,
        req.split_at,
        req.department_ids,
        req.note,
    )
    return [
        {
            "id": str(item.id),
            "draft_id": str(item.draft_id),
            "position": item.position,
            "title": item.title,
            "body_md": item.body_md,
            "source_start": item.source_start,
            "source_end": item.source_end,
            "heading": item.heading,
            "department_ids": item.department_ids or [],
            "department_suggestions": item.department_suggestions or [],
            "proposed_department": item.proposed_department,
            "status": item.status,
            "review_note": item.review_note,
        }
        for item in candidates
    ]


@router.post(
    "/pending-drafts/{id}/candidates/commit", status_code=status.HTTP_201_CREATED
)
async def commit_draft_candidates(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    result = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).commit_candidates(current_user, id, detailed=True)
    children = result["drafts"]
    return {
        "parent_draft_id": str(id),
        "drafts": [
            {
                "id": str(item.id),
                "title": item.title,
                "status": item.status,
                "source_ref": item.source_ref,
            }
            for item in children
        ],
        "draft_count": len(children),
        "results": result["results"],
    }


@router.get("/approver-rules")
async def list_approver_rules(
    current_user: User = Depends(
        require_permission("article.publish", scope="company")
    ),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    rules = await GovernanceRepository(db).list_approver_rules(
        current_user.company_domain
    )
    return [
        {
            "id": str(rule.id),
            "company_domain": rule.company_domain,
            "dept": rule.dept,
            "approver_id": str(rule.approver_id),
            "active": rule.active,
        }
        for rule in rules
    ]


@router.post("/approver-rules", status_code=status.HTTP_201_CREATED)
async def create_approver_rule(
    req: ApproverRuleRequest,
    current_user: User = Depends(
        require_permission("article.publish", scope="company")
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    dept = (
        await resolve_active_department(db, current_user.company_domain, req.dept)
    ).name
    approver = await UserRepository(db).get_by_id(req.approver_id, viewer=current_user)
    if (
        not approver
        or not approver.active
        or approver.company_domain != current_user.company_domain
    ):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422, detail="Approver must be an active user in your company"
        )
    probe = PendingDraft(
        company_domain=current_user.company_domain,
        dept=dept,
        created_by=current_user.id,
        status="pending",
    )
    service = GovernanceService(GovernanceRepository(db), ArticleRepository(db))
    if not service._can_review_draft(approver, probe):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail="Selected user does not have approval permission for this department",
        )
    rule = await db.scalar(
        select(ApproverRule).where(
            ApproverRule.company_domain == current_user.company_domain,
            ApproverRule.dept == dept,
        )
    )
    if rule is None:
        rule = ApproverRule(
            company_domain=current_user.company_domain,
            dept=dept,
            approver_id=approver.id,
            created_by=current_user.id,
            active=True,
        )
        db.add(rule)
    else:
        rule.approver_id = approver.id
        rule.created_by = current_user.id
        rule.active = True
    await db.commit()
    await db.refresh(rule)
    from src.repositories.audit import AuditRepository

    await AuditRepository(db).record(
        current_user.id, "approver_rule_update", "approver_rule", str(rule.id)
    )
    return {
        "id": str(rule.id),
        "company_domain": rule.company_domain,
        "dept": rule.dept,
        "approver_id": str(rule.approver_id),
        "active": rule.active,
    }


@router.delete("/approver-rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_approver_rule(
    rule_id: uuid.UUID,
    current_user: User = Depends(
        require_permission("article.publish", scope="company")
    ),
    db: AsyncSession = Depends(get_db),
) -> None:
    rule = await db.scalar(
        select(ApproverRule).where(
            ApproverRule.id == rule_id,
            ApproverRule.company_domain == current_user.company_domain,
        )
    )
    if not rule:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Approver rule not found")
    rule.active = False
    await db.commit()
    from src.repositories.audit import AuditRepository

    await AuditRepository(db).record(
        current_user.id, "approver_rule_delete", "approver_rule", str(rule.id)
    )


@router.get("/pending-drafts/{id}/eligible-approvers")
async def list_eligible_approvers(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    users = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).eligible_approvers(current_user, id)
    return [
        {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "dept": user.dept,
            "role": user.role,
        }
        for user in users
    ]


@router.post("/pending-drafts/{id}/approve")
async def approve_draft(
    id: uuid.UUID,
    req: ApproveRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    article = await service.approve_draft(
        user=current_user,
        draft_id=id,
        dept=req.dept,
        department_ids=req.department_ids,
        update_article_id=req.update_article_id,
        treat_as_new=req.treat_as_new,
        review_note=req.review_note,
        visibility=req.visibility,
        explicit_user_ids=req.explicit_user_ids,
        denied_user_ids=req.denied_user_ids,
    )
    return {
        "id": str(article.id),
        "title": article.title,
        "status": article.status,
        "version": article.version,
    }


@router.post("/pending-drafts/{id}/reject")
async def reject_draft(
    id: uuid.UUID,
    req: RejectRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    draft = await service.reject_draft(current_user, id, req.review_note)
    return {"id": str(draft.id), "title": draft.title, "status": draft.status}


async def _reload_actor(db: AsyncSession, actor_id: uuid.UUID) -> User | None:
    """Re-fetch the acting user by PRIMITIVE id after a failed decision.

    MEASURED, and the primitive id is the whole point. `approve_draft` calls `db.rollback()`
    when it refuses a draft (governance.py:1424), and a rollback EXPIRES the entire identity
    map — `expire_on_commit=False` governs commits, not rollbacks. Every attribute of the
    acting user then becomes a lazy load, so in async context:

      * the next draft's permission check reads `user.roles` -> MissingGreenlet
        (`rbac.py:224 in has_permission`), and
      * even reading `actor.id` to RELOAD the user raises the same error.

    That second point is why an earlier version of this helper, which took the `User`
    instance, could not work: it had to touch the expired object to recover from the
    expiry. Observed three times over — 2 of 5 drafts approved, then 24 MissingGreenlets.

    So the caller captures the id as a `uuid.UUID` BEFORE the first decision, and this
    function never touches an ORM instance it did not just load.
    """
    return await UserRepository(db).get_by_id(actor_id)


class BulkDecideRequest(BaseModel):
    """Approve or reject many drafts in one request.

    WHY THIS EXISTS. Every draft endpoint above is `/{id}`-scoped, so a 122-deep queue cost
    122 separate approvals — four interactions each, roughly 500 clicks and 4-10 hours of
    specialist time, while the connector beat task refilled the queue every 600 seconds.
    That is why 270 synced documents produced 7 articles: the ingest side converts at 45%,
    and pending->approved converts at 6%.

    Deliberately NOT a new approval path. Each id goes through the same
    `GovernanceService.approve_draft` / `reject_draft` a single review calls, so every
    permission check, similarity gate, ACL-mapping gate and split-candidate gate applies
    unchanged. This endpoint only removes the per-draft round trip.
    """

    draft_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)
    decision: str = Field(pattern="^(approve|reject)$")
    #: Applied to every draft in the batch. A draft needing a decision this cannot express
    #: — choosing an update target, resolving a split — is reported as blocked rather than
    #: guessed at.
    dept: str | None = None
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    visibility: str | None = Field(default=None, pattern="^(public|department)$")
    review_note: str | None = Field(default=None, max_length=2000)


@router.post("/pending-drafts/bulk-decide")
async def bulk_decide_drafts(
    req: BulkDecideRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Decide up to 100 drafts, reporting each outcome separately.

    PARTIAL SUCCESS IS THE POINT. A queue this size always contains drafts that cannot be
    decided in bulk — one needs a split committed, another needs its connector ACL mapped,
    a third needs a human to choose which article it updates. Failing the whole batch on the
    first of those would make the endpoint useless exactly when it is most needed, and
    skipping them silently would hide work the reviewer must still do.

    A refused draft must not poison the ones after it. `approve_draft` rolls back on refusal
    (governance.py:1424), which expires the whole identity map, so the acting user is
    re-loaded by primitive id after every failure — see `_reload_actor` for the measurement.
    The reviewer gets a shorter queue plus an exact list of what still needs attention.
    """
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)

    # Captured as a plain UUID BEFORE any decision runs. After a refusal the `current_user`
    # instance is expired, and reading even `.id` off it raises MissingGreenlet — so the
    # recovery path must not depend on the object it is recovering from.
    actor_id: uuid.UUID = current_user.id

    if req.decision == "reject" and not (req.review_note or "").strip():
        # reject_draft requires a note; catching it here reports one clear error instead of
        # the same validation failure repeated for every id in the batch.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "review_note_required",
                "message": "A review note is required when rejecting drafts.",
            },
        )

    decided: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    # dict.fromkeys: de-duplicate while keeping the reviewer's order, so a repeated id
    # cannot be decided twice and the report reads in the order they selected.
    #
    # NO savepoint wrapper, deliberately. `approve_draft` is ALREADY per-draft atomic: it
    # commits at governance.py:1421 and rolls back at :1424. Wrapping a self-committing call
    # in `db.begin_nested()` closes the savepoint underneath it, and the reload after the
    # commit then dies with "Can't operate on closed transaction inside context manager" —
    # measured. Each iteration is therefore its own transaction by virtue of the service,
    # which is exactly the isolation this endpoint needs.
    for draft_id in dict.fromkeys(req.draft_ids):
        try:
            if req.decision == "approve":
                article = await service.approve_draft(
                    user=current_user,
                    draft_id=draft_id,
                    dept=req.dept,
                    department_ids=req.department_ids,
                    review_note=req.review_note,
                    visibility=req.visibility,
                )
                decided.append(
                    {
                        "draft_id": str(draft_id),
                        "article_id": str(article.id),
                        "title": article.title,
                        "version": article.version,
                    }
                )
            else:
                draft = await service.reject_draft(
                    current_user, draft_id, req.review_note or ""
                )
                decided.append(
                    {"draft_id": str(draft_id), "title": draft.title, "status": draft.status}
                )
        except HTTPException as exc:
            # The service already explains itself, including the structured codes the UI
            # keys off (batch_review_required, update_confirmation_required,
            # external_acl_mapping_required). Pass them through rather than flattening to
            # a string the frontend cannot branch on.
            detail = exc.detail
            blocked.append(
                {
                    "draft_id": str(draft_id),
                    "status_code": exc.status_code,
                    "code": detail.get("code") if isinstance(detail, dict) else None,
                    "reason": detail.get("message") if isinstance(detail, dict) else detail,
                }
            )
            reloaded = await _reload_actor(db, actor_id)
            if reloaded is None:
                # The acting user vanished mid-batch (deactivated, deleted). Stop rather
                # than attempt further decisions with an unusable actor.
                break
            current_user = reloaded
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Bulk decision failed for a draft",
                draft_id=str(draft_id),
                decision=req.decision,
            )
            blocked.append(
                {
                    "draft_id": str(draft_id),
                    "status_code": 500,
                    "code": "unexpected_error",
                    "reason": str(exc)[:200],
                }
            )
            reloaded = await _reload_actor(db, actor_id)
            if reloaded is None:
                break
            current_user = reloaded

    # Audited AFTER the loop, in its own transaction. It cannot share one with the
    # decisions: each approve_draft/reject_draft already committed its own work, so by the
    # time we get here there is nothing left to join. The audit row is therefore a record
    # OF the batch rather than part of it — if this insert failed, the decisions would
    # still stand, which is the right way round for an irreversible publish.
    #
    # Direct-ORM style because AuditLog is already imported and used this way five times in
    # this file; importing AuditRepository would put a second audit convention beside it.
    db.add(
        AuditLog(
            # actor_id, not current_user.id: after a refusal the instance is expired and
            # reading .id off it raises MissingGreenlet.
            user_id=actor_id,
            action=f"draft_bulk_{req.decision}",
            target_type="pending_draft",
            # No single target id: the batch IS the subject, so the decided count goes in
            # the target slot and the ids live in the detail payload.
            target_id=str(len(decided)),
            # "success" even when some drafts are blocked: the request itself succeeded and
            # the blocked ids are in detail_json. Inventing a fourth outcome value would
            # break the audit-log filter, which only knows success/failure/applied.
            outcome="success",
            detail_json={
                "requested": len(set(req.draft_ids)),
                "decided": [item["draft_id"] for item in decided],
                "blocked": blocked,
            },
        )
    )
    await db.commit()
    logger.info(
        "Bulk draft decision",
        decision=req.decision,
        requested=len(set(req.draft_ids)),
        decided=len(decided),
        blocked=len(blocked),
    )
    return {
        "decision": req.decision,
        "requested": len(set(req.draft_ids)),
        "decided_count": len(decided),
        "blocked_count": len(blocked),
        "decided": decided,
        "blocked": blocked,
    }


@router.post("/pending-drafts/{id}/restructure")
async def restructure_draft(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    service = GovernanceService(gov_repo, ArticleRepository(db))
    draft = await gov_repo.get_draft_for_user(id, current_user)
    if not draft:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Draft not found")
    if draft.status != "pending":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400, detail="Only pending drafts can be restructured"
        )
    if (
        draft.company_domain != current_user.company_domain
        and not service._is_global_publisher(current_user)
    ):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Draft is outside your company")
    if not service._can_review_draft(current_user, draft):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail="Only users with review or publish permission can restructure this draft",
        )

    # Retry is deliberately asynchronous. A slow provider must not hold the
    # review request open until its HTTP timeout; the Pending Draft poller will
    # show queued -> processing -> completed/fallback.
    draft.restructure_status = "queued"
    draft.restructure_error = None
    draft.restructure_candidate_md = None
    draft.restructure_decision = "not_reviewed"
    draft = await gov_repo.update_draft(draft)
    from src.workers.tasks import dispatch_restructure_pending_draft

    if not dispatch_restructure_pending_draft(
        str(draft.id), current_user.company_domain, str(current_user.id)
    ):
        draft.restructure_status = "fallback_formatting"
        draft.restructure_model = "lossless-markdown"
        draft.restructure_error = "AI formatting could not be queued; the lossless reading view is still available."
        draft = await gov_repo.update_draft(draft)
    return {
        "id": str(draft.id),
        "restructured_body_md": draft.restructured_body_md,
        "restructure_status": draft.restructure_status,
        "restructure_model": draft.restructure_model,
        "restructure_error": draft.restructure_error,
        "restructure_candidate_md": draft.restructure_candidate_md,
        "restructure_decision": draft.restructure_decision,
        "restructure_report": asdict(
            build_restructure_report(
                draft.summary or "",
                draft.restructure_candidate_md or draft.restructured_body_md or "",
            )
        ),
        "restructure_chunk_count": len(
            split_into_chunks(draft.restructured_body_md or "")
        ),
    }


@router.post("/pending-drafts/{id}/restructure-decision")
async def decide_restructure(
    id: uuid.UUID,
    req: RestructureDecisionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    service = GovernanceService(GovernanceRepository(db), ArticleRepository(db))
    draft = await service.decide_restructure(current_user, id, req.decision)
    report_body = draft.restructure_candidate_md or draft.restructured_body_md or ""
    return {
        "id": str(draft.id),
        "restructured_body_md": draft.restructured_body_md,
        "restructure_candidate_md": draft.restructure_candidate_md,
        "restructure_decision": draft.restructure_decision,
        "restructure_status": draft.restructure_status,
        "restructure_model": draft.restructure_model,
        "restructure_error": draft.restructure_error,
        "restructure_report": asdict(
            build_restructure_report(draft.summary or "", report_body)
        ),
        "restructure_chunk_count": len(split_into_chunks(report_body)),
    }


@router.get("/pending-drafts/{id}/comparison")
async def compare_pending_draft(
    id: uuid.UUID,
    article_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    draft = await GovernanceRepository(db).get_draft_for_user(id, current_user)
    if not draft or draft.status != "pending":
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Pending draft not found")
    service = GovernanceService(GovernanceRepository(db), ArticleRepository(db))
    if (
        draft.assigned_approver_id and draft.assigned_approver_id != current_user.id
    ) or not service._can_review_draft(current_user, draft):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403, detail="You are not authorized to compare this draft"
        )
    article = await ArticleRepository(db).get_by_id(article_id, user=current_user)
    if not article or article.status == "deleted":
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Comparison article not found")
    from src.domain.permissions import PermissionService

    if not PermissionService.can_view_article(current_user, article):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail="You are not authorized to view the comparison article",
        )
    return {
        "id": str(article.id),
        "title": article.title,
        "body_md": article.body_md,
        "version": article.version,
        "status": article.status,
        "lifecycle_status": article.lifecycle_status,
    }


@router.get("/gaps")
async def list_search_gaps(
    status: str | None = Query(None),
    current_user: User = Depends(require_permission("governance.read")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    gaps = await service.list_gaps(current_user, status)
    return [_gap_response(gap) for gap in gaps]


@router.post("/gaps/{id}/assign")
async def assign_gap(
    id: uuid.UUID,
    req: AssignRequest,
    current_user: User = Depends(require_permission("article.review")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    gap = await service.assign_gap(current_user, id, req.dept)
    return _gap_response(gap)


@router.post("/gaps/{id}/dismiss")
async def dismiss_gap(
    id: uuid.UUID,
    current_user: User = Depends(require_permission("article.review")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    gap = await service.dismiss_gap(current_user, id)
    return _gap_response(gap)


@router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user_id: uuid.UUID | None = Query(None),
    action: str | None = Query(None, min_length=1, max_length=50),
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=422, detail="start_time must be before end_time"
        )
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    logs = await service.list_audit_logs(
        current_user,
        limit,
        offset=offset,
        user_id=user_id,
        action=action,
        start_time=start_time,
        end_time=end_time,
    )
    return [_audit_response(audit) for audit in logs]


@router.get("/request-failures")
async def get_request_failures(
    limit: int = Query(50, ge=1, le=500),
    path: str | None = Query(None, min_length=1, max_length=255),
    request_id: str | None = Query(None, min_length=1, max_length=100),
    min_status: int = Query(500, ge=400, le=599),
    since: datetime | None = Query(None),
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Recent failed requests, WITH the exception that caused each one.

    Exists so a 500 can be diagnosed without CloudWatch. The middleware always caught
    the exception and logged it, but only the status code was persisted, which meant
    reading production errors required an AWS role switch. Now the same information is
    queryable here.

    `path` matches the resolved route template (`/upload-source`, not `/upload-source?x=1`),
    because that is what the metric records — bounded cardinality is why it is stored that
    way. `request_id` looks up the exact failure a user reports, since the id is already
    returned to them in the `X-Request-ID` response header.
    """
    conditions = [ApiRequestMetric.status_code >= min_status]
    if path:
        conditions.append(ApiRequestMetric.path.ilike(f"%{path}%"))
    if request_id:
        conditions.append(ApiRequestMetric.request_id == request_id)
    if since:
        conditions.append(ApiRequestMetric.created_at >= since)

    rows = await db.execute(
        select(ApiRequestMetric)
        .where(*conditions)
        .order_by(ApiRequestMetric.created_at.desc())
        .limit(limit)
    )
    failures = rows.scalars().all()

    return {
        "count": len(failures),
        "failures": [
            {
                "request_id": failure.request_id,
                "at": failure.created_at.isoformat() if failure.created_at else None,
                "method": failure.method,
                "path": failure.path,
                "status_code": failure.status_code,
                "duration_ms": failure.duration_ms,
                "error_type": failure.error_type,
                # NULL for anything recorded before the detail columns existed, and for
                # failures raised as deliberate HTTPExceptions rather than crashes.
                "error_detail": failure.error_detail,
            }
            for failure in failures
        ],
    }


@router.get("/health-metrics")
async def get_health_metrics(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    gov_repo = GovernanceRepository(db)
    art_repo = ArticleRepository(db)
    service = GovernanceService(gov_repo, art_repo)
    metrics = await service.get_dashboard_metrics(current_user)
    health_company = (
        None
        if AuthorizationService.has_permission(
            current_user, "governance.read", requested_scope="global"
        )
        else current_user.company_domain
    )
    # Every remote provider counts toward the connector-sync signal. Filtering on
    # SharePoint alone reported "no active connector" on a tenant running only
    # OneDrive or Google Drive.
    connector_filters = [
        Connector.system.in_(sorted(REMOTE_PROVIDERS)),
        Connector.status == "active",
    ]
    index_filters = [
        Article.status == "published",
        Article.lifecycle_status == "active",
        Article.index_status.in_(["pending", "processing", "failed"]),
    ]
    if health_company:
        connector_filters.append(Connector.company_domain == health_company)
        index_filters.append(Article.company_domain == health_company)
    connector_count = await db.scalar(
        select(func.count(Connector.id)).where(*connector_filters)
    )
    queued_indexes = await db.scalar(
        select(func.count(Article.id)).where(*index_filters)
    )
    metrics["dependencies"] = {
        "r2": {"configured": _r2_is_configured()},
        "sharepoint": {
            "configured": bool(connector_count),
            "active_connectors": int(connector_count or 0),
        },
        # Same numbers under a provider-neutral name. `sharepoint` is retained
        # because the deployed frontend reads that key.
        "connectors": {
            "configured": bool(connector_count),
            "active_connectors": int(connector_count or 0),
        },
        "indexing": {"pending_or_failed_articles": int(queued_indexes or 0)},
        "llm": {"configured": bool(resolve_provider())},
    }
    return metrics


def _index_job_response(job: IndexReprocessJob) -> dict[str, Any]:
    return {
        "id": str(job.id),
        "company_domain": job.company_domain,
        "status": job.status,
        "total": job.total,
        "completed": job.completed,
        "failed": job.failed,
        "retry_count": job.retry_count,
        "last_error": job.last_error,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "created_at": job.created_at,
    }


@router.post("/index/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def start_index_reprocess(
    req: IndexReprocessRequest,
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if req.article_ids:
        visible = (
            (
                await db.execute(
                    select(Article.id).where(
                        Article.id.in_(req.article_ids),
                        Article.company_domain == current_user.company_domain,
                        Article.status == "published",
                        Article.lifecycle_status == "active",
                    )
                )
            )
            .scalars()
            .all()
        )
        if len(visible) != len(set(req.article_ids)):
            raise HTTPException(
                status_code=403,
                detail="One or more selected Articles are outside your reprocess scope",
            )
    job = IndexReprocessJob(
        company_domain=current_user.company_domain,
        requested_by=current_user.id,
        target_article_ids=[str(item) for item in req.article_ids] or None,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    from src.workers.tasks import reprocess_index_job_task

    if settings.JOB_MODE == "inline":
        await _run_inline_index_reprocess(job.id)
    else:
        reprocess_index_job_task.delay(str(job.id))
    return _index_job_response(job)


@router.get("/index/reprocess/{id}")
async def get_index_reprocess(
    id: uuid.UUID,
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    job = await db.scalar(
        select(IndexReprocessJob).where(
            IndexReprocessJob.id == id,
            IndexReprocessJob.company_domain == current_user.company_domain,
        )
    )
    if not job:
        raise HTTPException(status_code=404, detail="Index reprocess job not found")
    return _index_job_response(job)


@router.post("/index/reprocess/{id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_index_reprocess(
    id: uuid.UUID,
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    job = await db.scalar(
        select(IndexReprocessJob).where(
            IndexReprocessJob.id == id,
            IndexReprocessJob.company_domain == current_user.company_domain,
        )
    )
    if not job:
        raise HTTPException(status_code=404, detail="Index reprocess job not found")
    if job.status != "failed":
        raise HTTPException(
            status_code=409, detail="Only failed index jobs can be retried"
        )
    job.status = "queued"
    job.retry_count += 1
    job.last_error = None
    await db.commit()
    from src.workers.tasks import reprocess_index_job_task

    if settings.JOB_MODE == "inline":
        await _run_inline_index_reprocess(job.id)
    else:
        reprocess_index_job_task.delay(str(job.id))
    return _index_job_response(job)


@router.get("/eval-sets")
async def list_eval_sets(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    rows = (await db.execute(select(EvalSet).where(EvalSet.company_domain == current_user.company_domain).order_by(EvalSet.created_at.desc()))).scalars().all()
    return [{"id": str(item.id), "name": item.name, "version": item.version, "environment": item.environment, "status": item.status, "approved_by": str(item.approved_by) if item.approved_by else None, "approved_at": item.approved_at} for item in rows]


@router.post("/eval-sets", status_code=status.HTTP_201_CREATED)
async def create_eval_set(
    req: EvalSetCreate,
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    item = EvalSet(company_domain=current_user.company_domain, name=req.name.strip(), version=req.version.strip(), environment=req.environment.strip(), status="draft")
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return {"id": str(item.id), "name": item.name, "version": item.version, "environment": item.environment, "status": item.status}


@router.post("/eval-sets/{id}/approve")
async def approve_eval_set(id: uuid.UUID, current_user: User = Depends(require_permission("governance.read", scope="global")), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    item = await db.scalar(select(EvalSet).where(EvalSet.id == id, EvalSet.company_domain == current_user.company_domain))
    if not item:
        raise HTTPException(status_code=404, detail="Evaluation set not found")
    item.status = "approved"
    item.approved_by = current_user.id
    item.approved_at = datetime.utcnow()
    await db.commit()
    return {"id": str(item.id), "status": item.status, "approved_at": item.approved_at}


@router.get("/eval-report")
async def eval_report(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rows = (await db.execute(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(500))).scalars().all()
    if not rows:
        return {"sample_count": 0, "kpis": {"citation_recall": 0, "groundedness": 0, "correctness": 0, "latency_ms": 0}, "permission_leakage": 0, "verdict": "NO-GO", "reason": "No evaluation runs exist"}
    citation = sum(item.context_recall for item in rows) / len(rows)
    grounded = sum(item.faithfulness for item in rows) / len(rows)
    correctness = sum(item.answer_correctness for item in rows) / len(rows)
    latency = sum(item.latency_ms for item in rows) / len(rows)
    permission_leakage = sum(1 for item in rows if item.permission_leakage)
    go = citation >= 0.95 and grounded >= 0.90 and correctness >= 0.90 and latency <= 4000 and permission_leakage == 0
    return {"sample_count": len(rows), "kpis": {"citation_recall": round(citation, 4), "groundedness": round(grounded, 4), "correctness": round(correctness, 4), "latency_ms": round(latency)}, "permission_leakage": permission_leakage, "verdict": "GO" if go else "NO-GO", "thresholds": {"citation_recall": 0.95, "groundedness": 0.90, "correctness": 0.90, "latency_ms": 4000}}


@router.get("/eval-runs")
async def get_eval_runs(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    ops_repo = OpsRepository(db)
    return await ops_repo.list_eval_runs()


@router.get("/eval-questions")
async def list_eval_questions(
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    questions = await OpsRepository(db).list_eval_questions()
    return [
        {
            "id": str(item.id),
            "question": item.question,
            "expected_answer": item.expected_answer,
            "expected_chunk_ids": json.loads(item.expected_chunk_ids or "[]"),
            "category": item.category,
            "eval_set_id": str(item.eval_set_id) if item.eval_set_id else None,
        }
        for item in questions
    ]


@router.post("/eval-questions", status_code=status.HTTP_201_CREATED)
async def create_eval_question(
    req: EvalQuestionCreate,
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    item = await OpsRepository(db).create_eval_question(
        EvalQuestion(
            question=req.question,
            expected_answer=req.expected_answer,
            expected_chunk_ids=json.dumps(req.expected_chunk_ids),
            category=req.category,
            eval_set_id=req.eval_set_id,
        )
    )
    return {"id": str(item.id), "question": item.question, "category": item.category}


@router.post("/eval-questions/{id}/run")
async def run_eval_question(
    id: uuid.UUID,
    current_user: User = Depends(require_permission("governance.read", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    ops_repo = OpsRepository(db)
    question = await ops_repo.get_eval_question(id)
    if not question:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Evaluation question not found")

    gov_repo = GovernanceRepository(db)
    search_service = SearchService(ChunkRepository(db), gov_repo)
    started = time.perf_counter()
    retrieved = await search_service.search(current_user, question.question, limit=10)
    expected_ids = json.loads(question.expected_chunk_ids or "[]")
    retrieval_score = context_recall(
        [item["chunk_id"] for item in retrieved], expected_ids
    )
    context = "\n".join(item["parent_text"] for item in retrieved)
    answer = await AIService(AIRepository(db), search_service, gov_repo).ask(
        current_user, question.question
    )
    faithfulness_score = lexical_faithfulness(answer["answer"], context)
    correctness_score = answer_correctness(answer["answer"], question.expected_answer)
    permission_leakage = _permission_leakage_detected(
        retrieved, answer.get("citations")
    )
    run = await ops_repo.create_eval_run(
        EvalRun(
            eval_question_id=question.id,
            retrieval_version=settings.RETRIEVAL_VERSION,
            prompt_version=settings.PROMPT_VERSION,
            context_recall=retrieval_score,
            faithfulness=faithfulness_score,
            answer_correctness=correctness_score,
            latency_ms=round((time.perf_counter() - started) * 1000),
            permission_leakage=permission_leakage,
        )
    )
    return {
        "id": str(run.id),
        "eval_question_id": str(question.id),
        "context_recall": retrieval_score,
        "faithfulness": faithfulness_score,
        "answer_correctness": correctness_score,
        "latency_ms": run.latency_ms,
        "permission_leakage": permission_leakage,
        "created_at": run.created_at,
    }


@router.get("/feature-flags")
async def list_feature_flags(
    current_user: User = Depends(require_permission("role.manage", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    flags = {
        flag.key: flag
        for flag in await FeatureFlagRepository(db).list()
        if flag.key in MANAGED_FEATURE_FLAGS
    }
    response = [
        {
            "id": str(flag.id),
            "key": flag.key,
            "enabled": flag.enabled,
            "rollout_percent": flag.rollout_percent,
            "role": flag.role,
            "department": flag.department,
            "label": MANAGED_FEATURE_FLAGS.get(flag.key, {}).get("label", flag.key),
            "description": MANAGED_FEATURE_FLAGS.get(flag.key, {}).get(
                "description", ""
            ),
        }
        for flag in flags.values()
    ]
    for key, metadata in MANAGED_FEATURE_FLAGS.items():
        if key not in flags:
            response.append(
                {
                    "id": None,
                    "key": key,
                    "enabled": metadata["default_enabled"],
                    "rollout_percent": 100,
                    "role": None,
                    "department": None,
                    "label": metadata["label"],
                    "description": metadata["description"],
                }
            )
    return sorted(response, key=lambda item: item["key"])


@router.put("/feature-flags/{key}")
async def update_feature_flag(
    key: str,
    req: FeatureFlagUpdate,
    current_user: User = Depends(require_permission("role.manage", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    if not 0 <= req.rollout_percent <= 100:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422, detail="rollout_percent must be between 0 and 100"
        )
    if key not in MANAGED_FEATURE_FLAGS and not AuthorizationService.has_permission(
        current_user, "role.manage", requested_scope="global"
    ):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403, detail="CEOs can only manage approved company features"
        )
    flag = await FeatureFlagRepository(db).upsert(
        key, req.enabled, req.rollout_percent, req.role, req.department
    )
    from src.repositories.audit import AuditRepository

    await AuditRepository(db).record(
        current_user.id, "feature_flag_update", "feature_flag", key
    )
    return {
        "id": str(flag.id),
        "key": flag.key,
        "enabled": flag.enabled,
        "rollout_percent": flag.rollout_percent,
        "role": flag.role,
        "department": flag.department,
        "label": MANAGED_FEATURE_FLAGS.get(flag.key, {}).get("label", flag.key),
        "description": MANAGED_FEATURE_FLAGS.get(flag.key, {}).get("description", ""),
    }


# ---------------------------------------------------------------------------
# Approval agent
#
# Gated on article.publish at global scope, not governance.read. A rule is a standing
# instruction to publish or discard company-wide content, so writing one has to require
# the authority it grants -- reading the governance queue plainly does not.
# ---------------------------------------------------------------------------


class ApprovalRuleRequest(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    instruction: str = Field(min_length=1, max_length=5000)
    active: bool = True
    priority: int = Field(default=100, ge=0, le=10_000)
    connector_id: uuid.UUID | None = None
    dept: str | None = Field(default=None, max_length=100)
    file_extensions: list[str] | None = Field(default=None, max_length=25)
    max_similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    # Both default to False here as well as in the model and the database: a rule created
    # by a client that omits them must not acquire authority by omission.
    can_approve: bool = False
    can_reject: bool = False


class ApprovalAgentRunRequest(BaseModel):
    # Defaults to a dry run. Deciding hundreds of documents unattended should be
    # something a caller asks for explicitly, not what happens if a field is forgotten.
    dry_run: bool = True
    limit: int | None = Field(default=None, ge=1, le=500)


def _approval_rule_payload(rule: ApprovalRule) -> dict[str, Any]:
    return {
        "id": str(rule.id),
        "name": rule.name,
        "instruction": rule.instruction,
        "active": rule.active,
        "priority": rule.priority,
        "connector_id": str(rule.connector_id) if rule.connector_id else None,
        "dept": rule.dept,
        "file_extensions": rule.file_extensions,
        "max_similarity_score": rule.max_similarity_score,
        "can_approve": rule.can_approve,
        "can_reject": rule.can_reject,
        "created_by": str(rule.created_by) if rule.created_by else None,
        "created_at": rule.created_at,
        "updated_at": rule.updated_at,
    }


def _normalise_extensions(values: list[str] | None) -> list[str] | None:
    """Store extensions the one way the matcher looks them up."""
    if not values:
        return None
    cleaned = set()
    for value in values:
        text = value.strip().lstrip(".").lower()
        if text:
            cleaned.add("." + text)
    return sorted(cleaned) or None


@router.get("/approval-rules")
async def list_approval_rules(
    current_user: User = Depends(require_permission("article.publish", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    rules = (
        await db.execute(
            select(ApprovalRule)
            .where(ApprovalRule.company_domain == current_user.company_domain)
            .order_by(ApprovalRule.priority, ApprovalRule.created_at)
        )
    ).scalars().all()
    return [_approval_rule_payload(rule) for rule in rules]


@router.post("/approval-rules", status_code=status.HTTP_201_CREATED)
async def create_approval_rule(
    payload: ApprovalRuleRequest,
    current_user: User = Depends(require_permission("article.publish", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rule = ApprovalRule(
        company_domain=current_user.company_domain,
        name=payload.name.strip(),
        instruction=payload.instruction.strip(),
        active=payload.active,
        priority=payload.priority,
        connector_id=payload.connector_id,
        dept=payload.dept.strip() if payload.dept else None,
        file_extensions=_normalise_extensions(payload.file_extensions),
        max_similarity_score=payload.max_similarity_score,
        can_approve=payload.can_approve,
        can_reject=payload.can_reject,
        # The agent runs as this person. A rule outlives the session that created it, so
        # this is who it will still be acting as next month.
        created_by=current_user.id,
    )
    db.add(rule)
    await db.flush()
    db.add(
        AuditLog(
            user_id=current_user.id,
            action="approval_rule_create",
            target_type="approval_rule",
            target_id=str(rule.id),
            outcome="success",
            detail_json={"can_approve": rule.can_approve, "can_reject": rule.can_reject},
        )
    )
    await db.commit()
    return _approval_rule_payload(rule)


@router.patch("/approval-rules/{rule_id}")
async def update_approval_rule(
    rule_id: uuid.UUID,
    payload: ApprovalRuleRequest,
    current_user: User = Depends(require_permission("article.publish", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rule = await db.get(ApprovalRule, rule_id)
    if not rule or rule.company_domain != current_user.company_domain:
        raise HTTPException(status_code=404, detail="Rule not found")
    rule.name = payload.name.strip()
    rule.instruction = payload.instruction.strip()
    rule.active = payload.active
    rule.priority = payload.priority
    rule.connector_id = payload.connector_id
    rule.dept = payload.dept.strip() if payload.dept else None
    rule.file_extensions = _normalise_extensions(payload.file_extensions)
    rule.max_similarity_score = payload.max_similarity_score
    rule.can_approve = payload.can_approve
    rule.can_reject = payload.can_reject
    db.add(
        AuditLog(
            user_id=current_user.id,
            action="approval_rule_update",
            target_type="approval_rule",
            target_id=str(rule.id),
            outcome="success",
            detail_json={"can_approve": rule.can_approve, "can_reject": rule.can_reject},
        )
    )
    await db.commit()
    return _approval_rule_payload(rule)


@router.delete("/approval-rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_approval_rule(
    rule_id: uuid.UUID,
    current_user: User = Depends(require_permission("article.publish", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> None:
    rule = await db.get(ApprovalRule, rule_id)
    if not rule or rule.company_domain != current_user.company_domain:
        raise HTTPException(status_code=404, detail="Rule not found")
    await db.delete(rule)
    db.add(
        AuditLog(
            user_id=current_user.id,
            action="approval_rule_delete",
            target_type="approval_rule",
            target_id=str(rule_id),
            outcome="success",
        )
    )
    await db.commit()


@router.post("/approval-agent/run")
async def run_approval_agent(
    payload: ApprovalAgentRunRequest,
    current_user: User = Depends(require_permission("article.publish", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Apply the active rules to the pending queue.

    Defaults to a dry run, which is how a rule is meant to be introduced: write it, run
    it against the real queue, read the reasons it gives, and only then grant it
    can_approve or can_reject.
    """
    from src.domain.approval_agent import run as run_agent

    return await run_agent(
        db, current_user.company_domain, limit=payload.limit, dry_run=payload.dry_run
    )


class KnowledgePurgeRequest(BaseModel):
    # Defaults to a dry run, like the approval agent above. Deleting an entire corpus
    # should be something a caller asks for explicitly, not what happens if a field is
    # forgotten by a script.
    dry_run: bool = True
    #: Must equal the company_domain being purged. A boolean alone is too easy to send by
    #: accident from a saved request; typing the tenant name proves the operator knows
    #: WHICH corpus they are erasing, which is the mistake worth preventing when several
    #: environments share a client.
    confirm: str | None = Field(default=None, max_length=255)


@router.post("/knowledge/purge")
async def purge_knowledge(
    payload: KnowledgePurgeRequest,
    current_user: User = Depends(require_permission("role.manage", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Delete every article, document, chunk and sync record for the caller's tenant.

    IRREVERSIBLE. Intended for resetting a test corpus, which is otherwise a long manual
    job that does not even work: deleting articles by hand leaves the connector's
    revision/content-hash cache intact, so the next sync decides every provider file is
    unchanged and re-imports nothing.

    Keeps users, roles, departments, access groups, tag vocabulary, feature flags and the
    audit log. Keeps connector rows too, so the SharePoint grant survives and the operator
    does not have to reconnect after each reset — only their synchronisation state is
    reset, which is what makes the next sync re-import everything.

    Scoped to `current_user.company_domain`. A global-scope permission is required because
    the operation is unrecoverable, NOT because it crosses tenants — it does not.
    """
    from src.domain.kb_purge import delete_purged_objects, purge_knowledge_base

    company_domain = current_user.company_domain
    if not company_domain:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "tenant_unresolved",
                "message": "This account has no company domain, so no corpus can be scoped.",
            },
        )

    if not payload.dry_run and payload.confirm != company_domain:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "confirmation_mismatch",
                "message": (
                    "Set confirm to the company domain being purged to run this for real."
                ),
                "expected": company_domain,
            },
        )

    counts = await purge_knowledge_base(db, company_domain, dry_run=payload.dry_run)

    if payload.dry_run:
        # Nothing was written, so there is nothing to audit and nothing to commit.
        return {"dry_run": True, "company_domain": company_domain, **counts.as_dict()}

    # The audit row lands in the SAME transaction as the deletions, so the record of the
    # purge cannot survive without the purge or vice versa. audit_logs is deliberately not
    # one of the purged tables.
    db.add(
        AuditLog(
            user_id=current_user.id,
            action="knowledge_purge",
            target_type="knowledge_base",
            target_id=company_domain,
            outcome="success",
            detail_json=counts.as_dict(),
        )
    )
    await db.commit()

    # Object storage has no rollback, so the objects go only after the rows are durable.
    # Until this point a failed commit left the rows intact and their bytes already
    # destroyed; now a failure leaves both, and the orphan sweep reclaims the keys.
    await delete_purged_objects(counts)

    return {"dry_run": False, "company_domain": company_domain, **counts.as_dict()}


class FactoryResetRequest(BaseModel):
    #: Defaults to a dry run, like the knowledge purge above. A destructive path is
    #: something a caller asks for, never what happens when a field is forgotten.
    dry_run: bool = True
    #: Must equal FACTORY_RESET_CONFIRM_PHRASE. A boolean is too easy to resend from
    #: saved request history; typing the phrase proves the operator knows this is not
    #: the tenant-scoped content purge.
    confirm: str | None = Field(default=None, max_length=255)


#: Deliberately not the company domain — that is the knowledge purge's phrase, and
#: reusing it would let a saved purge request execute a full reset.
FACTORY_RESET_CONFIRM_PHRASE = "RESET ENTIRE DATABASE"


@router.post("/system/factory-reset")
async def factory_reset_database(
    payload: FactoryResetRequest,
    current_user: User = Depends(require_permission("role.manage", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Erase every table except identity, for every tenant. IRREVERSIBLE.

    Returns the deployment to a just-released state: users, roles, permissions,
    departments and SSO identity links survive; all knowledge, drafts, connectors,
    audit history, feature flags and the LLM provider configuration do not.

    Four independent gates, because no single one is enough for an operation with
    no undo:

    1. `role.manage` at global scope — the strongest permission the RBAC model has.
    2. `FACTORY_RESET_ENABLED`, so the capability does not exist in a deployment
       that never asked for it.
    3. An email allowlist, checked against the verified account.
    4. An exact confirmation phrase, and `dry_run` defaults to true.

    Gates 2 and 3 live in the API environment, which a compromised session cannot
    edit. That is the point: permission alone would mean any future global admin
    inherits the ability to destroy the deployment.
    """
    from src.domain.factory_reset import (
        delete_reset_objects,
        factory_reset,
        is_reset_operator,
    )

    if not settings.FACTORY_RESET_ENABLED:
        # 404, not 403: an endpoint that is switched off should not advertise that it
        # exists and is merely refusing this caller.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Not Found"
        )
    if not is_reset_operator(current_user.email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "not_reset_operator",
                "message": "This account is not authorised to reset the database.",
            },
        )
    if not payload.dry_run and payload.confirm != FACTORY_RESET_CONFIRM_PHRASE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "confirmation_mismatch",
                "message": "Set confirm to the exact phrase to run this for real.",
                "expected": FACTORY_RESET_CONFIRM_PHRASE,
            },
        )

    # Every table here FORCEs row security. Without a global-admin context the
    # DELETEs are silently FILTERED rather than refused, and the run would report
    # success having removed only the caller's own tenant.
    await set_database_context(db, None, True, user_id=str(current_user.id))

    counts = await factory_reset(db, dry_run=payload.dry_run)
    if payload.dry_run:
        return {"dry_run": True, **counts.as_dict()}

    # AFTER the deletes, not before: `audit_logs` is one of the cleared tables, so a
    # row written first would be erased by the reset it was recording. Same
    # transaction, so the record cannot survive without the reset or vice versa.
    db.add(
        AuditLog(
            user_id=current_user.id,
            action="factory_reset",
            target_type="database",
            target_id="all",
            outcome="success",
            detail_json=counts.as_dict(),
        )
    )
    await db.commit()

    # Object storage has no rollback, so the bytes go only once the rows are durable.
    await delete_reset_objects(counts)

    return {"dry_run": False, **counts.as_dict()}
