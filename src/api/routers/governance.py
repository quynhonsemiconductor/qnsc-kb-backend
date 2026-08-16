import uuid
import json
from datetime import datetime
from dataclasses import asdict
from typing import Any
from fastapi import APIRouter, Depends, Query, status, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.api.deps import get_db, get_current_user, require_permission
from src.models import User
from src.models.article import Article
from src.models.user import Department
from src.models.governance import (
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
from src.models.ops import EvalQuestion, EvalRun, IndexReprocessJob, Connector
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

router = APIRouter()


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
    visibility: str | None = Field(default=None, pattern="^(public|department|users)$")
    explicit_user_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)
    denied_user_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)


class AssignRequest(BaseModel):
    dept: str


class AssignApproverRequest(BaseModel):
    approver_id: uuid.UUID | None = None
    use_rule: bool = False


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
    ).assign_approver(current_user, id, req.approver_id, req.use_rule)
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
    children = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).commit_candidates(current_user, id)
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
    try:
        from src.workers.tasks import restructure_pending_draft_task

        restructure_pending_draft_task.delay(
            str(draft.id), current_user.company_domain, str(current_user.id)
        )
    except Exception:
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
        user_id=user_id,
        action=action,
        start_time=start_time,
        end_time=end_time,
    )
    return [_audit_response(audit) for audit in logs]


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
    connector_filters = [Connector.system == "sharepoint", Connector.status == "active"]
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
    run = await ops_repo.create_eval_run(
        EvalRun(
            eval_question_id=question.id,
            retrieval_version=settings.RETRIEVAL_VERSION,
            prompt_version=settings.PROMPT_VERSION,
            context_recall=retrieval_score,
            faithfulness=faithfulness_score,
            answer_correctness=correctness_score,
        )
    )
    return {
        "id": str(run.id),
        "eval_question_id": str(question.id),
        "context_recall": retrieval_score,
        "faithfulness": faithfulness_score,
        "answer_correctness": correctness_score,
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
