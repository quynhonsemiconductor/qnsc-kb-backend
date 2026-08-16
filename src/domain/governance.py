import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Sequence
from fastapi import HTTPException
from src.models.governance import (
    PendingDraft,
    DraftTransition,
    DraftCandidate,
    ApproverRule,
    IngestionFingerprint,
    Gap,
    AuditLog,
)
from src.models.article import (
    Article,
    ArticleTag,
    ArticleUserPermission,
    ArticleVersion,
    DocumentSource,
)
from src.models.user import User, AccessGroup, Department
from src.repositories.user import UserRepository
from src.models.connectors import ExternalDocument, ExternalGroupMapping
from src.models.ops import Connector, NotificationQueue
from sqlalchemy import delete, select
from src.repositories.governance import GovernanceRepository
from src.repositories.article import ArticleRepository
from src.domain.events import event_bus
from src.domain.content_restructure import _fallback_text, restructure_document
from src.domain.rbac import AuthorizationService
from src.domain.source_storage import delete_source
from src.domain.departments import resolve_active_department, resolve_active_departments


class GovernanceService:
    def __init__(self, gov_repo: GovernanceRepository, article_repo: ArticleRepository):
        self.gov_repo = gov_repo
        self.article_repo = article_repo

    async def _get_draft_for_user(
        self, draft_id: uuid.UUID, user: User, *, for_update: bool = False
    ) -> PendingDraft | None:
        scoped_loader = getattr(self.gov_repo, "get_draft_for_user", None)
        if scoped_loader is not None:
            return await scoped_loader(draft_id, user, for_update=for_update)
        # Lightweight repository doubles used by domain unit tests predate
        # the scoped loader; production repositories always take the query
        # path above.
        return await self.gov_repo.get_draft(draft_id)

    async def log_audit(
        self,
        user_id: uuid.UUID | None,
        action: str,
        target_type: str,
        target_id: str | None = None,
        outcome: str = "success",
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            outcome=outcome,
        )
        return await self.gov_repo.log_audit(log)

    async def _transition_draft(
        self,
        draft: PendingDraft,
        actor: User | None,
        to_status: str,
        reason: str | None = None,
    ) -> None:
        allowed = {
            "draft": {"pending", "rejected"},
            "pending": {"approved", "rejected"},
            "rejected": {"pending"},
            "approved": set(),
        }
        if to_status not in allowed.get(draft.status, set()):
            raise HTTPException(
                status_code=409,
                detail=f"Draft cannot transition from {draft.status} to {to_status}",
            )
        db = self.gov_repo.db
        if hasattr(db, "add"):
            db.add(
                DraftTransition(
                    draft_id=draft.id,
                    from_status=draft.status,
                    to_status=to_status,
                    actor_id=actor.id if actor else None,
                    reason=reason.strip() if reason and reason.strip() else None,
                    outcome="applied",
                )
            )
        draft.status = to_status

    def _can_approve_draft(self, user: User, draft: PendingDraft) -> bool:
        """Only the assigned approver or a privileged Admin/CEO may approve."""
        return bool(
            (draft.assigned_approver_id and draft.assigned_approver_id == user.id)
            or self._may_self_approve(user)
        )

    def _has_draft_permission(self, user: User, key: str, draft: PendingDraft) -> bool:
        """Evaluate a role scope against the draft's tenant and department."""
        return any(
            AuthorizationService.has_permission(user, key, draft, scope)
            for scope in ("own", "department", "company", "global")
        )

    def _can_review_draft(self, user: User, draft: PendingDraft) -> bool:
        return self._has_draft_permission(
            user, "article.review", draft
        ) or self._has_draft_permission(user, "article.publish", draft)

    def _may_self_approve(self, user: User) -> bool:
        """Admins and CEOs may approve their own submissions."""
        if user.role in {"Admin", "CEO"}:
            return True
        return any(
            role.active is not False
            and role.name in {"Admin", "CEO"}
            and role.company_domain in {None, user.company_domain}
            for role in getattr(user, "roles", [])
        )

    def _can_assign_approver(self, user: User, draft: PendingDraft) -> bool:
        # Assignment is optional. Any user who may review this draft may
        # choose to narrow it to one specific reviewer.
        return self._can_review_draft(user, draft)

    def _is_company_governance_lead(self, user: User) -> bool:
        """Return whether the identity is an Admin/CEO for its company."""
        return self._may_self_approve(user)

    def _draft_department_names(self, draft: PendingDraft) -> set[str]:
        metadata = getattr(draft, "content_metadata", None) or {}
        names = {str(draft.dept).strip()} if draft.dept else set()
        names.update(
            str(name).strip()
            for name in metadata.get("department_names", [])
            if str(name).strip()
        )
        return names

    def _draft_department_ids(self, draft: PendingDraft) -> set[str]:
        metadata = getattr(draft, "content_metadata", None) or {}
        return {str(item) for item in metadata.get("department_ids", []) if item}

    @staticmethod
    def _unmapped_external_acl_principals(metadata: dict) -> list[str]:
        """Return provider ACL principals that still need an approved mapping."""
        principals = [
            *(
                f"group:{item}"
                for item in metadata.get("unmapped_group_ids", []) or []
                if item
            ),
            *(
                f"user:{item}"
                for item in metadata.get("unmapped_source_user_ids", []) or []
                if item
            ),
            *(
                str(item)
                for item in metadata.get("unmapped_principal_ids", []) or []
                if item
            ),
        ]
        return sorted(set(principals))

    def _is_global_publisher(self, user: User) -> bool:
        return AuthorizationService.has_permission(
            user, "article.publish", requested_scope="global"
        )

    async def list_drafts(
        self, user: User, status: str | None = None
    ) -> Sequence[PendingDraft]:
        if self._is_global_publisher(user):
            return await self.gov_repo.list_drafts(status, assigned_approver_id=user.id)
        can_company_review = any(
            AuthorizationService.has_permission(user, key, requested_scope="company")
            for key in ("article.review", "article.publish", "governance.read")
        )
        if can_company_review:
            # Admin/CEO see all unassigned company drafts. Other reviewers
            # are scoped to departments they belong to, even when their
            # legacy role is granted a company-scoped review permission.
            if self._is_company_governance_lead(user):
                return await self.gov_repo.list_drafts(
                    status, user.company_domain, assigned_approver_id=user.id
                )
            member_departments = AuthorizationService.member_department_names(user)
            if user.dept:
                member_departments.add(user.dept)
            member_department_ids = {
                str(department.id)
                for department in getattr(user, "departments", [])
                if getattr(department, "active", True)
                and getattr(department, "company_domain", user.company_domain)
                == user.company_domain
            }
            return await self.gov_repo.list_drafts(
                status,
                user.company_domain,
                depts=sorted(member_departments),
                assigned_approver_id=user.id,
            )
        raise HTTPException(
            status_code=403, detail="Not authorized to view the approval queue"
        )

    async def submit_draft(
        self, user: User, draft_id: uuid.UUID, reason: str | None = None
    ) -> PendingDraft:
        """Move an authored/uploaded draft into the review queue."""
        draft = await self._get_draft_for_user(draft_id, user, for_update=True)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        derived_candidate = (getattr(draft, "content_metadata", None) or {}).get(
            "submission_kind"
        ) == "split_candidate"
        if (
            draft.created_by != user.id
            and not self._is_global_publisher(user)
            and not (derived_candidate and self._can_review_draft(user, draft))
        ):
            raise HTTPException(
                status_code=403,
                detail="Only the draft author or a global publisher can submit this draft",
            )
        if draft.status != "draft":
            raise HTTPException(
                status_code=409,
                detail=f"Draft cannot be submitted from status: {draft.status}",
            )
        await self._transition_draft(
            draft, user, "pending", reason or "Submitted for independent approval"
        )
        rule_loader = getattr(self.gov_repo, "get_approver_rule", None)
        rule = (
            await rule_loader(draft.company_domain, draft.dept) if rule_loader else None
        )
        if rule and rule.active:
            approver = await UserRepository(self.gov_repo.db).get_by_id(
                rule.approver_id, viewer=user
            )
            if (
                approver
                and approver.active
                and approver.company_domain == draft.company_domain
                and self._can_review_draft(approver, draft)
            ):
                if (
                    not draft.created_by
                    or approver.id != draft.created_by
                    or self._may_self_approve(approver)
                ):
                    draft.assigned_approver_id = approver.id
                    draft.assigned_by = user.id
                    draft.assigned_at = datetime.utcnow()
                    self.gov_repo.db.add(
                        NotificationQueue(
                            recipient_user_id=approver.id,
                            type="in_app",
                            payload={
                                "event": "draft_assigned_by_rule",
                                "draft_id": str(draft.id),
                                "approver_id": str(approver.id),
                                "assigned_by": str(user.id),
                            },
                        )
                    )
                    self.gov_repo.db.add(
                        AuditLog(
                            user_id=user.id,
                            action="assign_approver_rule",
                            target_type="draft",
                            target_id=str(draft.id),
                        )
                    )
        updated = await self.gov_repo.update_draft(draft)
        await self.log_audit(user.id, "submit", "draft", str(draft.id))
        return updated

    async def list_candidates(
        self, user: User, draft_id: uuid.UUID
    ) -> Sequence[DraftCandidate]:
        draft = await self._get_draft_for_user(draft_id, user)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        if not self._can_review_draft(user, draft) and draft.created_by != user.id:
            raise HTTPException(
                status_code=403, detail="Not authorized to review this document"
            )
        loader = getattr(self.gov_repo, "list_candidates", None)
        return (
            await loader(draft_id, user)
            if loader
            else list(getattr(draft, "candidates", []) or [])
        )

    async def review_candidate(
        self,
        user: User,
        draft_id: uuid.UUID,
        operation: str,
        candidate_id: uuid.UUID,
        other_candidate_id: uuid.UUID | None = None,
        title: str | None = None,
        split_at: int | None = None,
        department_ids: list[uuid.UUID] | None = None,
        note: str | None = None,
    ) -> Sequence[DraftCandidate]:
        draft = await self._get_draft_for_user(draft_id, user, for_update=True)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        if draft.status != "pending":
            raise HTTPException(
                status_code=409, detail="Only pending documents can be batch reviewed"
            )
        if draft.restructure_status in {"queued", "processing"}:
            raise HTTPException(
                status_code=409,
                detail="AI formatting is still in progress; candidates will be ready after it finishes",
            )
        if not self._can_review_draft(user, draft):
            raise HTTPException(
                status_code=403, detail="Not authorized to batch review this document"
            )
        candidates = list(await self.list_candidates(user, draft_id))
        by_id = {item.id: item for item in candidates}
        candidate = by_id.get(candidate_id)
        if not candidate or candidate.status != "candidate":
            raise HTTPException(status_code=404, detail="Candidate not found")
        if operation == "rename":
            if not title or not title.strip():
                raise HTTPException(
                    status_code=422, detail="A candidate title is required"
                )
            candidate.title = title.strip()[:255]
        elif operation == "set_departments":
            departments = await resolve_active_departments(
                self.gov_repo.db,
                draft.company_domain,
                department_ids or [],
                required=False,
            )
            candidate.department_ids = [
                str(department.id) for department in departments
            ]
        elif operation == "discard":
            candidate.status = "discarded"
            candidate.review_note = (note or "Discarded during batch review")[:2000]
        elif operation == "merge":
            other = by_id.get(other_candidate_id) if other_candidate_id else None
            if not other or other.status != "candidate" or other.id == candidate.id:
                raise HTTPException(
                    status_code=422, detail="Choose another active candidate to merge"
                )
            first, second = sorted((candidate, other), key=lambda item: item.position)
            first.body_md = f"{first.body_md.rstrip()}\n\n{second.body_md.lstrip()}"
            first.source_end = max(first.source_end, second.source_end)
            first.title = (title or first.title)[:255]
            second.status = "discarded"
            second.review_note = (note or f"Merged into candidate {first.position}")[
                :2000
            ]
        elif operation == "split":
            boundary = split_at or len(candidate.body_md) // 2
            if boundary < 1 or boundary >= len(candidate.body_md):
                raise HTTPException(
                    status_code=422,
                    detail="The split position must be inside the candidate text",
                )
            for item in sorted(
                (
                    item
                    for item in candidates
                    if item.status == "candidate" and item.position > candidate.position
                ),
                key=lambda item: item.position,
                reverse=True,
            ):
                item.position += 1
            left = candidate.body_md[:boundary].rstrip()
            right = candidate.body_md[boundary:].lstrip()
            original_end = candidate.source_end
            candidate.body_md = left
            candidate.source_end = candidate.source_start + len(left)
            db = self.gov_repo.db
            db.add(
                DraftCandidate(
                    draft_id=draft.id,
                    position=candidate.position + 1,
                    title=(title or f"{candidate.title} — continued")[:255],
                    body_md=right,
                    source_start=candidate.source_end,
                    source_end=original_end,
                    heading=candidate.heading,
                    department_ids=candidate.department_ids,
                    department_suggestions=candidate.department_suggestions,
                    proposed_department=candidate.proposed_department,
                    status="candidate",
                )
            )
        else:
            raise HTTPException(
                status_code=422, detail="Unsupported candidate operation"
            )
        await self.gov_repo.db.commit()
        await self.log_audit(
            user.id, f"candidate_{operation}", "draft_candidate", str(candidate.id)
        )
        return await self.list_candidates(user, draft_id)

    async def commit_candidates(
        self, user: User, draft_id: uuid.UUID
    ) -> list[PendingDraft]:
        parent = await self._get_draft_for_user(draft_id, user, for_update=True)
        if not parent:
            raise HTTPException(status_code=404, detail="Draft not found")
        if parent.status != "pending":
            raise HTTPException(
                status_code=409, detail="Only pending documents can be committed"
            )
        if parent.restructure_status in {"queued", "processing"}:
            raise HTTPException(
                status_code=409,
                detail="AI formatting is still in progress; candidates cannot be committed yet",
            )
        if not self._can_review_draft(user, parent):
            raise HTTPException(
                status_code=403, detail="Not authorized to commit batch candidates"
            )
        candidates = [
            item
            for item in await self.list_candidates(user, draft_id)
            if item.status == "candidate"
        ]
        if not candidates:
            raise HTTPException(
                status_code=422, detail="Keep at least one candidate before committing"
            )
        db = self.gov_repo.db
        children: list[PendingDraft] = []
        base_metadata = dict(parent.content_metadata or {})
        selected_candidate_ids = {
            department_id
            for item in candidates
            for department_id in (item.department_ids or [])
        }
        available_departments = []
        if selected_candidate_ids:
            available_departments = list(
                (
                    await db.execute(
                        select(Department).where(
                            Department.company_domain == parent.company_domain,
                            Department.active.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )
        departments_by_id = {
            str(department.id): department for department in available_departments
        }
        for item in sorted(candidates, key=lambda value: value.position):
            selected_departments = [
                departments_by_id[department_id]
                for department_id in (item.department_ids or [])
                if department_id in departments_by_id
            ]
            # Existing pre-routing candidates retain the parent route until a reviewer
            # opens them. New formatted candidates always carry an explicit selection.
            if not selected_departments and item.department_ids:
                raise HTTPException(
                    status_code=422,
                    detail=f"Choose at least one active department for candidate {item.position}",
                )
            child_hash = hashlib.sha256(
                f"{parent.source_hash}:{item.position}:{item.body_md}".encode("utf-8")
            ).hexdigest()
            metadata = {
                **base_metadata,
                "submission_kind": "split_candidate",
                "source_position": {
                    "start": item.source_start,
                    "end": item.source_end,
                    "heading": item.heading,
                },
                "department_ids": [
                    str(department.id) for department in selected_departments
                ]
                or list(base_metadata.get("department_ids") or []),
            }
            child = PendingDraft(
                title=item.title,
                company_domain=parent.company_domain,
                dept=(
                    selected_departments[0].name
                    if selected_departments
                    else parent.dept
                ),
                source_ref=f"{parent.source_ref}#candidate-{item.position}",
                source_hash=child_hash,
                summary=item.body_md,
                restructured_body_md=item.body_md,
                restructure_status="lossless_ready",
                restructure_model="structure-aware-splitter",
                storage_key=parent.storage_key,
                original_filename=parent.original_filename,
                mime_type=parent.mime_type,
                page_texts=parent.page_texts,
                created_by=parent.created_by,
                status="draft",
                tags=parent.tags,
                content_metadata=metadata,
            )
            db.add(child)
            await db.flush()
            db.add(
                DraftTransition(
                    draft_id=child.id,
                    from_status=None,
                    to_status="draft",
                    actor_id=user.id,
                    reason="Created from batch-reviewed candidate",
                    outcome="applied",
                )
            )
            children.append(child)
            item.status = "committed"
        await self._transition_draft(
            parent, user, "rejected", "Split candidates committed as independent drafts"
        )
        parent.review_note = "Split candidates committed as independent drafts"
        await db.commit()
        for child in children:
            await self.submit_draft(user, child.id, "Committed from batch review")
        await self.log_audit(user.id, "batch_commit", "draft", str(parent.id))
        return children

    async def assign_approver(
        self,
        user: User,
        draft_id: uuid.UUID,
        approver_id: uuid.UUID | None = None,
        use_rule: bool = False,
    ) -> PendingDraft:
        draft = await self._get_draft_for_user(draft_id, user)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        if draft.status != "pending":
            raise HTTPException(
                status_code=409, detail="Only pending drafts can be assigned"
            )
        if (
            draft.company_domain != user.company_domain
            and not self._is_global_publisher(user)
        ):
            raise HTTPException(status_code=403, detail="Draft is outside your company")
        if not self._can_assign_approver(user, draft):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to assign approvers for this draft",
            )

        if use_rule:
            rule_loader = getattr(self.gov_repo, "get_approver_rule", None)
            rule = (
                await rule_loader(draft.company_domain, draft.dept)
                if rule_loader
                else None
            )
            if not rule or not rule.active:
                raise HTTPException(
                    status_code=404,
                    detail="No active approver rule exists for this department",
                )
            approver_id = rule.approver_id
        if not approver_id:
            raise HTTPException(
                status_code=422,
                detail="approver_id is required unless use_rule is true",
            )

        approver = await UserRepository(self.gov_repo.db).get_by_id(
            approver_id, viewer=user
        )
        if (
            not approver
            or not approver.active
            or (approver.company_domain != draft.company_domain)
        ):
            raise HTTPException(
                status_code=422,
                detail="Approver must be an active user in the draft's company",
            )
        if (
            draft.created_by
            and approver.id == draft.created_by
            and not self._may_self_approve(approver)
        ):
            raise HTTPException(
                status_code=422, detail="A submitter cannot approve their own draft"
            )
        if not self._can_review_draft(approver, draft):
            raise HTTPException(
                status_code=422,
                detail="Selected user does not have approval permission for this draft",
            )

        draft.assigned_approver_id = approver.id
        draft.assigned_by = user.id
        draft.assigned_at = datetime.utcnow()
        updated = await self.gov_repo.update_draft(draft)
        self.gov_repo.db.add(
            NotificationQueue(
                recipient_user_id=approver.id,
                type="in_app",
                payload={
                    "event": "draft_assigned",
                    "draft_id": str(draft.id),
                    "approver_id": str(approver.id),
                    "assigned_by": str(user.id),
                },
            )
        )
        await self.gov_repo.db.commit()
        await self.log_audit(
            user.id,
            "assign_approver_rule" if use_rule else "assign_approver",
            "draft",
            str(draft.id),
        )
        return updated

    async def eligible_approvers(
        self, user: User, draft_id: uuid.UUID
    ) -> Sequence[User]:
        draft = await self._get_draft_for_user(draft_id, user)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        if (
            draft.company_domain != user.company_domain
            and not self._is_global_publisher(user)
        ):
            raise HTTPException(status_code=403, detail="Draft is outside your company")
        if not self._can_assign_approver(user, draft):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to assign approvers for this draft",
            )
        users = await UserRepository(self.gov_repo.db).list_users(
            limit=500,
            viewer=user,
            company_domain=draft.company_domain,
            active=True,
        )
        return [
            candidate
            for candidate in users
            if (candidate.id != draft.created_by or self._may_self_approve(candidate))
            and self._can_review_draft(candidate, draft)
        ]

    async def approve_draft(
        self,
        user: User,
        draft_id: uuid.UUID,
        category: str | None = None,
        dept: str | None = None,
        update_article_id: uuid.UUID | None = None,
        treat_as_new: bool = False,
        sensitivity: str | None = None,
        access_group_ids: list[uuid.UUID] | None = None,
        department_ids: list[uuid.UUID] | None = None,
        review_note: str | None = None,
        visibility: str | None = None,
        explicit_user_ids: list[uuid.UUID] | None = None,
        denied_user_ids: list[uuid.UUID] | None = None,
    ) -> Article:
        """Publish one draft atomically.

        The lock is deliberately taken before validation.  A second request
        waits, then sees the committed ``approved`` state instead of creating
        a second article from the same draft.
        """
        db = self.gov_repo.db
        if hasattr(db, "execute"):
            draft = await self._get_draft_for_user(draft_id, user, for_update=True)
        else:  # Lightweight unit-test repository compatibility.
            draft = await self.gov_repo.get_draft(draft_id)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        try:
            metadata = getattr(draft, "content_metadata", None) or {}
            category = category or str(metadata.get("type") or "SOP")
            dept = dept or str(metadata.get("dept") or draft.dept or "")
            sensitivity = sensitivity or str(metadata.get("sensitivity") or "public")
            visibility = visibility or str(
                metadata.get("visibility")
                or ("public" if sensitivity == "public" else "department")
            )
            if explicit_user_ids is None:
                try:
                    explicit_user_ids = [
                        uuid.UUID(str(user_id))
                        for user_id in metadata.get("explicit_user_ids", [])
                    ]
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="The submitted explicit-user selection is invalid",
                    ) from exc
            if denied_user_ids is None:
                try:
                    denied_user_ids = [
                        uuid.UUID(str(user_id))
                        for user_id in metadata.get("denied_user_ids", [])
                    ]
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="The submitted explicit-deny selection is invalid",
                    ) from exc
            if access_group_ids is None:
                try:
                    access_group_ids = [
                        uuid.UUID(str(group_id))
                        for group_id in metadata.get("access_group_ids", [])
                    ]
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="The submitted access-group selection is invalid",
                    ) from exc
            if draft.status != "pending":
                raise HTTPException(
                    status_code=409,
                    detail=f"Draft cannot be approved from status: {draft.status}",
                )
            if (
                draft.company_domain != user.company_domain
                and not self._is_global_publisher(user)
            ):
                raise HTTPException(
                    status_code=403, detail="Draft is outside your company"
                )
            if not self._can_approve_draft(user, draft):
                raise HTTPException(
                    status_code=403,
                    detail="Only the assigned approver or an Admin can approve this draft",
                )
            external_document = None
            if draft.external_document_id:
                external_document = await db.get(
                    ExternalDocument, draft.external_document_id
                )
                if not external_document:
                    raise HTTPException(
                        status_code=409,
                        detail="The connector source record is missing; approval is blocked",
                    )
                unresolved_acl = self._unmapped_external_acl_principals(
                    external_document.metadata_json or {}
                )
                if unresolved_acl:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "external_acl_mapping_required",
                            "message": "Map every provider permission principal before approving this source",
                            "principals": unresolved_acl,
                        },
                    )
            active_candidates = [
                item
                for item in (getattr(draft, "candidates", []) or [])
                if item.status == "candidate"
            ]
            if len(active_candidates) > 1:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "batch_review_required",
                        "message": "Review and commit split candidates before approving this document",
                    },
                )
            if (
                draft.requires_update_confirmation
                and not update_article_id
                and not treat_as_new
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "update_confirmation_required",
                        "matches": draft.similarity_matches or [],
                    },
                )
            if metadata.get("submission_kind") == "manual_update":
                expected_target = str(metadata.get("suggested_update_article_id") or "")
                if (
                    treat_as_new
                    or not update_article_id
                    or str(update_article_id) != expected_target
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="A manually submitted article change must update its original article",
                    )
            if not dept.strip():
                raise HTTPException(
                    status_code=422,
                    detail="An active department is required before publishing",
                )
            dept = (
                await resolve_active_department(db, draft.company_domain, dept)
            ).name
            if draft.dept and dept != draft.dept:
                raise HTTPException(
                    status_code=422,
                    detail="An approved article must remain in the submitted department",
                )
            try:
                submitted_department_ids = (
                    department_ids
                    if department_ids is not None
                    else metadata.get("department_ids", [])
                )
                department_ids = [
                    uuid.UUID(str(item)) for item in submitted_department_ids
                ]
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The submitted department selection is invalid",
                ) from exc
            if department_ids:
                selected_departments = await resolve_active_departments(
                    db, draft.company_domain, department_ids
                )
                if dept not in {department.name for department in selected_departments}:
                    raise HTTPException(
                        status_code=422,
                        detail="The submitted primary department must be selected",
                    )
            else:
                selected_departments = [
                    await resolve_active_department(db, draft.company_domain, dept)
                ]
            if category not in {
                "POLICY",
                "SOP",
                "DECISION",
                "FAQ",
                "RCA",
                "HOWTO",
                "PLAYBOOK",
                "REFERENCE",
            }:
                raise HTTPException(status_code=422, detail="Invalid article type")
            if sensitivity not in {"public", "internal", "confidential", "restricted"}:
                raise HTTPException(
                    status_code=422, detail="Invalid article sensitivity"
                )
            if visibility not in {"public", "department", "users"}:
                raise HTTPException(
                    status_code=422, detail="Invalid Article visibility"
                )
            explicit_user_ids = list(dict.fromkeys(explicit_user_ids or []))
            denied_user_ids = list(dict.fromkeys(denied_user_ids or []))
            if set(explicit_user_ids) & set(denied_user_ids):
                raise HTTPException(
                    status_code=422,
                    detail="A user cannot be both explicitly allowed and denied",
                )
            selected_explicit_users = list(
                await UserRepository(db).get_by_ids(
                    explicit_user_ids + denied_user_ids, draft.company_domain
                )
            )
            if len(selected_explicit_users) != len(
                set(explicit_user_ids + denied_user_ids)
            ):
                raise HTTPException(
                    status_code=422,
                    detail="Every explicit Article user must belong to the Article company",
                )
            if visibility == "users" and not explicit_user_ids:
                raise HTTPException(
                    status_code=422,
                    detail="Explicit-user visibility requires at least one user",
                )

            selected_groups: list[AccessGroup] = []
            if access_group_ids:
                selected_groups = list(
                    await UserRepository(db).get_groups_by_ids(
                        access_group_ids, draft.company_domain
                    )
                )
                if len({group.id for group in selected_groups}) != len(
                    set(access_group_ids)
                ):
                    raise HTTPException(
                        status_code=422, detail="One or more access groups do not exist"
                    )
            # New content is authorized by role/permission/department. If a
            # legacy draft carries a non-public flag without a real ACL, make
            # it compatible with the current resource-based model instead of
            # blocking publication on a removed UI control.
            if (
                sensitivity != "public"
                and not selected_groups
                and not explicit_user_ids
                and not draft.external_document_id
            ):
                sensitivity = "public"
            external_source_user_ids: set[uuid.UUID] = set()
            if external_document:
                external_metadata = external_document.metadata_json or {}
                mapped_ids = external_metadata.get("mapped_access_group_ids", [])
                selected_groups = (
                    list(
                        (
                            await db.execute(
                                select(AccessGroup).where(
                                    AccessGroup.id.in_(mapped_ids),
                                    AccessGroup.company_domain == draft.company_domain,
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                    if mapped_ids
                    else []
                )
                if len({group.id for group in selected_groups}) != len(set(mapped_ids)):
                    raise HTTPException(
                        status_code=422,
                        detail="The connector ACL contains an invalid access group",
                    )
                try:
                    external_source_user_ids = {
                        uuid.UUID(str(item))
                        for item in external_metadata.get("mapped_source_user_ids", [])
                    }
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="The connector ACL contains an invalid mapped user",
                    ) from exc
                selected_external_users = list(
                    await UserRepository(db).get_by_ids(
                        list(external_source_user_ids), draft.company_domain
                    )
                )
                if len(selected_external_users) != len(external_source_user_ids):
                    raise HTTPException(
                        status_code=422,
                        detail="The connector ACL contains a user outside this company",
                    )
                explicit_user_ids = list(
                    dict.fromkeys([*explicit_user_ids, *external_source_user_ids])
                )
                if set(explicit_user_ids) & set(denied_user_ids):
                    raise HTTPException(
                        status_code=422,
                        detail="A connector-mapped user cannot be explicitly denied",
                    )
                selected_explicit_users = list(
                    await UserRepository(db).get_by_ids(
                        explicit_user_ids + denied_user_ids, draft.company_domain
                    )
                )
                if len(selected_explicit_users) != len(
                    set(explicit_user_ids + denied_user_ids)
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="Every explicit Article user must belong to the Article company",
                    )
                # A provider ACL is restrictive input, never a reason to make
                # the Article public. Empty/unmapped ACLs remain closed by
                # using a restricted Article with no effective grant.
                if selected_groups:
                    sensitivity = "restricted"
                    visibility = "department"
                elif external_source_user_ids:
                    sensitivity = "restricted"
                    visibility = "users"
                else:
                    sensitivity = "restricted"
                    visibility = "department"
                connector = await db.get(Connector, external_document.connector_id)
                if not connector or connector.company_domain != draft.company_domain:
                    raise HTTPException(
                        status_code=422,
                        detail="The connector document is outside this draft's company",
                    )

            update_target = None
            if update_article_id:
                update_target = (
                    await self.article_repo.get_by_id_for_update(
                        update_article_id, user=user
                    )
                    if hasattr(self.article_repo, "get_by_id_for_update")
                    else await self.article_repo.get_by_id(update_article_id, user=user)
                )
                if not update_target or (
                    update_target.company_domain != user.company_domain
                    and not AuthorizationService.has_permission(
                        user, "article.read", requested_scope="global"
                    )
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="The selected update target is not accessible",
                    )
                from src.domain.permissions import PermissionService

                if not PermissionService.can_view_article(
                    user, update_target
                ) or not any(
                    AuthorizationService.has_permission(
                        user, "article.edit", update_target, scope
                    )
                    or AuthorizationService.has_permission(
                        user, "article.publish", update_target, scope
                    )
                    for scope in ("own", "department", "company", "global")
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="Not authorized to supersede the selected article",
                    )
                if update_target.lifecycle_status != "active":
                    raise HTTPException(
                        status_code=409,
                        detail="The selected update target is already inactive",
                    )

            requested_external_id = str(metadata.get("external_id") or "").strip() or (
                update_target.external_id if update_target else None
            )
            if requested_external_id:
                external_stmt = select(Article).where(
                    Article.company_domain == draft.company_domain,
                    Article.external_id == requested_external_id,
                )
                if update_target:
                    external_stmt = external_stmt.where(Article.id != update_target.id)
                if await db.scalar(external_stmt):
                    raise HTTPException(
                        status_code=409,
                        detail="An article with this external ID already exists in this company",
                    )

            next_review = None
            if metadata.get("next_review"):
                try:
                    next_review = datetime.fromisoformat(
                        str(metadata["next_review"]).replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=422, detail="The submitted review date is invalid"
                    ) from exc
            elif update_target:
                next_review = update_target.next_review
            next_version = (update_target.version + 1) if update_target else 1
            body_md = (
                draft.restructured_body_md
                or draft.summary
                or f"Draft imported from {draft.source_ref}. Content pending edit."
            )
            draft_tags = list(
                dict.fromkeys(
                    str(tag).strip() for tag in (draft.tags or []) if str(tag).strip()
                )
            )[:20]
            if update_target and draft.tags is None:
                draft_tags = [tag.tag for tag in update_target.tags]
            if update_target:
                # Revisions keep the stable article identity. Replacing the
                # row previously broke bookmarks, comments, external links,
                # ACL references, and the article's version history.
                created_article = update_target
                permission_state_before = (
                    str(created_article.dept or ""),
                    tuple(
                        sorted(
                            str(department.id)
                            for department in getattr(
                                created_article, "departments", []
                            )
                        )
                    ),
                    str(created_article.sensitivity or ""),
                    str(created_article.visibility or "department"),
                    tuple(
                        sorted(
                            str(group.id)
                            for group in getattr(created_article, "access_groups", [])
                        )
                    ),
                    tuple(
                        sorted(
                            (str(item.user_id), str(item.effect))
                            for item in getattr(created_article, "user_permissions", [])
                        )
                    ),
                )
                created_article.title = draft.title
                created_article.body_md = body_md
                created_article.external_id = requested_external_id
                created_article.dept = dept
                created_article.departments = selected_departments
                created_article.domain = str(
                    metadata.get("domain") or created_article.domain
                )
                created_article.type = category
                created_article.sensitivity = sensitivity
                created_article.visibility = visibility
                created_article.language = str(
                    metadata.get("language") or created_article.language
                )
                created_article.next_review = next_review
                created_article.status = "published"
                created_article.lifecycle_status = "active"
                created_article.version = next_version
                created_article.last_reviewed = datetime.utcnow()
                created_article.index_status = "pending"
                created_article.index_error = None
                created_article.related_article_ids = (
                    draft.related_article_ids
                    if draft.related_article_ids is not None
                    else created_article.related_article_ids
                )
                if metadata.get("source_position"):
                    created_article.source_position = metadata.get("source_position")
                created_article.access_groups = selected_groups
                connector_permissions = [
                    item
                    for item in getattr(created_article, "user_permissions", [])
                    if item.source == "sharepoint"
                ]
                source_allow_ids = {
                    item.user_id
                    for item in connector_permissions
                    if item.effect == "allow"
                }
                connector_permissions.extend(
                    ArticleUserPermission(
                        user_id=external_user_id, effect="allow", source="sharepoint"
                    )
                    for external_user_id in external_source_user_ids
                    if external_user_id not in source_allow_ids
                )
                created_article.user_permissions = (
                    connector_permissions
                    + [
                        ArticleUserPermission(user_id=explicit_user_id, effect="allow")
                        for explicit_user_id in explicit_user_ids
                        if explicit_user_id not in external_source_user_ids
                    ]
                    + [
                        ArticleUserPermission(user_id=denied_user_id, effect="deny")
                        for denied_user_id in denied_user_ids
                    ]
                )
                permission_state_after = (
                    str(created_article.dept or ""),
                    tuple(
                        sorted(
                            str(department.id) for department in selected_departments
                        )
                    ),
                    str(created_article.sensitivity or ""),
                    str(created_article.visibility or "department"),
                    tuple(sorted(str(group.id) for group in selected_groups)),
                    tuple(
                        sorted(
                            [(str(user_id), "allow") for user_id in explicit_user_ids]
                            + [(str(user_id), "deny") for user_id in denied_user_ids]
                        )
                    ),
                )
                draft.update_target_article_id = created_article.id
                await db.execute(
                    delete(ArticleTag).where(
                        ArticleTag.article_id == created_article.id
                    )
                )
                db.add(
                    AuditLog(
                        user_id=user.id,
                        action="update",
                        target_type="article",
                        target_id=str(created_article.id),
                    )
                )
                if permission_state_before != permission_state_after:
                    db.add(
                        AuditLog(
                            user_id=user.id,
                            action="permission_change",
                            target_type="article",
                            target_id=str(created_article.id),
                        )
                    )
            else:
                created_article = Article(
                    title=draft.title,
                    body_md=body_md,
                    external_id=requested_external_id,
                    dept=dept,
                    domain=str(metadata.get("domain") or "Ingestion"),
                    type=category,
                    sensitivity=sensitivity,
                    language=str(metadata.get("language") or "en"),
                    next_review=next_review,
                    owner_id=user.id,
                    status="published",
                    version=next_version,
                    company_domain=draft.company_domain,
                    lifecycle_status="active",
                    last_reviewed=datetime.utcnow(),
                    related_article_ids=draft.related_article_ids,
                    source_position=metadata.get("source_position"),
                    access_groups=selected_groups,
                    visibility=visibility,
                    user_permissions=[
                        ArticleUserPermission(
                            user_id=explicit_user_id,
                            effect="allow",
                            source=(
                                "sharepoint"
                                if explicit_user_id in external_source_user_ids
                                else None
                            ),
                        )
                        for explicit_user_id in explicit_user_ids
                    ]
                    + [
                        ArticleUserPermission(user_id=denied_user_id, effect="deny")
                        for denied_user_id in denied_user_ids
                    ],
                    departments=selected_departments,
                )
                db.add(created_article)
                await db.flush()
                db.add(
                    AuditLog(
                        user_id=user.id,
                        action="create",
                        target_type="article",
                        target_id=str(created_article.id),
                    )
                )
            db.add_all(
                ArticleTag(article_id=created_article.id, tag=tag) for tag in draft_tags
            )
            db.add(
                ArticleVersion(
                    article_id=created_article.id,
                    version=next_version,
                    snapshot={
                        "title": created_article.title,
                        "body_md": created_article.body_md,
                        "dept": created_article.dept,
                        "department_ids": [
                            str(department.id) for department in selected_departments
                        ],
                        "domain": created_article.domain,
                        "type": created_article.type,
                        "sensitivity": created_article.sensitivity,
                        "visibility": created_article.visibility,
                        "explicit_user_ids": [str(item) for item in explicit_user_ids],
                        "denied_user_ids": [str(item) for item in denied_user_ids],
                        "language": created_article.language,
                        "external_id": created_article.external_id,
                        "tags": draft_tags,
                    },
                    edited_by=user.id,
                )
            )
            if draft.storage_key:
                db.add(
                    DocumentSource(
                        article_id=created_article.id,
                        source_system=(
                            connector.system if external_document else "upload"
                        ),
                        source_ref=draft.source_ref,
                        source_hash=draft.source_hash,
                        storage_key=draft.storage_key,
                        original_filename=draft.original_filename or draft.title,
                        mime_type=draft.mime_type,
                        page_texts=draft.page_texts,
                    )
                )
            if external_document:
                external_document.article_id = created_article.id
            fingerprint = await db.scalar(
                select(IngestionFingerprint).where(
                    IngestionFingerprint.company_domain == draft.company_domain,
                    IngestionFingerprint.source_hash == draft.source_hash,
                )
            )
            if fingerprint:
                fingerprint.status = "approved"
                fingerprint.article_id = created_article.id
            else:
                db.add(
                    IngestionFingerprint(
                        company_domain=draft.company_domain,
                        source_hash=draft.source_hash,
                        status="approved",
                        article_id=created_article.id,
                        created_by=draft.created_by,
                    )
                )
            await self._transition_draft(draft, user, "approved", review_note)
            draft.reviewed_by = user.id
            draft.reviewed_at = datetime.utcnow()
            draft.review_note = (
                review_note.strip() if review_note and review_note.strip() else None
            )
            db.add_all(
                [
                    AuditLog(
                        user_id=user.id,
                        action="approve",
                        target_type="draft",
                        target_id=str(draft.id),
                    ),
                    AuditLog(
                        user_id=user.id,
                        action="publish",
                        target_type="article",
                        target_id=str(created_article.id),
                    ),
                ]
            )
            if draft.created_by and draft.created_by != user.id:
                db.add(
                    NotificationQueue(
                        recipient_user_id=draft.created_by,
                        type="in_app",
                        payload={
                            "event": "draft_approved",
                            "draft_id": str(draft.id),
                            "article_id": str(created_article.id),
                            "reviewer_id": str(user.id),
                        },
                    )
                )
            await db.commit()
        except Exception:
            if hasattr(db, "rollback"):
                await db.rollback()
            raise

        published = await self.article_repo.get_by_id(created_article.id)
        if not published:
            raise HTTPException(
                status_code=500, detail="Published article could not be reloaded"
            )
        await event_bus.publish(
            "ArticleUpdated" if update_target else "ArticlePublished",
            {"article_id": str(published.id)},
        )
        return published

    async def restructure_draft(
        self, user: User, draft_id: uuid.UUID, enabled: bool = True
    ) -> PendingDraft:
        draft = await self._get_draft_for_user(draft_id, user)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        if draft.status != "pending":
            raise HTTPException(
                status_code=400, detail="Only pending drafts can be restructured"
            )
        if (
            draft.company_domain != user.company_domain
            and not self._is_global_publisher(user)
        ):
            raise HTTPException(status_code=403, detail="Draft is outside your company")
        # Preparing the reading view is available to any authorized reviewer or
        # publisher. Assignment only narrows which reviewer may act.
        if not self._can_review_draft(user, draft):
            raise HTTPException(
                status_code=403,
                detail="Only users with review or publish permission can restructure this draft",
            )

        source_text = draft.summary or "\n\n".join(
            str(page.get("text", ""))
            for page in (draft.page_texts or [])
            if page.get("text")
        )
        result = await restructure_document(draft.title, source_text, enabled=enabled)
        draft.restructured_body_md = result.body_md
        draft.restructure_candidate_md = result.candidate_body_md
        draft.restructure_decision = (
            "pending_review"
            if result.candidate_body_md
            else ("ai_ready" if result.status == "llm" else "lossless_ready")
        )
        draft.restructure_status = result.status
        draft.restructure_model = result.model
        draft.restructure_error = result.error
        updated = await self.gov_repo.update_draft(draft)
        await self.log_audit(user.id, "restructure", "draft", str(draft.id))
        return updated

    async def decide_restructure(
        self, user: User, draft_id: uuid.UUID, decision: str
    ) -> PendingDraft:
        """Let an authorized reviewer choose the AI candidate or safe fallback."""
        draft = await self._get_draft_for_user(draft_id, user)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        if draft.status != "pending":
            raise HTTPException(
                status_code=400,
                detail="Only pending drafts can change the reading-view decision",
            )
        if (
            draft.company_domain != user.company_domain
            and not self._is_global_publisher(user)
        ):
            raise HTTPException(status_code=403, detail="Draft is outside your company")
        if draft.assigned_approver_id and draft.assigned_approver_id != user.id:
            raise HTTPException(
                status_code=403, detail="This draft is assigned to another approver"
            )
        if not self._can_review_draft(user, draft):
            raise HTTPException(
                status_code=403,
                detail="You do not have approval permission for this draft",
            )
        if decision not in {"keep_ai", "keep_lossless"}:
            raise HTTPException(
                status_code=422, detail="Decision must be keep_ai or keep_lossless"
            )

        if decision == "keep_ai":
            if not draft.restructure_candidate_md:
                raise HTTPException(
                    status_code=409, detail="There is no retained AI candidate to keep"
                )
            draft.restructured_body_md = draft.restructure_candidate_md
            draft.restructure_status = "llm_reviewed"
            draft.restructure_decision = "ai_kept"
        else:
            draft.restructured_body_md = _fallback_text(
                draft.summary or draft.restructured_body_md or ""
            )
            draft.restructure_status = "fallback_formatting"
            draft.restructure_decision = "lossless_kept"
        updated = await self.gov_repo.update_draft(draft)
        await self.log_audit(user.id, f"restructure_{decision}", "draft", str(draft.id))
        return updated

    async def reject_draft(
        self, user: User, draft_id: uuid.UUID, review_note: str
    ) -> PendingDraft:
        draft = await self._get_draft_for_user(draft_id, user)
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")

        if draft.status != "pending":
            raise HTTPException(
                status_code=400,
                detail=f"Draft cannot be rejected from status: {draft.status}",
            )
        if (
            draft.company_domain != user.company_domain
            and not self._is_global_publisher(user)
        ):
            raise HTTPException(status_code=403, detail="Draft is outside your company")
        if not self._can_approve_draft(user, draft):
            raise HTTPException(
                status_code=403,
                detail="Only the assigned approver or an Admin can reject this draft",
            )

        await self._transition_draft(draft, user, "rejected", review_note)
        draft.reviewed_by = user.id
        draft.reviewed_at = datetime.utcnow()
        draft.review_note = review_note.strip()
        storage_key = draft.storage_key if not draft.external_document_id else None
        if hasattr(self.gov_repo.db, "execute"):
            await self.gov_repo.db.execute(
                delete(IngestionFingerprint).where(
                    IngestionFingerprint.company_domain == draft.company_domain,
                    IngestionFingerprint.source_hash == draft.source_hash,
                )
            )
        self.gov_repo.db.add(
            NotificationQueue(
                recipient_user_id=draft.created_by,
                type="in_app",
                payload={
                    "event": "draft_rejected",
                    "draft_id": str(draft.id),
                    "created_by": str(draft.created_by) if draft.created_by else None,
                    "reviewer_id": str(user.id),
                },
            )
        )
        updated_draft = await self.gov_repo.update_draft(draft)
        if storage_key:
            try:
                await asyncio.to_thread(delete_source, storage_key)
            except Exception:
                # Rejection must remain successful; an operator/GC job can
                # remove an unavailable object-store key later.
                pass

        # Log Audit Trail
        await self.log_audit(
            user_id=user.id,
            action="reject",
            target_type="draft",
            target_id=str(draft.id),
        )

        return updated_draft

    # Gap Queue
    async def list_gaps(self, user: User, status: str | None = None) -> Sequence[Gap]:
        if not AuthorizationService.has_permission(
            user, "governance.read", requested_scope="company"
        ):
            raise HTTPException(status_code=403, detail="Not authorized to view gaps")
        company_domain = (
            None
            if AuthorizationService.has_permission(
                user, "governance.read", requested_scope="global"
            )
            else user.company_domain
        )
        return await self.gov_repo.list_gaps(status, company_domain)

    async def assign_gap(self, user: User, gap_id: uuid.UUID, dept: str) -> Gap:
        if not AuthorizationService.has_permission(
            user, "governance.read", requested_scope="company"
        ):
            raise HTTPException(status_code=403, detail="Not authorized to manage gaps")

        company_domain = (
            None
            if AuthorizationService.has_permission(
                user, "governance.read", requested_scope="global"
            )
            else user.company_domain
        )
        gap = await self.gov_repo.get_gap(gap_id, company_domain)
        if not gap:
            raise HTTPException(status_code=404, detail="Gap not found")

        gap.dept = (
            await resolve_active_department(self.gov_repo.db, gap.company_domain, dept)
        ).name
        gap.status = "assigned"
        updated_gap = await self.gov_repo.update_gap(gap)

        # Log Audit Trail
        await self.log_audit(
            user_id=user.id, action="assign", target_type="gap", target_id=str(gap.id)
        )

        return updated_gap

    async def dismiss_gap(self, user: User, gap_id: uuid.UUID) -> Gap:
        if not AuthorizationService.has_permission(
            user, "governance.read", requested_scope="company"
        ):
            raise HTTPException(status_code=403, detail="Not authorized to manage gaps")

        company_domain = (
            None
            if AuthorizationService.has_permission(
                user, "governance.read", requested_scope="global"
            )
            else user.company_domain
        )
        gap = await self.gov_repo.get_gap(gap_id, company_domain)
        if not gap:
            raise HTTPException(status_code=404, detail="Gap not found")

        gap.status = "dismissed"
        updated_gap = await self.gov_repo.update_gap(gap)

        # Log Audit Trail
        await self.log_audit(
            user_id=user.id, action="dismiss", target_type="gap", target_id=str(gap.id)
        )

        return updated_gap

    # Dashboard Metrics
    async def get_dashboard_metrics(self, user: User) -> dict:
        if not AuthorizationService.has_permission(
            user, "governance.read", requested_scope="global"
        ):
            raise HTTPException(
                status_code=403,
                detail="Only global administrators can access cross-company metrics",
            )
        return await self.gov_repo.get_health_metrics()

    async def list_audit_logs(
        self,
        user: User,
        limit: int = 100,
        *,
        user_id: uuid.UUID | None = None,
        action: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> Sequence[AuditLog]:
        if not AuthorizationService.has_permission(
            user, "governance.read", requested_scope="global"
        ):
            raise HTTPException(
                status_code=403, detail="Only Admins can view full audit logs"
            )
        return await self.gov_repo.list_audits(
            limit=limit,
            user_id=user_id,
            action=action,
            start_time=start_time,
            end_time=end_time,
        )
