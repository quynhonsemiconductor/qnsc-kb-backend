"""Publish drafts of departments that do not require a review.

A department with `review_required = false` has been opted out of the review queue by an
Admin/CEO (`Department.auto_publish_by`). This sweep approves its prepared, pending drafts
AS that person, through `GovernanceService.approve_draft` -- the same call a manual approval
makes -- so every check that exists there still applies and nothing is published by a
shortcut written here. A draft those checks refuse (unmapped connector permissions, a
document that may update an existing article, split candidates, ...) simply stays in the
queue for a human, which is the right outcome for exactly the drafts that need judgement.

Never auto-published, whatever the department says: a draft marked restricted/confidential,
or in a department an operator listed as high risk (`compute_draft_risk_tier`). Skipping
review for ordinary documents does not extend to the sensitive ones.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()

NOTE = "Auto-published: department {name} does not require review."


async def run(db: Any, company_domain: str, *, limit: int | None = None) -> dict:
    from sqlalchemy import select

    from src.core.config import settings
    from src.domain.approval_agent import HIGH_RISK, compute_draft_risk_tier
    from src.domain.governance import GovernanceService
    from src.models.governance import AuditLog, PendingDraft
    from src.models.user import Department
    from src.repositories.article import ArticleRepository
    from src.repositories.governance import GovernanceRepository
    from src.repositories.user import UserRepository

    summary: dict[str, Any] = {"approved": 0, "left_for_review": 0, "skipped": 0}
    # Plain values, not ORM rows: approving commits, which expires loaded objects.
    departments = {
        row.name: row.auto_publish_by
        for row in (
            await db.execute(
                select(Department).where(
                    Department.company_domain == company_domain,
                    Department.active.is_(True),
                    Department.review_required.is_(False),
                    Department.auto_publish_by.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    }
    if not departments:
        return summary

    drafts = list(
        (
            await db.execute(
                select(PendingDraft)
                .where(
                    PendingDraft.company_domain == company_domain,
                    PendingDraft.status == "pending",
                    PendingDraft.dept.in_(list(departments)),
                    PendingDraft.restructure_status.notin_(
                        ["queued", "processing", "deferred"]
                    ),
                )
                .order_by(PendingDraft.created_at)
                .limit(limit or settings.APPROVAL_AGENT_BATCH_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    service = GovernanceService(GovernanceRepository(db), ArticleRepository(db))
    users = UserRepository(db)
    for draft in drafts:
        if compute_draft_risk_tier(draft) == HIGH_RISK:
            summary["skipped"] += 1
            continue
        draft_id = draft.id
        department_name = str(draft.dept)
        # Fetched per draft, like the approval agent does for a rule's author: the
        # repository eager-loads the roles the authorization checks read.
        actor = await users.get_by_id(departments[department_name])
        if actor is None or not getattr(actor, "active", True):
            summary["left_for_review"] += 1
            continue
        actor_id = actor.id
        try:
            await service.approve_draft(
                actor, draft_id, review_note=NOTE.format(name=department_name)
            )
        except Exception as exc:  # noqa: BLE001 - one draft must not stop the rest
            # Refused by an existing check (or failed): it stays pending for a person.
            await db.rollback()
            logger.info(
                "Department auto-publish left a draft for review",
                draft_id=str(draft_id),
                department=department_name,
                reason=str(getattr(exc, "detail", exc))[:300],
            )
            summary["left_for_review"] += 1
            continue
        db.add(
            AuditLog(
                user_id=actor_id,
                action="department_auto_publish",
                target_type="draft",
                target_id=str(draft_id),
                outcome="success",
                detail_json={"department": department_name},
            )
        )
        await db.commit()
        summary["approved"] += 1
    return summary
