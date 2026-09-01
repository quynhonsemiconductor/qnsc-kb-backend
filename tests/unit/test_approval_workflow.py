import uuid
import asyncio

import pytest
from fastapi import HTTPException

from src.domain.governance import GovernanceService
from src.models.connectors import ExternalDocument
from src.models.governance import PendingDraft
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import Department, User


def reviewer(email: str, company: str = "acme.test") -> User:
    user = User(id=uuid.uuid4(), email=email, name=email, company_domain=company, role="Reviewer", active=True)
    permission = Permission(key="article.review", name="Review")
    role = Role(name="Reviewer", company_domain=company)
    role.permissions.append(RolePermission(permission=permission, scope="company"))
    user.roles.append(role)
    return user


class FakeGovernanceRepository:
    def __init__(self, draft: PendingDraft, external_document: ExternalDocument | None = None):
        self.draft = draft
        class Db:
            def __init__(self, source):
                self.source = source
            def add(self, _item):
                pass
            async def commit(self):
                pass
            async def get(self, model, object_id):
                if self.source and object_id == self.source.id:
                    return self.source
                return None
        self.db = Db(external_document)
        self.audits = []

    async def get_draft(self, draft_id):
        return self.draft if draft_id == self.draft.id else None

    async def update_draft(self, draft):
        self.draft = draft
        return draft

    async def log_audit(self, audit):
        self.audits.append(audit)
        return audit

    async def list_drafts(self, status=None, company_domain=None, dept=None, depts=None, assigned_approver_id=None):
        drafts = [self.draft]
        if assigned_approver_id:
            drafts = [item for item in drafts if item.assigned_approver_id in {None, assigned_approver_id}]
        return drafts


class FakeArticleRepository:
    pass


def test_reviewer_cannot_approve_without_assignment():
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    draft = PendingDraft(id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="upload://procedure.pdf", source_hash="a" * 64, created_by=author.id, status="pending")
    service = GovernanceService(FakeGovernanceRepository(draft), FakeArticleRepository())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.approve_draft(reviewer("reviewer@acme.test"), draft.id))

    assert exc.value.status_code == 403
    assert "assigned approver" in str(exc.value.detail)


def test_reviewer_may_optionally_assign_an_approver():
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    draft = PendingDraft(id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="upload://procedure.pdf", source_hash="a" * 64, created_by=author.id, status="pending")
    service = GovernanceService(FakeGovernanceRepository(draft), FakeArticleRepository())

    assert service._can_assign_approver(reviewer("reviewer@acme.test"), draft)


def test_draft_transition_rejects_invalid_state_change():
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    draft = PendingDraft(id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="upload://procedure.pdf", source_hash="a" * 64, created_by=author.id, status="approved")
    service = GovernanceService(FakeGovernanceRepository(draft), FakeArticleRepository())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service._transition_draft(draft, reviewer("reviewer@acme.test"), "pending"))

    assert exc.value.status_code == 409


def test_author_can_submit_draft_for_assignment():
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    draft = PendingDraft(id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="upload://procedure.pdf", source_hash="a" * 64, created_by=author.id, dept="Engineering", status="draft")
    repo = FakeGovernanceRepository(draft)
    service = GovernanceService(repo, FakeArticleRepository())

    submitted = asyncio.run(service.submit_draft(author, draft.id))

    assert submitted.status == "pending"


def test_reviewer_queue_is_limited_to_member_department_and_assignment():
    review_user = reviewer("reviewer@acme.test")
    review_user.dept = "Engineering"
    review_user.departments = [Department(id=uuid.uuid4(), name="Engineering", company_domain="acme.test", active=True)]
    draft = PendingDraft(
        id=uuid.uuid4(), title="Engineering procedure", company_domain="acme.test", dept="Engineering",
        source_ref="upload://engineering.pdf", source_hash="b" * 64, status="pending",
    )
    service = GovernanceService(FakeGovernanceRepository(draft), FakeArticleRepository())

    visible = asyncio.run(service.list_drafts(review_user, "pending"))

    assert [item.id for item in visible] == [draft.id]

    draft.assigned_approver_id = uuid.uuid4()
    assert asyncio.run(service.list_drafts(review_user, "pending")) == []

    draft.assigned_approver_id = review_user.id
    assert [item.id for item in asyncio.run(service.list_drafts(review_user, "pending"))] == [draft.id]


def test_only_assigned_reviewer_can_reject(monkeypatch):
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    assigned = reviewer("assigned@acme.test")
    other = reviewer("other@acme.test")
    draft = PendingDraft(id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="upload://procedure.pdf", source_hash="a" * 64, created_by=author.id, assigned_approver_id=assigned.id, status="pending")
    service = GovernanceService(FakeGovernanceRepository(draft), FakeArticleRepository())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.reject_draft(other, draft.id, "Not ready"))
    assert exc.value.status_code == 403

    rejected = asyncio.run(service.reject_draft(assigned, draft.id, "Missing required steps"))
    assert rejected.status == "rejected"
    assert rejected.reviewed_by == assigned.id


def test_any_authorized_reviewer_can_restructure(monkeypatch):
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    assigned = reviewer("assigned@acme.test")
    other_reviewer = reviewer("other@acme.test")
    draft = PendingDraft(
        id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="upload://procedure.pdf",
        source_hash="a" * 64, created_by=author.id, assigned_approver_id=assigned.id, status="pending",
        summary="Original extracted text",
    )
    repo = FakeGovernanceRepository(draft)
    service = GovernanceService(repo, FakeArticleRepository())

    async def fake_restructure(title, source_text, enabled=True):
        from src.domain.content_restructure import RestructureResult

        return RestructureResult("# Reading view\n\nOriginal extracted text", "llm", "test-model")

    monkeypatch.setattr("src.domain.governance.restructure_document", fake_restructure)

    result = asyncio.run(service.restructure_draft(other_reviewer, draft.id))

    assert result.restructured_body_md == "# Reading view\n\nOriginal extracted text"
    assert result.restructure_status == "llm"
    assert result.assigned_approver_id == assigned.id


def test_manual_revision_cannot_be_published_as_a_new_article():
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    assigned = reviewer("assigned@acme.test")
    original_article_id = uuid.uuid4()
    draft = PendingDraft(
        id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="manual-update://procedure",
        source_hash="a" * 64, created_by=author.id, assigned_approver_id=assigned.id, status="pending",
        requires_update_confirmation=True,
        content_metadata={
            "submission_kind": "manual_update",
            "suggested_update_article_id": str(original_article_id),
            "sensitivity": "public",
        },
    )
    service = GovernanceService(FakeGovernanceRepository(draft), FakeArticleRepository())

    with pytest.raises(HTTPException, match="must update its original article"):
        asyncio.run(service.approve_draft(assigned, draft.id, treat_as_new=True))


def test_admin_or_ceo_can_self_approve():
    admin = User(id=uuid.uuid4(), email="admin@acme.test", name="Admin", company_domain="acme.test", role="Admin")
    ceo = User(id=uuid.uuid4(), email="ceo@acme.test", name="CEO", company_domain="acme.test", role="CEO")
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    draft = PendingDraft(id=uuid.uuid4(), title="Procedure", company_domain="acme.test", source_ref="upload://procedure.pdf", source_hash="a" * 64, created_by=author.id, status="pending")
    service = GovernanceService(FakeGovernanceRepository(draft), FakeArticleRepository())

    assert service._may_self_approve(admin)
    assert service._may_self_approve(ceo)
    assert not service._may_self_approve(author)


def test_unmapped_external_acl_blocks_approval():
    author = User(id=uuid.uuid4(), email="author@acme.test", name="Author", company_domain="acme.test", role="Staff")
    assigned = reviewer("assigned@acme.test")
    external = ExternalDocument(
        id=uuid.uuid4(), connector_id=uuid.uuid4(), corpus_id="drive-1", external_id="file-1", name="policy.md",
        metadata_json={
            "sharepoint_acl_present": True,
            "mapped_department_ids": [],
            "unmapped_group_ids": ["provider-group"],
            "mapped_source_user_ids": [],
            "unmapped_source_user_ids": [],
            "unmapped_principal_ids": [],
        },
    )
    draft = PendingDraft(
        id=uuid.uuid4(), title="Policy", company_domain="acme.test", source_ref="sharepoint://drive-1/file-1",
        source_hash="a" * 64, created_by=author.id, assigned_approver_id=assigned.id, status="pending",
        external_document_id=external.id,
    )
    service = GovernanceService(FakeGovernanceRepository(draft, external), FakeArticleRepository())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.approve_draft(assigned, draft.id))

    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "external_acl_mapping_required"
    assert exc.value.detail["principals"] == ["group:provider-group"]
    assert draft.status == "pending"
