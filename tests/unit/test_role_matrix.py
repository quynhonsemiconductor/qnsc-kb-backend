import uuid

import pytest

from src.domain.rbac import AuthorizationService
from src.models.user import User


ROLE_EXPECTATIONS = {
    "Admin": {
        "article.read": "global",
        "article.review": "global",
        "article.publish": "global",
        "user.manage": "global",
        "role.manage": "global",
        "permission.manage": "global",
        "connector.manage": "global",
        "governance.read": "global",
        "ai.ask": "global",
    },
    "CEO": {
        "article.read": "company",
        "article.review": "company",
        "article.publish": "company",
        "user.manage": "company",
        "connector.manage": "company",
        "governance.read": "company",
        "ai.ask": "company",
    },
    "Reviewer": {
        "article.read": "company",
        "article.review": "company",
        "governance.read": "company",
        "ai.ask": "company",
    },
    "Staff": {
        "article.read": "company",
        "article.create": "own",
        "article.edit": "own",
        "ai.ask": "company",
    },
}


@pytest.mark.parametrize("role,permissions", ROLE_EXPECTATIONS.items())
def test_seeded_role_scope_matrix(role, permissions):
    user = User(role=role, company_domain="acme.test", dept="Engineering")
    for key, scope in permissions.items():
        assert AuthorizationService.has_permission(user, key, requested_scope=scope)

    for key in {"user.manage", "role.manage", "permission.manage", "connector.manage", "governance.read"} - permissions.keys():
        assert not AuthorizationService.has_permission(user, key, requested_scope="global")


def test_reviewer_approval_scope_requires_assignment_but_admin_can_override():
    from src.domain.governance import GovernanceService
    from src.models.governance import PendingDraft

    reviewer = User(id=uuid.uuid4(), role="Reviewer", company_domain="acme.test")
    admin = User(id=uuid.uuid4(), role="Admin", company_domain="acme.test")
    draft = PendingDraft(company_domain="acme.test", status="pending", created_by=uuid.uuid4())

    service = GovernanceService(object(), object())
    assert not service._can_approve_draft(reviewer, draft)
    assert service._can_approve_draft(admin, draft)
