import uuid
import pytest
from fastapi import HTTPException
from sqlalchemy import inspect

from src.domain.rbac import AuthorizationService, DEFAULT_ROLE_PERMISSIONS
from src.api.routers.auth import _apply_department_ownership, _ensure_owned_departments_are_memberships, _validate_role_assignment_authority
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User, Department, DepartmentManager
from src.models.article import Article


def make_user(role_name: str, permission_key: str, scope: str = "company", **attrs) -> User:
    user = User(role="Staff", company_domain="acme.test", **attrs)
    permission = Permission(key=permission_key, name=permission_key)
    role = Role(name=role_name, company_domain="acme.test")
    role.permissions.append(RolePermission(permission=permission, scope=scope))
    user.roles.append(role)
    return user


def test_multiple_roles_union_permissions():
    user = make_user("Reader", "article.read", dept="Ops")
    department = Department(id=uuid.uuid4(), company_domain="acme.test", name="Ops", active=True)
    user.department_ownerships.append(DepartmentManager(department=department, user=user, active=True))
    second_permission = Permission(key="article.publish", name="Publish")
    second_role = Role(name="Publisher", company_domain="acme.test")
    second_role.permissions.append(RolePermission(permission=second_permission, scope="department"))
    user.roles.append(second_role)
    assert AuthorizationService.has_permission(user, "article.read")
    assert AuthorizationService.has_permission(user, "article.publish", Article(company_domain="acme.test", dept="Ops"), "department")


def test_department_scope_requires_explicit_ownership():
    user = make_user("Department content manager", "article.edit", scope="department", dept="Ops")
    article = Article(company_domain="acme.test", dept="Ops")
    assert not AuthorizationService.has_permission(user, "article.edit", article, "department")


def test_scope_is_checked_against_resource():
    user = make_user("Editor", "article.edit", "own")
    own = Article(company_domain="acme.test", owner_id=user.id)
    other = Article(company_domain="acme.test", owner_id=uuid.uuid4())
    assert AuthorizationService.has_permission(user, "article.edit", own, "own")
    assert not AuthorizationService.has_permission(user, "article.edit", other, "own")


def test_restrict_metadata_does_not_mark_article_relationships_dirty():
    user = User(role="Staff", company_domain="acme.test", dept="Ops")
    visible = Department(id=uuid.uuid4(), company_domain="acme.test", name="Ops", active=True)
    hidden = Department(id=uuid.uuid4(), company_domain="acme.test", name="Security", active=True)
    user.departments = [visible]
    article = Article(company_domain="acme.test", dept="Ops", departments=[visible, hidden])

    AuthorizationService.restrict_article_metadata(user, article)

    assert [department.name for department in article.departments] == ["Ops"]
    assert not inspect(article).attrs.departments.history.has_changes()


def test_global_admin_bypass():
    admin = User(role="Admin", company_domain="admin.test")
    assert AuthorizationService.has_permission(admin, "permission.manage", requested_scope="global")


def test_authorization_fingerprint_changes_when_group_changes():
    user = make_user("Reader", "article.read", dept="Ops")
    first = AuthorizationService.authorization_fingerprint(user)
    from src.models.user import AccessGroup
    user.groups.append(AccessGroup(name="Security", bitmask_position=4))
    second = AuthorizationService.authorization_fingerprint(user)
    assert first != second


def test_company_manager_cannot_delegate_permissions_they_do_not_hold():
    manager = make_user("Company manager", "user.manage")
    role = Role(name="Permission manager", company_domain="acme.test")
    role.permissions.append(RolePermission(permission=Permission(key="permission.manage", name="Permission manager"), scope="company"))

    with pytest.raises(HTTPException, match="broader than your own"):
        _validate_role_assignment_authority(manager, [role])


def test_company_manager_can_delegate_an_equal_company_permission():
    manager = make_user("Company manager", "user.manage")
    role = Role(name="User manager", company_domain="acme.test")
    role.permissions.append(RolePermission(permission=Permission(key="user.manage", name="User manager"), scope="company"))

    _validate_role_assignment_authority(manager, [role])


def test_inactive_role_permissions_are_not_effective():
    user = User(role="Staff", company_domain="acme.test")
    permission = Permission(key="article.edit", name="Edit")
    role = Role(name="Former editor", company_domain="acme.test", active=False)
    role.permissions.append(RolePermission(permission=permission, scope="company"))
    user.roles.append(role)

    assert AuthorizationService.get_effective_permissions(user) == []
    assert not AuthorizationService.has_permission(user, "article.edit", requested_scope="company")


def test_department_ownership_is_not_a_role():
    assert "Department Owner" not in DEFAULT_ROLE_PERMISSIONS


def test_owned_department_must_also_be_a_membership():
    membership = Department(id=uuid.uuid4(), company_domain="acme.test", name="Engineering", active=True)
    other = Department(id=uuid.uuid4(), company_domain="acme.test", name="Security", active=True)

    with pytest.raises(HTTPException, match="member of every department"):
        _ensure_owned_departments_are_memberships([membership], [other])


def test_ownership_sync_reuses_existing_assignment_and_deactivates_removed_rows():
    department = Department(id=uuid.uuid4(), company_domain="acme.test", name="Engineering", active=True)
    user = User(role="Staff", company_domain="acme.test")
    reviewer_role = Role(name="Reviewer", company_domain="acme.test")
    user.roles = [reviewer_role]
    existing = DepartmentManager(department=department, user=user, active=True)
    user.department_ownerships = [existing]

    _apply_department_ownership(user, [department])
    assert user.department_ownerships == [existing]
    assert existing.active is True

    _apply_department_ownership(user, [])
    assert user.department_ownerships == [existing]
    assert existing.active is False
    assert [role.name for role in user.roles] == ["Reviewer"]
