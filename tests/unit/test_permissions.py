import pytest
import uuid
from src.models.user import User, Department, DepartmentManager
from src.models.article import Article
from src.domain.permissions import PermissionService
from src.domain.rbac import AuthorizationService

def test_can_view_article_rules():
    # Admin can always view
    admin = User(role="Admin")
    engineering = Department(id=uuid.uuid4(), company_domain="local", name="Engineering", active=True)
    audience = Department(id=uuid.uuid4(), company_domain="local", name="Security", active=True)
    restricted_article = Article(company_domain="local", dept="Engineering", departments=[audience], sensitivity="restricted", owner_id=uuid.uuid4())
    assert PermissionService.can_view_article(admin, restricted_article) is True

    # Restricted content requires a shared audience department. Sharing only
    # the primary department name through `Article.dept` is not enough.
    staff = User(role="Staff", company_domain="local", departments=[engineering], id=uuid.uuid4())
    assert PermissionService.can_view_article(staff, restricted_article) is False

    # Staff inside the article's audience department can view it
    staff_with_access = User(role="Staff", company_domain="local", departments=[engineering, audience], id=uuid.uuid4())
    assert PermissionService.can_view_article(staff_with_access, restricted_article) is True


def test_ceo_company_access_is_consistent_with_article_and_chat_scope():
    ceo = User(role="CEO", company_domain="acme.test")
    article = Article(company_domain="acme.test", sensitivity="internal")
    assert PermissionService.can_view_article(ceo, article) is True


def test_department_scope_uses_explicit_ownership_not_a_role_name():
    owner = User(role="Staff", company_domain="acme.test")
    department = Department(company_domain="acme.test", name="Operations", active=True)
    owner.departments.append(department)
    owner.department_ownerships.append(DepartmentManager(department=department, active=True))
    from src.models.rbac import Permission, Role, RolePermission
    role = Role(name="Department content manager", company_domain="acme.test")
    role.permissions.append(RolePermission(permission=Permission(key="article.read", name="Read"), scope="department"))
    owner.roles.append(role)
    article = Article(company_domain="acme.test", dept="Operations", sensitivity="internal")
    assert PermissionService.can_view_article(owner, article) is True


def test_unpublished_articles_are_not_reader_visible():
    department = Department(id=uuid.uuid4(), company_domain="local", name="Engineering", active=True)
    reader = User(role="Staff", company_domain="local", departments=[department], id=uuid.uuid4())
    owner = User(role="Staff", company_domain="local", departments=[department], id=uuid.uuid4())
    for state in ("draft", "pending_review", "archived"):
        article = Article(
            company_domain="local",
            status=state,
            dept="Engineering",
            departments=[department],
            sensitivity="public",
            owner_id=owner.id,
            lifecycle_status="active",
        )
        assert PermissionService.can_view_article(reader, article) is False
    published = Article(company_domain="local", status="published", dept="Engineering", departments=[department], sensitivity="public", lifecycle_status="active")
    assert PermissionService.can_view_article(reader, published) is True


def test_department_membership_blocks_article_from_another_department():
    engineering = Department(company_domain="acme.test", name="Engineering", active=True)
    security = Department(company_domain="acme.test", name="Security", active=True)
    member = User(role="Staff", company_domain="acme.test", departments=[engineering])
    engineering_article = Article(company_domain="acme.test", dept="Engineering", departments=[engineering], sensitivity="public")
    security_article = Article(company_domain="acme.test", dept="Security", departments=[security], sensitivity="public")

    assert PermissionService.can_view_article(member, engineering_article) is True
    assert PermissionService.can_view_article(member, security_article) is False


def test_article_response_hides_non_member_department_labels():
    engineering = Department(company_domain="acme.test", name="Engineering", active=True)
    security = Department(company_domain="acme.test", name="Security", active=True)
    member = User(role="Staff", company_domain="acme.test", departments=[engineering])
    article = Article(company_domain="acme.test", dept="Security", departments=[engineering, security], sensitivity="public")

    assert PermissionService.can_view_article(member, article) is True
    AuthorizationService.restrict_article_metadata(member, article)
    assert article.dept == "Engineering"
    assert [department.name for department in article.departments] == ["Engineering"]
