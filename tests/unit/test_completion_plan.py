import uuid

from sqlalchemy import inspect
from sqlalchemy.orm.attributes import set_committed_value

from src.domain.permissions import PermissionService
from src.domain.rbac import AuthorizationService
from src.models.article import Article
from src.models.user import AccessGroup, Department, User


def test_metadata_restriction_is_display_only_and_does_not_dirty_relationships():
    hr = Department(id=uuid.uuid4(), company_domain="local", name="HR", active=True)
    it = Department(id=uuid.uuid4(), company_domain="local", name="IT", active=True)
    user = User(id=uuid.uuid4(), role="Staff", company_domain="local", departments=[hr])
    article = Article(
        id=uuid.uuid4(), company_domain="local", dept="HR", departments=[hr, it],
        sensitivity="internal", status="published", lifecycle_status="active",
    )
    # Mimic a row loaded from the database: the original scalar value is
    # committed before the display-only redaction runs.
    set_committed_value(article, "dept", "HR")

    AuthorizationService.restrict_article_metadata(user, article)

    assert [item.name for item in article.departments] == ["HR"]
    assert not inspect(article).attrs.departments.history.has_changes()
    assert not inspect(article).attrs.dept.history.has_changes()


def test_access_group_allows_cross_department_read_without_org_membership():
    audit = AccessGroup(id=uuid.uuid4(), name="Audit-IT", company_domain="local", bitmask_position=7)
    hr = Department(id=uuid.uuid4(), company_domain="local", name="HR", active=True)
    user = User(id=uuid.uuid4(), role="Staff", company_domain="local", departments=[hr], groups=[audit])
    article = Article(
        id=uuid.uuid4(), company_domain="local", dept="IT", departments=[],
        access_groups=[audit], sensitivity="internal", status="published", lifecycle_status="active",
    )

    assert PermissionService.can_view_article(user, article) is True
