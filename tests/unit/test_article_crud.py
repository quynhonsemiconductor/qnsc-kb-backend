import asyncio
import uuid

import pytest
from fastapi import HTTPException

from src.domain.articles import ArticleService
from src.domain.events import event_bus
from src.models.article import Article, ArticleVersion
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User


def make_user(permission_key: str) -> User:
    user = User(
        id=uuid.uuid4(),
        email="editor@acme.test",
        name="Editor",
        company_domain="acme.test",
        role="Staff",
        active=True,
    )
    role = Role(name="Editor", company_domain="acme.test", active=True)
    role.permissions.append(
        RolePermission(
            permission=Permission(key=permission_key, name=permission_key),
            scope="company",
        )
    )
    user.roles.append(role)
    return user


def make_article() -> Article:
    return Article(
        id=uuid.uuid4(),
        title="Operations policy",
        body_md="Current policy",
        company_domain="acme.test",
        dept="Engineering",
        domain="Operations",
        type="POLICY",
        sensitivity="public",
        status="published",
        lifecycle_status="active",
        version=2,
    )


class FakeSession:
    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def refresh(self, _obj, attribute_names=None):
        return None


class FakeArticleRepository:
    def __init__(self, article: Article, version: ArticleVersion | None = None):
        self.article = article
        self.version = version
        self.created_versions = []
        self.db = FakeSession()

    async def get_by_id(self, *_args, **_kwargs):
        return self.article

    async def soft_delete(self, _article_id, user=None, **_kwargs):
        self.article.status = "deleted"
        return True

    async def update(self, article, **_kwargs):
        self.article = article
        return article

    async def get_version_by_number(self, *_args, **_kwargs):
        return self.version

    async def create_version(self, version, **_kwargs):
        self.created_versions.append(version)
        return version


class FakeAuditRepository:
    def __init__(self):
        self.records = []

    async def record(self, *args, **_kwargs):
        self.records.append(args)


def test_soft_delete_requires_permission_and_records_audit(monkeypatch):
    article = make_article()
    repository = FakeArticleRepository(article)
    audit = FakeAuditRepository()
    service = ArticleService(repository, object(), audit)
    events = []

    async def publish(event_type, payload):
        events.append((event_type, payload))

    monkeypatch.setattr(event_bus, "publish", publish)

    editor = make_user("article.delete")
    asyncio.run(service.soft_delete_article(editor, article.id))

    assert article.status == "deleted"
    assert audit.records[0][0] == editor.id
    assert audit.records[0][1:] == ("delete", "article", str(article.id))
    assert events == [("ArticleDeleted", {"article_id": str(article.id)})]


def test_soft_delete_rejects_user_without_delete_permission():
    article = make_article()
    repository = FakeArticleRepository(article)
    service = ArticleService(repository, object(), FakeAuditRepository())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.soft_delete_article(make_user("article.read"), article.id))

    assert exc.value.status_code == 403
    assert article.status == "published"


def test_restore_version_creates_a_new_version_and_requeues_indexing():
    article = make_article()
    old_version = ArticleVersion(
        id=uuid.uuid4(),
        article_id=article.id,
        version=1,
        snapshot={
            "title": "Original policy",
            "body_md": "Original body",
            "dept": "Engineering",
            "domain": "Operations",
            "type": "POLICY",
            "sensitivity": "public",
            "language": "en",
        },
        edited_by=uuid.uuid4(),
    )
    repository = FakeArticleRepository(article, old_version)
    audit = FakeAuditRepository()
    service = ArticleService(repository, object(), audit)

    restored = asyncio.run(
        service.restore_version(make_user("article.edit"), article.id, 1)
    )

    assert restored.title == "Original policy"
    assert restored.body_md == "Original body"
    assert restored.version == 3
    assert restored.index_status == "pending"
    assert repository.created_versions[-1].version == 3
    assert audit.records[-1][1:] == ("restore_version", "article", str(article.id))


def test_history_and_single_version_reject_user_without_article_access():
    article = make_article()
    service = ArticleService(
        FakeArticleRepository(
            article, ArticleVersion(article_id=article.id, version=1, snapshot={})
        ),
        object(),
    )
    unauthorized = make_user("article.read")

    with pytest.raises(HTTPException) as history_error:
        asyncio.run(service.get_history(unauthorized, article.id))
    with pytest.raises(HTTPException) as version_error:
        asyncio.run(service.get_version(unauthorized, article.id, 1))

    assert history_error.value.status_code == 403
    assert version_error.value.status_code == 403
