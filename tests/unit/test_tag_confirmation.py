import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.models.article import Article, ArticleTag
from src.models.user import User
from src.models.rbac import Permission, Role, RolePermission


def make_tag_user(permission_key: str) -> User:
    user = User(
        id=uuid.uuid4(),
        email="tagger@acme.test",
        name="Tagger",
        password_hash="hash",
        company_domain="acme.test",
        role="Staff",
        active=True,
    )
    role = Role(name="Tagger", company_domain="acme.test", active=True)
    role.permissions.append(RolePermission(permission=Permission(key=permission_key, name=permission_key), scope="company"))
    user.roles.append(role)
    return user


def make_tag_article(user: User) -> Article:
    return Article(
        id=uuid.uuid4(),
        title="Access policy",
        body_md="Use the approved access policy.",
        dept="Engineering",
        company_domain="acme.test",
        domain="Security",
        type="POLICY",
        sensitivity="public",
        status="published",
        lifecycle_status="active",
        index_status="ready",
        owner_id=user.id,
        tags=[ArticleTag(tag="existing")],
    )
from src.models.user import User


def test_ai_tagging_returns_proposals_without_persisting(monkeypatch):
    from src.api.routers import articles

    user = User(id=uuid.uuid4(), email="owner@acme.test", name="Owner", password_hash="hash", company_domain="acme.test", role="Staff")
    article = Article(
        id=uuid.uuid4(),
        title="Access policy",
        body_md="Use the approved access policy.",
        dept="Engineering",
        company_domain="acme.test",
        domain="Security",
        type="POLICY",
        sensitivity="public",
        status="published",
        lifecycle_status="active",
        owner_id=user.id,
        tags=[ArticleTag(tag="existing")],
    )

    class FakeArticleRepository:
        def __init__(self, _db):
            pass

        async def get_by_id(self, article_id, user=None):
            return article if article_id == article.id else None

        async def sync_tags(self, *_args):
            raise AssertionError("suggestion generation must not persist tags")

    async def fake_complete(*_args, **_kwargs):
        return json.dumps({"articles": [{"id": str(article.id), "tags": ["Access Control"]}]}), 0, "test", "test"

    monkeypatch.setattr(articles, "ArticleRepository", FakeArticleRepository)
    monkeypatch.setattr(articles, "resolve_provider", lambda: SimpleNamespace(model="test-model"))
    monkeypatch.setattr(articles, "complete", fake_complete)

    result = asyncio.run(articles.auto_tag_articles(articles.AutoTagRequest(article_ids=[article.id]), user, object()))

    assert result.updated_count == 0
    assert result.requires_confirmation is True
    assert result.results[0]["current_tags"] == ["existing"]
    assert result.results[0]["proposed_tags"] == ["existing", "access control"]


def test_tag_confirmation_requires_explicit_sets_and_requeues_published_article(monkeypatch):
    from src.api.routers import articles

    user = make_tag_user("article.edit")
    article = make_tag_article(user)
    sync_calls = []
    audit_records = []
    events = []

    class FakeArticleRepository:
        def __init__(self, _db):
            pass

        async def get_by_id(self, article_id, user=None):
            return article if article_id == article.id else None

        async def sync_tags(self, article_id, tags, *, commit=True):
            sync_calls.append((article_id, tags, commit))

    class FakeAuditRepository:
        def __init__(self, _db):
            pass

        async def record(self, *args, **kwargs):
            audit_records.append((args, kwargs))

    class FakeDB:
        commits = 0

        async def commit(self):
            self.commits += 1

    async def publish(event_type, payload):
        events.append((event_type, payload))

    monkeypatch.setattr(articles, "ArticleRepository", FakeArticleRepository)
    monkeypatch.setattr(articles, "AuditRepository", FakeAuditRepository)
    monkeypatch.setattr(articles, "event_bus", SimpleNamespace(publish=publish))

    result = asyncio.run(articles.confirm_article_tags(
        articles.ConfirmTagsRequest(items=[
            articles.ConfirmTagItem(article_id=article.id, tags=[" Access Control ", "access   control", "bad<script>"])
        ]),
        user,
        FakeDB(),
    ))

    assert result == {"confirmed": [{"article_id": str(article.id), "tags": ["access control"]}], "confirmed_count": 1}
    assert sync_calls == [(article.id, ["access control"], False)]
    assert article.index_status == "pending"
    assert audit_records[0][0][1:] == ("tags_confirm", "article", str(article.id))
    assert events == [("ArticleUpdated", {"article_id": str(article.id)})]


def test_tag_endpoints_reject_authenticated_user_without_edit_permission(monkeypatch):
    from src.api.routers import articles

    owner = make_tag_user("article.edit")
    unauthorized = make_tag_user("article.read")
    article = make_tag_article(owner)

    class FakeArticleRepository:
        def __init__(self, _db):
            pass

        async def get_by_id(self, article_id, user=None):
            return article if article_id == article.id else None

    monkeypatch.setattr(articles, "ArticleRepository", FakeArticleRepository)
    monkeypatch.setattr(articles, "resolve_provider", lambda: SimpleNamespace(model="test-model"))

    with pytest.raises(HTTPException) as auto_exc:
        asyncio.run(articles.auto_tag_articles(articles.AutoTagRequest(article_ids=[article.id]), unauthorized, object()))
    assert auto_exc.value.status_code == 403

    with pytest.raises(HTTPException) as confirm_exc:
        asyncio.run(articles.confirm_article_tags(
            articles.ConfirmTagsRequest(items=[articles.ConfirmTagItem(article_id=article.id, tags=["secret"])]),
            unauthorized,
            object(),
        ))
    assert confirm_exc.value.status_code == 403
