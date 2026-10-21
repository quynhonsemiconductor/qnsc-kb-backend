"""Reviewers must be able to see which approved documents the AI cannot answer from yet.

An approved draft becomes a published Article whose chunks are embedded in the background, so
for a while it exists but is invisible to Ask. Both the list filter and the summary count must
respect the same authorization predicate as every other read: a count that included another
department's documents would reveal that they exist.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from src.api.routers.articles import article_index_summary, list_articles
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import Department, User
from src.repositories.article import ArticleRepository


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return self

    def __iter__(self):
        return iter(self._rows)


class _DB:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _Result(self.rows)


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect())).lower()


def _reader() -> User:
    user = User(
        id=uuid.uuid4(), role="Staff", company_domain="acme.test", dept="Engineering"
    )
    user.departments = [
        Department(
            id=uuid.uuid4(), company_domain="acme.test", name="Engineering", active=True
        )
    ]
    role = Role(name="Reader", company_domain="acme.test")
    role.permissions = [
        RolePermission(
            permission=Permission(key="article.read", name="Read"), scope="company"
        )
    ]
    user.roles = [role]
    return user


def test_counts_are_restricted_to_what_the_caller_may_read() -> None:
    db = _DB([("ready", 3)])
    asyncio.run(ArticleRepository(db).index_status_counts(_reader()))
    sql = _sql(db.statement)
    assert "article_user_permissions" in sql
    assert "articles.company_domain" in sql
    assert "articles.lifecycle_status" in sql
    assert "articles.status = " in sql
    assert "group by articles.index_status" in sql


def test_summary_reports_every_state_and_what_is_not_searchable() -> None:
    db = _DB([("ready", 7), ("pending", 2), ("processing", 1), ("failed", 3)])
    summary = asyncio.run(article_index_summary(_reader(), db))
    assert summary.model_dump() == {
        "pending": 2,
        "processing": 1,
        "ready": 7,
        "failed": 3,
        "total": 13,
        "not_ready": 6,
    }


def test_summary_of_an_empty_corpus_is_all_zero() -> None:
    summary = asyncio.run(article_index_summary(_reader(), _DB([])))
    assert summary.total == 0
    assert summary.not_ready == 0
    assert summary.ready == 0


def test_summary_does_not_count_an_unknown_state_as_a_known_one() -> None:
    summary = asyncio.run(
        article_index_summary(_reader(), _DB([("legacy", 4), ("ready", 1)]))
    )
    assert summary.total == 5
    assert summary.ready == 1
    assert summary.not_ready == 0


def test_list_filters_by_index_state_inside_the_authorization_predicate() -> None:
    db = _DB()
    asyncio.run(
        ArticleRepository(db).list_articles(
            _reader(), status="published", index_status="processing"
        )
    )
    sql = _sql(db.statement)
    assert "articles.index_status = " in sql
    assert "article_user_permissions" in sql


def test_list_without_the_filter_does_not_constrain_index_state() -> None:
    db = _DB()
    asyncio.run(ArticleRepository(db).list_articles(_reader()))
    assert "articles.index_status = " not in _sql(db.statement)


@pytest.mark.parametrize("state", ["pending", "processing", "ready", "failed"])
def test_endpoint_accepts_only_known_states(state: str) -> None:
    from typing import get_args

    annotation = list_articles.__annotations__["index_status"]
    assert state in str(annotation)
    assert "Literal" in str(annotation) or get_args(annotation)


def test_summary_route_is_registered_before_the_id_route() -> None:
    from src.api.routers.articles import router

    paths = [route.path for route in router.routes]
    assert paths.index("/index-summary") < paths.index("/{id}")
