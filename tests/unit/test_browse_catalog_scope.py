"""/knowledge/catalog must keep the department column switchable.

The endpoint scopes topics and the visible total by the selected department, but
the department list itself used the same scoped filters — pick HR and every other
department vanished, with no way back. The department scope is the one filter the
department column must NOT apply.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from src.api.routers.knowledge import _browse_filters
from src.models.article import Article
from src.models.user import User
from src.repositories.article import ArticleRepository


def _compiled(filters) -> str:
    statement = select(Article.id).where(*filters)
    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def _staff_user() -> User:
    return User(
        id=uuid.uuid4(), role="Staff", company_domain="acme.test", dept="Engineering"
    )


def test_the_department_column_filters_carry_no_department_scope():
    repo = ArticleRepository(None)
    user = _staff_user()

    department_filters = _browse_filters(repo, user, dept=None, query="handbook")
    sql = _compiled(department_filters)

    assert "HR" not in sql
    assert "handbook" in sql  # the search still narrows the counts


def test_the_topic_and_total_scope_keeps_the_department():
    repo = ArticleRepository(None)
    user = _staff_user()

    scoped = _browse_filters(repo, user, dept="HR", query="handbook")
    sql = _compiled(scoped)

    assert "HR" in sql
    assert "handbook" in sql
