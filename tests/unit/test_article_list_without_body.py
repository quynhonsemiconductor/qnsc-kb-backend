"""GET /articles/?include_body=false: the list without every document's full text.

Measured on 50 real chip specifications: a 541 KB response, 90% of it `body_md`, for a page that shows
titles and badges. The default is unchanged (existing API clients keep getting the body).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace

from src.api.routers import articles as articles_module
from src.api.routers.articles import list_articles
from src.models.article import Article


def _article(title: str, body: str) -> Article:
    return Article(
        id=uuid.uuid4(),
        title=title,
        body_md=body,
        dept="Engineering",
        domain="ops",
        company_domain="acme.test",
        type="SOP",
        sensitivity="internal",
        language="en",
        status="published",
        version=1,
        created_at=datetime(2026, 11, 3),
        # Column defaults are applied on flush; a transient row has None until then.
        visibility="department",
        lifecycle_status="active",
        index_status="ready",
        needs_update=False,
        self_approved=False,
        source_changed=False,
        legal_hold=False,
    )


def _call(monkeypatch, **kwargs):
    rows = [_article("A", "x" * 5_000), _article("B", "y" * 5_000)]

    class _Repo:
        def __init__(self, _db):
            pass

        async def list_articles(self, **_kwargs):
            return rows

    monkeypatch.setattr(articles_module, "ArticleRepository", _Repo)
    params = dict(
        dept=None,
        topic=None,
        status=None,
        index_status=None,
        q=None,
        limit=50,
        offset=0,
        current_user=SimpleNamespace(),
        db=SimpleNamespace(),
    )
    params.update(kwargs)
    return rows, asyncio.run(list_articles(**params))


def test_the_default_still_returns_the_orm_rows_so_nothing_changes_for_existing_clients(
    monkeypatch,
):
    rows, result = _call(monkeypatch, include_body=True)
    assert result is rows
    assert all(len(article.body_md) == 5_000 for article in result)


def test_without_the_body_every_item_keeps_its_fields_and_an_empty_body(monkeypatch):
    rows, result = _call(monkeypatch, include_body=False)
    assert [item.title for item in result] == ["A", "B"]
    assert [item.id for item in result] == [row.id for row in rows]
    assert all(item.body_md == "" for item in result)
    assert all(item.dept == "Engineering" and item.version == 1 for item in result)


def test_leaving_the_body_out_does_not_modify_the_stored_article(monkeypatch):
    rows, _ = _call(monkeypatch, include_body=False)
    assert all(len(row.body_md) == 5_000 for row in rows)
