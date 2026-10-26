"""Which figures and tables an answer shows for the pages it cites.

The selection is pure, so it is pinned directly. What a fake cannot prove — that the real
query applies the visibility predicate — is checked by reading the compiled SQL.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from src.domain import page_evidence
from src.models.user import User
from src.domain.page_evidence import attach_page_evidence, pick_page_evidence

ARTICLE = str(uuid.uuid4())
OTHER = str(uuid.uuid4())


def _row(kind: str, page: int, label: str, article: str = ARTICLE, data=None):
    return {
        "id": uuid.uuid4(),
        "article_id": article,
        "kind": kind,
        "page_number": page,
        "label": label,
        "structured_data": data,
    }


def _cite(page: int | None, article: str = ARTICLE, **extra):
    return {"article_id": article, "page_number": page, **extra}


def _pick(citations, rows, **kw):
    return pick_page_evidence(
        citations, rows, per_citation=kw.pop("per", 3), total=kw.pop("total", 8), **kw
    )


def test_a_cited_page_shows_its_figure_with_a_preview_url():
    figure = _row("figure", 55, "Figure 1 on page 55")
    chosen = _pick([_cite(55)], [figure])
    item = chosen[0][0]
    assert item["evidence_id"] == str(figure["id"])
    assert (
        item["preview_url"]
        == f"/api/v1/articles/{ARTICLE}/evidence/{figure['id']}/preview"
    )
    assert item["table"] is None


def test_a_table_carries_its_bounded_rows():
    table = _row(
        "table", 29, "Table 1", data={"columns": ["a", "b"], "rows": [["1", "2"]] * 3}
    )
    item = _pick([_cite(29)], [table])[0][0]
    assert item["table"]["columns"] == ["a", "b"]
    assert item["table"]["total_rows"] == 3


def test_only_the_cited_page_and_article_count():
    rows = [
        _row("figure", 56, "other page"),
        _row("figure", 55, "other article", article=OTHER),
    ]
    assert _pick([_cite(55)], rows) == {}


def test_a_citation_without_a_page_or_that_is_already_evidence_adds_nothing():
    rows = [_row("figure", 55, "f")]
    assert _pick([_cite(None)], rows) == {}
    assert _pick([_cite(55, evidence_id=str(uuid.uuid4()))], rows) == {}
    # A retrieved result carries its evidence nested rather than flat.
    assert _pick([_cite(55, evidence={"id": str(uuid.uuid4())})], rows) == {}


def test_the_same_evidence_is_shown_once_at_the_first_citation_reaching_it():
    rows = [_row("figure", 55, "f")]
    chosen = _pick([_cite(55), _cite(55)], rows)
    assert list(chosen) == [0]


def test_unique_off_lists_the_evidence_for_every_item_on_the_page():
    rows = [_row("figure", 55, "f")]
    assert list(_pick([_cite(55), _cite(55)], rows, unique=False)) == [0, 1]


def test_evidence_the_answer_already_shows_as_a_citation_is_not_repeated():
    shown = _row("figure", 55, "shown")
    extra = _row("table", 55, "extra")
    chosen = _pick([_cite(55), _cite(55, evidence_id=str(shown["id"]))], [shown, extra])
    assert [item["label"] for item in chosen[0]] == ["extra"]


def test_a_whole_page_image_is_dropped_when_the_page_has_a_figure_or_table():
    rows = [
        _row("page_visual", 55, "Visual evidence on page 55"),
        _row("figure", 55, "Figure 1"),
    ]
    assert [i["kind"] for i in _pick([_cite(55)], rows)[0]] == ["figure"]


def test_a_whole_page_image_is_kept_for_a_page_with_nothing_else():
    rows = [_row("page_visual", 2, "Visual evidence on page 2")]
    assert [i["kind"] for i in _pick([_cite(2)], rows)[0]] == ["page_visual"]


def test_one_claim_and_one_answer_are_bounded():
    rows = [_row("figure", 55, f"Figure {n}") for n in range(6)] + [
        _row("figure", 56, "p56")
    ]
    chosen = _pick([_cite(55), _cite(56)], rows, per=3, total=4)
    assert len(chosen[0]) == 3
    assert len(chosen[1]) == 1
    assert sum(len(v) for v in chosen.values()) == 4


def test_figures_and_tables_on_a_page_are_listed_in_label_order():
    rows = [_row("figure", 5, "B"), _row("table", 5, "Z"), _row("figure", 5, "A")]
    assert [i["label"] for i in _pick([_cite(5)], rows)[0]] == ["A", "B", "Z"]


def _user() -> User:
    user = User(
        id=uuid.uuid4(),
        email="reader@qnsc.vn",
        name="Reader",
        password_hash="x",
        company_domain="qnsc.vn",
        role="Staff",
        active=True,
    )
    user.roles = []
    user.departments = []
    user.department_ownerships = []
    return user


class _DB:
    def __init__(self, rows):
        self.rows = rows
        self.sql: list[str] = []

    async def execute(self, statement, *a, **k):
        self.sql.append(
            " ".join(
                str(statement.compile(dialect=postgresql.dialect())).lower().split()
            )
        )
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.rows))


def _evidence(row):
    return SimpleNamespace(**{k: v for k, v in row.items()})


def test_attaching_applies_the_visibility_predicate_and_the_page_filter():
    db = _DB([_evidence(_row("figure", 55, "Figure 1"))])
    citations = [_cite(55)]
    asyncio.run(
        attach_page_evidence(
            db,
            _user(),
            citations,
        )
    )
    sql = db.sql[0]
    assert "document_evidence.page_number in" in sql
    assert "articles.status =" in sql
    assert "articles.index_status =" in sql
    assert (
        "articles.lifecycle_status =" in sql
    )  # from the shared authorization predicate
    assert citations[0]["page_evidence"][0]["label"] == "Figure 1"


def test_nothing_is_queried_when_disabled_or_when_no_citation_has_a_page(monkeypatch):
    db = _DB([])
    user = _user()
    asyncio.run(attach_page_evidence(db, user, [_cite(None)]))
    assert db.sql == []
    monkeypatch.setattr(page_evidence.settings, "ANSWER_PAGE_EVIDENCE_ENABLED", False)
    asyncio.run(attach_page_evidence(db, user, [_cite(55)]))
    assert db.sql == []
