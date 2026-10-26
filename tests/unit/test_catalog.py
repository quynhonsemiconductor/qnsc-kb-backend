"""Answering "how many documents do we have" by counting, not by retrieving.

Pure helpers are tested directly. The SQL is checked by compiling it, because the unit venv
has no Postgres: what matters here is that the authorization predicate is part of every
count. And `AIService.ask` runs end to end with a provider that fails the test if it is
called, because the contract of this path is that it costs no model call.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql

from src.core.config import settings
from src.domain.catalog import (
    OTHER_KEY,
    CatalogDocument,
    CatalogFilter,
    CatalogResult,
    catalog_artifacts,
    catalog_conditions,
    escape_markdown,
    file_format,
    filter_for_question,
    format_catalog_answer,
    parse_period,
    summarise_sources,
    top_groups,
)
from src.models.article import Article
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User
from src.domain.ai_service import AIService
from tests.unit.test_agent import FakeSearch, _llm, _Repo, _hit, _provider, _user, run

NOW = datetime(2026, 10, 28, 15, 30)  # a Wednesday


# ── formats and tallies ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("filename", "mime", "expected"),
    [
        ("Quy trinh.PDF", None, "pdf"),
        ("a.docx", "application/pdf", "docx"),  # the extension is what the uploader saw
        ("noext", "application/pdf", "pdf"),
        (None, "text/csv; charset=utf-8", "csv"),
        ("scan.JPEG", None, "jpg"),
        ("scan.tiff", None, "tif"),
        ("archive.tar.gz", None, "gz"),
        ("weird.", None, ""),
        (None, None, ""),
        ("noext", "application/x-unknown", ""),
    ],
)
def test_a_source_file_is_named_by_its_extension_then_its_mime_type(
    filename, mime, expected
):
    assert file_format(filename, mime) == expected


def test_formats_and_pages_are_tallied_and_a_missing_page_count_is_zero():
    formats, pages = summarise_sources(
        [
            ("a.pdf", None, 10),
            ("b.pdf", None, None),
            ("c.docx", None, 3),
            (None, None, -4),
        ]
    )
    assert dict(formats) == {"pdf": 2, "docx": 1, "": 1}
    assert pages == 13


def test_groups_beyond_the_limit_are_folded_into_one_other_row_that_keeps_the_total():
    groups = [("a", 10), ("b", 8), ("c", 5), ("d", 3), ("e", 1)]

    kept, cut = top_groups(groups, 3)

    assert cut is True
    assert kept == [("a", 10), ("b", 8), (OTHER_KEY, 9)]
    assert sum(count for _key, count in kept) == sum(count for _key, count in groups)


def test_nothing_is_folded_when_the_groups_fit():
    kept, cut = top_groups([("b", 2), ("a", 2), ("c", 0)], 5)

    assert (kept, cut) == ([("a", 2), ("b", 2)], False)


def test_an_unknown_format_is_not_listed_twice_when_the_groups_are_folded():
    kept, _cut = top_groups(
        [("pdf", 9), (OTHER_KEY, 4), ("docx", 3), ("xls", 2), ("csv", 1)], 3
    )

    assert [key for key, _ in kept].count(OTHER_KEY) == 1
    assert sum(count for _key, count in kept) == 19


# ── periods ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("folded", "since", "period"),
    [
        ("bao nhieu tai lieu hom nay ", datetime(2026, 10, 28), "today"),
        ("tai lieu nhap tuan nay ", datetime(2026, 10, 26), "week"),  # Monday
        ("tai lieu nhap thang nay ", datetime(2026, 10, 1), "month"),
        ("tai lieu nhap nam nay ", datetime(2026, 1, 1), "year"),
        ("tai lieu 7 ngay qua ", datetime(2026, 10, 22), "days:7"),
        ("documents added in the last 30 days ", datetime(2026, 9, 29), "days:30"),
        ("how many documents added this month ", datetime(2026, 10, 1), "month"),
    ],
)
def test_a_period_in_the_wording_becomes_a_start_date(folded, since, period):
    assert parse_period(folded, NOW) == (since, None, period)


def test_no_period_means_the_whole_collection_not_a_default_window():
    assert parse_period("bao nhieu tai lieu ", NOW) == (None, None, "")


def test_an_absurd_number_of_days_is_ignored():
    assert parse_period("tai lieu 99999 ngay qua ", NOW) == (None, None, "")


def test_the_filter_takes_type_and_department_from_the_plan_and_the_period_from_the_words():
    flt = filter_for_question(
        "Có bao nhiêu SOP nhập tháng này?", ["SOP"], ["Kỹ thuật"], NOW
    )

    assert flt.doc_types == ["SOP"]
    assert flt.departments == ["Kỹ thuật"]
    assert flt.since == datetime(2026, 10, 1)
    assert flt.is_narrowed is True
    assert CatalogFilter().is_narrowed is False


# ── titles are text, never markup ─────────────────────────────────────────────────────────


def test_a_title_cannot_inject_markup_or_a_link_into_the_answer():
    hostile = "[click](http://evil.test) **bold** | <b>x</b>" + chr(10) + "next"

    assert escape_markdown(hostile) == (
        r"\[click\]\(http://evil.test\) \*\*bold\*\* \| \<b\>x\</b\> next"
    )


# ── SQL: the authorization predicate is part of every count ───────────────────────────────


def _reader(scope: str = "company") -> User:
    user = _user()
    role = user.roles[0]
    role.permissions.append(
        RolePermission(
            permission=Permission(key="article.read", name="Read"), scope=scope
        )
    )
    user.dept = "Kỹ thuật"
    return user


def _sql(conditions) -> tuple[str, dict]:
    compiled = (
        select(func.count(Article.id))
        .where(*conditions)
        .compile(dialect=postgresql.dialect())
    )
    return str(compiled), compiled.params


def test_a_user_who_may_not_read_articles_counts_nothing():
    sql, _params = _sql(catalog_conditions(_user(), CatalogFilter()))

    assert "false" in sql.lower()


def test_every_count_carries_the_visibility_rules_and_only_counts_published_documents():
    sql, params = _sql(catalog_conditions(_reader(), CatalogFilter()))

    assert "lifecycle_status" in sql
    assert "published" in params.values()
    assert "deleted" in params.values()  # the same status guard every article read uses
    assert "company_domain" in sql  # a tenant boundary, not just a department one


def test_the_filters_narrow_the_count_without_replacing_authorization():
    flt = CatalogFilter(
        doc_types=["SOP"], departments=["Kỹ thuật"], since=datetime(2026, 10, 1)
    )

    sql, params = _sql(catalog_conditions(_reader(), flt))

    assert "lifecycle_status" in sql and "company_domain" in sql
    assert ["SOP"] in params.values() or any("SOP" in str(v) for v in params.values())
    assert datetime(2026, 10, 1) in params.values()


# ── wording and tables ────────────────────────────────────────────────────────────────────


def _result(**overrides) -> CatalogResult:
    base = dict(
        total=42,
        by_type=[("SOP", 30), ("POLICY", 12)],
        by_department=[("Kỹ thuật", 25), (OTHER_KEY, 17)],
        by_language=[("vi", 40), ("en", 2)],
        by_format=[("pdf", 35), ("docx", 5), (OTHER_KEY, 2)],
        by_month=[("2026-10", 5), ("2026-09", 8)],
        pages=1234,
        evidence={"table": 79, "figure": 23, "page_visual": 27},
        recent=[
            CatalogDocument(
                str(uuid.uuid4()), "Quy trình [A]", "SOP", "Kỹ thuật", NOW, 12
            )
        ],
        generated_at=NOW,
    )
    base.update(overrides)
    return CatalogResult(**base)


def test_the_vietnamese_answer_states_the_count_and_every_breakdown():
    text = format_catalog_answer(_result(), "vi")

    assert "**42** tài liệu" in text
    assert "SOP 30" in text and "POLICY 12" in text
    assert "Kỹ thuật 25" in text and "Khác 17" in text
    assert "pdf 35" in text
    assert "1234 trang" in text and "79 bảng" in text and "23 hình" in text
    assert "2026-10 5" in text
    assert "2026-10-28 15:30" in text
    assert OTHER_KEY not in text


def test_the_english_answer_uses_english_wording():
    text = format_catalog_answer(_result(), "en")

    assert "**42** documents" in text
    assert "By type" in text and "Other 17" in text
    assert "1234 pages" in text


def test_the_scope_is_stated_when_the_count_is_narrowed():
    flt = CatalogFilter(doc_types=["SOP"], departments=["Kỹ thuật"], period="month")

    text = format_catalog_answer(_result(filter=flt), "vi")

    assert "loại SOP" in text and "của phòng ban Kỹ thuật" in text
    assert "nhập trong tháng này" in text


def test_an_empty_collection_says_so_and_has_no_tables():
    empty = CatalogResult(total=0, generated_at=NOW)

    assert "chưa có tài liệu nào" in format_catalog_answer(empty, "vi")
    assert "no documents" in format_catalog_answer(empty, "en")
    assert catalog_artifacts(empty, "vi") == []


def test_a_partial_count_is_flagged():
    assert "Khác" in format_catalog_answer(_result(partial=True), "vi")
    assert "folded into" in format_catalog_answer(_result(partial=True), "en")


def test_the_answer_cannot_be_hijacked_by_a_hostile_title_or_department():
    result = _result(
        by_department=[("**x** [y](http://evil.test)", 1)],
        recent=[CatalogDocument("id", "# heading", "SOP", "d", NOW, 1)],
    )

    text = format_catalog_answer(result, "vi")

    assert "[y](http://evil.test)" not in text
    assert "\\[y\\]\\(http://evil.test\\)" in text


def test_the_artifacts_are_a_statistics_table_and_a_linked_document_table():
    result = _result()

    summary, recent = catalog_artifacts(result, "vi")

    assert summary["type"] == "table" and recent["type"] == "table"
    assert summary["columns"] == ["Nhóm", "Giá trị", "Số lượng"]
    assert ["Loại", "SOP", 30] in summary["rows"]
    assert ["Phòng ban", "Khác", 17] in summary["rows"]
    assert ["Đã trích", "bảng", 79] in summary["rows"]
    assert ["Tổng số trang", "", 1234] in summary["rows"]
    assert summary["link_column"] is None and summary["row_links"] is None

    assert recent["link_column"] == 0
    assert len(recent["rows"]) == len(recent["row_links"]) == 1
    assert recent["row_links"][0] == {"article_id": result.recent[0].article_id}
    assert recent["rows"][0][0] == "Quy trình [A]"  # the table shows the real title
    assert recent["rows"][0][3] == "2026-10-28"
    assert recent["total_rows"] == 1 and recent["truncated"] is False


def test_a_table_is_cut_at_the_bound_and_says_so():
    docs = [
        CatalogDocument(str(uuid.uuid4()), f"d{i}", "SOP", "k", NOW, 1)
        for i in range(250)
    ]

    _summary, recent = catalog_artifacts(_result(recent=docs), "vi")

    assert len(recent["rows"]) == len(recent["row_links"]) == 200
    assert recent["total_rows"] == 250 and recent["truncated"] is True


# ── end to end through AIService.ask ──────────────────────────────────────────────────────


class _DbRepo(_Repo):
    def __init__(self):
        super().__init__()
        self.rolled_back = False
        self.db = SimpleNamespace(rollback=self._rollback)

    async def _rollback(self):
        self.rolled_back = True


def _service(search):
    repo = _DbRepo()
    return AIService(repo, search, object()), repo


def _fail_if_called(monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("a catalog question must not call a model")

    monkeypatch.setattr("src.domain.ai_service.complete", boom)
    monkeypatch.setattr("src.domain.agent.plan.complete", boom)


def test_a_catalog_question_is_counted_with_no_model_call_and_no_search(monkeypatch):
    _fail_if_called(monkeypatch)
    seen = {}

    async def fake_build(db, user, flt, **_k):
        seen["filter"] = flt
        return _result()

    monkeypatch.setattr("src.domain.ai_service.build_catalog", fake_build)
    search = FakeSearch({}, [_hit("x")])
    service, repo = _service(search)
    streamed: list[str] = []

    async def on_token(piece):
        streamed.append(piece)

    events = []

    async def sink(event):
        events.append(event)

    payload = run(
        service.ask(
            _user(),
            "Thống kê toàn bộ tài liệu hiện có",
            on_token=on_token,
            on_event=sink,
        )
    )

    assert search.calls == []
    assert "**42** tài liệu" in payload["answer_grounded"]
    assert payload["citations"] == [] and payload["refused"] is False
    assert payload["retrieval_version"] == "catalog"
    assert [item["type"] for item in payload["artifacts"]] == ["table", "table"]
    assert "".join(streamed) == payload["answer"]
    assert repo.logged[0].llm_model == "catalog" and repo.logged[0].tokens_used == 0
    assert repo.cached == []  # the collection changes whenever a document is published
    steps = {(e["step"], e["status"]) for e in events if e["type"] == "step"}
    assert ("understand", "skipped") in steps and ("catalog", "done") in steps
    assert payload["agent_plan"]["intent"] == "catalog"
    assert not seen["filter"].is_narrowed


def test_the_type_and_period_in_the_question_narrow_the_count(monkeypatch):
    _fail_if_called(monkeypatch)
    seen = {}

    async def fake_build(db, user, flt, **_k):
        seen["filter"] = flt
        return _result(filter=flt)

    monkeypatch.setattr("src.domain.ai_service.build_catalog", fake_build)
    service, _repo = _service(FakeSearch({}))

    payload = run(service.ask(_user(), "Có bao nhiêu SOP nhập tháng này?"))

    assert seen["filter"].doc_types == ["SOP"]
    assert seen["filter"].period == "month"
    assert "loại SOP" in payload["answer_grounded"]


def test_a_failing_count_falls_back_to_retrieval_instead_of_an_error(monkeypatch):
    prompts = _provider(monkeypatch, "<<<GROUNDED>>>\nCó tài liệu [C1].")
    _llm(monkeypatch, "plan", ["{}"])

    async def broken(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr("src.domain.ai_service.build_catalog", broken)
    search = FakeSearch({}, [_hit("Có tài liệu.")])
    service, repo = _service(search)

    payload = run(service.ask(_user(), "Thống kê toàn bộ tài liệu hiện có"))

    assert repo.rolled_back is True
    assert search.calls and prompts
    assert payload["retrieval_version"] != "catalog"
    assert any(
        step["step"] == "catalog" and step["status"] == "skipped"
        for step in payload["agent_trace"]
    )


def test_the_routing_can_be_switched_off_and_the_question_is_retrieved_as_before(
    monkeypatch,
):
    monkeypatch.setattr(settings, "AGENT_CORPUS_ROUTING_ENABLED", False)
    _provider(monkeypatch, "<<<GROUNDED>>>\nCó [C1].")
    plan_calls = _llm(monkeypatch, "plan", ["{}"])
    called = []

    async def never(*_a, **_k):
        called.append(1)

    monkeypatch.setattr("src.domain.ai_service.build_catalog", never)
    search = FakeSearch({}, [_hit("Có.")])
    service, _repo = _service(search)

    run(service.ask(_user(), "Thống kê toàn bộ tài liệu hiện có"))

    assert called == [] and len(plan_calls) == 1 and search.calls


def test_a_question_that_only_mentions_documents_is_still_retrieved(monkeypatch):
    _provider(monkeypatch, "<<<GROUNDED>>>\nBa mươi ngày [C1].")
    called = []

    async def never(*_a, **_k):
        called.append(1)

    monkeypatch.setattr("src.domain.ai_service.build_catalog", never)
    search = FakeSearch({}, [_hit("Ba mươi ngày.")])
    service, _repo = _service(search)

    run(service.ask(_user(), "Thời hạn nộp hồ sơ là bao lâu?"))

    assert called == [] and search.calls


def test_the_planner_choosing_catalog_for_an_odd_phrasing_is_counted_too(monkeypatch):
    _fail_if_called_after_plan = _llm(
        monkeypatch,
        "plan",
        [
            '{"intent": "catalog", "sub_queries": ["kho có gì"], "doc_types": ["POLICY"]}'
        ],
    )

    async def boom(*_a, **_k):
        raise AssertionError("no answer-model call for a catalog question")

    monkeypatch.setattr("src.domain.ai_service.complete", boom)
    seen = {}

    async def fake_build(db, user, flt, **_k):
        seen["filter"] = flt
        return _result()

    monkeypatch.setattr("src.domain.ai_service.build_catalog", fake_build)
    service, _repo = _service(FakeSearch({}))

    payload = run(
        service.ask(
            _user(),
            "Cho mình biết trong kho tri thức này đang chứa những gì nhé, kể cả chính sách?",
        )
    )

    assert len(_fail_if_called_after_plan) == 1  # the planner, once
    assert seen["filter"].doc_types == ["POLICY"]
    assert payload["retrieval_version"] == "catalog"
