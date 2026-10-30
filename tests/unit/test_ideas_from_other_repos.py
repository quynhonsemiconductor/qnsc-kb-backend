"""Ideas taken from graphiti, mem0 and RAGFlow: a winner ends the loser's validity, a lesson is
compared with what is stored before it is saved, chunking can be inspected, and the asker can
choose how hard an answer looks.
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from src.domain import lessons, temporal
from src.domain.agent.core import Budget, Tracer
from src.domain.agent.depth import current_depth, llm_calls, retrieval_rounds
from src.domain.agent.plan import AgentPlan
from src.domain.agent.retrieve import agentic_retrieve
from src.domain.chunk_quality import HUGE_CHARS, TINY_CHARS, build_chunk_view
from src.domain.lessons import LessonDecision, lesson_body, parse_decision
from src.domain.temporal import Validity, with_supersession


def _run(coro):
    return asyncio.run(coro)


# --- a winner ends the loser's validity -----------------------------------------------


def _v(issue=None, expiry=None):
    return Validity(
        date.fromisoformat(issue) if issue else None,
        date.fromisoformat(expiry) if expiry else None,
    )


def test_the_losing_article_expires_the_day_before_the_winner_was_issued():
    validity = {"old": _v("2020-01-01"), "new": _v("2025-06-01")}

    out = with_supersession(validity, [(["old", "new"], "new")])

    assert out["old"] == _v("2020-01-01", "2025-05-31")
    assert out["new"] == validity["new"]


def test_an_earlier_expiry_is_kept_and_a_later_one_is_cut_back():
    early = with_supersession(
        {"old": _v("2020-01-01", "2024-01-01"), "new": _v("2025-06-01")},
        [(["old", "new"], "new")],
    )
    late = with_supersession(
        {"old": _v("2020-01-01", "2030-01-01"), "new": _v("2025-06-01")},
        [(["old", "new"], "new")],
    )

    assert early["old"].expiry == date(2024, 1, 1)
    assert late["old"].expiry == date(2025, 5, 31)


def test_without_the_winners_issue_date_nothing_can_be_said():
    validity = {"old": _v("2020-01-01"), "new": _v()}

    assert with_supersession(validity, [(["old", "new"], "new")]) == validity


def test_a_loser_with_no_dates_of_its_own_still_gets_an_end():
    out = with_supersession({"new": _v("2025-06-01")}, [(["old", "new"], "new")])

    assert out["old"] == _v(None, "2025-05-31")


def test_before_the_winner_existed_the_loser_is_still_in_force():
    validity = with_supersession(
        {"old": _v("2020-01-01"), "new": _v("2025-06-01")}, [(["old", "new"], "new")]
    )

    assert temporal.bucket(validity["old"], date(2024, 1, 1)) == temporal.IN_FORCE
    assert temporal.bucket(validity["old"], date(2025, 7, 1)) == temporal.OUT_OF_FORCE


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _SeqDb:
    """Returns each prepared result in turn: decisions first, then article dates."""

    def __init__(self, *results):
        self.results = list(results)

    async def execute(self, _stmt):
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Rows(item)


def _hit(article_id):
    return {"article_id": article_id, "chunk_id": f"c-{article_id}"}


def test_a_decision_puts_the_winner_first_for_a_date_after_it_was_issued():
    db = _SeqDb(
        [(["old", "new"], "new")],
        [
            ("old", {"issue_date": "2020-01-01"}),
            ("new", {"issue_date": "2025-06-01"}),
        ],
    )
    results = [_hit("old"), _hit("new")]

    out = _run(temporal.apply_time_focus(db, results, date(2025, 8, 1), "qnsc.vn"))

    assert [r["article_id"] for r in out] == ["new", "old"]


def test_the_winner_is_looked_up_even_when_retrieval_did_not_return_it():
    db = _SeqDb(
        [(["old", "new"], "new")],
        [
            ("old", {"issue_date": "2020-01-01"}),
            ("new", {"issue_date": "2025-06-01"}),
            ("other", {"issue_date": "2024-01-01"}),
        ],
    )
    results = [_hit("old"), _hit("other")]

    out = _run(temporal.apply_time_focus(db, results, date(2025, 8, 1), "qnsc.vn"))

    assert [r["article_id"] for r in out] == ["other", "old"]


def test_unreadable_decisions_leave_the_order_alone():
    db = _SeqDb(RuntimeError("db down"))
    results = [_hit("old"), _hit("new")]

    assert (
        _run(temporal.apply_time_focus(db, results, date(2025, 8, 1), "qnsc.vn"))
        is results
    )


# --- lessons ---------------------------------------------------------------------------


def test_a_lesson_has_a_fixed_shape():
    assert lesson_body(" Do X. ", None) == "## Lesson\n\nDo X."
    assert lesson_body("Do X.", "  ") == "## Lesson\n\nDo X."
    assert lesson_body("Do X.", "Because Y.") == (
        "## Lesson\n\nDo X.\n\n## Context\n\nBecause Y."
    )


def test_the_models_decision_is_parsed_defensively():
    assert parse_decision('{"decision": "update", "target": 2, "reason": "r"}', 3) == (
        "update",
        2,
        "r",
    )
    assert parse_decision('```json\n{"decision": "add"}\n```', 3)[0] == "add"
    assert parse_decision('{"decision": "skip", "target": 1}', 3)[:2] == ("skip", 1)


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "not json",
        "[]",
        '{"decision": "update"}',  # no target
        '{"decision": "update", "target": 9}',  # no such article
        '{"decision": "delete", "target": 1}',  # not an option
    ],
)
def test_anything_unusable_is_a_plain_add(reply):
    assert parse_decision(reply, 3)[0] == "add"


def test_a_skip_without_a_target_points_at_the_closest_candidate():
    assert parse_decision('{"decision": "skip"}', 3)[:2] == ("skip", 1)


class _Article:
    def __init__(self, title, body):
        self.title, self.body_md = title, body


class _LessonDb:
    async def get(self, _model, article_id):
        return _Article(f"Title {article_id}", f"Body of {article_id}")


def _matches(*scored):
    async def fake(_db, _user, _content):
        return [
            {"article_id": aid, "title": f"Title {aid}", "score": score}
            for aid, score in scored
        ]

    return fake


def test_a_lesson_with_nothing_similar_is_added(monkeypatch):
    monkeypatch.setattr(lessons, "find_similar_documents", _matches())

    assert _run(lessons.decide_lesson(_LessonDb(), None, "T", "B")) == LessonDecision(
        "add"
    )


def test_the_models_update_names_the_existing_article(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(
        lessons, "find_similar_documents", _matches(("a1", 0.5), ("a2", 0.4))
    )
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def fake(messages, **_k):
        assert "EXISTING ARTICLE 2" in messages[1]["content"]
        return (
            '{"decision": "update", "target": 2, "reason": "same subject"}',
            1,
            "m",
            "p",
        )

    monkeypatch.setattr(llm_client, "complete", fake)

    decision = _run(lessons.decide_lesson(_LessonDb(), None, "T", "B"))

    assert decision == LessonDecision("update", "a2", "Title a2", "same subject")


def test_the_models_skip_saves_nothing(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(lessons, "find_similar_documents", _matches(("a1", 0.9)))
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def fake(*_a, **_k):
        return '{"decision": "skip", "target": 1, "reason": "known"}', 1, "m", "p"

    monkeypatch.setattr(llm_client, "complete", fake)

    assert _run(lessons.decide_lesson(_LessonDb(), None, "T", "B")).action == "skip"


def test_without_a_model_a_very_close_match_is_an_update_and_a_loose_one_an_add(
    monkeypatch,
):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)

    monkeypatch.setattr(lessons, "find_similar_documents", _matches(("a1", 0.8)))
    close = _run(lessons.decide_lesson(_LessonDb(), None, "T", "B"))
    monkeypatch.setattr(lessons, "find_similar_documents", _matches(("a1", 0.3)))
    loose = _run(lessons.decide_lesson(_LessonDb(), None, "T", "B"))

    assert close.action == "update" and close.article_id == "a1"
    assert loose.action == "add"


def test_a_failing_model_or_search_never_blocks_saving(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(lessons, "find_similar_documents", _matches(("a1", 0.3)))
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def boom(*_a, **_k):
        raise RuntimeError("timeout")

    monkeypatch.setattr(llm_client, "complete", boom)
    assert _run(lessons.decide_lesson(_LessonDb(), None, "T", "B")).action == "add"

    monkeypatch.setattr(lessons, "find_similar_documents", boom)
    assert _run(lessons.decide_lesson(_LessonDb(), None, "T", "B")).action == "add"


def _submit(monkeypatch, decision):
    from src.api.routers import articles, knowledge

    created = []

    async def fake_decide(*_a, **_k):
        return decision

    async def fake_create(article_in, current_user, db):
        created.append(article_in)
        return {"id": "new-draft"}

    monkeypatch.setattr(lessons, "decide_lesson", fake_decide)
    monkeypatch.setattr(articles, "create_article", fake_create)
    request = knowledge.LessonRequest(
        title="Bus ID", lesson="IDs are 6 bits.", context="After rework.", tags=["chip"]
    )
    result = _run(knowledge.submit_lesson(request, object(), object()))
    return result, created


def test_a_new_lesson_is_submitted_for_review_as_a_tagged_department_article(
    monkeypatch,
):
    result, created = _submit(monkeypatch, LessonDecision("add"))

    assert result["decision"] == "add" and result["result"] == {"id": "new-draft"}
    article = created[0]
    assert (
        article.body_md == "## Lesson\n\nIDs are 6 bits.\n\n## Context\n\nAfter rework."
    )
    assert article.tags == ["chip", lessons.LESSON_TAG]
    assert article.visibility == "department"


def test_an_update_creates_nothing_and_says_which_article_to_update(monkeypatch):
    result, created = _submit(
        monkeypatch, LessonDecision("update", "a1", "Bus IDs", "same subject")
    )

    assert created == []
    assert result["decision"] == "update" and result["article_id"] == "a1"
    assert "update_article" in result["next"] and "a1" in result["next"]


def test_a_known_lesson_creates_nothing(monkeypatch):
    result, created = _submit(monkeypatch, LessonDecision("skip", "a1", "Bus IDs"))

    assert created == [] and result["decision"] == "skip"
    assert "already says this" in result["next"]


# --- chunk view -------------------------------------------------------------------------


def _child(cid, index, text, parent="p1", emb="e2", chunking="c3", page=None):
    return SimpleNamespace(
        id=cid,
        chunk_index=index,
        chunk_text=text,
        parent_chunk_id=parent,
        embedding_version=emb,
        chunking_version=chunking,
        page_number=page,
        heading=None,
    )


def _parent(pid, heading="H"):
    return SimpleNamespace(
        id=pid,
        heading=heading,
        section_ref=None,
        page_number=1,
        chunk_type="text",
        text="parent text",
    )


def _view(children, parents=None):
    return build_chunk_view(parents or [_parent("p1")], children, "e2", "c3")


def test_a_healthy_article_has_no_flags():
    view = _view([_child("c1", 0, "x" * 200), _child("c2", 1, "y" * 200)])

    assert view["summary"]["children"] == 2 and view["summary"]["avg_chars"] == 200
    assert not any(view["summary"]["flags"].values())


def test_tiny_huge_and_duplicate_passages_are_flagged():
    big = "z" * (HUGE_CHARS + 1)
    dupe = "the same sentence repeated here for the check"
    view = _view(
        [
            _child("c1", 0, "tiny"),
            _child("c2", 1, big),
            _child("c3", 2, dupe),
            _child("c4", 3, dupe.upper()),
        ]
    )

    flags = {c["id"]: c["flags"] for c in view["parents"][0]["children"]}
    assert "tiny" in flags["c1"] and len("tiny") < TINY_CHARS
    assert "huge" in flags["c2"]
    assert "duplicate" in flags["c3"] and "duplicate" in flags["c4"]
    assert view["summary"]["flags"]["duplicate"] == 2


def test_an_older_embedding_or_chunking_version_is_flagged():
    view = _view([_child("c1", 0, "x" * 100, emb="e1", chunking="c2")])

    assert set(view["parents"][0]["children"][0]["flags"]) == {
        "stale_embedding",
        "old_chunking",
    }


def test_passages_come_in_document_order_under_their_parents():
    parents = [_parent("p2", "Second"), _parent("p1", "First"), _parent("p3", "Empty")]
    children = [
        _child("c3", 2, "x" * 100, parent="p2"),
        _child("c1", 0, "x" * 100, parent="p1"),
        _child("c2", 1, "y" * 100, parent="p1"),
    ]

    view = _view(children, parents)

    assert [p["heading"] for p in view["parents"]] == ["First", "Second", "Empty"]
    assert [c["index"] for c in view["parents"][0]["children"]] == [0, 1]
    assert view["parents"][2]["children"] == []


def test_an_article_with_no_chunks_gives_an_empty_summary():
    view = build_chunk_view([], [], "e2", "c3")

    assert view["summary"]["children"] == 0 and view["summary"]["avg_chars"] == 0
    assert view["parents"] == []


def test_the_chunks_endpoint_reads_the_article_through_its_authorization(monkeypatch):
    from src.api.routers import articles

    seen = {}

    class _Service:
        def __init__(self, *_a, **_k):
            pass

        async def get_article(self, user, article_id):
            seen["checked"] = (user, article_id)
            return SimpleNamespace(id=article_id)

    class _Db:
        def __init__(self):
            self.queries = 0

        async def execute(self, _stmt):
            self.queries += 1

            class _R:
                def scalars(self):
                    return self

                def all(self):
                    return []

            return _R()

    monkeypatch.setattr(articles, "ArticleService", _Service)
    db = _Db()

    view = _run(articles.get_article_chunks("art-1", "the-user", db))

    assert seen["checked"] == ("the-user", "art-1")
    assert db.queries == 2 and view["summary"]["children"] == 0


# --- depth ------------------------------------------------------------------------------


def test_standard_depth_changes_nothing():
    assert retrieval_rounds(2) == 2 and llm_calls(6) == 6


def test_quick_looks_once_and_thorough_looks_harder():
    token = current_depth.set("quick")
    try:
        assert retrieval_rounds(2) == 0 and llm_calls(6) == 3
        assert llm_calls(2) == 3  # never below what a plan and an answer need
    finally:
        current_depth.reset(token)
    token = current_depth.set("thorough")
    try:
        assert retrieval_rounds(2) == 4 and llm_calls(6) == 10
    finally:
        current_depth.reset(token)


def test_the_ask_request_defaults_to_standard_and_rejects_unknown_depths():
    from pydantic import ValidationError

    from src.api.routers.ai import AskRequest

    assert AskRequest(question="hello?").depth == "standard"
    assert AskRequest(question="hello?", depth="quick").depth == "quick"
    with pytest.raises(ValidationError):
        AskRequest(question="hello?", depth="extreme")


class _Search:
    def __init__(self):
        self.calls = 0
        self.vector_search_degraded = False

    async def search(self, user, query, filters=None, limit=5):
        self.calls += 1
        return []


def _retrieve(depth):
    token = current_depth.set(depth)
    try:
        search = _Search()
        _run(
            agentic_retrieve(
                search,
                SimpleNamespace(company_domain="qnsc.vn"),
                "q",
                AgentPlan(sub_queries=["q"]),
                tracer=Tracer(),
                budget=Budget(0),
            )
        )
        return search.calls
    finally:
        current_depth.reset(token)


def test_a_quick_answer_never_runs_a_retry_round_for_an_uncovered_question():
    assert _retrieve("quick") == 1
    # standard retries once without filters when nothing is found and a reword has no budget
    assert _retrieve("standard") >= _retrieve("quick")
