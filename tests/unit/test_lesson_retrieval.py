"""record_lesson compares a lesson with the PASSAGES the assistant would find, not only with
whole articles by text overlap (src/domain/lessons.py)."""

import asyncio

import src.domain.llm_client as llm_client
from src.domain import lessons
from src.domain.lessons import LessonDecision


def _run(coro):
    return asyncio.run(coro)


class _Article:
    def __init__(self, title, body):
        self.title, self.body_md = title, body


class _Db:
    async def get(self, _model, article_id):
        return _Article(f"Title {article_id}", f"START of {article_id} " + "x" * 5000)


def _hit(article_id, text, score=0.8, title=None):
    return {
        "article_id": article_id,
        "title": title or f"Title {article_id}",
        "chunk_text": text,
        "score": score,
    }


def _retrieval(monkeypatch, hits=None, fail=False):
    """Make the assistant's search return `hits` (or raise)."""
    from src.domain.search_service import SearchService

    async def fake_search(self, user, query, filters=None, limit=5):
        if fail:
            raise RuntimeError("search down")
        return hits or []

    monkeypatch.setattr(SearchService, "search", fake_search)


def _lexical(monkeypatch, *scored):
    async def fake(_db, _user, _content):
        return [{"article_id": a, "title": f"Title {a}", "score": s} for a, s in scored]

    monkeypatch.setattr(lessons, "find_similar_documents", fake)


def _model(monkeypatch, reply, seen=None):
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def fake(messages, **_k):
        if seen is not None:
            seen.append(messages[1]["content"])
        return reply, 1, "m", "p"

    monkeypatch.setattr(llm_client, "complete", fake)


def test_the_model_is_shown_the_passage_that_covers_the_lesson_not_the_start_of_the_article(
    monkeypatch,
):
    _retrieval(
        monkeypatch, [_hit("spec", "PADOUTSET is at offset 0x10, write-1-to-set.")]
    )
    _lexical(monkeypatch)
    seen: list[str] = []
    _model(monkeypatch, '{"decision": "skip", "target": 1, "reason": "known"}', seen)

    decision = _run(lessons.decide_lesson(_Db(), None, "PADOUTSET", "It is at 0x10."))

    assert decision == LessonDecision("skip", "spec", "Title spec", "known")
    assert "PADOUTSET is at offset 0x10" in seen[0]
    assert "START of spec" not in seen[0]


def test_a_lesson_about_a_documented_register_is_an_update_even_when_text_overlap_is_nil(
    monkeypatch,
):
    """A short lesson against a long specification scores ~0 on whole-document text
    similarity, so before this it was always added as a new article."""
    _retrieval(monkeypatch, [_hit("spec", "UART0 base 0x8002_0000, APB_M8.")])
    _lexical(monkeypatch)  # nothing above the text-similarity threshold
    _model(
        monkeypatch, '{"decision": "update", "target": 1, "reason": "same register"}'
    )

    decision = _run(lessons.decide_lesson(_Db(), None, "UART0 base", "UART0 is at ..."))

    assert decision.action == "update" and decision.article_id == "spec"


def test_one_article_with_several_passages_is_one_candidate_with_the_passages_together(
    monkeypatch,
):
    _retrieval(
        monkeypatch,
        [
            _hit("a", "first passage"),
            _hit("b", "other article"),
            _hit("a", "second passage"),
        ],
    )
    _lexical(monkeypatch)
    seen: list[str] = []
    _model(monkeypatch, '{"decision": "add"}', seen)

    _run(lessons.decide_lesson(_Db(), None, "T", "B"))

    assert seen[0].count("EXISTING ARTICLE") == 2
    assert "first passage" in seen[0] and "second passage" in seen[0]
    assert seen[0].index("first passage") < seen[0].index("EXISTING ARTICLE 2")


def test_a_pending_draft_found_only_by_text_is_still_a_candidate(monkeypatch):
    _retrieval(monkeypatch, [])  # drafts are not searchable until published
    _lexical(monkeypatch, ("draft", 0.7))
    _model(monkeypatch, '{"decision": "skip", "target": 1, "reason": "already queued"}')

    decision = _run(lessons.decide_lesson(_Db(), None, "T", "B"))

    assert decision.action == "skip" and decision.article_id == "draft"


def test_at_most_three_candidates_and_retrieval_comes_first(monkeypatch):
    _retrieval(monkeypatch, [_hit("r1", "p1"), _hit("r2", "p2")])
    _lexical(monkeypatch, ("l1", 0.9), ("l2", 0.8), ("r1", 0.7))
    seen: list[str] = []
    _model(monkeypatch, '{"decision": "add"}', seen)

    _run(lessons.decide_lesson(_Db(), None, "T", "B"))

    assert seen[0].count("EXISTING ARTICLE") == 3
    assert (
        seen[0].index("Title r1")
        < seen[0].index("Title r2")
        < seen[0].index("Title l1")
    )
    assert "Title l2" not in seen[0]


def test_without_a_model_retrieval_alone_never_makes_an_update(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)
    _retrieval(monkeypatch, [_hit("spec", "text", score=1.0)])
    _lexical(monkeypatch)

    assert _run(lessons.decide_lesson(_Db(), None, "T", "B")) == LessonDecision("add")


def test_without_a_model_a_very_close_text_match_is_still_an_update(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)
    _retrieval(monkeypatch, [_hit("spec", "text", score=1.0)])
    _lexical(monkeypatch, ("twin", 0.9))

    decision = _run(lessons.decide_lesson(_Db(), None, "T", "B"))

    assert decision.action == "update" and decision.article_id == "twin"


def test_a_failing_retrieval_falls_back_to_text_similarity(monkeypatch):
    _retrieval(monkeypatch, fail=True)
    _lexical(monkeypatch, ("a1", 0.5))
    _model(monkeypatch, '{"decision": "update", "target": 1, "reason": "r"}')

    decision = _run(lessons.decide_lesson(_Db(), None, "T", "B"))

    assert decision == LessonDecision("update", "a1", "Title a1", "r")


def test_nothing_found_anywhere_is_an_add(monkeypatch):
    _retrieval(monkeypatch, [])
    _lexical(monkeypatch)

    assert _run(lessons.decide_lesson(_Db(), None, "T", "B")) == LessonDecision("add")


def test_checking_a_lesson_does_not_log_a_knowledge_gap(monkeypatch):
    """The search may find nothing, which for a reader is a gap worth reporting; for a lesson
    being filed it is the normal case and must not pollute the gap report."""
    from src.domain.search_service import SearchService
    from src.repositories import governance

    logged = []

    class _Gov:
        def __init__(self, db):
            self.db = db

        async def log_gap(self, **kwargs):
            logged.append(kwargs)

    async def search_finding_nothing(self, user, query, filters=None, limit=5):
        await self._record_gap(user, query)
        return []

    monkeypatch.setattr(governance, "GovernanceRepository", _Gov)
    monkeypatch.setattr(SearchService, "search", search_finding_nothing)
    _lexical(monkeypatch)

    _run(lessons.decide_lesson(_Db(), None, "T", "B"))

    assert logged == []
    # ... while an ordinary SearchService still records one (so the test can fail).
    plain = SearchService(object(), _Gov(object()))
    _run(plain._record_gap(type("U", (), {"company_domain": "d", "dept": "x"})(), "q"))
    assert len(logged) == 1
