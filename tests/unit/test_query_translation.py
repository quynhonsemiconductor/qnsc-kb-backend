"""A Vietnamese question is also searched in English (src/rag/query_translation.py), and each passage keeps the better
of its scores (SearchService.best_score_per_chunk)."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

import src.domain.llm_client as llm_client
from src.core.config import settings
from src.domain import search_service as module
from src.domain.search_service import SearchService, best_score_per_chunk
from src.rag import query_translation
from src.rag.reranker import normalize_query
from tests.unit.test_multi_query_retrieval import (
    _admin_user,
    _article,
    _chunk,
    _FakeChunkRepo,
)


@pytest.fixture(autouse=True)
def _fresh_state():
    query_translation._cache.clear()
    query_translation._cooldown_until = 0.0
    yield
    query_translation._cache.clear()
    query_translation._cooldown_until = 0.0


def _model(monkeypatch, reply, calls=None):
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def fake(messages, **kwargs):
        if calls is not None:
            calls.append((messages, kwargs))
        if isinstance(reply, Exception):
            raise reply
        return reply, 1, "m", "p"

    monkeypatch.setattr(llm_client, "complete", fake)


def _translate(q):
    return asyncio.run(query_translation.translate_query(q))


# -- the translation itself ----------------------------------------------------------------


def test_only_questions_with_vietnamese_letters_are_translated():
    assert query_translation.needs_translation("Địa chỉ base của GPIO là bao nhiêu?")
    assert query_translation.needs_translation("thanh ghi nào xoá cờ ngắt")
    assert not query_translation.needs_translation("What is the base address of GPIO?")
    assert not query_translation.needs_translation("PADOUTSET offset")
    assert not query_translation.needs_translation("")


def test_an_english_question_costs_no_model_call(monkeypatch):
    calls: list = []
    _model(monkeypatch, "x", calls)

    assert _translate("What is the UART base address?") is None
    assert calls == []


def test_a_vietnamese_question_is_translated_with_a_budget_for_a_reasoning_model(
    monkeypatch,
):
    calls: list = []
    _model(monkeypatch, "What is the base address of GPIO?", calls)

    assert _translate("Địa chỉ base của GPIO là bao nhiêu?") == (
        "What is the base address of GPIO?"
    )
    messages, kwargs = calls[0]
    assert "English" in messages[0]["content"]
    assert "exactly as" in messages[0]["content"]
    assert kwargs["max_tokens"] >= 400 and kwargs["thinking"] is False
    assert kwargs["timeout"] == settings.QUERY_TRANSLATION_TIMEOUT_SECONDS


def test_a_question_is_translated_once(monkeypatch):
    calls: list = []
    _model(monkeypatch, "Where is UART0?", calls)

    for _ in range(3):
        assert _translate("UART0 nằm ở địa chỉ nào?") == "Where is UART0?"

    assert len(calls) == 1


def test_a_reply_that_adds_nothing_is_no_translation(monkeypatch):
    _model(monkeypatch, "UART0 nằm ở địa chỉ nào?")
    assert _translate("UART0 nằm ở địa chỉ nào?") is None
    _model(monkeypatch, "   ")
    assert _translate("Còn UART1 thì sao?") is None


def test_only_the_first_line_of_a_chatty_reply_is_used(monkeypatch):
    _model(monkeypatch, '"Where is UART0?"\nNote: translated from Vietnamese.')
    assert _translate("UART0 nằm ở địa chỉ nào?") == "Where is UART0?"


def test_a_failing_model_means_no_translation_and_a_pause_before_the_next_try(
    monkeypatch,
):
    calls: list = []
    _model(monkeypatch, RuntimeError("provider down"), calls)

    assert _translate("UART0 nằm ở địa chỉ nào?") is None
    assert _translate("Còn UART1 thì sao?") is None  # no second timeout paid

    assert len(calls) == 1


def test_without_a_provider_nothing_is_translated(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)
    assert _translate("UART0 nằm ở địa chỉ nào?") is None


# -- scoring ------------------------------------------------------------------------------


def test_each_chunk_keeps_its_best_score_and_the_order_follows_it():
    a, b, c = (SimpleNamespace(id=n) for n in "abc")
    first = [(a, 0.9), (b, 0.5), (c, 0.1)]
    second = [(c, 0.8), (b, 0.6), (a, 0.2)]

    merged = best_score_per_chunk([first, second])

    assert [(x.id, s) for x, s in merged] == [("a", 0.9), ("c", 0.8), ("b", 0.6)]


def test_one_scoring_is_returned_untouched():
    one = [(SimpleNamespace(id="a"), 0.9), (SimpleNamespace(id="b"), 0.1)]
    assert best_score_per_chunk([one]) is one


# -- inside the real search -----------------------------------------------------------------


def _scored_by_query(monkeypatch):
    """A lexical scorer that knows only English: a chunk scores for the query whose text it holds."""

    async def fake_embed(text):
        return [0.1, 0.2, 0.3]

    def fake_rerank(query, chunks, limit=5, identifiers=None, vector_ranks=None):
        scored = [(c, c.scores.get(query, 0.0)) for c in chunks]
        return sorted(scored, key=lambda p: p[1], reverse=True)[:limit]

    monkeypatch.setattr(module, "embed_query", fake_embed)
    monkeypatch.setattr(module, "rerank_chunks_with_scores", fake_rerank)


def _chunk_scoring(article, scores, text="chunk"):
    chunk = _chunk(article, 0.0, text=text)
    chunk.scores = scores
    return chunk


def _search(repo, query):
    service = SearchService(chunk_repo=repo, gov_repo=None, feature_flags=None)
    return asyncio.run(service.search(_admin_user(), query, limit=10))


def test_a_passage_only_the_translation_matches_reaches_the_results(monkeypatch):
    """The English passage scores 0 for the Vietnamese words and is dropped by the context gate
    unless it is also scored against the translation."""
    monkeypatch.setattr(settings, "QUERY_TRANSLATION_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", False)
    _scored_by_query(monkeypatch)
    question = "Địa chỉ base của GPIO là bao nhiêu?"
    _model(monkeypatch, "What is the base address of GPIO?")
    vi_q = normalize_query(question)
    en_q = normalize_query("What is the base address of GPIO?")
    article = _article()
    english_passage = _chunk_scoring(
        article, {vi_q: 0.0, en_q: 0.9}, "GPIO0 0x8000_C000"
    )
    vietnamese_decoy = _chunk_scoring(article, {vi_q: 0.5, en_q: 0.0}, "địa chỉ khác")
    repo = _FakeChunkRepo(
        {vi_q: [vietnamese_decoy], en_q: [english_passage, vietnamese_decoy]}
    )

    results = _search(repo, question)

    assert set(repo.calls) == {vi_q, en_q}
    assert results[0]["chunk_id"] == str(english_passage.id)
    assert {r["chunk_id"] for r in results} == {
        str(english_passage.id),
        str(vietnamese_decoy.id),
    }


def test_a_vietnamese_only_passage_is_still_found_by_the_question_as_asked(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_TRANSLATION_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", False)
    _scored_by_query(monkeypatch)
    question = "Quy trình thanh toán công tác phí thế nào?"
    _model(monkeypatch, "What is the business-travel expense process?")
    vi_q = normalize_query(question)
    en_q = normalize_query("What is the business-travel expense process?")
    article = _article()
    policy = _chunk_scoring(article, {vi_q: 0.8, en_q: 0.0}, "Hồ sơ thanh toán")
    repo = _FakeChunkRepo({vi_q: [policy], en_q: []})

    results = _search(repo, question)

    assert [r["chunk_id"] for r in results] == [str(policy.id)]
    assert results[0]["score"] == pytest.approx(0.8)


def test_disabled_it_searches_exactly_the_one_query_and_makes_no_call(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_TRANSLATION_ENABLED", False)
    _scored_by_query(monkeypatch)
    calls: list = []
    _model(monkeypatch, "never used", calls)
    question = "Địa chỉ base của GPIO là bao nhiêu?"
    vi_q = normalize_query(question)
    article = _article()
    repo = _FakeChunkRepo({vi_q: [_chunk_scoring(article, {vi_q: 0.9})]})

    _search(repo, question)

    assert repo.calls == [vi_q] and calls == []


def test_a_failed_translation_leaves_the_search_as_it_was(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_TRANSLATION_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", False)
    _scored_by_query(monkeypatch)
    _model(monkeypatch, RuntimeError("down"))
    question = "Địa chỉ base của GPIO là bao nhiêu?"
    vi_q = normalize_query(question)
    article = _article()
    repo = _FakeChunkRepo({vi_q: [_chunk_scoring(article, {vi_q: 0.9})]})

    results = _search(repo, question)

    assert repo.calls == [vi_q] and len(results) == 1
