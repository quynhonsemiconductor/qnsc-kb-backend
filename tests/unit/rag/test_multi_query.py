"""Agentic multi-query retrieval must never break ordinary search, only improve it.

Same failure-mode contract as contextual_header.py/auto_tagging.py: every degrade path
(no provider, a failed call, a malformed or repeated reply) returns the empty/None value
the caller (search_service.py) already treats as "fall back to the single-query search
this feature does not change."
"""
from __future__ import annotations

import asyncio

import pytest

from src.rag.multi_query import generate_followup_query, generate_subqueries


@pytest.fixture
def llm(monkeypatch):
    from src.domain import llm_client

    state: dict = {"reply": "phân cụm dữ liệu\nunsupervised learning", "raises": None, "provider": object()}

    async def fake_complete(messages, **kwargs):
        state["messages"] = messages
        state["kwargs"] = kwargs
        if state["raises"] is not None:
            raise state["raises"]
        return state["reply"], 10, "fake-model", "fake"

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: state["provider"])
    return state


# --- generate_subqueries -----------------------------------------------------------


def test_returns_parsed_lines_as_distinct_queries(llm):
    result = asyncio.run(generate_subqueries("học không giám sát"))
    assert result == ["phân cụm dữ liệu", "unsupervised learning"]


def test_no_provider_configured_returns_empty_without_calling(llm):
    llm["provider"] = None
    assert asyncio.run(generate_subqueries("query")) == []
    assert "messages" not in llm


def test_provider_failure_returns_empty_not_raises(llm):
    llm["raises"] = RuntimeError("boom")
    assert asyncio.run(generate_subqueries("query")) == []


def test_blank_question_returns_empty_without_calling(llm):
    assert asyncio.run(generate_subqueries("   ")) == []
    assert "messages" not in llm


def test_blank_and_duplicate_lines_are_dropped(llm):
    llm["reply"] = "\n".join(["", "  ", "học không giám sát", "phân cụm dữ liệu", "phân cụm dữ liệu"])
    result = asyncio.run(generate_subqueries("Học không giám sát"))
    # The reply's own restatement of the question is folded/diacritic-matched against
    # the ORIGINAL question and dropped, and the repeated line collapses to one.
    assert result == ["phân cụm dữ liệu"]


def test_capped_at_max_subqueries(llm, monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "MULTI_QUERY_MAX_SUBQUERIES", 1)
    llm["reply"] = "one\ntwo\nthree"
    assert asyncio.run(generate_subqueries("query")) == ["one"]


def test_max_subqueries_zero_returns_empty_without_calling(llm, monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "MULTI_QUERY_MAX_SUBQUERIES", 0)
    assert asyncio.run(generate_subqueries("query")) == []
    assert "messages" not in llm


def test_bullet_and_numbering_markers_are_stripped(llm):
    llm["reply"] = "- clustering\n* dimensionality reduction"
    assert asyncio.run(generate_subqueries("unsupervised learning")) == [
        "clustering",
        "dimensionality reduction",
    ]


def test_thinking_is_explicitly_disabled(llm):
    """Same latency-budget fix as the main answer path (llm_client.py): this call is on
    the live search request, so hidden reasoning tokens must not be left to the
    provider's own default."""
    asyncio.run(generate_subqueries("query"))
    assert llm["kwargs"]["thinking"] is False


# --- generate_followup_query -------------------------------------------------------


def test_returns_the_proposed_query(llm):
    llm["reply"] = "phân cụm k-means"
    result = asyncio.run(
        generate_followup_query("học không giám sát", ["học không giám sát"], ["Chương 10 - Học có giám sát"])
    )
    assert result == "phân cụm k-means"


def test_none_response_returns_none(llm):
    llm["reply"] = "NONE"
    assert asyncio.run(generate_followup_query("q", ["q"], [])) is None


def test_a_query_repeating_the_question_is_rejected(llm):
    llm["reply"] = "Học Không Giám Sát"
    assert asyncio.run(generate_followup_query("học không giám sát", [], [])) is None


def test_a_query_repeating_an_already_tried_query_is_rejected(llm):
    llm["reply"] = "phân cụm dữ liệu"
    result = asyncio.run(
        generate_followup_query("question", ["phân cụm dữ liệu"], [])
    )
    assert result is None


def test_no_provider_configured_returns_none_without_calling(llm):
    llm["provider"] = None
    assert asyncio.run(generate_followup_query("q", [], [])) is None
    assert "messages" not in llm


def test_provider_failure_returns_none_not_raises(llm):
    llm["raises"] = RuntimeError("boom")
    assert asyncio.run(generate_followup_query("q", [], [])) is None


def test_blank_question_returns_none_without_calling(llm):
    assert asyncio.run(generate_followup_query("  ", [], [])) is None
    assert "messages" not in llm


def test_prompt_includes_question_tried_queries_and_weak_titles(llm):
    asyncio.run(
        generate_followup_query(
            "học không giám sát", ["phân cụm"], ["Chương 10 - Học có giám sát"]
        )
    )
    user_message = llm["messages"][1]["content"]
    assert "học không giám sát" in user_message
    assert "phân cụm" in user_message
    assert "Chương 10 - Học có giám sát" in user_message


def test_thinking_is_explicitly_disabled_for_followup(llm):
    asyncio.run(generate_followup_query("q", [], []))
    assert llm["kwargs"]["thinking"] is False
