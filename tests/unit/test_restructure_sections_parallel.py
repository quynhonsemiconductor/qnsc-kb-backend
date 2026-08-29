"""Restructuring must section ordinary documents and format them concurrently.

A 26,633-character upload took 304 s. It was ONE LLM call regenerating the whole
document, so wall clock is bound by output tokens and grows with length — and it cleared
RESTRUCTURE_TIMEOUT_SECONDS of 300 by a margin too thin to rely on. Past that edge the
result is not an error but a silent fallback to unformatted text.

Sectioning already existed, but only above RESTRUCTURE_MAX_CHARS (120,000), which no real
document reached. And the one path that did split ran the sections SEQUENTIALLY — an
`await` inside a list comprehension — so the documents most in need of the split paid the
sum of their parts.

Two properties matter and are easy to get wrong together: the sections must actually run
at the same time, and they must come back in DOCUMENT order regardless of which finished
first. Reassembling a document out of order would produce fluent, plausible, wrong text.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.config import settings
from src.domain import content_restructure as cr


@pytest.fixture
def fake_llm(monkeypatch):
    """Record concurrency and let sections finish out of order."""
    state = {"active": 0, "peak": 0, "seen": []}

    async def _fake(title, section, model, department_descriptions=None):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        state["seen"].append(section[:12])
        # Later sections finish sooner, so completion order differs from argument order.
        await asyncio.sleep(0.05 / (len(state["seen"])))
        state["active"] -= 1
        return cr._result(section, f"[{section[:12]}]", "llm", model)

    monkeypatch.setattr(cr, "_restructure_single_document", _fake)
    return state


def _document(sections: int, section_chars: int) -> str:
    """Paragraph-separated text that _split_oversized_source will cut cleanly."""
    para = ("noi dung tai lieu " * (section_chars // 18)).strip()
    return "\n\n".join(f"{chr(ord('A') + i)} {para}" for i in range(sections))


def test_sections_run_concurrently(fake_llm, monkeypatch):
    """The regression: an await inside a list comprehension is sequential."""
    monkeypatch.setattr(settings, "RESTRUCTURE_SECTION_CHARS", 2000)
    monkeypatch.setattr(settings, "RESTRUCTURE_MAX_CONCURRENCY", 4)

    asyncio.run(
        cr._restructure_sections("doc", ["a" * 100] * 4, "m", None)
    )

    assert fake_llm["peak"] > 1, "sections were formatted one after another"


def test_concurrency_is_bounded(fake_llm, monkeypatch):
    """The ceiling is the provider's rate limit; a burst is refused, not queued."""
    monkeypatch.setattr(settings, "RESTRUCTURE_MAX_CONCURRENCY", 2)

    asyncio.run(cr._restructure_sections("doc", ["a" * 100] * 8, "m", None))

    assert fake_llm["peak"] <= 2


def test_results_come_back_in_document_order(fake_llm):
    """gather preserves argument order, not completion order. Out-of-order reassembly
    would read fluently and say the wrong thing."""
    sections = [f"{letter} section body" for letter in "ABCDE"]

    results = asyncio.run(cr._restructure_sections("doc", sections, "m", None))

    assert [r.body_md for r in results] == [f"[{s[:12]}]" for s in sections]


def test_an_ordinary_document_is_now_sectioned(fake_llm, monkeypatch):
    """Previously only documents over RESTRUCTURE_MAX_CHARS were split, so nothing real
    ever was."""
    monkeypatch.setattr(settings, "RESTRUCTURE_SECTION_CHARS", 2000)
    monkeypatch.setattr(cr, "resolve_provider", lambda *_a, **_k: _Provider())

    text = _document(sections=4, section_chars=2000)
    assert len(text) < settings.RESTRUCTURE_MAX_CHARS

    asyncio.run(cr.restructure_document("doc", text, enabled=True))

    assert len(fake_llm["seen"]) > 1, "a 4-section document went through as one call"


def test_a_short_document_still_takes_the_single_call_path(fake_llm, monkeypatch):
    """Splitting a short document would add seams for no gain."""
    monkeypatch.setattr(settings, "RESTRUCTURE_SECTION_CHARS", 8000)
    monkeypatch.setattr(cr, "resolve_provider", lambda *_a, **_k: _Provider())

    asyncio.run(cr.restructure_document("doc", "ngan gon " * 20, enabled=True))

    assert len(fake_llm["seen"]) == 1


class _Provider:
    name = "glm"
    model = "glm-4.5-flash"
