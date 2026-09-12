"""Contextual chunk headers must never break indexing, only improve it when they land.

Mirrors the failure-mode contract auto_tagging.py/entity_extraction.py already have:
every degrade path (no provider, a failed call) returns "" rather than raising, because
this runs inline in indexing and a broken header must not turn into a broken index.
"""
from __future__ import annotations

import asyncio

import pytest

from src.rag.contextual_header import apply_header, generate_section_context


@pytest.fixture
def llm(monkeypatch):
    from src.domain import llm_client

    state: dict = {"reply": "This document covers the leave policy.", "raises": None, "provider": object()}

    async def fake_complete(messages, **kwargs):
        state["messages"] = messages
        if state["raises"] is not None:
            raise state["raises"]
        return state["reply"], 10, "fake-model", "fake"

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: state["provider"])
    return state


def test_returns_the_cleaned_header_from_a_well_formed_reply(llm):
    result = asyncio.run(generate_section_context("Employee Handbook", "Leave Policy", "content"))
    assert result == "This document covers the leave policy."


def test_no_provider_configured_returns_empty_without_calling(llm):
    llm["provider"] = None
    assert asyncio.run(generate_section_context("Doc", "Section", "content")) == ""


def test_provider_failure_returns_empty_not_raises(llm):
    llm["raises"] = RuntimeError("boom")
    assert asyncio.run(generate_section_context("Doc", "Section", "content")) == ""


def test_header_is_capped_to_max_length(llm):
    llm["reply"] = "x" * 1000
    result = asyncio.run(generate_section_context("Doc", "Section", "content"))
    assert len(result) == 400


def test_prompt_includes_title_heading_and_excerpt(llm):
    asyncio.run(generate_section_context("Employee Handbook", "Leave Policy", "Submit forms early."))
    user_message = llm["messages"][1]["content"]
    assert "Employee Handbook" in user_message
    assert "Leave Policy" in user_message
    assert "Submit forms early." in user_message


def test_missing_heading_falls_back_to_a_placeholder(llm):
    asyncio.run(generate_section_context("Doc", None, "content"))
    assert "(untitled section)" in llm["messages"][1]["content"]


# --- apply_header: embedding-input-only join, never touches stored text ---------------


def test_apply_header_prepends_with_a_blank_line_separator():
    assert apply_header("Context.", "Chunk text.") == "Context.\n\nChunk text."


def test_apply_header_with_no_header_returns_the_chunk_text_unchanged():
    assert apply_header("", "Chunk text.") == "Chunk text."
