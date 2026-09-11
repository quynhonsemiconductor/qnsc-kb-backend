"""Automatic tag suggestion must never be the reason ingestion breaks.

Every failure mode here (no provider, a failed call, an unparseable reply, a malformed
tag) has to degrade to an empty list rather than raise -- this runs inline in the same
background pass that formats a freshly uploaded document, and a broken suggestion must
never turn into a broken upload.
"""
from __future__ import annotations

import asyncio

import pytest

from src.domain.auto_tagging import suggest_tags_for_document


@pytest.fixture
def llm(monkeypatch):
    from src.domain import llm_client

    state: dict = {"reply": '{"tags": ["database", "backup"]}', "raises": None, "provider": object()}

    async def fake_complete(messages, **kwargs):
        state["messages"] = messages
        if state["raises"] is not None:
            raise state["raises"]
        return state["reply"], 10, "fake-model", "fake"

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: state["provider"])
    return state


def _suggest(llm, title="Doc", body="content", doc_type=""):
    return asyncio.run(suggest_tags_for_document(title, body, doc_type))


def test_returns_cleaned_tags_from_a_well_formed_reply(llm):
    assert _suggest(llm) == ["database", "backup"]


def test_no_provider_configured_returns_empty_without_calling(llm):
    llm["provider"] = None
    assert _suggest(llm) == []


def test_provider_failure_returns_empty_not_raises(llm):
    llm["raises"] = RuntimeError("provider is down")
    assert _suggest(llm) == []


@pytest.mark.parametrize("reply", ["not json", "", "[1,2,3]", '"just a string"', '{"tags": "not-a-list"}'])
def test_unparseable_reply_returns_empty(llm, reply):
    llm["reply"] = reply
    assert _suggest(llm) == []


def test_code_fence_is_stripped_before_parsing(llm):
    llm["reply"] = '```json\n{"tags": ["fire-safety"]}\n```'
    assert _suggest(llm) == ["fire-safety"]


def test_tags_are_lowercased_and_whitespace_normalized(llm):
    llm["reply"] = '{"tags": ["  Database   Backup  "]}'
    assert _suggest(llm) == ["database backup"]


def test_invalid_characters_drop_the_tag_not_the_whole_reply(llm):
    llm["reply"] = '{"tags": ["valid-tag", "bad/tag!", "also_valid"]}'
    assert _suggest(llm) == ["valid-tag", "also_valid"]


def test_duplicate_tags_after_cleaning_are_deduplicated(llm):
    llm["reply"] = '{"tags": ["Database", "database", "DATABASE"]}'
    assert _suggest(llm) == ["database"]


def test_overlong_tag_is_dropped(llm):
    llm["reply"] = '{"tags": ["' + ("x" * 51) + '", "short"]}'
    assert _suggest(llm) == ["short"]


def test_result_is_capped_at_max_tags(llm):
    llm["reply"] = '{"tags": [' + ", ".join(f'"tag{i}"' for i in range(20)) + ']}'
    result = _suggest(llm)
    assert len(result) == 8
    assert result == [f"tag{i}" for i in range(8)]


def test_non_dict_json_payload_returns_empty(llm):
    llm["reply"] = "[1, 2, 3]"
    assert _suggest(llm) == []


def test_missing_tags_key_returns_empty(llm):
    llm["reply"] = "{}"
    assert _suggest(llm) == []


# --- catalogue governance: mirrors auto_tag_articles's own rule in articles.py --------


def test_no_catalogue_argument_suggests_freely(llm):
    """Default (no catalogue passed) is unchanged: existing callers keep working."""
    llm["reply"] = '{"tags": ["anything", "goes"]}'
    assert asyncio.run(suggest_tags_for_document("Doc", "content")) == ["anything", "goes"]


def test_empty_catalogue_suggests_nothing(llm):
    """An empty (not None) catalogue means the tenant has no approved vocabulary yet --
    suggest nothing rather than seed it from unreviewed AI output."""
    llm["reply"] = '{"tags": ["database", "backup"]}'
    assert asyncio.run(suggest_tags_for_document("Doc", "content", catalogue=set())) == []


def test_catalogue_keeps_only_approved_tags(llm):
    llm["reply"] = '{"tags": ["database", "backup", "unapproved-tag"]}'
    result = asyncio.run(suggest_tags_for_document("Doc", "content", catalogue={"database", "backup"}))
    assert result == ["database", "backup"]


def test_catalogue_match_is_accent_insensitive(llm):
    llm["reply"] = '{"tags": ["an toàn"]}'
    result = asyncio.run(suggest_tags_for_document("Doc", "content", catalogue={"an toan"}))
    assert result == ["an toàn"]
