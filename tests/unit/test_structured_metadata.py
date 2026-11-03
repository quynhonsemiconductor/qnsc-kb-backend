"""Parsing discipline for document-identity extraction: never store an unusable value.

The two things worth pinning: a date the model could not put in YYYY-MM-DD must be
dropped rather than stored as a string that merely looks like a date, and every one of
the four keys must always be present in the result (as a clean string or explicit None)
so a caller never has to guard against a missing key.
"""
from __future__ import annotations

import asyncio

import pytest

from src.domain.structured_metadata import (
    StructuredMetadataUnavailable,
    extract_structured_metadata,
)


@pytest.fixture
def llm(monkeypatch):
    from src.domain import llm_client

    state: dict = {"reply": "{}", "raises": None}

    async def fake_complete(messages, **kwargs):
        if state["raises"] is not None:
            raise state["raises"]
        return state["reply"], 12, "fake-model", "fake"

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: object())
    return state


def _extract(llm, title="Doc", body="body"):
    return asyncio.run(extract_structured_metadata(title, body))


def test_a_full_reply_is_returned_as_is(llm):
    llm["reply"] = (
        '{"document_number": "SOP-114", "issue_date": "2026-01-15", '
        '"expiry_date": "2027-01-15", "signed_by": "Jane Doe"}'
    )
    assert _extract(llm) == {
        "document_number": "SOP-114",
        "issue_date": "2026-01-15",
        "expiry_date": "2027-01-15",
        "signed_by": "Jane Doe",
    }


def test_every_key_is_present_even_when_the_model_omits_it(llm):
    llm["reply"] = '{"document_number": "SOP-114"}'
    result = _extract(llm)
    assert set(result.keys()) == {"document_number", "issue_date", "expiry_date", "signed_by"}
    assert result["issue_date"] is None


@pytest.mark.parametrize("placeholder", ["null", "None", "N/A", "n/a", "unknown", ""])
def test_placeholder_values_become_none(llm, placeholder):
    llm["reply"] = f'{{"document_number": "{placeholder}"}}'
    assert _extract(llm)["document_number"] is None


def test_a_date_not_in_iso_format_is_dropped_not_stored(llm):
    llm["reply"] = '{"issue_date": "15 January 2026"}'
    assert _extract(llm)["issue_date"] is None


def test_a_valid_iso_date_is_kept(llm):
    llm["reply"] = '{"issue_date": "2026-01-15"}'
    assert _extract(llm)["issue_date"] == "2026-01-15"


def test_document_number_is_not_validated_as_a_date(llm):
    """Only issue_date/expiry_date go through date validation; free text is fine here."""
    llm["reply"] = '{"document_number": "not-a-date-at-all"}'
    assert _extract(llm)["document_number"] == "not-a-date-at-all"


def test_a_markdown_code_fence_is_stripped_before_parsing(llm):
    llm["reply"] = '```json\n{"document_number": "SOP-1"}\n```'
    assert _extract(llm)["document_number"] == "SOP-1"


@pytest.mark.parametrize("reply", ["not json", "", "[1, 2, 3]", '"just a string"'])
def test_unusable_replies_raise_rather_than_fabricate(llm, reply):
    llm["reply"] = reply
    with pytest.raises(StructuredMetadataUnavailable):
        _extract(llm)


def test_provider_failure_raises(llm):
    llm["raises"] = RuntimeError("provider is down")
    with pytest.raises(StructuredMetadataUnavailable):
        _extract(llm)


def test_no_provider_configured_raises(llm, monkeypatch):
    from src.domain import llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: None)
    with pytest.raises(StructuredMetadataUnavailable):
        _extract(llm)
