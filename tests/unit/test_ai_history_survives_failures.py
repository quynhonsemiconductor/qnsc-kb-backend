"""A turn must still be in the conversation after a reload, even when it went wrong.

Two ways a turn used to vanish:

* the question was persisted before generation, but a failure returned from the SSE
  handler WITHOUT persisting the reply — so a reload showed a question with no answer and
  no sign that anything had failed;
* an answer that genuinely found nothing to cite was indistinguishable, on reload, from an
  answer whose sources had been revoked, so it was replaced with "no longer available".
  The only thing separating them was a hardcoded list of English and Vietnamese phrase
  fragments, which no model-authored wording is obliged to match.

The fix records provenance at write time. These tests hold that line, and — just as
importantly — hold the fail-closed rule that motivated the masking in the first place.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from src.api.routers import ai as ai_router
from tests.unit.test_ai_safety import make_ai_user


def _history(**overrides):
    """One stored assistant message, with no citations."""
    message = SimpleNamespace(
        id=uuid.uuid4(),
        role="assistant",
        content="Toi khong tim thay tai lieu nao lien quan den cau hoi nay.",
        grounded_content="Toi khong tim thay tai lieu nao lien quan den cau hoi nay.",
        extended_content=None,
        citations="[]",
        action_data=None,
        usage_log_id=None,
        created_at=None,
    )
    for key, value in overrides.items():
        setattr(message, key, value)

    class FakeRepository:
        def __init__(self, _db):
            pass

        async def get_conversation(self, *_args, **_kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        async def list_messages(self, *_args, **_kwargs):
            return [message]

    return FakeRepository


def _load(monkeypatch, repository):
    monkeypatch.setattr(ai_router, "AIRepository", repository)
    return asyncio.run(
        ai_router.get_conversation_messages(uuid.uuid4(), make_ai_user(), object())
    )


def test_an_answer_recorded_as_uncited_survives_a_reload(monkeypatch):
    """This is the "no relevant documents" answer. It was never a cited answer."""
    response = _load(
        monkeypatch, _history(action_data={"grounding": "uncited"})
    )

    assert response[0]["content"].startswith("Toi khong tim thay")
    assert not response[0]["content"].startswith("This historical answer")
    assert response[0]["answer_grounded"].startswith("Toi khong tim thay")


def test_a_failed_turn_survives_a_reload_and_is_marked_as_failed(monkeypatch):
    response = _load(
        monkeypatch,
        _history(
            content="AI generation failed. Please try again.",
            grounded_content=None,
            action_data={"failed": True, "grounding": "uncited"},
        ),
    )

    assert response[0]["failed"] is True
    assert response[0]["content"] == "AI generation failed. Please try again."


def test_an_ordinary_answer_is_not_marked_as_failed(monkeypatch):
    response = _load(monkeypatch, _history(action_data={"grounding": "uncited"}))

    assert response[0]["failed"] is False


def test_an_uncited_answer_with_no_provenance_still_fails_closed(monkeypatch):
    """Rows written before provenance existed keep the old, safe behaviour."""
    response = _load(monkeypatch, _history(content="UNCITED PRIVATE ANSWER"))

    assert response[0]["content"].startswith("This historical answer is no longer")
    assert response[0]["answer_grounded"] == ""


def test_provenance_records_why_an_answer_has_no_citations():
    assert ai_router._answer_provenance({"citations": []}) == {"grounding": "uncited"}
    assert ai_router._answer_provenance({}) == {"grounding": "uncited"}


def test_provenance_leaves_a_cited_answer_alone():
    """Nothing to explain: it has sources, so revocation checks must still apply."""
    assert ai_router._answer_provenance({"citations": [{"chunk_id": "x"}]}) is None


def test_provenance_preserves_an_action_and_adds_to_it():
    stored = ai_router._answer_provenance(
        {"citations": [], "action_data": {"action": "article_updated", "article_id": "a"}}
    )

    assert stored == {
        "action": "article_updated",
        "article_id": "a",
        "grounding": "uncited",
    }


def test_provenance_never_overwrites_an_existing_grounding_claim():
    stored = ai_router._answer_provenance(
        {"citations": [], "action_data": {"grounding": "something-else"}}
    )

    assert stored == {"grounding": "something-else"}


def test_recording_a_failed_turn_never_masks_the_error_from_the_reader():
    """The reader must get the error event even if the database write fails."""
    calls: list[str] = []

    class BrokenRepository:
        async def add_message(self, *_args, **_kwargs):
            raise RuntimeError("database is down")

    class BrokenSession:
        async def rollback(self):
            calls.append("rollback")

    # Must not raise: the SSE handler yields the error event immediately after this.
    asyncio.run(
        ai_router._record_failed_turn(
            BrokenRepository(), BrokenSession(), uuid.uuid4(), "AI generation failed."
        )
    )

    assert calls == ["rollback"]


def test_a_failed_turn_is_written_as_an_uncited_assistant_message():
    recorded: dict = {}

    class Repository:
        async def add_message(self, conversation_id, role, content, **kwargs):
            recorded.update(
                {
                    "conversation_id": conversation_id,
                    "role": role,
                    "content": content,
                    **kwargs,
                }
            )

    class Session:
        async def rollback(self):
            return None

    conversation_id = uuid.uuid4()
    asyncio.run(
        ai_router._record_failed_turn(
            Repository(), Session(), conversation_id, "Upstream refused."
        )
    )

    assert recorded["conversation_id"] == conversation_id
    assert recorded["role"] == "assistant"
    assert recorded["content"] == "Upstream refused."
    assert recorded["action_data"] == {"failed": True, "grounding": "uncited"}
