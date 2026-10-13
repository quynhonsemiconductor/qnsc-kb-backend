"""Provider rate limits are managed, not surfaced; an empty reply is a failed call.

From the develop worker logs (Groq): "groq rate limit reached" on tag suggestion and restructure, and
"Automatic tag suggestion returned unparseable JSON ... reply=" with an EMPTY reply.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
import structlog

from src.core.config import settings
from src.domain import auto_tagging, content_restructure, llm_client
from src.domain.llm_client import ProviderEmptyReplyError, ProviderRateLimitError


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(llm_client, "_rate_limited_until", 0.0)
    monkeypatch.setattr(settings, "LLM_RATE_LIMIT_INLINE_WAIT_SECONDS", 30.0)
    monkeypatch.setattr(settings, "LLM_RATE_LIMIT_MAX_INLINE_RETRIES", 2)
    monkeypatch.setattr(settings, "LLM_EMPTY_REPLY_RETRIES", 2)
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", 2)
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds, *a, **k):
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)
    return sleeps


def ok(text="hello"):
    return (text, 5, "m", "groq")


def scripted(*outcomes):
    """A stand-in for _complete_once that yields each outcome in turn (exceptions are raised)."""
    calls = []
    queue = list(outcomes)

    async def fake(messages, **_kw):
        calls.append(1)
        outcome = queue.pop(0) if queue else ok()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return fake, calls


MSG = [{"role": "user", "content": "x"}]


def test_a_short_retry_after_is_waited_out_and_the_call_succeeds(monkeypatch, clean):
    fake, calls = scripted(ProviderRateLimitError("groq rate limit reached", retry_after=7), ok("done"))
    monkeypatch.setattr(llm_client, "_complete_once", fake)

    text, *_ = asyncio.run(llm_client.complete(MSG))

    assert text == "done" and len(calls) == 2
    assert any(7 <= s <= 8.5 for s in clean), f"it must wait the Retry-After it was given, plus jitter: {clean}"


def test_a_long_retry_after_is_raised_with_the_wait_and_the_provider_is_left_alone(monkeypatch):
    fake, calls = scripted(ProviderRateLimitError("groq rate limit reached", retry_after=600))
    monkeypatch.setattr(llm_client, "_complete_once", fake)

    with pytest.raises(ProviderRateLimitError) as caught:
        asyncio.run(llm_client.complete(MSG))
    assert caught.value.retry_after == 600 and len(calls) == 1

    # The next caller does not spend a request finding out the same thing.
    with pytest.raises(ProviderRateLimitError) as again:
        asyncio.run(llm_client.complete(MSG))
    assert len(calls) == 1, "still inside the provider's wait: no new request"
    assert 590 <= again.value.retry_after <= 601


def test_inline_retries_are_bounded(monkeypatch):
    fake, calls = scripted(*[ProviderRateLimitError("rate", retry_after=1)] * 10)
    monkeypatch.setattr(llm_client, "_complete_once", fake)
    with pytest.raises(ProviderRateLimitError):
        asyncio.run(llm_client.complete(MSG))
    assert len(calls) == 1 + settings.LLM_RATE_LIMIT_MAX_INLINE_RETRIES


def test_concurrent_calls_are_capped(monkeypatch):
    in_flight = {"now": 0, "max": 0}

    async def slow(messages, **_kw):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.get_running_loop().create_future() if False else await _tick()
        in_flight["now"] -= 1
        return ok()

    async def _tick():
        for _ in range(5):
            await asyncio.sleep(0)

    monkeypatch.setattr(llm_client, "_complete_once", slow)

    async def many():
        await asyncio.gather(*(llm_client.complete(MSG) for _ in range(8)))

    asyncio.run(many())
    assert in_flight["max"] == 2, f"LLM_MAX_CONCURRENCY=2 but {in_flight['max']} calls ran at once"


def test_an_empty_reply_is_retried_then_succeeds(monkeypatch):
    fake, calls = scripted(ok(""), ok("   \n"), ok("tags"))
    monkeypatch.setattr(llm_client, "_complete_once", fake)
    text, *_ = asyncio.run(llm_client.complete(MSG))
    assert text == "tags" and len(calls) == 3


def test_a_persistently_empty_reply_is_a_failed_call_not_an_empty_answer(monkeypatch):
    fake, calls = scripted(*[ok("")] * 10)
    monkeypatch.setattr(llm_client, "_complete_once", fake)
    with pytest.raises(ProviderEmptyReplyError):
        asyncio.run(llm_client.complete(MSG))
    assert len(calls) == 1 + settings.LLM_EMPTY_REPLY_RETRIES


# ── callers ──────────────────────────────────────────────────────────────────────────────────
def test_tag_suggestion_reports_an_empty_reply_as_a_failed_call_not_unparseable_json(monkeypatch):
    import src.domain.llm_client as client

    async def empty_after_retries(messages, **_kw):
        raise ProviderEmptyReplyError("groq returned an empty reply 3 times")

    monkeypatch.setattr(client, "complete", empty_after_retries)
    monkeypatch.setattr(client, "resolve_provider", lambda *_a, **_k: SimpleNamespace(name="groq"))

    with structlog.testing.capture_logs() as logs:
        tags = asyncio.run(auto_tagging.suggest_tags_for_document("T", "body", "doc"))

    assert tags == []
    events = [entry["event"] for entry in logs]
    assert "Automatic tag suggestion call failed" in events
    assert not any("unparseable" in event for event in events)


def test_tag_suggestion_given_a_bare_empty_string_does_not_call_it_a_parse_error(monkeypatch):
    import src.domain.llm_client as client

    async def empty(messages, **_kw):
        return "", 0, "m", "groq"

    monkeypatch.setattr(client, "complete", empty)
    monkeypatch.setattr(client, "resolve_provider", lambda *_a, **_k: SimpleNamespace(name="groq"))
    with structlog.testing.capture_logs() as logs:
        assert asyncio.run(auto_tagging.suggest_tags_for_document("T", "body", "doc")) == []
    events = [entry["event"] for entry in logs]
    assert "Automatic tag suggestion returned an empty reply" in events
    assert not any("unparseable" in event for event in events)


def test_restructure_reports_a_rate_limit_as_deferred_with_the_wait(monkeypatch):
    async def limited(*_a, **_k):
        raise ProviderRateLimitError("groq rate limit reached", retry_after=45)

    monkeypatch.setattr(content_restructure, "complete", limited)
    monkeypatch.setattr(content_restructure, "resolve_provider", lambda *_a, **_k: SimpleNamespace(name="groq", model="m"))

    result = asyncio.run(content_restructure._restructure_single_document("T", "x" * 200, "m"))

    assert result.status == "deferred"
    assert result.retry_after == 45
    assert "deferred" in result.error.lower()


def test_any_section_deferred_defers_the_whole_document(monkeypatch):
    async def sections(title, secs, model, deps):
        return [
            content_restructure._result("a", "a", "llm", "m"),
            content_restructure._result("b", "b", "deferred", "m", "AI formatting deferred"),
        ]

    monkeypatch.setattr(content_restructure, "_restructure_sections", sections)
    monkeypatch.setattr(content_restructure, "resolve_provider", lambda *_a, **_k: SimpleNamespace(name="groq", model="m"))
    monkeypatch.setattr(content_restructure, "is_already_structured", lambda _t: False)
    monkeypatch.setattr(settings, "RESTRUCTURE_SECTION_CHARS", 100)
    monkeypatch.setattr(settings, "RESTRUCTURE_MAX_SECTIONS", 10)
    monkeypatch.setattr(settings, "RESTRUCTURE_MAX_CHARS", 10_000)
    monkeypatch.setattr(content_restructure, "_split_oversized_source", lambda text, n: ["a", "b"])

    result = asyncio.run(content_restructure.restructure_document("T", "y" * 500, enabled=True))

    assert result.status == "deferred" and result.retry_after


# ── the task re-queues instead of failing ────────────────────────────────────────────────────
class FakeDB:
    def __init__(self, draft):
        self.draft, self.commits, self.rollbacks = draft, 0, 0

    async def execute(self, _stmt):
        return SimpleNamespace(scalar_one_or_none=lambda: self.draft)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def draft(status="processing", metadata=None):
    return SimpleNamespace(
        id=uuid.uuid4(), status="pending", restructure_status=status, restructure_error=None, content_metadata=metadata,
        restructure_decision="x", restructure_model="m",
    )


@pytest.fixture
def scheduled(monkeypatch):
    from src.workers import tasks

    calls = []
    monkeypatch.setattr(tasks, "_schedule_restructure", lambda *a: calls.append(a))
    return calls


def test_a_rate_limited_draft_is_deferred_with_a_message_and_a_scheduled_retry(scheduled):
    from src.workers import tasks

    d = draft()
    asyncio.run(tasks._defer_restructure(FakeDB(d), str(d.id), "acme.test", "u1", 45))

    assert d.restructure_status == "deferred"
    assert "deferred" in d.restructure_error.lower() and "45" in d.restructure_error
    assert d.content_metadata["restructure_deferrals"] == 1
    assert scheduled == [(str(d.id), "acme.test", "u1", 45)]


def test_after_the_deferral_budget_the_lossless_fallback_is_used_and_nothing_is_scheduled(scheduled, monkeypatch):
    from src.workers import tasks

    monkeypatch.setattr(settings, "RESTRUCTURE_MAX_DEFERRALS", 2)
    d = draft(metadata={"restructure_deferrals": 2})
    asyncio.run(tasks._defer_restructure(FakeDB(d), str(d.id), "acme.test", "u1", 45))

    assert d.restructure_status == "fallback_formatting"
    assert "kept rate limiting" in d.restructure_error
    assert scheduled == []


def test_a_draft_decided_in_the_meantime_is_not_touched(scheduled):
    from src.workers import tasks

    d = draft(status="llm")
    db = FakeDB(d)
    asyncio.run(tasks._defer_restructure(db, str(d.id), "acme.test", "u1", 45))
    assert d.restructure_status == "llm" and scheduled == [] and db.rollbacks == 1


def test_deferred_counts_as_ai_busy_and_a_lost_retry_becomes_stalled():
    from datetime import datetime, timedelta

    from src.api.routers.governance import _visible_restructure_state

    now = datetime(2026, 1, 1, 12, 0, 0)
    fresh = SimpleNamespace(restructure_status="deferred", restructure_error="AI formatting deferred", updated_at=now - timedelta(minutes=2), created_at=now)
    old = SimpleNamespace(restructure_status="deferred", restructure_error="x", updated_at=now - timedelta(minutes=30), created_at=now)
    assert _visible_restructure_state(fresh, now)[0] == "deferred"
    assert _visible_restructure_state(old, now)[0] == "stalled"
