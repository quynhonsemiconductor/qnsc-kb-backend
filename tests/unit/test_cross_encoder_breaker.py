"""The reranker's circuit breaker and its startup budget check."""
from __future__ import annotations

import asyncio
import threading
import uuid
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.domain import search_service as module
from src.domain.search_service import SearchService
from src.rag import cross_encoder
from src.rag.cross_encoder import CircuitBreaker


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def small_numbers(monkeypatch):
    monkeypatch.setattr(settings, "CROSS_ENCODER_BREAKER_THRESHOLD", 3)
    monkeypatch.setattr(settings, "CROSS_ENCODER_BREAKER_COOLDOWN_SECONDS", 100.0)
    cross_encoder.breaker.record_success()


def test_it_opens_only_after_consecutive_timeouts():
    breaker = CircuitBreaker(Clock())
    breaker.record_timeout()
    breaker.record_timeout()
    assert breaker.allow(), "two timeouts are under the threshold"
    breaker.record_timeout()
    assert not breaker.allow()


def test_a_success_resets_the_count():
    breaker = CircuitBreaker(Clock())
    breaker.record_timeout()
    breaker.record_timeout()
    breaker.record_success()
    breaker.record_timeout()
    breaker.record_timeout()
    assert breaker.allow(), "the successes in between mean these are not consecutive"


def test_after_the_cooldown_exactly_one_probe_is_let_through():
    clock = Clock()
    breaker = CircuitBreaker(clock)
    for _ in range(3):
        breaker.record_timeout()
    clock.now += 99
    assert not breaker.allow()
    clock.now += 2
    assert breaker.allow(), "the first caller after the cooldown is the probe"
    assert not breaker.allow(), "everyone else keeps skipping while the probe is out"


def test_a_successful_probe_closes_it_and_a_failed_one_reopens_it():
    clock = Clock()
    breaker = CircuitBreaker(clock)
    for _ in range(3):
        breaker.record_timeout()
    clock.now += 101
    assert breaker.allow()
    breaker.record_timeout()
    assert not breaker.allow(), "a timed-out probe re-opens it for another full cooldown"
    clock.now += 101
    assert breaker.allow()
    breaker.record_success()
    assert breaker.allow() and not breaker.is_open


# ── through the real SearchService ───────────────────────────────────────────────────────────
def _admin():
    from src.models.rbac import Permission, Role, RolePermission
    from src.models.user import User

    user = User(id=uuid.uuid4(), email="a@acme.test", name="A", company_domain="acme.test", role="Admin", active=True)
    role = Role(name="Admin", company_domain="acme.test", active=True)
    role.permissions.append(RolePermission(permission=Permission(key="article.read", name="Read"), scope="company"))
    user.roles.append(role)
    return user


def _chunk(article, score, label):
    return SimpleNamespace(
        id=uuid.uuid4(), parent_chunk_id=uuid.uuid4(), article=article, chunk_text=label, chunk_type="text",
        chunking_version=None, page_number=None, fake_score=score,
        parent_chunk=SimpleNamespace(text=label, section_ref="S", heading="H", page_number=None, child_chunks=[]),
    )


def _search(chunks):
    class Repo:
        db = None

        async def hybrid_search(self, **_kw):
            return list(chunks)

    service = SearchService(chunk_repo=Repo(), gov_repo=None, feature_flags=None)

    async def go():
        return await service.search(_admin(), "what is the leave policy? and who approves it?", limit=10)

    return asyncio.run(go())


@pytest.fixture
def enabled(monkeypatch):
    async def fake_embed(_text):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(module, "embed_query", fake_embed)
    monkeypatch.setattr(module, "rerank_chunks_with_scores",
                        lambda q, chunks, limit=5: [(c, c.fake_score) for c in sorted(chunks, key=lambda c: -c.fake_score)[:limit]])
    monkeypatch.setattr(settings, "CROSS_ENCODER_RERANKER_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0)
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 0.05)


def _article():
    return SimpleNamespace(
        id=uuid.uuid4(), title="T", dept="Eng", domain=None, type="document", sensitivity="internal",
        status="published", lifecycle_status="active", visibility="company", company_domain="acme.test",
        owner_id=uuid.uuid4(), sources=[], user_permissions=[], departments=[], owner=None, last_reviewed=None,
    )


def test_after_n_timeouts_searches_stop_calling_the_reranker(enabled, monkeypatch):
    calls = []
    release = threading.Event()

    def slow(query, passages):
        calls.append(1)
        release.wait(2)
        return [0.0] * len(passages)

    monkeypatch.setattr(module, "cross_encoder_score", slow)
    chunks = [_chunk(_article(), 0.9, "a"), _chunk(_article(), 0.5, "b")]
    try:
        for _ in range(3):
            _search(chunks)
            release.set()
            release.clear()
        assert len(calls) == 3
        _search(chunks)
        assert len(calls) == 3, "the 4th search must not even try the reranker while the circuit is open"
    finally:
        release.set()
    assert cross_encoder.breaker.is_open


def test_a_working_reranker_never_opens_the_circuit(enabled, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(module, "cross_encoder_score", lambda q, p: [0.0] * len(p))
    chunks = [_chunk(_article(), 0.9, "a"), _chunk(_article(), 0.5, "b")]
    for _ in range(5):
        _search(chunks)
    assert not cross_encoder.breaker.is_open


# ── the startup budget check ─────────────────────────────────────────────────────────────────
def _model(seconds_per_call):
    import time

    class Model:
        def predict(self, pairs):
            time.sleep(seconds_per_call)
            return [0.0] * len(pairs)

    return Model()


def test_budget_check_warns_when_the_pool_cannot_fit_the_timeout(monkeypatch):
    from structlog.testing import capture_logs

    monkeypatch.setattr(settings, "CROSS_ENCODER_CANDIDATE_POOL", 30)
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(cross_encoder._model, "get", lambda: _model(0.2))  # 0.1 s/pair x 30 = 3.0 s+

    with capture_logs() as logs:
        per_pair = cross_encoder.check_budget()

    assert per_pair == pytest.approx(0.1, abs=0.03)
    warned = [entry for entry in logs if entry["log_level"] == "warning"]
    assert warned and "cannot score its candidate pool" in warned[0]["event"]
    assert warned[0]["candidate_pool"] == 30


def test_budget_check_is_quiet_when_it_fits(monkeypatch):
    from structlog.testing import capture_logs

    monkeypatch.setattr(settings, "CROSS_ENCODER_CANDIDATE_POOL", 10)
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(cross_encoder._model, "get", lambda: _model(0.002))

    with capture_logs() as logs:
        cross_encoder.check_budget()

    assert not [entry for entry in logs if entry["log_level"] == "warning"]
