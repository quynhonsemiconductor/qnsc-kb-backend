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


# ── the probe can never wedge the circuit ────────────────────────────────────────────────────
def _open(breaker, clock):
    for _ in range(3):
        breaker.record_timeout()
    clock.now += 101  # past the cooldown


def test_a_probe_that_ends_in_a_non_timeout_error_reopens_and_the_next_cooldown_allows_again():
    """The wedge: probe ends by CrossEncoderUnavailable / 'still scoring' / cancellation, nothing records it,
    and every later caller is refused forever."""
    clock = Clock()
    breaker = CircuitBreaker(clock)
    _open(breaker, clock)

    probe = breaker.begin()
    assert probe is not None and probe.probe
    breaker.finish(probe)  # ended with neither record_success nor record_timeout

    assert breaker.begin() is None, "re-opened for another full cooldown, not let straight through"
    clock.now += 101
    again = breaker.begin()
    assert again is not None and again.probe, "and the next cooldown allows a probe again"


def test_finish_is_a_no_op_after_the_probe_recorded_a_result():
    clock = Clock()
    breaker = CircuitBreaker(clock)
    _open(breaker, clock)
    probe = breaker.begin()
    breaker.record_success(probe)
    breaker.finish(probe)
    assert breaker.begin() is not None and not breaker.is_open


def test_a_closed_circuit_attempt_that_ends_late_cannot_abort_someone_elses_probe():
    clock = Clock()
    breaker = CircuitBreaker(clock)
    early = breaker.begin()  # started while closed
    assert early is not None and not early.probe
    _open(breaker, clock)
    probe = breaker.begin()
    breaker.finish(early)  # the old, non-probe attempt finally ends
    assert breaker.begin() is None, "the probe is still out; one caller at a time"
    breaker.record_success(probe)
    assert not breaker.is_open


def test_a_lost_probe_expires_on_its_own():
    clock = Clock()
    breaker = CircuitBreaker(clock)
    _open(breaker, clock)
    assert breaker.begin() is not None  # the probe, never finished (a bug elsewhere, a killed task)
    clock.now += 50
    assert breaker.begin() is None
    clock.now += 60
    assert breaker.begin() is not None, "after a full cooldown a stale probe is considered lost"


def test_through_search_a_probe_that_raises_a_non_timeout_error_does_not_wedge_it(enabled, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cross_encoder.breaker, "_clock", lambda: now["t"])
    for _ in range(3):
        cross_encoder.breaker.record_timeout()
    now["t"] += 101

    from src.rag.cross_encoder import CrossEncoderUnavailable

    def refuses(query, passages):
        raise CrossEncoderUnavailable("cross-encoder is still scoring an earlier search")

    monkeypatch.setattr(module, "cross_encoder_score", refuses)
    chunks = [_chunk(_article(), 0.9, "a"), _chunk(_article(), 0.5, "b")]
    try:
        _search(chunks)  # this search is the probe, and it ends with a non-timeout error
        assert cross_encoder.breaker.begin() is None, "re-opened, not half-open forever"
        now["t"] += 101
        calls = []
        monkeypatch.setattr(module, "cross_encoder_score", lambda q, p: calls.append(1) or [0.0] * len(p))
        _search(chunks)
        assert calls, "the next cooldown lets a probe through again"
        assert not cross_encoder.breaker.is_open, "and its success closes the circuit"
    finally:
        monkeypatch.setattr(cross_encoder.breaker, "_clock", __import__("time").monotonic)
        cross_encoder.breaker.record_success()


def test_a_cancelled_probe_does_not_wedge_it(enabled, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cross_encoder.breaker, "_clock", lambda: now["t"])
    for _ in range(3):
        cross_encoder.breaker.record_timeout()
    now["t"] += 101

    def cancelled(query, passages):
        raise asyncio.CancelledError()

    monkeypatch.setattr(module, "cross_encoder_score", cancelled)
    chunks = [_chunk(_article(), 0.9, "a"), _chunk(_article(), 0.5, "b")]
    try:
        try:
            _search(chunks)
        except BaseException:
            pass  # a cancelled request propagates; what matters is the breaker's state
        assert cross_encoder.breaker.begin() is None
        now["t"] += 101
        assert cross_encoder.breaker.begin() is not None
    finally:
        monkeypatch.setattr(cross_encoder.breaker, "_clock", __import__("time").monotonic)
        cross_encoder.breaker.record_success()


def test_check_budget_waits_for_the_scoring_lock(monkeypatch):
    monkeypatch.setattr(cross_encoder._model, "get", lambda: _model(0.001))
    finished = threading.Event()

    cross_encoder._scoring.acquire()
    worker = threading.Thread(target=lambda: (cross_encoder.check_budget(), finished.set()))
    worker.start()
    try:
        assert not finished.wait(0.3), "check_budget must not run while a search is scoring"
    finally:
        cross_encoder._scoring.release()
    assert finished.wait(5)
    worker.join(5)
