"""A search that arrives while the reranker is busy waits for its turn instead of silently losing the reranker.

The reranker scores one search at a time. It used to refuse every other search at once, so with several assistants
asking together most answers were ordered by the lexical scorer alone, and the same six questions gave different top
results asked together and asked one after another (measured on the local stack: "PWM base address" returned
QNSC_MAS_INTMAP together and QNSC_MAS_PWM one after another).
"""

from __future__ import annotations

import threading
import time

import pytest

from src.core.config import settings
from src.rag import cross_encoder
from src.rag.cross_encoder import CrossEncoderUnavailable
from tests.unit.test_search_bounds_the_cross_encoder import (  # noqa: F401 - fixture and helpers
    _article,
    _chunk,
    _run,
    cross_encoder_on,
)
from src.domain import search_service as module


class _Model:
    """Takes `hold` seconds per call and records how many calls ran."""

    def __init__(self, hold: float):
        self.hold, self.calls = hold, 0
        self.started = threading.Event()

    def predict(self, pairs):
        self.calls += 1
        self.started.set()
        time.sleep(self.hold)
        return [0.5] * len(pairs)


def _occupy(model: _Model) -> threading.Thread:
    """Start a scoring call and return once it holds the reranker."""
    thread = threading.Thread(
        target=lambda: cross_encoder.score("q", ["a"], wait_seconds=0)
    )
    thread.start()
    assert model.started.wait(5)
    model.started.clear()
    return thread


def test_a_search_waits_for_its_turn_and_is_then_scored(monkeypatch):
    model = _Model(hold=0.3)
    monkeypatch.setattr(cross_encoder._model, "get", lambda: model)
    first = _occupy(model)

    started = time.monotonic()
    scores = cross_encoder.score("q", ["b", "c"], wait_seconds=3.0)

    assert scores == [0.5, 0.5]
    assert time.monotonic() - started >= 0.15, "it must have waited for the first call"
    first.join(5)
    assert model.calls == 2


def test_a_search_gives_up_after_its_wait_and_leaves_no_job_behind(monkeypatch):
    model = _Model(hold=0.5)
    monkeypatch.setattr(cross_encoder._model, "get", lambda: model)
    first = _occupy(model)

    with pytest.raises(CrossEncoderUnavailable, match="still scoring"):
        cross_encoder.score("q", ["b"], wait_seconds=0.05)
    first.join(5)

    time.sleep(0.2)
    assert (
        model.calls == 1
    ), "a caller that gave up must not run its job once the reranker frees up"


def test_the_default_wait_is_the_setting(monkeypatch):
    model = _Model(hold=0.3)
    monkeypatch.setattr(cross_encoder._model, "get", lambda: model)
    monkeypatch.setattr(settings, "CROSS_ENCODER_QUEUE_WAIT_SECONDS", 0.0)
    first = _occupy(model)
    try:
        with pytest.raises(CrossEncoderUnavailable):
            cross_encoder.score("q", ["b"])  # 0 = refuse at once, as before
    finally:
        first.join(5)

    monkeypatch.setattr(settings, "CROSS_ENCODER_QUEUE_WAIT_SECONDS", 3.0)
    second = _occupy(model)
    assert cross_encoder.score("q", ["b"]) == [0.5]
    second.join(5)


def test_the_search_deadline_includes_the_wait_for_a_turn(
    cross_encoder_on, monkeypatch
):
    """Scoring that takes longer than CROSS_ENCODER_TIMEOUT_SECONDS only because it queued is not abandoned."""
    article = _article()
    high, low = _chunk(article, 0.9, "lexically first"), _chunk(
        article, 0.5, "lexically second"
    )

    def queued_then_scored(query, passages):
        time.sleep(0.4)  # waited 0.3 s for its turn, scored in 0.1 s
        return [0.0, 1.0]

    monkeypatch.setattr(module, "cross_encoder_score", queued_then_scored)
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(settings, "CROSS_ENCODER_QUEUE_WAIT_SECONDS", 1.0)

    ids, _ = _run([high, low])

    assert ids == [str(low.id), str(high.id)]
    assert (
        not cross_encoder.breaker.is_open
    ), "waiting for a turn is not a reranker timeout"
