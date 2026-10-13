"""How the warm-up is classified, and that a late success is not forgotten.

Follow-ups to test_readiness_waits_for_embedding_model.py:
* ``timeout`` used to be sticky: the load kept running after the readiness timeout, finished, and the
  probe still said ``degraded`` for the life of the process;
* only running out of patience is a timeout. An error raised by the load itself (a socket timeout
  while downloading) is a failure, even if its text says "timed out";
* the task must be referenced somewhere, or the loop can drop it before it records the outcome.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from src.api import main
from src.core import warmup
from src.core.warmup import EmbeddingWarmup


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    state = EmbeddingWarmup()
    monkeypatch.setattr(main, "embedding_warmup", state)
    monkeypatch.setattr(main.settings, "CROSS_ENCODER_RERANKER_ENABLED", False)
    return state


def test_a_load_that_finishes_after_the_timeout_ends_ready_not_degraded(fresh_state, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(main.settings, "EMBEDDING_WARMUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(main, "warm_up_embeddings", lambda: release.wait(5))
    fresh_state.begin()

    async def scenario():
        task = asyncio.create_task(main._preload_embedding_model())
        await asyncio.sleep(0.3)
        during = (fresh_state.status, fresh_state.degraded, fresh_state.holds_readiness)
        release.set()
        await task
        return during

    try:
        during = asyncio.run(scenario())
    finally:
        release.set()

    assert during == (warmup.TIMEOUT, True, False), "after the timeout it is degraded and no longer holds readiness"
    assert fresh_state.status == warmup.READY
    assert fresh_state.degraded is False
    assert fresh_state.seconds >= 0.3, "the duration is the real one, including the wait after the timeout"


def test_a_late_failure_is_recorded_as_failed(fresh_state, monkeypatch):
    release = threading.Event()

    def load():
        release.wait(5)
        raise RuntimeError("model.onnx is truncated")

    monkeypatch.setattr(main.settings, "EMBEDDING_WARMUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(main, "warm_up_embeddings", load)
    fresh_state.begin()

    async def scenario():
        task = asyncio.create_task(main._preload_embedding_model())
        await asyncio.sleep(0.2)
        assert fresh_state.status == warmup.TIMEOUT
        release.set()
        await task

    try:
        asyncio.run(scenario())
    finally:
        release.set()

    assert fresh_state.status == warmup.FAILED
    assert "truncated" in fresh_state.error


def test_a_socket_timeout_inside_the_load_is_a_failure_not_a_timeout(fresh_state, monkeypatch):
    """TimeoutError raised BY the load (a download that timed out) must not be mistaken for the
    readiness timeout: nothing was still loading, it simply failed."""
    def load():
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(main.settings, "EMBEDDING_WARMUP_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(main, "warm_up_embeddings", load)
    fresh_state.begin()

    asyncio.run(main._preload_embedding_model())

    assert fresh_state.status == warmup.FAILED
    assert "timed out" in fresh_state.error


def test_a_load_inside_the_timeout_is_simply_ready(fresh_state, monkeypatch):
    monkeypatch.setattr(main.settings, "EMBEDDING_WARMUP_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(main, "warm_up_embeddings", lambda: None)
    fresh_state.begin()

    asyncio.run(main._preload_embedding_model())

    assert fresh_state.status == warmup.READY


def test_initialize_resources_keeps_a_reference_to_the_preload_task():
    import inspect

    source = inspect.getsource(main.initialize_resources)
    assert "_preload_task = asyncio.create_task(" in source, (
        "the event loop holds tasks weakly; an unreferenced preload task can be collected before it records its outcome"
    )
