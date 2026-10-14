"""`/health/ready` must say 503 while the embedding model loads, and must not wait forever.

A deploy is declared healthy by polling `/health/ready`. It used to answer 200 as soon as the
database and Redis did, while the model (seconds to a minute on a cold, shared-CPU task) was still
loading in the background, so a freshly started task was reported ready and then made the first
searches wait. The load is now recorded (src/core/warmup.py) and the probe reads it.

What is deliberately NOT asserted as a failure: a load that raises, or times out, leaves readiness
at 200 with ``degraded: true``. Keyword search works without the model; holding the probe forever
would turn a degraded feature into an outage and fail every deploy.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api import main
from src.core import warmup
from src.core.warmup import EmbeddingWarmup


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    state = EmbeddingWarmup()
    monkeypatch.setattr(main, "embedding_warmup", state)
    monkeypatch.setattr(warmup, "embedding_warmup", state)
    return state


@pytest.fixture
def backing_services_up(monkeypatch):
    """Database and Redis answer, so the only thing that can hold the probe is the model."""
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=connection)
    connection.__aexit__ = AsyncMock(return_value=False)
    connection.execute = AsyncMock()
    import src.api.deps as deps

    monkeypatch.setattr(deps, "engine", MagicMock(connect=MagicMock(return_value=connection)))
    redis = MagicMock(ping=AsyncMock(), aclose=AsyncMock())
    with patch("redis.asyncio.Redis.from_url", return_value=redis):
        yield


def ready():
    return asyncio.run(main.health_ready())


def test_503_while_the_model_is_loading(fresh_state, backing_services_up):
    fresh_state.begin()

    with pytest.raises(HTTPException) as caught:
        ready()

    assert caught.value.status_code == 503
    assert caught.value.detail["embedding"]["state"] == "warming"


def test_200_once_the_model_has_loaded(fresh_state, backing_services_up):
    fresh_state.begin()
    fresh_state.finish(warmup.READY)

    body = ready()

    assert body["status"] == "ready"
    assert body["embedding"]["state"] == "ready"
    assert body["degraded"] is False
    assert body["embedding"]["seconds"] >= 0


def test_200_when_there_is_no_model_to_wait_for(fresh_state, backing_services_up):
    """Mock embeddings never call begin(); readiness must not hang on them."""
    body = ready()

    assert body["embedding"]["state"] == "disabled"


def test_a_failed_load_does_not_hold_readiness_but_is_reported(fresh_state, backing_services_up, monkeypatch):
    fresh_state.begin()
    monkeypatch.setattr(main, "warm_up_embeddings", MagicMock(side_effect=RuntimeError("no model.onnx")))

    asyncio.run(main._preload_embedding_model())
    body = ready()

    assert fresh_state.status == warmup.FAILED
    assert "no model.onnx" in fresh_state.error
    assert body["status"] == "ready"
    assert body["degraded"] is True


def test_a_load_that_outlasts_the_timeout_stops_holding_readiness(fresh_state, backing_services_up, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(main.settings, "EMBEDDING_WARMUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(main, "warm_up_embeddings", lambda: release.wait(5))
    fresh_state.begin()

    async def scenario():
        task = asyncio.create_task(main._preload_embedding_model())
        await asyncio.sleep(0.3)  # past the timeout, load still running
        body = await main.health_ready()
        release.set()
        await task
        return body

    try:
        body = asyncio.run(scenario())
    finally:
        release.set()

    assert body["degraded"] is True and body["embedding"]["state"] == warmup.TIMEOUT
    # and once the load really ends, the verdict follows it (see test_embedding_warmup_outcomes.py)
    assert fresh_state.status == warmup.READY


def test_the_probe_goes_green_exactly_when_the_load_ends(fresh_state, backing_services_up, monkeypatch):
    """The end-to-end shape: 503 during the load, 200 after, through the real task."""
    release = threading.Event()
    monkeypatch.setattr(main.settings, "EMBEDDING_WARMUP_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(main, "warm_up_embeddings", lambda: release.wait(5))

    async def scenario():
        fresh_state.begin()
        task = asyncio.create_task(main._preload_embedding_model())
        await asyncio.sleep(0.05)
        with pytest.raises(HTTPException) as during:
            await main.health_ready()
        release.set()
        await task
        return during.value.status_code, await main.health_ready()

    status_during, body_after = asyncio.run(scenario())

    assert status_during == 503
    assert body_after["embedding"]["state"] == "ready"


def test_liveness_never_reads_the_warm_up(fresh_state):
    """/health/live and /livez answer while the model loads; ECS and the image HEALTHCHECK use them."""
    fresh_state.begin()

    assert asyncio.run(main.health_live()) == {"status": "alive"}
    assert asyncio.run(main.livez()) == {"status": "alive"}


def test_the_deploy_probe_does_not_wait_for_the_model(fresh_state, backing_services_up):
    """/health/deps is what the deploy polls (fixed 120 s, shared workflow). A cold model load
    must not be able to fail a deploy, so it answers 200 while /health/ready answers 503."""
    fresh_state.begin()

    body = asyncio.run(main.health_deps())
    with pytest.raises(HTTPException) as caught:
        ready()

    assert body["status"] == "ready"
    assert caught.value.status_code == 503


def test_the_deploy_probe_still_fails_when_a_dependency_is_down(fresh_state, monkeypatch):
    """It is not liveness: ci requires a deploy gate to notice a missing database or cache."""
    import src.api.deps as deps

    broken = MagicMock(connect=MagicMock(side_effect=RuntimeError("database is down")))
    monkeypatch.setattr(deps, "engine", broken)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(main.health_deps())

    assert caught.value.status_code == 503
