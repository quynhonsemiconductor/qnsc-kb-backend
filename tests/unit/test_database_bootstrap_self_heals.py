"""The API boots before the database and heals without a restart.

After a node reboot every pod starts at once, and the API can come up before PostgreSQL. Startup used to
try the database bootstrap three times and then raise, which exited the process: the pod went into a
CrashLoopBackOff whose restart delay (up to 5 min) outlasted the outage. Now the bootstrap retries in the
background with capped backoff, liveness answers throughout, and readiness answers 503 until the database
answers AND the bootstrap has finished, then 200, all in the same process.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api import main


class _Database:
    """The engine readiness pings: down until `up = True`."""

    def __init__(self) -> None:
        self.up = False

    def connect(self):
        if not self.up:
            raise ConnectionRefusedError("connect ECONNREFUSED")
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=connection)
        connection.__aexit__ = AsyncMock(return_value=False)
        connection.execute = AsyncMock()
        return connection


@pytest.fixture
def database(monkeypatch):
    import src.api.deps as deps

    db = _Database()
    monkeypatch.setattr(deps, "engine", db)
    monkeypatch.setattr(main, "database_bootstrapped", False)
    redis = MagicMock(ping=AsyncMock(), aclose=AsyncMock())
    with patch("redis.asyncio.Redis.from_url", return_value=redis):
        yield db


def _deps_status() -> tuple[int, dict]:
    try:
        return 200, asyncio.run(main.health_deps())
    except HTTPException as exc:
        return exc.status_code, exc.detail


def test_readiness_goes_from_down_to_bootstrapping_to_ready_without_a_restart(database):
    assert _deps_status() == (503, {"status": "not_ready", "database": "unavailable"})

    database.up = True
    assert _deps_status() == (503, {"status": "not_ready", "database": "bootstrapping"})

    main.database_bootstrapped = True
    code, body = _deps_status()
    assert code == 200
    assert body["database"] == "ok"


def test_liveness_answers_while_the_database_is_down(database):
    assert asyncio.run(main.livez()) == {"status": "alive"}
    assert asyncio.run(main.health_live()) == {"status": "alive"}


def test_bootstrap_retries_with_capped_backoff_until_the_database_answers(monkeypatch):
    monkeypatch.setattr(main, "database_bootstrapped", False)
    calls = {"n": 0}

    async def init_db():
        calls["n"] += 1
        if calls["n"] <= 6:
            raise ConnectionRefusedError("the database system is starting up")

    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(main, "init_db", init_db)
    for name in (
        "verify_rls_policies",
        "verify_embedding_column_width",
        "reconcile_published_indexes",
    ):
        monkeypatch.setattr(main, name, AsyncMock())
    monkeypatch.setattr(main, "engine", MagicMock(dispose=AsyncMock()))
    monkeypatch.setattr(main.asyncio, "sleep", sleep)

    asyncio.run(main._bootstrap_database())

    assert main.database_bootstrapped is True
    assert calls["n"] == 7
    assert sleeps == [2, 4, 8, 16, 30, 30]  # doubling, capped, never giving up


def test_startup_does_not_wait_for_the_database(monkeypatch):
    """initialize_resources returns at once and leaves the bootstrap running in the background."""
    import src.lib.observability as observability

    monkeypatch.setattr(type(main.settings), "validate_production", lambda self: None)
    monkeypatch.setattr(main, "configure_tracing", lambda: None)
    monkeypatch.setattr(observability, "setup_logging", lambda: None)
    monkeypatch.setattr(main.settings, "EMBEDDING_MODEL", "mock")

    async def scenario():
        gate = asyncio.Event()

        async def bootstrap():
            await gate.wait()

        monkeypatch.setattr(main, "_bootstrap_database", bootstrap)
        await asyncio.wait_for(main.initialize_resources(), timeout=5)
        task = main._bootstrap_task
        assert task is not None and not task.done()
        gate.set()
        await task

    asyncio.run(scenario())


def test_the_engine_pre_pings_pooled_connections():
    """A connection left dead by a database restart is replaced, not handed to a request."""
    import src.api.deps as deps

    assert deps.engine.pool._pre_ping is True
