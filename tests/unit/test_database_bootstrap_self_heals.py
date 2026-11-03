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


@pytest.fixture
def bootstrap(monkeypatch):
    """`_bootstrap_database` with its steps, sleep, dispose, loop start and process exit stubbed."""
    monkeypatch.setattr(main, "database_bootstrapped", False)
    state = {
        "sleeps": [],
        "loops_started_after_attempt": None,
        "exit": None,
        "attempts": 0,
    }

    async def sleep(seconds: float) -> None:
        state["sleeps"].append(seconds)

    def start_loops() -> None:
        state["loops_started_after_attempt"] = state["attempts"]

    def exit_process(code: int) -> None:
        state["exit"] = code

    for name in (
        "verify_rls_policies",
        "verify_embedding_column_width",
        "reconcile_published_indexes",
    ):
        monkeypatch.setattr(main, name, AsyncMock())
    # A dispose that fails on half-closed sockets must not end the retry loop.
    monkeypatch.setattr(
        main,
        "engine",
        MagicMock(dispose=AsyncMock(side_effect=OSError("socket closed"))),
    )
    monkeypatch.setattr(main.asyncio, "sleep", sleep)
    monkeypatch.setattr(main, "_start_inline_loops", start_loops)
    monkeypatch.setattr(main, "_exit_process", exit_process)

    def init_db_failing(times: int, error: Exception):
        async def init_db():
            state["attempts"] += 1
            if state["attempts"] <= times:
                raise error

        monkeypatch.setattr(main, "init_db", init_db)

    state["init_db_failing"] = init_db_failing
    return state


def test_bootstrap_retries_with_capped_backoff_until_the_database_answers(bootstrap):
    bootstrap["init_db_failing"](6, ConnectionRefusedError("connect ECONNREFUSED"))

    asyncio.run(main._bootstrap_database())

    assert main.database_bootstrapped is True
    assert bootstrap["attempts"] == 7
    assert bootstrap["sleeps"] == [
        2,
        4,
        8,
        16,
        30,
        30,
    ]  # doubling, capped, never giving up
    assert bootstrap["exit"] is None


def test_inline_loops_start_only_after_the_bootstrap_succeeds(bootstrap, monkeypatch):
    """They read and write the database: not while it is unreachable, not during reconciliation."""
    bootstrap["init_db_failing"](3, ConnectionRefusedError("connect ECONNREFUSED"))
    reconciled = {"loops_started_before": None}

    async def reconcile():
        reconciled["loops_started_before"] = (
            bootstrap["loops_started_after_attempt"] is not None
        )

    monkeypatch.setattr(main, "reconcile_published_indexes", reconcile)

    asyncio.run(main._bootstrap_database())

    assert reconciled["loops_started_before"] is False
    assert bootstrap["loops_started_after_attempt"] == 4  # the first successful attempt


def test_a_deterministic_error_is_not_retried_it_exits_for_crashloopbackoff(bootstrap):
    """AUTO_CREATE_SCHEMA set, a missing RLS policy: retrying would fail the same way forever."""
    bootstrap["init_db_failing"](
        1, RuntimeError("AUTO_CREATE_SCHEMA is disabled; run Alembic")
    )

    asyncio.run(main._bootstrap_database())

    assert bootstrap["exit"] == 1
    assert bootstrap["sleeps"] == []
    assert bootstrap["loops_started_after_attempt"] is None
    assert main.database_bootstrapped is False


def test_connectivity_errors_are_transient_through_sqlalchemy_wrapping():
    import asyncpg
    from sqlalchemy import exc as sa_exc

    starting_up = asyncpg.CannotConnectNowError("the database system is starting up")
    wrapped = sa_exc.DBAPIError("SELECT 1", {}, starting_up)
    assert main._is_transient_database_error(wrapped)
    assert main._is_transient_database_error(
        asyncpg.TooManyConnectionsError("too many clients")
    )
    assert main._is_transient_database_error(TimeoutError())
    assert main._is_transient_database_error(ConnectionRefusedError())

    assert not main._is_transient_database_error(
        RuntimeError("AUTO_CREATE_SCHEMA is disabled")
    )
    assert not main._is_transient_database_error(
        sa_exc.ProgrammingError("SELECT 1", {}, Exception("relation does not exist"))
    )


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


def test_tls_failures_are_permanent_even_though_they_are_oserrors():
    import ssl

    from sqlalchemy import exc as sa_exc

    bad_ca = ssl.SSLCertVerificationError(
        "certificate verify failed: unable to get local issuer"
    )
    assert not main._is_transient_database_error(bad_ca)
    assert not main._is_transient_database_error(
        sa_exc.DBAPIError("SELECT 1", {}, bad_ca)
    )
    # asyncpg's own wording when the server will not speak TLS.
    rejected = ConnectionError("PostgreSQL server at db:5432 rejected SSL upgrade")
    assert not main._is_transient_database_error(rejected)
    # A plain connection reset is still transient.
    assert main._is_transient_database_error(ConnectionResetError("reset by peer"))


def test_a_suppressed_context_is_not_followed():
    """`raise X from None` says the earlier error is not the cause; it must not decide."""
    try:
        try:
            raise ConnectionRefusedError("refused")
        except ConnectionRefusedError:
            raise RuntimeError("AUTO_CREATE_SCHEMA is disabled") from None
    except RuntimeError as err:
        suppressed = err
    assert not main._is_transient_database_error(suppressed)

    try:
        try:
            raise ConnectionRefusedError("refused")
        except ConnectionRefusedError:
            raise RuntimeError("wrapped while handling")
    except RuntimeError as err:
        implicit = err
    assert main._is_transient_database_error(implicit)


def test_a_failure_after_success_exits_rather_than_leaving_readiness_stuck(
    bootstrap, monkeypatch
):
    bootstrap["init_db_failing"](0, RuntimeError("unused"))

    def broken_start() -> None:
        raise RuntimeError("could not schedule the loops")

    monkeypatch.setattr(main, "_start_inline_loops", broken_start)

    asyncio.run(main._bootstrap_database())

    assert bootstrap["exit"] == 1
    assert main.database_bootstrapped is False


def test_lifespan_teardown_surfaces_a_bootstrap_that_died(monkeypatch):
    """A test (or anything) whose bootstrap failed must see the failure, not a quiet shutdown."""
    import src.lib.observability as observability

    monkeypatch.setattr(type(main.settings), "validate_production", lambda self: None)
    monkeypatch.setattr(main, "configure_tracing", lambda: None)
    monkeypatch.setattr(observability, "setup_logging", lambda: None)
    monkeypatch.setattr(main.settings, "EMBEDDING_MODEL", "mock")

    async def dies() -> None:
        raise RuntimeError("bootstrap blew up")

    monkeypatch.setattr(main, "_bootstrap_database", dies)

    async def scenario():
        async with main.lifespan(main.app):
            await asyncio.sleep(0)  # let the task run and fail

    with pytest.raises(RuntimeError, match="bootstrap blew up"):
        asyncio.run(scenario())
