"""A pooled connection the server has dropped must not fail the request that draws it.

Built from `engine_options()`, the options the application's own engine uses. The server end of every
pooled connection is terminated, as a failover or an idle timeout does, and the next query must still
succeed. The control engine, identical but without the check, shows the failure this prevents.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from src.api import deps
from src.workers.loop import sync_run

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs a real PostgreSQL to drop connections from",
)


def _terminate_others(admin_engine, database: str) -> int:
    async def go():
        async with admin_engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
                    "WHERE datname = :db AND pid <> pg_backend_pid() "
                    "AND application_name = 'it-engine'"
                ),
                {"db": database},
            )
            return int(result.scalar_one())

    return sync_run(go())


def _engine(**overrides):
    options = deps.engine_options()
    options["connect_args"] = {
        **options["connect_args"],
        "server_settings": {"application_name": "it-engine"},
    }
    options.update(overrides)
    return create_async_engine(os.environ["DATABASE_URL"], **options)


async def _fill_and_query(engine, connections: int = 4):
    # Hold several connections at once so the pool genuinely contains more than one.
    opened = [await engine.connect() for _ in range(connections)]
    for connection in opened:
        await connection.execute(text("SELECT 1"))
    for connection in opened:
        await connection.close()


def test_a_dropped_pooled_connection_does_not_fail_the_next_request():
    engine, admin = _engine(), create_async_engine(os.environ["DATABASE_URL"])
    database = engine.url.database
    try:
        sync_run(_fill_and_query(engine))
        assert _terminate_others(admin, database) >= 1, "nothing was pooled to drop"

        async def next_request():
            async with engine.connect() as connection:
                return (await connection.execute(text("SELECT 42"))).scalar_one()

        assert sync_run(next_request()) == 42
    finally:
        sync_run(engine.dispose())
        sync_run(admin.dispose())


def test_without_the_check_the_same_drop_does_fail_a_request():
    """The control: proves the test drops something that matters, so the one above is not vacuous."""
    engine, admin = _engine(pool_pre_ping=False), create_async_engine(
        os.environ["DATABASE_URL"]
    )
    database = engine.url.database
    try:
        sync_run(_fill_and_query(engine))
        assert _terminate_others(admin, database) >= 1

        async def next_request():
            async with engine.connect() as connection:
                return (await connection.execute(text("SELECT 42"))).scalar_one()

        with pytest.raises((DBAPIError, ConnectionError, OSError)):
            sync_run(next_request())
    finally:
        sync_run(engine.dispose())
        sync_run(admin.dispose())
