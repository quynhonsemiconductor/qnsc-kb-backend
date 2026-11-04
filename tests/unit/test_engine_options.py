"""The shared database engine checks its connections and exposes its pool.

A database restart or failover leaves a pool full of dead connections. Without `pool_pre_ping`
every request that draws one fails: restarting Postgres under 10 concurrent requests returned
8 x 500 before the first healthy answer. And under load the first thing to run dry is the pool
itself ("QueuePool limit of size 5 overflow 10 reached"), which could only be changed by editing code.
"""

from __future__ import annotations

from src.api import deps
from src.core.config import settings


def test_pooled_connections_are_checked_before_use():
    assert deps.engine_options()["pool_pre_ping"] is True


def test_idle_connections_are_recycled_rather_than_trusted_forever():
    assert deps.engine_options()["pool_recycle"] == settings.DB_POOL_RECYCLE_SECONDS > 0


def test_the_pool_is_configurable_and_the_defaults_are_the_ones_it_always_had():
    options = deps.engine_options()
    assert (options["pool_size"], options["max_overflow"], options["pool_timeout"]) == (
        5,
        10,
        30.0,
    )
    assert settings.DB_POOL_SIZE == 5 and settings.DB_MAX_OVERFLOW == 10


def test_the_pool_settings_flow_through(monkeypatch):
    monkeypatch.setattr(settings, "DB_POOL_SIZE", 20)
    monkeypatch.setattr(settings, "DB_MAX_OVERFLOW", 30)
    monkeypatch.setattr(settings, "DB_POOL_TIMEOUT_SECONDS", 7.5)
    options = deps.engine_options()
    assert (options["pool_size"], options["max_overflow"], options["pool_timeout"]) == (
        20,
        30,
        7.5,
    )


def test_the_five_second_ceilings_on_connecting_and_querying_are_kept():
    # Raising these would let one slow request hold a pooled connection; deps.py explains why they are 5 s.
    connect_args = deps.engine_options()["connect_args"]
    assert connect_args["timeout"] == 5 and connect_args["command_timeout"] == 5


def test_the_module_level_engine_is_built_from_these_options():
    pool = deps.engine.pool
    assert pool.size() == settings.DB_POOL_SIZE
    assert pool._pre_ping is True
