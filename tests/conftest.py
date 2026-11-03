import os

# BEFORE src.core.config is imported anywhere, because `settings = Settings()` runs at
# import time and reads the process environment exactly once. The Settings default for
# ENVIRONMENT is `production` — deliberately, so a deployment that forgets the variable
# fails closed instead of booting with the committed development secrets — and under that
# default the suite's module-level `settings` would demand externally supplied secrets,
# HTTPS origins and R2 credentials that no test has.
#
# setdefault, not assignment: `ENVIRONMENT=production poetry run pytest` still exercises
# the production branch, which is what the hardening tests need.
os.environ.setdefault("ENVIRONMENT", "development")

import sys

import pytest

# After the environment defaults above: importing it builds `settings`. Imported here so the guard
# below always applies, whatever order the test modules import things in.
import src.api.main  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _never_exit_the_test_process():
    """`_bootstrap_database` ends the process (os._exit) on a non-retryable error, so Kubernetes
    shows CrashLoopBackOff. Under pytest that would kill the whole run without a report, from a
    background task any test that enters the app lifespan starts. Raise instead.

    Not via `monkeypatch`: sharing the test's monkeypatch instance would move its undo after
    other fixtures' teardown, which some of them depend on.
    """
    main = sys.modules["src.api.main"]

    def refuse(code: int) -> None:
        raise RuntimeError(f"_exit_process({code}) called during a test")

    original = main._exit_process
    main._exit_process = refuse
    try:
        yield
    finally:
        main._exit_process = original


@pytest.fixture(autouse=True)
def _engine_pool_per_event_loop():
    """Drop the shared engine's pooled connections after every test.

    `src.api.deps.engine` is module-level, and asyncpg binds each connection to the event loop
    that opened it. Tests drive the app with `asyncio.run` (and each TestClient has its own loop),
    so a connection pooled by one test was handed to the next one's loop and failed with
    "RuntimeError: Event loop is closed" (seen in CI: the lifespan bootstrap of
    test_source_preview_headers drew such a connection). `close=False`: the old loop is gone,
    so the connections are only dereferenced, never awaited on a dead loop.
    """
    yield
    deps = sys.modules.get("src.api.deps")
    engine = getattr(deps, "engine", None)
    sync_engine = getattr(engine, "sync_engine", None)
    if sync_engine is not None:
        sync_engine.dispose(close=False)
