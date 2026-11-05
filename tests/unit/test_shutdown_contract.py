"""PLATFORM-CONTRACT.md §2/§3 conformance: the /readyz body, draining, and the
bounded, ordered shutdown sequence (audit issue #180, gaps A1/A2/B4)."""

import asyncio
import logging
import time

import httpx
import pytest

from src.api import main as api_main
from src.core.shutdown import (
    ShutdownConfigurationError,
    ShutdownPlan,
    resolve_shutdown_plan,
)
from src.lib.observability import PROBE_PATHS, ProbeAccessFilter, is_probe_path


@pytest.fixture(autouse=True)
def _fresh_shutdown_state():
    state = api_main.shutdown_state
    state.draining = False
    state.deadline = None
    yield
    state.draining = False
    state.deadline = None


def _get(path):
    async def go():
        transport = httpx.ASGITransport(app=api_main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            return await client.get(path)

    return asyncio.run(go())


# --- the plan ------------------------------------------------------------------------


def test_the_defaults_are_the_contract_values_outside_a_pod(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    plan = resolve_shutdown_plan(None, None, in_pod := False)
    assert plan == ShutdownPlan(timeout_seconds=25.0, endpoint_delay_seconds=0.0)


def test_in_a_pod_the_default_delay_is_five_seconds(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert resolve_shutdown_plan(None, None, True).endpoint_delay_seconds == 5.0


def test_explicit_values_win_over_the_defaults():
    plan = resolve_shutdown_plan(30000, 1000, False)
    assert plan == ShutdownPlan(timeout_seconds=30.0, endpoint_delay_seconds=1.0)


def test_a_delay_at_or_above_the_deadline_refuses_to_boot():
    with pytest.raises(ShutdownConfigurationError):
        resolve_shutdown_plan(5000, 5000, False)
    with pytest.raises(ShutdownConfigurationError):
        resolve_shutdown_plan(5000, 6000, False)


def test_a_negative_value_refuses_to_boot():
    with pytest.raises(ShutdownConfigurationError):
        resolve_shutdown_plan(-1, None, False)
    with pytest.raises(ShutdownConfigurationError):
        resolve_shutdown_plan(None, -5, False)


# --- probe paths ----------------------------------------------------------------------


def test_probe_paths_match_the_whole_path_only():
    assert is_probe_path("/readyz")
    assert is_probe_path("/readyz?force=1")
    assert not is_probe_path("/readyz/extra")
    assert not is_probe_path("/api/v1/readyz")
    assert is_probe_path("/favicon.ico")
    # The legacy ECS aliases are deliberately not probes: the deploy pipeline
    # polls them and their request lines are useful when it fails.
    assert "/health/ready" not in PROBE_PATHS


def test_the_access_filter_drops_probe_lines_only():
    probe = logging.LogRecord("uvicorn.access", logging.INFO, "", 1, "", (), None)
    probe.args = ("10.0.0.5:8000", "GET", "/readyz", "1.1", 503)
    normal = logging.LogRecord("uvicorn.access", logging.INFO, "", 1, "", (), None)
    normal.args = ("10.0.0.5:8000", "GET", "/api/v1/articles", "1.1", 200)
    unknown = logging.LogRecord("uvicorn.access", logging.INFO, "", 1, "", (), None)
    assert not ProbeAccessFilter().filter(probe)
    assert ProbeAccessFilter().filter(normal)
    assert ProbeAccessFilter().filter(unknown)


# --- /readyz --------------------------------------------------------------------------


def test_readyz_answers_the_contract_body_when_every_check_is_up(monkeypatch):
    async def checks():
        return {"database": "up", "cache": "up", "embedding": "up"}

    monkeypatch.setattr(api_main, "_readiness_checks", checks)
    response = _get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "shuttingDown": False,
        "checks": {"database": "up", "cache": "up", "embedding": "up"},
    }


def test_a_down_check_is_503_and_its_cause_stays_out_of_the_body(monkeypatch):
    async def checks():
        return {"database": "down", "cache": "up", "embedding": "up"}

    monkeypatch.setattr(api_main, "_readiness_checks", checks)
    response = _get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["database"] == "down"
    # The endpoint is unauthenticated: the check's cause was logged, not returned.
    assert "refused" not in response.text and "bootstrap pending" not in response.text


def test_draining_answers_503_with_the_fixed_body(monkeypatch):
    monkeypatch.setattr(api_main.shutdown_state, "draining", True)
    response = _get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "error", "shuttingDown": True, "checks": {}}


def test_the_v1_aliases_behave_like_their_targets(monkeypatch):
    async def checks():
        return {"database": "up"}

    monkeypatch.setattr(api_main, "_readiness_checks", checks)
    assert _get("/v1/readyz").status_code == 200
    assert _get("/v1/healthz").json() == {"status": "alive"}


def test_a_hung_dependency_reads_as_down_within_the_budget(monkeypatch):
    monkeypatch.setattr(api_main, "READY_TIMEOUT_S", 0.02)

    async def hanging():
        await asyncio.sleep(2.0)

    started = time.monotonic()
    result = asyncio.run(api_main._readiness_checks(checks=[("database", hanging)]))
    assert result == {"database": "down"}
    assert time.monotonic() - started < 1.0


def test_a_bootstrap_that_has_not_finished_holds_the_database_check(monkeypatch):
    async def answers_but_bootstrap_pending():
        return None

    monkeypatch.setattr(api_main, "database_bootstrapped", False)
    result = asyncio.run(
        api_main._readiness_checks(checks=[("database", answers_but_bootstrap_pending)])
    )
    assert result == {"database": "down"}


# --- the bounded shutdown sequence ------------------------------------------------------


def test_lifespan_shutdown_sets_draining_and_stops_the_inline_work(monkeypatch):
    async def noop():
        return None

    monkeypatch.setattr(api_main, "initialize_resources", noop)
    monkeypatch.setattr(api_main, "shutdown_metrics", lambda: None)
    monkeypatch.setattr(api_main, "shutdown_tracing", lambda: None)
    # Other tests in the run may have left startup state behind; this test
    # owns the lifespan globals so its assertions are about its own work.
    monkeypatch.setattr(api_main, "_bootstrap_task", None)
    monkeypatch.setattr(api_main, "_preload_task", None)
    monkeypatch.setattr(api_main, "_inline_tasks", [])

    async def go():
        state = api_main.shutdown_state
        async with api_main.lifespan(api_main.app):
            assert not state.draining
            work = asyncio.ensure_future(asyncio.sleep(30))
            api_main._inline_tasks.append(work)
        assert state.draining
        assert work.cancelled()

    asyncio.run(go())


def test_a_drain_past_the_deadline_exits_one(monkeypatch):
    exits: list[int] = []
    monkeypatch.setattr(api_main, "hard_exit", lambda code: exits.append(code))
    monkeypatch.setattr(api_main, "MIN_CLEANUP_SECONDS", 0.0)

    async def stubborn():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(0.5)  # work that refuses to stop promptly

    async def go():
        task = asyncio.ensure_future(stubborn())
        await asyncio.sleep(0)  # let it reach its sleep
        monkeypatch.setattr(api_main, "_bootstrap_task", None)
        monkeypatch.setattr(api_main, "_preload_task", None)
        monkeypatch.setattr(api_main, "_inline_tasks", [task])
        state = api_main.shutdown_state
        state.begin(api_main.SHUTDOWN_PLAN, time.monotonic())
        state.deadline = time.monotonic() - 0.01  # the budget is already spent
        try:
            await api_main._run_shutdown_cleanup(state)
        finally:
            task.cancel()

    asyncio.run(go())
    assert exits == [1]


def test_the_api_image_boots_through_the_contract_launcher():
    dockerfile = open("Dockerfile", encoding="utf-8").read()
    assert 'CMD ["python", "-m", "src.api.serve"]' in dockerfile
    serve = open("src/api/serve.py", encoding="utf-8").read()
    # uvicorn's in-flight wait is derived from the same deadline, not a free-
    # floating Dockerfile flag that drifts away from SHUTDOWN_TIMEOUT_MS.
    assert "timeout_graceful_shutdown" in serve
