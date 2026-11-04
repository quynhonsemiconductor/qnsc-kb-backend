"""The Investigate statements, executed against a real, migrated PostgreSQL.

The unit tests compile these statements and shape their results, but a compiled statement is
not a proven one: Postgres rejects things SQLAlchemy happily renders (a bound parameter that
differs between SELECT and GROUP BY, a window function in the wrong place), and a renamed
column only shows up when something runs the query. This runs every statement for real.

Rows are written in the year 2001 and queried through a window placed there, so the test
neither sees nor disturbs anything another test left in the same database, and it deletes
what it wrote.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.domain import investigate as inv
from src.models.ai import AiUsageLog
from src.models.ops import ApiRequestMetric

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs a real PostgreSQL; the point is that Postgres accepts these statements",
)

#: Noon on 2001-01-01. Nothing else writes telemetry with this timestamp.
NOON = datetime(2001, 1, 1, 12, 0, 0)


@pytest_asyncio.fixture
async def session():
    from src.api.deps import set_database_context

    engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as db:
        await set_database_context(db, None, True)
        try:
            yield db
        finally:
            await db.rollback()
            await db.execute(
                delete(ApiRequestMetric).where(
                    ApiRequestMetric.request_id.like("it-inv-%")
                )
            )
            await db.execute(
                delete(AiUsageLog).where(AiUsageLog.llm_model.like("it-inv-%"))
            )
            await db.commit()
    await engine.dispose()


def _metric(
    n: int,
    method: str,
    path: str,
    status: int,
    ms: float,
    *,
    minute: int,
    error: str | None = None,
):
    return ApiRequestMetric(
        request_id=f"it-inv-{n}",
        method=method,
        path=path,
        status_code=status,
        duration_ms=ms,
        error_type=error,
        error_detail=f"{error}: boom" if error else None,
        created_at=datetime(2001, 1, 1, 10, minute, 0),
    )


@pytest.mark.asyncio
async def test_request_statements_run_and_count_what_was_written(session):
    rows = [
        # ten fast successes on one route
        *[_metric(i, "GET", "/it-inv-a", 200, 100.0 + i, minute=i) for i in range(10)],
        # two crashes on another
        _metric(20, "POST", "/it-inv-b", 500, 900.0, minute=20, error="ValueError"),
        _metric(21, "POST", "/it-inv-b", 500, 950.0, minute=21, error="ValueError"),
        # a user error, and a health probe that must not be counted
        _metric(30, "GET", "/it-inv-c", 404, 5.0, minute=30),
        _metric(31, "GET", "/health/ready", 200, 1.0, minute=31),
    ]
    session.add_all(rows)
    await session.commit()

    window = inv.make_window(24, NOON)
    payload = await inv.load_requests(session, window, 30)

    current = payload["current"]
    assert current["requests"] == 13  # the probe is excluded
    assert current["server_errors"] == 2
    assert current["client_errors"] == 1
    assert current["error_rate"] == round(2 / 13 * 100, 2)
    assert current["p95_ms"] is not None and current["p95_ms"] >= 900.0
    assert payload["previous"]["requests"] == 0
    assert payload["delta"]["error_rate_pp"] is None  # nothing before to compare with

    assert sum(point["requests"] for point in payload["series"]) == 13
    assert sum(point["server_errors"] for point in payload["series"]) == 2

    group = payload["failure_groups"][0]
    assert (group["method"], group["path"], group["status_code"]) == (
        "POST",
        "/it-inv-b",
        500,
    )
    assert group["error_type"] == "ValueError" and group["count"] == 2
    # The newest failure in the group is the one offered as the sample.
    assert group["sample_request_id"] == "it-inv-21"

    # Only routes with enough traffic are ranked as slow.
    slow_paths = [route["path"] for route in payload["slow_routes"]]
    assert "/it-inv-a" in slow_paths and "/it-inv-b" not in slow_paths


@pytest.mark.asyncio
async def test_the_previous_window_is_compared(session):
    session.add_all(
        [
            # previous window: 2000-12-31 12:00 -> 2001-01-01 12:00 is the current one, so go
            # back one more day for the earlier traffic.
            *[
                ApiRequestMetric(
                    request_id=f"it-inv-p{i}",
                    method="GET",
                    path="/it-inv-prev",
                    status_code=200,
                    duration_ms=50.0,
                    created_at=datetime(2000, 12, 31, 6, i, 0),
                )
                for i in range(5)
            ],
            _metric(40, "GET", "/it-inv-now", 200, 60.0, minute=1),
        ]
    )
    await session.commit()
    payload = await inv.load_requests(session, inv.make_window(24, NOON), 30)
    assert payload["previous"]["requests"] == 5
    assert payload["current"]["requests"] == 1
    assert payload["delta"]["requests_pct"] == -80.0


@pytest.mark.asyncio
async def test_ai_statements_run_and_count_what_was_written(session):
    def log(n: int, model: str, tokens: int, ms: int, prompt: str = "v1") -> AiUsageLog:
        return AiUsageLog(
            question="[redacted]",
            answer="[redacted]",
            tokens_used=tokens,
            latency_ms=ms,
            prompt_version=prompt,
            llm_model=model,
            retrieval_version="r1",
            reranker_version="x",
            created_at=datetime(2001, 1, 1, 9, n, 0),
        )

    session.add_all(
        [
            log(1, "it-inv-model", 1000, 800),
            log(2, "it-inv-model", 2000, 1200),
            log(3, "it-inv-cache", 0, 3, prompt="cached"),
        ]
    )
    await session.commit()
    payload = await inv.load_ai(session, inv.make_window(24, NOON))
    assert payload["current"]["requests"] == 3
    assert payload["current"]["tokens"] == 3000
    assert payload["current"]["cache_hit_rate"] == round(1 / 3 * 100, 2)
    assert sum(point["requests"] for point in payload["series"]) == 3
    models = {item["model"]: item for item in payload["models"]}
    assert (
        models["it-inv-model"]["requests"] == 2
        and models["it-inv-model"]["tokens"] == 3000
    )


@pytest.mark.asyncio
async def test_every_pipeline_statement_runs_on_the_migrated_schema(session):
    pipeline = await inv.load_pipeline(session, include_titles=True)
    for key in (
        "indexing",
        "dead_letters",
        "outbox",
        "notifications",
        "connector_jobs",
        "reprocess_jobs",
        "attention",
    ):
        assert key in pipeline
    assert pipeline["attention"]["level"] in {"ok", "warning"}
    assert isinstance(pipeline["indexing"]["failed_articles"], list)


@pytest.mark.asyncio
async def test_request_failures_can_be_narrowed_to_one_group(session):
    from src.api.routers.governance import get_request_failures

    session.add_all(
        [
            _metric(50, "POST", "/it-inv_x", 500, 10.0, minute=1, error="ValueError"),
            _metric(51, "POST", "/it-invZx", 500, 10.0, minute=2, error="ValueError"),
            _metric(52, "GET", "/it-inv_x", 500, 10.0, minute=3, error="KeyError"),
            _metric(53, "POST", "/it-inv_x", 500, 10.0, minute=4, error="ValueError"),
        ]
    )
    await session.commit()

    async def failures(**overrides):
        arguments = dict(
            limit=50,
            path=None,
            request_id=None,
            min_status=500,
            since=datetime(2001, 1, 1),
            until=None,
            route=None,
            method=None,
            error_type=None,
            current_user=None,
            db=session,
        )
        arguments.update(overrides)
        payload = await get_request_failures(**arguments)
        return {item["request_id"] for item in payload["failures"]}

    # `route` is an exact match. `path` is a substring ILIKE where `_` is a wildcard, so it
    # also finds "/it-invZx" -- the reason drill-down uses `route` and not `path`.
    assert await failures(route="/it-inv_x") == {"it-inv-50", "it-inv-52", "it-inv-53"}
    assert await failures(path="/it-inv_x") >= {"it-inv-50", "it-inv-51", "it-inv-52"}
    assert await failures(route="/it-inv_x", method="post") == {
        "it-inv-50",
        "it-inv-53",
    }
    assert await failures(route="/it-inv_x", error_type="KeyError") == {"it-inv-52"}
    # `until` is exclusive: the newest row sits at 10:04.
    assert await failures(route="/it-inv_x", until=datetime(2001, 1, 1, 10, 4, 0)) == {
        "it-inv-50",
        "it-inv-52",
    }
