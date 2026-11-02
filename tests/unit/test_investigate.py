"""The Investigate endpoints: a recent window, compared with the one before it.

`/governance/health-metrics` averages over every request ever recorded, so an outage that began
an hour ago barely moves it. These pin the properties that make the Investigate view usable
during an incident: bounded windows (the API engine times queries out at 5 s), health probes
kept out of the numbers, quiet buckets kept in the series, a verdict that ignores noise, and
nothing sensitive in what comes back.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from src.api import deps
from src.api.routers import investigate as router
from src.domain import investigate as inv
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User

NOW = datetime(2026, 11, 2, 12, 0, 0)


def run(coro):
    return asyncio.run(coro)


def sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


# ── windows ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, 24),
        (0, 1),
        (-5, 1),
        (1, 1),
        (24, 24),
        (168, 168),
        (999, 168),
        ("12", 12),
        ("x", 24),
    ],
)
def test_the_window_is_always_usable_and_never_past_the_cap(requested, expected):
    assert inv.clamp_window(requested) == expected


@pytest.mark.parametrize(
    ("hours", "width"), [(1, 60), (6, 300), (24, 1800), (72, 3600), (168, 10800)]
)
def test_the_bucket_width_keeps_a_chart_readable(hours, width):
    assert inv.bucket_seconds_for(hours) == width
    assert hours * 3600 / width <= inv.MAX_POINTS


def test_every_allowed_window_yields_a_bounded_number_of_points():
    for hours in range(1, inv.MAX_WINDOW_HOURS + 1):
        window = inv.make_window(hours, NOW)
        assert len(inv.fill_series([], window, {"n": 0})) <= inv.MAX_POINTS + 1


def test_the_previous_window_is_the_same_length_immediately_before():
    window = inv.make_window(6, NOW)
    assert window.until - window.since == timedelta(hours=6)
    assert window.since - window.previous_since == timedelta(hours=6)
    assert window.public()["hours"] == 6


# ── series ────────────────────────────────────────────────────────────────────────────────


def test_quiet_buckets_stay_in_the_series_as_zeroes():
    window = inv.make_window(1, NOW)  # 60 s buckets
    busy = inv.bucket_index(NOW - timedelta(minutes=30), window.bucket_seconds)
    series = inv.build_request_series([(busy, 10, 2, 1, 120.0)], window)
    by_time = {point["t"]: point for point in series}
    hit = by_time[inv.bucket_start(busy, 60).isoformat()]
    assert (
        hit["requests"],
        hit["server_errors"],
        hit["client_errors"],
        hit["p95_ms"],
    ) == (10, 2, 1, 120.0)
    quiet = [point for point in series if point is not hit]
    assert quiet and all(p["requests"] == 0 and p["p95_ms"] is None for p in quiet)
    # Consecutive and ordered, so a chart never has to sort or guess.
    stamps = [point["t"] for point in series]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_bucket_math_round_trips():
    index = inv.bucket_index(datetime(2026, 11, 2, 12, 17, 45), 1800)
    assert inv.bucket_start(index, 1800) == datetime(2026, 11, 2, 12, 0, 0)


# ── small helpers ─────────────────────────────────────────────────────────────────────────


def test_percent_and_change_do_not_divide_by_zero():
    assert inv.percent(1, 0) == 0.0
    assert inv.percent(1, 4) == 25.0
    assert inv.pct_change(5.0, 0) is None
    assert inv.pct_change(None, 3.0) is None
    assert inv.pct_change(150.0, 100.0) == 50.0


def test_error_text_is_one_bounded_line():
    assert inv.clean_error_text(None) is None
    assert inv.clean_error_text("  \n ") is None
    assert inv.clean_error_text("boom\n\n  at   line 3") == "boom at line 3"
    long = inv.clean_error_text("x" * 5000)
    assert long is not None and len(long) == inv.ERROR_TEXT_MAX and long.endswith("…")


# ── totals, delta, verdict ────────────────────────────────────────────────────────────────


def test_an_empty_window_reads_as_zeroes_not_as_an_error():
    totals = inv.build_totals(None)
    assert (
        totals["requests"] == 0
        and totals["error_rate"] == 0.0
        and totals["p95_ms"] is None
    )


def test_totals_report_server_and_client_errors_separately():
    totals = inv.build_totals((200, 10, 30, 40.0, 250.5, 900))
    assert totals["error_rate"] == 5.0
    assert (totals["server_errors"], totals["client_errors"]) == (10, 30)
    assert totals["p99_ms"] == 900.0


def test_the_error_rate_moves_in_points_and_only_when_both_windows_had_traffic():
    now, before = inv.build_totals((100, 5, 0, 1, 100, 1)), inv.build_totals(
        (100, 1, 0, 1, 50, 1)
    )
    delta = inv.build_delta(now, before)
    assert delta["error_rate_pp"] == 4.0 and delta["p95_pct"] == 100.0
    assert inv.build_delta(now, inv.build_totals(None))["error_rate_pp"] is None


def _verdict(requests, server, p95=100.0, before=(100, 0, 0, 1, 100, 1)):
    current = inv.build_totals((requests, server, 0, 1, p95, 1))
    return inv.build_verdict(
        current, inv.build_delta(current, inv.build_totals(before))
    )


def test_no_traffic_is_reported_as_such():
    assert _verdict(0, 0) == {"level": "ok", "reasons": ["no_traffic"]}


def test_a_handful_of_requests_never_raises_an_alarm():
    # One failure in four requests is 25%, and means nothing.
    assert _verdict(4, 1)["level"] == "ok"


def test_error_rate_thresholds():
    assert _verdict(1000, 5)["level"] == "ok"  # 0.5%
    warning = _verdict(1000, 15, before=(1000, 14, 0, 1, 100, 1))  # 1.5%
    assert warning["level"] == "warning" and warning["reasons"] == ["error_rate_high"]
    critical = _verdict(1000, 60, before=(1000, 55, 0, 1, 100, 1))  # 6%
    assert critical["level"] == "critical" and critical["reasons"] == [
        "error_rate_critical"
    ]


def test_a_sharp_rise_is_a_warning_even_below_the_absolute_threshold():
    # 0% -> 0.9% is under 1% but 0.9 points up; 0.1% -> 2.5% is a clear rise.
    result = _verdict(1000, 25, before=(1000, 1, 0, 1, 100, 1))
    assert "error_rate_rising" in result["reasons"] and result["level"] != "ok"


def test_slow_responses_are_a_warning():
    result = _verdict(1000, 0, p95=4500.0)
    assert result == {"level": "warning", "reasons": ["latency_high"]}


def test_critical_is_not_downgraded_by_a_later_warning():
    result = _verdict(1000, 80, p95=9000.0, before=(1000, 1, 0, 1, 100, 1))
    assert result["level"] == "critical"
    assert {"error_rate_critical", "error_rate_rising", "latency_high"} <= set(
        result["reasons"]
    )


# ── shaping ───────────────────────────────────────────────────────────────────────────────


def test_failure_groups_and_slow_routes_are_shaped_for_the_interface():
    first, last = NOW - timedelta(hours=2), NOW - timedelta(minutes=1)
    groups = inv.build_failure_groups(
        [("POST", "/upload-source", 500, "RuntimeError", 7, first, last, "req-9")]
    )
    assert groups == [
        {
            "method": "POST",
            "path": "/upload-source",
            "status_code": 500,
            "error_type": "RuntimeError",
            "count": 7,
            "first_seen": first.isoformat(),
            "last_seen": last.isoformat(),
            "sample_request_id": "req-9",
        }
    ]
    slow = inv.build_slow_routes([("GET", "/search", 40, 2100.0, 800.0, 4)])
    assert slow[0]["error_rate"] == 10.0 and slow[0]["p95_ms"] == 2100.0


def test_ai_figures_are_shaped_and_a_cold_start_is_not_an_error():
    assert inv.build_ai_totals(None)["cache_hit_rate"] == 0.0
    totals = inv.build_ai_totals((50, 12000, 800.0, 2400.0, 10))
    assert totals["cache_hit_rate"] == 20.0 and totals["tokens"] == 12000
    models = inv.build_ai_models([("glm-4.5", 30, 9000, 900.0), ("cache", 20, 0, 3.0)])
    assert [m["model"] for m in models] == ["glm-4.5", "cache"]


# ── background work ───────────────────────────────────────────────────────────────────────


def _pipeline(**override):
    base = {
        "indexing": {"failed": 0},
        "dead_letters": {"total": 0},
        "outbox": {"dead": 0, "failed": 0},
        "notifications": {"dead": 0, "failed": 0},
        "connector_jobs": {"failed_recent": 0},
        "reprocess_jobs": {"failed": 0},
    }
    base.update(override)
    return base


def test_nothing_failing_means_nothing_needs_attention():
    assert inv.build_attention(_pipeline()) == {"level": "ok", "codes": []}


def test_each_failing_area_is_named():
    attention = inv.build_attention(
        _pipeline(
            indexing={"failed": 2},
            dead_letters={"total": 1},
            outbox={"dead": 1, "failed": 0},
            notifications={"dead": 0, "failed": 3},
            connector_jobs={"failed_recent": 1},
            reprocess_jobs={"failed": 1},
        )
    )
    assert attention["level"] == "warning"
    assert attention["codes"] == [
        "indexing_failed",
        "dead_letters",
        "outbox_stuck",
        "notifications_failing",
        "connector_jobs_failed",
        "reprocess_failed",
    ]


def test_failed_articles_hide_titles_unless_the_reader_may_see_them():
    article = uuid.uuid4()
    rows = [
        (article, "Confidential merger plan", "Legal", "Embedding unavailable", NOW)
    ]
    hidden = inv.build_indexing([("failed", 1)], None, rows, include_titles=False)
    shown = inv.build_indexing([("failed", 1)], None, rows, include_titles=True)
    assert hidden["failed_articles"][0]["title"] is None
    assert hidden["failed_articles"][0]["article_id"] == str(article)
    assert shown["failed_articles"][0]["title"] == "Confidential merger plan"
    assert "Confidential" not in repr(hidden)


def test_dead_letters_return_the_error_but_never_the_payload():
    built = inv.build_dead_letters(
        [("ingestion", 3), ("connectors", 1)],
        [(uuid.uuid4(), "ingestion", "Embedding unavailable for 2 chunk(s)", NOW)],
    )
    assert built["total"] == 4 and built["recent"][0]["error"].startswith("Embedding")
    assert all("payload" not in item for item in built["recent"])


def test_missing_states_count_as_zero():
    assert inv.build_queue_counts([("failed", 2)], ("pending", "failed", "dead")) == {
        "pending": 0,
        "failed": 2,
        "dead": 0,
    }


# ── SQL ───────────────────────────────────────────────────────────────────────────────────


def test_health_probes_are_excluded_from_every_request_statement():
    window = inv.make_window(24, NOW)
    for statement in (
        inv.totals_statement(window.since, window.until),
        inv.series_statement(window),
        inv.failure_groups_statement(window.since, window.until),
        inv.slow_routes_statement(window.since, window.until),
    ):
        text = sql(statement)
        assert text.count("NOT LIKE") == len(inv.PROBE_PATH_PREFIXES)
        assert "created_at >=" in text and "created_at <" in text


def test_the_bucket_width_is_inlined_so_select_and_group_by_agree():
    window = inv.make_window(24, NOW)
    text = sql(inv.series_statement(window))
    # A bound width would be `$1` in SELECT and `$2` in GROUP BY, which Postgres rejects.
    assert "CAST(1800 AS NUMERIC)" in text
    assert text.count("floor(EXTRACT(epoch FROM api_request_metrics.created_at)") >= 2
    assert "GROUP BY floor(" in text


def test_failure_groups_are_server_errors_ranked_and_capped():
    window = inv.make_window(24, NOW)
    text = sql(inv.failure_groups_statement(window.since, window.until))
    assert "status_code >=" in text and "GROUP BY" in text and "DESC" in text
    assert f"LIMIT %(param_1)s" in text or "LIMIT" in text
    assert "array_agg(api_request_metrics.request_id ORDER BY" in text


def test_slow_routes_need_a_minimum_sample():
    window = inv.make_window(24, NOW)
    statement = inv.slow_routes_statement(window.since, window.until)
    assert "HAVING count(api_request_metrics.id) >=" in sql(statement)
    assert (
        inv.MIN_SLOW_ROUTE_REQUESTS
        in statement.compile(dialect=postgresql.dialect()).params.values()
    )


def test_only_live_published_articles_are_counted_for_indexing():
    for statement in (inv.indexing_counts_statement(), inv.indexing_failed_statement()):
        text = sql(statement)
        assert "articles.status =" in text and "articles.lifecycle_status =" in text


# ── loading, against a scripted database ──────────────────────────────────────────────────


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def one_or_none(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return list(self.rows)

    def scalar_one_or_none(self):
        return self.rows[0][0] if self.rows else None


class _ScriptedDb:
    """Answers each statement in turn, and checks it is the statement that was expected."""

    def __init__(self, script):
        self.script = list(script)
        self.seen: list[str] = []

    async def execute(self, statement):
        text = sql(statement)
        self.seen.append(text)
        marker, rows = self.script.pop(0)
        assert (
            marker in text
        ), f"expected a statement containing {marker!r}, got:\n{text}"
        return _Result(rows)


def test_requests_are_loaded_in_a_fixed_order_and_assembled():
    window = inv.make_window(24, NOW)
    bucket = inv.bucket_index(NOW - timedelta(hours=1), window.bucket_seconds)
    db = _ScriptedDb(
        [
            ("percentile_cont", [(1000, 60, 20, 30.0, 400.0, 900.0)]),  # current
            ("percentile_cont", [(1000, 5, 20, 30.0, 300.0, 800.0)]),  # previous
            ("GROUP BY floor(", [(bucket, 100, 10, 2, 350.0)]),
            ("array_agg", [("POST", "/x", 500, "ValueError", 40, NOW, NOW, "req-1")]),
            ("HAVING", [("GET", "/search", 50, 2500.0, 900.0, 2)]),
        ]
    )
    payload = run(inv.load_requests(db, window, 30))
    assert db.script == []
    assert (
        payload["current"]["error_rate"] == 6.0
        and payload["previous"]["error_rate"] == 0.5
    )
    assert payload["delta"]["error_rate_pp"] == 5.5
    assert payload["verdict"]["level"] == "critical"
    assert payload["retention_days"] == 30 and payload["excludes"] == list(
        inv.PROBE_PATH_PREFIXES
    )
    assert sum(p["requests"] for p in payload["series"]) == 100
    assert payload["failure_groups"][0]["sample_request_id"] == "req-1"
    assert payload["slow_routes"][0]["path"] == "/search"


def test_a_system_with_no_telemetry_loads_cleanly():
    window = inv.make_window(24, NOW)
    db = _ScriptedDb(
        [
            ("percentile_cont", []),
            ("percentile_cont", []),
            ("GROUP BY floor(", []),
            ("array_agg", []),
            ("HAVING", []),
        ]
    )
    payload = run(inv.load_requests(db, window, 30))
    assert payload["current"]["requests"] == 0
    assert payload["verdict"] == {"level": "ok", "reasons": ["no_traffic"]}
    assert payload["failure_groups"] == [] and payload["slow_routes"] == []


def test_the_pipeline_is_loaded_and_assembled():
    db = _ScriptedDb(
        [
            ("index_status", [("failed", 2), ("pending", 3)]),
            ("min(articles.updated_at)", [(NOW - timedelta(hours=3),)]),
            (
                "articles.title",
                [(uuid.uuid4(), "Doc", "HR", "Embedding unavailable", NOW)],
            ),
            ("outbox_events", [("dead", 1), ("pending", 4)]),
            ("min(outbox_events.created_at)", [(NOW,)]),
            ("dead_letter_jobs.source_queue, count", [("ingestion", 2)]),
            ("dead_letter_jobs.id", [(uuid.uuid4(), "ingestion", "boom", NOW)]),
            ("notification_queue", [("failed", 5)]),
            ("connector_jobs.status, count", [("running", 1), ("failed", 2)]),
            (
                "connectors.name",
                [(uuid.uuid4(), "SharePoint", "401 Unauthorized", NOW)],
            ),
            ("index_reprocess_jobs", [("running", 1)]),
        ]
    )
    pipeline = run(inv.load_pipeline(db, now=NOW, include_titles=False))
    assert db.script == []
    assert pipeline["indexing"]["failed"] == 2 and pipeline["indexing"]["pending"] == 3
    assert pipeline["indexing"]["failed_articles"][0]["title"] is None
    assert pipeline["outbox"] == {
        "pending": 4,
        "processing": 0,
        "failed": 0,
        "dead": 1,
        "oldest_waiting_at": NOW.isoformat(),
    }
    assert pipeline["notifications"]["failed"] == 5
    assert pipeline["connector_jobs"]["failed_recent"] == 2
    assert pipeline["reprocess_jobs"]["running"] == 1
    assert pipeline["attention"]["level"] == "warning"
    assert "indexing_failed" in pipeline["attention"]["codes"]


# ── access ────────────────────────────────────────────────────────────────────────────────


def _user(scope: str | None, permission: str = "governance.read") -> User:
    user = User(
        id=uuid.uuid4(),
        email="o@acme.test",
        name="Ops",
        company_domain="acme.test",
        role="Ops",
        active=True,
    )
    role = Role(name="Ops", company_domain="acme.test", active=True)
    if scope:
        role.permissions.append(
            RolePermission(
                permission=Permission(key=permission, name=permission), scope=scope
            )
        )
    user.roles.append(role)
    return user


def test_every_investigate_route_requires_global_governance_access():
    assert router.router.routes
    for route in router.router.routes:
        calls = [dep.call for dep in route.dependant.dependencies]
        assert any(
            getattr(call, "__name__", "") == "permission_checker" for call in calls
        ), f"{route.path} is not behind require_permission"
    checker = deps.require_permission("governance.read", scope="global")
    with pytest.raises(HTTPException) as denied:
        checker(_user("company"))
    assert denied.value.status_code == 403
    with pytest.raises(HTTPException):
        checker(_user(None))
    assert checker(_user("global")) is not None


def test_the_endpoints_pass_a_bounded_window_through():
    captured = {}

    async def fake_load(db, window, retention):
        captured["hours"], captured["retention"] = window.hours, retention
        return {"ok": True}

    original = router.investigate.load_requests
    router.investigate.load_requests = fake_load
    try:
        result = run(
            router.investigate_requests(
                window_hours=9999, current_user=_user("global"), db=SimpleNamespace()
            )
        )
    finally:
        router.investigate.load_requests = original
    assert result == {"ok": True}
    # The Query() bound rejects 9999 before this runs over HTTP; the domain clamps it as well.
    assert captured["hours"] == inv.MAX_WINDOW_HOURS
