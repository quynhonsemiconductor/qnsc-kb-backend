"""Windowed operational telemetry behind the Investigate area of the health dashboard.

`/governance/health-metrics` answers "what is the all-time average": its error rate and p95 are
taken over every request ever recorded, so an outage that began an hour ago moves them by a
fraction of a percent and the dashboard reads healthy. An operator investigating needs the
opposite: what is failing NOW, compared with what was normal, and which route, exception or
background job to look at first. This module computes that, over a bounded recent window.

Three things are deliberate:

* Every query is bounded by a window of at most `MAX_WINDOW_HOURS`. The shared API engine
  enforces a 5 s client-side `command_timeout` (api/deps.py), so an unbounded aggregate over a
  growing telemetry table is a request that eventually fails for the one person who needs it.
* Health probes (`/health/*`, `/livez`, `/metrics`) are excluded. A load balancer polling every
  few seconds is most of the traffic on a quiet system and would dilute the error rate and
  bury real routes in the slow list.
* Nothing here returns request bodies, task arguments, document text or user identities. Error
  messages are truncated to `ERROR_TEXT_MAX`, because a stored failure message can quote the
  input that caused it.

SQL is built by small `*_statement` functions and shaped by pure `build_*` functions, so the
statements can be compiled and inspected, and the shaping tested, without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from sqlalchemy import (
    Select,
    case,
    desc,
    extract,
    func,
    literal_column,
    select,
)
from sqlalchemy.dialects.postgresql import array_agg, aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai import AiUsageLog
from src.models.article import Article
from src.models.ops import (
    ApiRequestMetric,
    Connector,
    ConnectorJob,
    DeadLetterJob,
    IndexReprocessJob,
    NotificationQueue,
    OutboxEvent,
)

MAX_WINDOW_HOURS = 168
#: Candidate bucket widths in seconds. The smallest one that keeps a window to at most
#: `MAX_POINTS` points is used, so a chart always has a readable number of bars.
BUCKET_CANDIDATES = (60, 300, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)
MAX_POINTS = 72
TOP_N = 10
RECENT_N = 5
#: A route needs this many requests in the window before its p95 means anything.
MIN_SLOW_ROUTE_REQUESTS = 5
ERROR_TEXT_MAX = 300
#: Routes that exist for infrastructure, not users.
PROBE_PATH_PREFIXES = ("/health", "/livez", "/metrics")

# Thresholds for the overall verdict. Rates are percentages of server errors (5xx).
CRITICAL_ERROR_RATE = 5.0
WARNING_ERROR_RATE = 1.0
WARNING_P95_MS = 3000.0
WARNING_ERROR_RATE_RISE_PP = 2.0
#: Below this many requests a rate is noise: one failure in four requests is 25%.
MIN_REQUESTS_FOR_VERDICT = 20


# ── pure helpers ──────────────────────────────────────────────────────────────────────────


def clamp_window(hours: int | None) -> int:
    """A usable window in whole hours: 1 to `MAX_WINDOW_HOURS`, defaulting to 24."""
    try:
        value = int(hours) if hours is not None else 24
    except (TypeError, ValueError):
        value = 24
    return max(1, min(value, MAX_WINDOW_HOURS))


def bucket_seconds_for(hours: int) -> int:
    """The smallest bucket width that keeps a `hours` window to `MAX_POINTS` points or fewer."""
    total = max(1, hours) * 3600
    for width in BUCKET_CANDIDATES:
        if total / width <= MAX_POINTS:
            return width
    return BUCKET_CANDIDATES[-1]


@dataclass(frozen=True)
class Window:
    hours: int
    since: datetime
    until: datetime
    previous_since: datetime
    bucket_seconds: int

    def public(self) -> dict[str, Any]:
        return {
            "hours": self.hours,
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "bucket_seconds": self.bucket_seconds,
        }


def make_window(hours: int | None, now: datetime | None = None) -> Window:
    """The current window ending now, and the start of the equally long one before it."""
    span = clamp_window(hours)
    until = now or datetime.utcnow()
    since = until - timedelta(hours=span)
    return Window(
        hours=span,
        since=since,
        until=until,
        previous_since=since - timedelta(hours=span),
        bucket_seconds=bucket_seconds_for(span),
    )


def _epoch(moment: datetime) -> float:
    """Seconds since the epoch of a naive UTC timestamp, which is what the columns hold."""
    return (moment - datetime(1970, 1, 1)).total_seconds()


def bucket_index(moment: datetime, width: int) -> int:
    return int(_epoch(moment) // width)


def bucket_start(index: int, width: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(seconds=index * width)


def fill_series(
    rows: Iterable[tuple[int, dict[str, Any]]],
    window: Window,
    empty: dict[str, Any],
) -> list[dict[str, Any]]:
    """One point per bucket across the window, with `empty` where nothing happened.

    A chart that skips quiet buckets draws a line between the busy ones and reads as steady
    traffic; a quiet bucket is itself the signal (nothing arrived), so it is kept.
    """
    by_index = {int(index): values for index, values in rows}
    first = bucket_index(window.since, window.bucket_seconds)
    last = bucket_index(window.until, window.bucket_seconds)
    series: list[dict[str, Any]] = []
    for index in range(first, last + 1):
        point = {"t": bucket_start(index, window.bucket_seconds).isoformat()}
        point.update(by_index.get(index, empty))
        series.append(point)
    return series


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _int(value: Any) -> int:
    return int(value or 0)


def percent(part: int | float, whole: int | float) -> float:
    return round(part / whole * 100.0, 2) if whole else 0.0


def pct_change(current: float | None, previous: float | None) -> float | None:
    """Relative change in percent, or None when there is nothing to compare against."""
    if current is None or previous is None or previous == 0:
        return None
    return round((current - previous) / previous * 100.0, 1)


def clean_error_text(value: Any) -> str | None:
    """A stored error message, collapsed to one bounded line. Messages can quote their input."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text if len(text) <= ERROR_TEXT_MAX else text[: ERROR_TEXT_MAX - 1] + "…"


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment else None


# ── request telemetry ─────────────────────────────────────────────────────────────────────


def _not_probe() -> list[Any]:
    return [~ApiRequestMetric.path.startswith(prefix) for prefix in PROBE_PATH_PREFIXES]


def request_conditions(since: datetime, until: datetime) -> list[Any]:
    return [
        ApiRequestMetric.created_at >= since,
        ApiRequestMetric.created_at < until,
        *_not_probe(),
    ]


def _count_where(condition: Any) -> Any:
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def _is_server_error() -> Any:
    return ApiRequestMetric.status_code >= 500


def _is_client_error() -> Any:
    return (ApiRequestMetric.status_code >= 400) & (ApiRequestMetric.status_code < 500)


def _percentile(fraction: float, column: Any) -> Any:
    return func.percentile_cont(fraction).within_group(column)


def totals_statement(since: datetime, until: datetime) -> Select[Any]:
    duration = ApiRequestMetric.duration_ms
    return select(
        func.count(ApiRequestMetric.id),
        _count_where(_is_server_error()),
        _count_where(_is_client_error()),
        _percentile(0.5, duration),
        _percentile(0.95, duration),
        _percentile(0.99, duration),
    ).where(*request_conditions(since, until))


def series_statement(window: Window) -> Select[Any]:
    # The width is inlined as a literal, not bound: Postgres treats `$1` in the SELECT list and
    # `$2` in GROUP BY as different expressions and rejects the query as ungrouped.
    width: Any = literal_column(str(int(window.bucket_seconds)))
    index = func.floor(extract("epoch", ApiRequestMetric.created_at) / width)
    return (
        select(
            index.label("bucket"),
            func.count(ApiRequestMetric.id),
            _count_where(_is_server_error()),
            _count_where(_is_client_error()),
            _percentile(0.95, ApiRequestMetric.duration_ms),
        )
        .where(*request_conditions(window.since, window.until))
        .group_by(index)
        .order_by(index)
    )


def failure_groups_statement(
    since: datetime, until: datetime, min_status: int = 500
) -> Select[Any]:
    """Failures grouped by what failed and how, newest-and-most-frequent first."""
    count = func.count(ApiRequestMetric.id)
    last_seen = func.max(ApiRequestMetric.created_at)
    latest_request = array_agg(
        aggregate_order_by(
            ApiRequestMetric.request_id, ApiRequestMetric.created_at.desc()
        )
    )
    return (
        select(
            ApiRequestMetric.method,
            ApiRequestMetric.path,
            ApiRequestMetric.status_code,
            ApiRequestMetric.error_type,
            count,
            func.min(ApiRequestMetric.created_at),
            last_seen,
            latest_request[1],
        )
        .where(
            *request_conditions(since, until),
            ApiRequestMetric.status_code >= min_status,
        )
        .group_by(
            ApiRequestMetric.method,
            ApiRequestMetric.path,
            ApiRequestMetric.status_code,
            ApiRequestMetric.error_type,
        )
        .order_by(desc(count), desc(last_seen))
        .limit(TOP_N)
    )


def slow_routes_statement(since: datetime, until: datetime) -> Select[Any]:
    count = func.count(ApiRequestMetric.id)
    p95 = _percentile(0.95, ApiRequestMetric.duration_ms)
    return (
        select(
            ApiRequestMetric.method,
            ApiRequestMetric.path,
            count,
            p95,
            func.avg(ApiRequestMetric.duration_ms),
            _count_where(_is_server_error()),
        )
        .where(*request_conditions(since, until))
        .group_by(ApiRequestMetric.method, ApiRequestMetric.path)
        .having(count >= MIN_SLOW_ROUTE_REQUESTS)
        .order_by(desc(p95))
        .limit(TOP_N)
    )


def build_totals(row: Sequence[Any] | None) -> dict[str, Any]:
    requests, server, client, p50, p95, p99 = row or (0, 0, 0, None, None, None)
    requests, server, client = _int(requests), _int(server), _int(client)
    return {
        "requests": requests,
        "server_errors": server,
        "client_errors": client,
        "error_rate": percent(server, requests),
        "p50_ms": _number(p50),
        "p95_ms": _number(p95),
        "p99_ms": _number(p99),
    }


def build_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """How the current window moved against the one before it.

    Percent changes are None when the earlier window had no traffic to compare with. The error
    rate moves in percentage POINTS, because "0.5% to 1%" is a 100% rise and a shrug.
    """
    comparable = previous["requests"] > 0 and current["requests"] > 0
    return {
        "requests_pct": pct_change(current["requests"], previous["requests"]),
        "error_rate_pp": (
            round(current["error_rate"] - previous["error_rate"], 2)
            if comparable
            else None
        ),
        "p95_pct": pct_change(current["p95_ms"], previous["p95_ms"]),
    }


def build_verdict(current: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """One level and the reasons for it, as codes the interface turns into wording."""
    reasons: list[str] = []
    level = "ok"
    if current["requests"] >= MIN_REQUESTS_FOR_VERDICT:
        rate = current["error_rate"]
        if rate >= CRITICAL_ERROR_RATE:
            level, reasons = "critical", ["error_rate_critical"]
        elif rate >= WARNING_ERROR_RATE:
            level, reasons = "warning", ["error_rate_high"]
        rise = delta.get("error_rate_pp")
        if rise is not None and rise >= WARNING_ERROR_RATE_RISE_PP:
            reasons.append("error_rate_rising")
            level = "critical" if level == "critical" else "warning"
        if (current["p95_ms"] or 0.0) >= WARNING_P95_MS:
            reasons.append("latency_high")
            level = "critical" if level == "critical" else "warning"
    elif current["requests"] == 0:
        reasons = ["no_traffic"]
    return {"level": level, "reasons": reasons}


def build_failure_groups(rows: Iterable[Sequence[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "path": path,
            "status_code": _int(status),
            "error_type": error_type,
            "count": _int(count),
            "first_seen": _iso(first),
            "last_seen": _iso(last),
            "sample_request_id": sample,
        }
        for method, path, status, error_type, count, first, last, sample in rows
    ]


def build_slow_routes(rows: Iterable[Sequence[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "path": path,
            "requests": _int(count),
            "p95_ms": _number(p95),
            "avg_ms": _number(average),
            "error_rate": percent(_int(server), _int(count)),
        }
        for method, path, count, p95, average, server in rows
    ]


def build_request_series(
    rows: Iterable[Sequence[Any]], window: Window
) -> list[dict[str, Any]]:
    empty = {"requests": 0, "server_errors": 0, "client_errors": 0, "p95_ms": None}
    shaped = [
        (
            int(bucket),
            {
                "requests": _int(count),
                "server_errors": _int(server),
                "client_errors": _int(client),
                "p95_ms": _number(p95),
            },
        )
        for bucket, count, server, client, p95 in rows
    ]
    return fill_series(shaped, window, empty)


async def load_requests(
    db: AsyncSession, window: Window, retention_days: int
) -> dict[str, Any]:
    current = build_totals(
        (await db.execute(totals_statement(window.since, window.until))).one_or_none()
    )
    previous = build_totals(
        (
            await db.execute(totals_statement(window.previous_since, window.since))
        ).one_or_none()
    )
    series_rows = (await db.execute(series_statement(window))).all()
    group_rows = (
        await db.execute(failure_groups_statement(window.since, window.until))
    ).all()
    slow_rows = (
        await db.execute(slow_routes_statement(window.since, window.until))
    ).all()
    delta = build_delta(current, previous)
    return {
        "window": window.public(),
        "excludes": list(PROBE_PATH_PREFIXES),
        "retention_days": retention_days,
        "current": current,
        "previous": previous,
        "delta": delta,
        "verdict": build_verdict(current, delta),
        "series": build_request_series(series_rows, window),
        "failure_groups": build_failure_groups(group_rows),
        "slow_routes": build_slow_routes(slow_rows),
    }


# ── AI usage ──────────────────────────────────────────────────────────────────────────────


def ai_conditions(since: datetime, until: datetime) -> list[Any]:
    return [AiUsageLog.created_at >= since, AiUsageLog.created_at < until]


def ai_totals_statement(since: datetime, until: datetime) -> Select[Any]:
    return select(
        func.count(AiUsageLog.id),
        func.coalesce(func.sum(AiUsageLog.tokens_used), 0),
        func.avg(AiUsageLog.latency_ms),
        _percentile(0.95, AiUsageLog.latency_ms),
        func.coalesce(
            func.sum(case((AiUsageLog.prompt_version == "cached", 1), else_=0)), 0
        ),
    ).where(*ai_conditions(since, until))


def ai_series_statement(window: Window) -> Select[Any]:
    width: Any = literal_column(str(int(window.bucket_seconds)))
    index = func.floor(extract("epoch", AiUsageLog.created_at) / width)
    return (
        select(
            index.label("bucket"),
            func.count(AiUsageLog.id),
            func.coalesce(func.sum(AiUsageLog.tokens_used), 0),
            func.avg(AiUsageLog.latency_ms),
        )
        .where(*ai_conditions(window.since, window.until))
        .group_by(index)
        .order_by(index)
    )


def ai_models_statement(since: datetime, until: datetime) -> Select[Any]:
    count = func.count(AiUsageLog.id)
    return (
        select(
            AiUsageLog.llm_model,
            count,
            func.coalesce(func.sum(AiUsageLog.tokens_used), 0),
            func.avg(AiUsageLog.latency_ms),
        )
        .where(*ai_conditions(since, until))
        .group_by(AiUsageLog.llm_model)
        .order_by(desc(count))
        .limit(TOP_N)
    )


def build_ai_totals(row: Sequence[Any] | None) -> dict[str, Any]:
    requests, tokens, average, p95, cached = row or (0, 0, None, None, 0)
    requests = _int(requests)
    return {
        "requests": requests,
        "tokens": _int(tokens),
        "avg_latency_ms": _number(average),
        "p95_latency_ms": _number(p95),
        "cache_hit_rate": percent(_int(cached), requests),
    }


def build_ai_series(
    rows: Iterable[Sequence[Any]], window: Window
) -> list[dict[str, Any]]:
    empty = {"requests": 0, "tokens": 0, "avg_latency_ms": None}
    shaped = [
        (
            int(bucket),
            {
                "requests": _int(count),
                "tokens": _int(tokens),
                "avg_latency_ms": _number(average),
            },
        )
        for bucket, count, tokens, average in rows
    ]
    return fill_series(shaped, window, empty)


def build_ai_models(rows: Iterable[Sequence[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "requests": _int(count),
            "tokens": _int(tokens),
            "avg_latency_ms": _number(average),
        }
        for model, count, tokens, average in rows
    ]


async def load_ai(db: AsyncSession, window: Window) -> dict[str, Any]:
    current = build_ai_totals(
        (
            await db.execute(ai_totals_statement(window.since, window.until))
        ).one_or_none()
    )
    previous = build_ai_totals(
        (
            await db.execute(ai_totals_statement(window.previous_since, window.since))
        ).one_or_none()
    )
    return {
        "window": window.public(),
        "current": current,
        "previous": previous,
        "delta": {
            "requests_pct": pct_change(current["requests"], previous["requests"]),
            "tokens_pct": pct_change(current["tokens"], previous["tokens"]),
            "latency_pct": pct_change(
                current["avg_latency_ms"], previous["avg_latency_ms"]
            ),
        },
        "series": build_ai_series(
            (await db.execute(ai_series_statement(window))).all(), window
        ),
        "models": build_ai_models(
            (await db.execute(ai_models_statement(window.since, window.until))).all()
        ),
    }


# ── background work ───────────────────────────────────────────────────────────────────────

_ACTIVE_INDEX_STATES = ("pending", "processing", "failed")


def _live_articles() -> list[Any]:
    return [Article.status == "published", Article.lifecycle_status == "active"]


def indexing_counts_statement() -> Select[Any]:
    return (
        select(Article.index_status, func.count(Article.id))
        .where(*_live_articles(), Article.index_status.in_(_ACTIVE_INDEX_STATES))
        .group_by(Article.index_status)
    )


def indexing_oldest_waiting_statement() -> Select[Any]:
    return select(func.min(Article.updated_at)).where(
        *_live_articles(), Article.index_status.in_(("pending", "processing"))
    )


def indexing_failed_statement() -> Select[Any]:
    return (
        select(
            Article.id,
            Article.title,
            Article.dept,
            Article.index_error,
            Article.updated_at,
        )
        .where(*_live_articles(), Article.index_status == "failed")
        .order_by(desc(Article.updated_at))
        .limit(TOP_N)
    )


def dead_letter_by_queue_statement() -> Select[Any]:
    count = func.count(DeadLetterJob.id)
    return (
        select(DeadLetterJob.source_queue, count)
        .group_by(DeadLetterJob.source_queue)
        .order_by(desc(count))
        .limit(TOP_N)
    )


def dead_letter_recent_statement() -> Select[Any]:
    return (
        select(
            DeadLetterJob.id,
            DeadLetterJob.source_queue,
            DeadLetterJob.error,
            DeadLetterJob.failed_at,
        )
        .order_by(desc(DeadLetterJob.failed_at))
        .limit(RECENT_N)
    )


def outbox_counts_statement() -> Select[Any]:
    return (
        select(OutboxEvent.status, func.count(OutboxEvent.id))
        .where(OutboxEvent.status.in_(("pending", "processing", "failed", "dead")))
        .group_by(OutboxEvent.status)
    )


def outbox_oldest_waiting_statement() -> Select[Any]:
    return select(func.min(OutboxEvent.created_at)).where(
        OutboxEvent.status.in_(("pending", "processing", "failed"))
    )


def notification_counts_statement() -> Select[Any]:
    return (
        select(NotificationQueue.status, func.count(NotificationQueue.id))
        .where(NotificationQueue.status.in_(("pending", "failed", "dead")))
        .group_by(NotificationQueue.status)
    )


def connector_job_counts_statement(since: datetime) -> Select[Any]:
    return (
        select(ConnectorJob.status, func.count(ConnectorJob.id))
        .where(
            (ConnectorJob.status == "running")
            | ((ConnectorJob.status == "failed") & (ConnectorJob.updated_at >= since))
        )
        .group_by(ConnectorJob.status)
    )


def connector_job_failures_statement() -> Select[Any]:
    return (
        select(
            ConnectorJob.id,
            Connector.name,
            ConnectorJob.last_error,
            ConnectorJob.updated_at,
        )
        .join(Connector, Connector.id == ConnectorJob.connector_id)
        .where(ConnectorJob.status == "failed")
        .order_by(desc(ConnectorJob.updated_at))
        .limit(RECENT_N)
    )


def reprocess_counts_statement() -> Select[Any]:
    return (
        select(IndexReprocessJob.status, func.count(IndexReprocessJob.id))
        .where(IndexReprocessJob.status.in_(("queued", "running", "failed")))
        .group_by(IndexReprocessJob.status)
    )


def _counts(rows: Iterable[Sequence[Any]]) -> dict[str, int]:
    return {str(status): _int(count) for status, count in rows}


def build_indexing(
    counts: Iterable[Sequence[Any]],
    oldest_waiting: datetime | None,
    failed: Iterable[Sequence[Any]],
    *,
    include_titles: bool,
) -> dict[str, Any]:
    tally = _counts(counts)
    return {
        "pending": tally.get("pending", 0),
        "processing": tally.get("processing", 0),
        "failed": tally.get("failed", 0),
        "oldest_waiting_at": _iso(oldest_waiting),
        "failed_articles": [
            {
                "article_id": str(article_id),
                # A title can be as sensitive as the document, and this endpoint is gated on
                # governance, not on reading articles. Ids are enough to open the article.
                "title": title if include_titles else None,
                "dept": dept,
                "error": clean_error_text(error),
                "at": _iso(updated),
            }
            for article_id, title, dept, error, updated in failed
        ],
    }


def build_dead_letters(
    by_queue: Iterable[Sequence[Any]], recent: Iterable[Sequence[Any]]
) -> dict[str, Any]:
    queues = [{"queue": str(queue), "count": _int(count)} for queue, count in by_queue]
    return {
        "total": sum(item["count"] for item in queues),
        "by_queue": queues,
        "recent": [
            {
                "id": str(job_id),
                "queue": queue,
                "error": clean_error_text(error),
                "at": _iso(failed_at),
            }
            for job_id, queue, error, failed_at in recent
        ],
    }


def build_queue_counts(
    counts: Iterable[Sequence[Any]], states: Sequence[str]
) -> dict[str, int]:
    tally = _counts(counts)
    return {state: tally.get(state, 0) for state in states}


def build_connector_jobs(
    counts: Iterable[Sequence[Any]], failures: Iterable[Sequence[Any]]
) -> dict[str, Any]:
    tally = _counts(counts)
    return {
        "running": tally.get("running", 0),
        "failed_recent": tally.get("failed", 0),
        "recent_failures": [
            {
                "id": str(job_id),
                "connector": name,
                "error": clean_error_text(error),
                "at": _iso(updated),
            }
            for job_id, name, error, updated in failures
        ],
    }


def build_attention(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Which background areas need a person, as codes, and the worst level among them."""
    codes: list[str] = []
    if pipeline["indexing"]["failed"]:
        codes.append("indexing_failed")
    if pipeline["dead_letters"]["total"]:
        codes.append("dead_letters")
    if pipeline["outbox"]["dead"] or pipeline["outbox"]["failed"]:
        codes.append("outbox_stuck")
    if pipeline["notifications"]["dead"] or pipeline["notifications"]["failed"]:
        codes.append("notifications_failing")
    if pipeline["connector_jobs"]["failed_recent"]:
        codes.append("connector_jobs_failed")
    if pipeline["reprocess_jobs"]["failed"]:
        codes.append("reprocess_failed")
    return {"level": "warning" if codes else "ok", "codes": codes}


async def load_pipeline(
    db: AsyncSession, *, now: datetime | None = None, include_titles: bool = False
) -> dict[str, Any]:
    moment = now or datetime.utcnow()
    indexing = build_indexing(
        (await db.execute(indexing_counts_statement())).all(),
        (await db.execute(indexing_oldest_waiting_statement())).scalar_one_or_none(),
        (await db.execute(indexing_failed_statement())).all(),
        include_titles=include_titles,
    )
    outbox: dict[str, Any] = dict(
        build_queue_counts(
            (await db.execute(outbox_counts_statement())).all(),
            ("pending", "processing", "failed", "dead"),
        )
    )
    outbox["oldest_waiting_at"] = _iso(
        (await db.execute(outbox_oldest_waiting_statement())).scalar_one_or_none()
    )
    pipeline = {
        "generated_at": moment.isoformat(),
        "indexing": indexing,
        "dead_letters": build_dead_letters(
            (await db.execute(dead_letter_by_queue_statement())).all(),
            (await db.execute(dead_letter_recent_statement())).all(),
        ),
        "outbox": outbox,
        "notifications": build_queue_counts(
            (await db.execute(notification_counts_statement())).all(),
            ("pending", "failed", "dead"),
        ),
        "connector_jobs": build_connector_jobs(
            (
                await db.execute(
                    connector_job_counts_statement(moment - timedelta(hours=24))
                )
            ).all(),
            (await db.execute(connector_job_failures_statement())).all(),
        ),
        "reprocess_jobs": build_queue_counts(
            (await db.execute(reprocess_counts_statement())).all(),
            ("queued", "running", "failed"),
        ),
    }
    pipeline["attention"] = build_attention(pipeline)
    return pipeline
