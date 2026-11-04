"""OTLP metrics for the kb API — the same RED families rova and opshub emit.

WHY THESE NAMES AND LABELS, NOT THE SEMCONV INSTRUMENTATION:

The obvious move is `opentelemetry-instrumentation-fastapi`, which emits
`http.server.request.duration` (seconds) with `http.request.method`,
`http.response.status_code` and `http.route`. It is the wrong shape for THIS
estate in two ways, both verified against the live pipeline:

* the Adaptive Metrics `http_server_` rule keeps only `job`, `service_namespace`,
  `service_name`, `deployment_environment_name`, `method`, `status_class`, `route`
  and `le`. The instrumentation's attribute names are not in that list, so every
  dimension except the resource ones is aggregated away — no per-route and no
  per-status series;
* rova/opshub record their own `http.server.duration` histogram (unit ms) with
  `route`, `method` and `status_class` (app-platform's `HttpMetrics`). A Python
  semconv family would be a second spelling of the same concept, with no dashboard,
  SLO or alert able to query both.

So this module records exactly rova's family, under the same names and labels,
which is also what makes the same 5xx alert rule shape apply to kb:

    http.server.duration  -> http_server_duration_milliseconds_{bucket,sum,count}
    http.server.requests  -> http_server_requests_total
    http.server.errors    -> http_server_errors_total      (status >= 400)

The histogram uses the OpenTelemetry JS default boundaries (the set rova's SDK
emits with), in ms, so a p99 lines up with the Node products' dashboards. Long AI
routes read their quantile route-filtered; extra 30/45/60 s rungs would cost
buckets on every route to describe one.

GATING: nothing is exported unless OTEL_EXPORTER_OTLP_ENDPOINT is set and
OTEL_ENABLED is not explicitly false. Without that, both configure_metrics() and
record_http_request() are no-ops, which is what keeps unit tests silent.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.core.otel import resource_attributes, signal_endpoint, telemetry_disabled

try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        MetricReader,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource
except ImportError:  # Keep local/unit environments lightweight, like tracing.py.
    metrics = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: Metric names — the rova/opshub spelling (see the module docstring).
HTTP_SERVER_DURATION_METRIC = "http.server.duration"
HTTP_SERVER_REQUESTS_METRIC = "http.server.requests"
HTTP_SERVER_ERRORS_METRIC = "http.server.errors"

#: Celery worker task metrics. The worker is a separate process from the API
#: and had NO metrics at all, so a queue that stopped draining or a task that
#: started failing every run was invisible until a user noticed. Names follow
#: the OTel -> Prometheus convention (counters get `_total`): the exported
#: series are `celery_task_runs_total`, `celery_task_failures_total` and
#: `celery_task_duration_milliseconds_*`.
CELERY_TASK_RUNS_METRIC = "celery.task.runs"
CELERY_TASK_FAILURES_METRIC = "celery.task.failures"
CELERY_TASK_DURATION_METRIC = "celery.task.duration"

#: Histogram boundaries in milliseconds — the OpenTelemetry JS defaults, verbatim
#: (rova/opshub's SDK uses the same set unless a product overrides it). Chosen for
#: comparability with the Node products' p50/p95/p99, and because it is the
#: cheapest set that still separates the rungs a normal request lives in. No
#: 45s/60s rungs: kb's long AI calls are better read from the route-filtered
#: quantile than from extra buckets on every route.
HTTP_DURATION_BOUNDARIES_MS = (
    0,
    5,
    10,
    25,
    50,
    75,
    100,
    250,
    500,
    750,
    1000,
    2500,
    5000,
    7500,
    10000,
)

#: One export per minute (rova/opshub use the same interval): the Grafana Cloud
#: free tier is billed per active series, not per request, and a shorter interval
#: only grows cardinality through churn.
DEFAULT_EXPORT_INTERVAL_MS = 60_000

#: Longest route label kept. A framework template is bounded already; this is the
#: same tripwire rova's label guard uses, for the unmatched-path fallback.
MAX_ROUTE_LABEL_LENGTH = 128

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

#: Longest Celery task name kept. Celery task names are static decorators, so
#: this is a tripwire like MAX_ROUTE_LABEL_LENGTH, not an expected cut.
MAX_TASK_LABEL_LENGTH = 128

_provider: Any = None
_duration: Any = None
_requests: Any = None
_errors: Any = None
_task_runs: Any = None
_task_failures: Any = None
_task_duration: Any = None


def _metric_reader() -> Any:
    endpoint = signal_endpoint("metrics")
    if endpoint is None:
        return None
    return PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint),
        export_interval_millis=DEFAULT_EXPORT_INTERVAL_MS,
    )


def configure_metrics(
    metric_reader: MetricReader | None = None, *, process_instance_id: bool = False
) -> bool:
    """Build the MeterProvider and the RED instruments. Returns True when live.

    ``metric_reader`` is a test seam: a unit test passes an InMemoryMetricReader
    and needs no endpoint, no network and no exporter thread.

    ``process_instance_id`` adds ``service.instance.id=<pid>``. The API does NOT
    set it (the collector already identifies the pod), but the worker is a
    prefork parent with one child and a restart reuses the pod name: the pid is
    what separates this process's series from the previous one's in the overlap
    window after a restart.
    """
    global _provider, _duration, _requests, _errors
    global _task_runs, _task_failures, _task_duration

    if telemetry_disabled():
        return False
    if metrics is None:
        return False
    if _provider is not None:
        return True
    reader = metric_reader or _metric_reader()
    if reader is None:
        return False

    attributes = resource_attributes()
    if process_instance_id:
        attributes["service.instance.id"] = str(os.getpid())
    provider = MeterProvider(
        resource=Resource.create(attributes),
        metric_readers=[reader],
    )
    # The instruments are taken from THIS provider, not from the global meter
    # provider: the OTel API allows setting that global exactly once per process,
    # which makes a second configuration (a test, a reload) keep resolving through
    # a provider that has been shut down. kb has no third-party instrumentation
    # that reads the global, so pinning the instruments to the provider is both
    # smaller and deterministic.
    meter = provider.get_meter("qnsc-kb")

    _provider = provider
    _duration = meter.create_histogram(
        HTTP_SERVER_DURATION_METRIC,
        unit="ms",
        description="Inbound HTTP request duration",
        explicit_bucket_boundaries_advisory=HTTP_DURATION_BOUNDARIES_MS,
    )
    _requests = meter.create_counter(
        HTTP_SERVER_REQUESTS_METRIC, description="Inbound HTTP requests"
    )
    _errors = meter.create_counter(
        HTTP_SERVER_ERRORS_METRIC, description="Inbound HTTP requests that failed"
    )

    # Created in the same provider, and only exported by the process that
    # configures it: the API never records a task, the worker never records an
    # HTTP request, so each process emits only its own families.
    _task_runs = meter.create_counter(
        CELERY_TASK_RUNS_METRIC,
        description="Celery tasks that finished, by task and outcome",
    )
    _task_failures = meter.create_counter(
        CELERY_TASK_FAILURES_METRIC,
        description="Celery tasks that finished in a non-success state",
    )
    _task_duration = meter.create_histogram(
        CELERY_TASK_DURATION_METRIC,
        unit="ms",
        description="Celery task wall-clock duration",
    )
    return True


def status_class(status_code: int) -> str:
    """Bucket a status code. The four classes rova uses, plus ``other``.

    rova's ``HttpMetrics`` folds anything below 200 into ``2xx`` and anything at
    or above 600 into ``5xx``; neither is a status a compliant server sends, and
    calling an informational (1xx) response a success or a 7xx a server error
    would misreport it in the 5xx alert. A distinct bounded value keeps the class
    total without lying — and 1xx/600+ stay well inside the cardinality guard.
    """
    if 200 <= status_code < 300:
        return "2xx"
    if 300 <= status_code < 400:
        return "3xx"
    if 400 <= status_code < 500:
        return "4xx"
    if 500 <= status_code < 600:
        return "5xx"
    return "other"


def method_label(method: str) -> str:
    upper = (method or "").upper()
    return upper if upper in _METHODS else "OTHER"


def route_label(route: str | None) -> str:
    # Middleware passes the resolved route template (bounded by construction);
    # the fallback covers a request that matched no route.
    label = (route or "").strip() or "/unmatched"
    return label[:MAX_ROUTE_LABEL_LENGTH]


def record_http_request(
    route: str | None, method: str, status_code: int, duration_ms: float
) -> None:
    """Record one inbound request. A no-op until configure_metrics() succeeded."""
    if _duration is None:
        return
    labels = {
        "route": route_label(route),
        "method": method_label(method),
        "status_class": status_class(status_code),
    }
    try:
        _duration.record(duration_ms, labels)
        _requests.add(1, labels)
        # Same rule as rova: the errors counter counts every >= 400, so it is the
        # dashboard's error rate, not the 5xx share (the histogram carries 5xx).
        if status_code >= 400:
            _errors.add(1, labels)
    except Exception:  # Telemetry must never break a request.
        logger.debug("Could not record the HTTP metric", exc_info=True)


def task_label(task_name: str | None) -> str:
    """The bounded task-name label. A missing name is not dropped: an empty one
    is what makes a misconfigured signal visible instead of silent."""
    label = (task_name or "").strip() or "unknown"
    return label[:MAX_TASK_LABEL_LENGTH]


def record_task_finished(
    task_name: str | None, state: str | None, duration_ms: float
) -> None:
    """Record one finished Celery task. A no-op until configure_metrics() ran.

    Called from Celery's ``task_postrun`` signal (celery_app.py), which fires
    once per task whether it succeeded, failed or was revoked. ``state`` is
    Celery's task state.

    ONLY ``FAILURE`` increments the failure counter. ``RETRY`` is a task that
    will run again (not a terminal failure), and ``REVOKED`` is a deliberate
    cancellation — by ``expires`` (the stale-beat guard), operator shutdown or
    an admin revoke — not the task's own failure. Both remain visible in the
    runs counter under their exact state, so the dashboard can show them
    without them inflating the pageable failure rate.
    """
    if _task_runs is None:
        return
    task = task_label(task_name)
    outcome = (state or "UNKNOWN").strip().upper()
    try:
        _task_runs.add(1, {"task": task, "state": outcome})
        if outcome == "FAILURE":
            _task_failures.add(1, {"task": task, "state": outcome})
        _task_duration.record(duration_ms, {"task": task})
    except Exception:  # Telemetry must never break a worker task.
        logger.debug("Could not record the Celery task metric", exc_info=True)


def shutdown_metrics() -> None:
    """Flush the last interval and stop the exporter reader.

    Called from the lifespan's finally block. Without it the final 60-second
    window of metrics is lost on every deploy/scale-in.
    """
    provider = _provider
    if provider is None:
        return
    try:
        provider.force_flush()
    finally:
        provider.shutdown()


def reset_metrics_for_testing() -> None:
    """Forget the provider and instruments so a test can configure again."""
    global _provider, _duration, _requests, _errors
    global _task_runs, _task_failures, _task_duration
    provider = _provider
    _provider = _duration = _requests = _errors = None
    _task_runs = _task_failures = _task_duration = None
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:  # A reader that was never started has nothing to stop.
            logger.debug("MeterProvider shutdown failed", exc_info=True)
