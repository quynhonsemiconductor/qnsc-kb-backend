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

The histogram uses the same bucket boundaries as opshub, in ms, so a p99 is a
number rather than a clamp at 10 s on AI routes that legitimately run long.

GATING: nothing is exported unless OTEL_EXPORTER_OTLP_ENDPOINT is set and
OTEL_ENABLED is not explicitly false. Without that, both configure_metrics() and
record_http_request() are no-ops, which is what keeps unit tests silent.
"""

from __future__ import annotations

import logging
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

#: Histogram boundaries in milliseconds, opshub's set: the OTel JS defaults plus
#: the rungs that cover the real timeout budgets. Above 60 s the request is not a
#: served request any more, so the overflow bucket describes nothing.
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
    15000,
    30000,
    45000,
    60000,
)

#: One export per minute (rova/opshub use the same interval): the Grafana Cloud
#: free tier is billed per active series, not per request, and a shorter interval
#: only grows cardinality through churn.
DEFAULT_EXPORT_INTERVAL_MS = 60_000

#: Longest route label kept. A framework template is bounded already; this is the
#: same tripwire rova's label guard uses, for the unmatched-path fallback.
MAX_ROUTE_LABEL_LENGTH = 128

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

_provider: Any = None
_duration: Any = None
_requests: Any = None
_errors: Any = None


def _metric_reader() -> Any:
    endpoint = signal_endpoint("metrics")
    if endpoint is None:
        return None
    return PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint),
        export_interval_millis=DEFAULT_EXPORT_INTERVAL_MS,
    )


def configure_metrics(metric_reader: MetricReader | None = None) -> bool:
    """Build the MeterProvider and the RED instruments. Returns True when live.

    ``metric_reader`` is a test seam: a unit test passes an InMemoryMetricReader
    and needs no endpoint, no network and no exporter thread.
    """
    global _provider, _duration, _requests, _errors

    if telemetry_disabled():
        return False
    if metrics is None:
        return False
    if _provider is not None:
        return True
    reader = metric_reader or _metric_reader()
    if reader is None:
        return False

    provider = MeterProvider(
        resource=Resource.create(resource_attributes()),
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
    return True


def status_class(status_code: int) -> str:
    """Bucket a status code the way rova's HttpMetrics does."""
    if status_code >= 500:
        return "5xx"
    if status_code >= 400:
        return "4xx"
    if status_code >= 300:
        return "3xx"
    return "2xx"


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


def reset_metrics_for_testing() -> None:
    """Forget the provider and instruments so a test can configure again."""
    global _provider, _duration, _requests, _errors
    provider = _provider
    _provider = _duration = _requests = _errors = None
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:  # A reader that was never started has nothing to stop.
            logger.debug("MeterProvider shutdown failed", exc_info=True)
