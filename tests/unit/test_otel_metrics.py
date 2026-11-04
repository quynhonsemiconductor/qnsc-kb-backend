from __future__ import annotations

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from src.core import otel, otel_metrics
from src.core.config import settings


@pytest.fixture(autouse=True)
def _reset_metrics():
    otel_metrics.reset_metrics_for_testing()
    yield
    otel_metrics.reset_metrics_for_testing()


def _data_points(reader: InMemoryMetricReader):
    """Every data point across every instrument the reader has collected."""
    data = reader.get_metrics_data()
    if data is None:
        return {}
    points = {}
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                points.setdefault(metric.name, []).extend(metric.data.data_points)
    return points


# ── endpoints and identity ─────────────────────────────────────────────────


def test_signal_endpoint_appends_the_signal_path(monkeypatch):
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4318")
    assert otel.signal_endpoint("metrics") == "http://alloy:4318/v1/metrics"
    assert otel.signal_endpoint("traces") == "http://alloy:4318/v1/traces"


def test_signal_endpoint_tolerates_a_trailing_slash(monkeypatch):
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4318/")
    assert otel.signal_endpoint("metrics") == "http://alloy:4318/v1/metrics"


def test_signal_endpoint_is_none_without_an_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", None)
    assert otel.signal_endpoint("metrics") is None


def test_signal_endpoint_does_not_double_a_full_signal_url(monkeypatch):
    monkeypatch.setattr(
        settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4318/v1/metrics"
    )
    assert otel.signal_endpoint("metrics") == "http://alloy:4318/v1/metrics"


def test_signal_endpoint_resolves_every_signal_from_one_full_url(monkeypatch):
    """A caller may configure any one full signal URL; the rest still resolve."""
    monkeypatch.setattr(
        settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4318/v1/traces"
    )
    assert otel.signal_endpoint("metrics") == "http://alloy:4318/v1/metrics"
    assert otel.signal_endpoint("traces") == "http://alloy:4318/v1/traces"
    assert otel.signal_endpoint("logs") == "http://alloy:4318/v1/logs"


def test_histogram_uses_the_otel_js_default_boundaries():
    assert otel_metrics.HTTP_DURATION_BOUNDARIES_MS == (
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


def test_deployment_environment_prefers_deployment_env(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ENV", "dev")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=prod,service.namespace=kb"
    )
    assert otel.deployment_environment() == "dev"


def test_deployment_environment_falls_back_to_the_resource_attribute(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.namespace=kb,deployment.environment=dev,service.version=sha-abc",
    )
    assert otel.deployment_environment() == "dev"


def test_deployment_environment_never_reads_environment(monkeypatch):
    """kb pins ENVIRONMENT=production in dev; it must not become the env label."""
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert otel.deployment_environment() == "unknown"


def test_resource_attributes_use_the_injected_service_name(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "kb-api")
    monkeypatch.setenv("DEPLOYMENT_ENV", "dev")
    assert otel.resource_attributes() == {
        "service.name": "kb-api",
        "service.namespace": "kb",
        "deployment.environment.name": "dev",
    }


def test_resource_attributes_fall_back_to_the_project_name(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.setenv("DEPLOYMENT_ENV", "dev")
    assert otel.resource_attributes()["service.name"] == settings.PROJECT_NAME


# ── the gate ───────────────────────────────────────────────────────────────


def test_configure_metrics_is_off_without_an_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", None)
    assert otel_metrics.configure_metrics() is False
    # Recording must stay a no-op rather than raise.
    otel_metrics.record_http_request("/api/v1/health", "GET", 200, 3.0)


def test_configure_metrics_honours_an_explicit_otel_enabled_false(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "false")
    assert otel_metrics.configure_metrics(metric_reader=InMemoryMetricReader()) is False


def test_configure_metrics_is_idempotent():
    reader = InMemoryMetricReader()
    assert otel_metrics.configure_metrics(metric_reader=reader) is True
    assert otel_metrics.configure_metrics(metric_reader=InMemoryMetricReader()) is True


# ── the recorded series ────────────────────────────────────────────────────


def test_records_the_rova_histogram_and_counters(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ENV", "dev")
    reader = InMemoryMetricReader()
    assert otel_metrics.configure_metrics(metric_reader=reader) is True

    otel_metrics.record_http_request("/api/v1/articles/{article_id}", "GET", 503, 12.5)

    points = _data_points(reader)
    duration = points[otel_metrics.HTTP_SERVER_DURATION_METRIC]
    assert duration[0].attributes == {
        "route": "/api/v1/articles/{article_id}",
        "method": "GET",
        "status_class": "5xx",
    }
    assert duration[0].sum == 12.5
    assert points[otel_metrics.HTTP_SERVER_REQUESTS_METRIC][0].value == 1
    assert points[otel_metrics.HTTP_SERVER_ERRORS_METRIC][0].value == 1


def test_4xx_counts_as_an_error_but_not_as_5xx():
    reader = InMemoryMetricReader()
    assert otel_metrics.configure_metrics(metric_reader=reader) is True

    otel_metrics.record_http_request("/api/v1/auth/me", "GET", 401, 2.0)

    points = _data_points(reader)
    assert points[otel_metrics.HTTP_SERVER_ERRORS_METRIC][0].value == 1
    attributes = points[otel_metrics.HTTP_SERVER_DURATION_METRIC][0].attributes
    assert attributes["status_class"] == "4xx"


def test_a_successful_request_does_not_touch_the_errors_counter():
    reader = InMemoryMetricReader()
    assert otel_metrics.configure_metrics(metric_reader=reader) is True

    otel_metrics.record_http_request("/api/v1/health", "GET", 200, 1.0)

    points = _data_points(reader)
    assert otel_metrics.HTTP_SERVER_REQUESTS_METRIC in points
    # A counter with no data point has no series at all — that is the point.
    assert otel_metrics.HTTP_SERVER_ERRORS_METRIC not in points


def test_methods_and_routes_are_bounded():
    assert otel_metrics.method_label("get") == "GET"
    assert otel_metrics.method_label("TRACE") == "OTHER"
    assert otel_metrics.route_label(None) == "/unmatched"
    assert otel_metrics.route_label("") == "/unmatched"
    long_route = "/" + "a" * (otel_metrics.MAX_ROUTE_LABEL_LENGTH * 2)
    assert len(otel_metrics.route_label(long_route)) == (
        otel_metrics.MAX_ROUTE_LABEL_LENGTH
    )


def test_route_template_is_kept_verbatim():
    assert (
        otel_metrics.route_label("/api/v1/articles/{article_id}")
        == "/api/v1/articles/{article_id}"
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (100, "other"),
        (199, "other"),
        (200, "2xx"),
        (299, "2xx"),
        (302, "3xx"),
        (404, "4xx"),
        (500, "5xx"),
        (599, "5xx"),
        (600, "other"),
        (700, "other"),
    ],
)
def test_status_class_buckets_and_the_other_fallthrough(code, expected):
    assert otel_metrics.status_class(code) == expected
