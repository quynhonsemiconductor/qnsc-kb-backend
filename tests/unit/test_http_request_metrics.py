"""The request middleware feed, exercised through the real app middleware chain.

These go through `TestClient(main.app)` rather than calling `record_request_metric`
directly, because the failure this pins is about the CHAIN: which route label a
404 gets, whether an unhandled 500 still reaches the recorder with its request id,
and whether a broken recorder can turn a healthy response into an error. Calling
the function directly asserts none of that.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from src.api import main
from src.core import otel_metrics


class _NoDatabase:
    """A session that no-ops the write and fails the commit.

    `record_request_metric` persists the row and catches a storage failure; giving
    it a session that fails there proves the metric is recorded BEFORE the write,
    without needing a Postgres for this test.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def add(self, *args, **kwargs):
        return None

    async def commit(self):
        raise RuntimeError("no database in this test")


def _data_points(reader: InMemoryMetricReader) -> dict:
    data = reader.get_metrics_data()
    if data is None:
        return {}
    points: dict = {}
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                points.setdefault(metric.name, []).extend(metric.data.data_points)
    return points


@pytest.fixture
def client(monkeypatch):
    reader = InMemoryMetricReader()
    otel_metrics.reset_metrics_for_testing()
    assert otel_metrics.configure_metrics(metric_reader=reader) is True
    monkeypatch.setattr(main, "SessionLocal", _NoDatabase)
    yield TestClient(main.app, raise_server_exceptions=False), reader
    otel_metrics.reset_metrics_for_testing()


def test_a_404_is_recorded_with_the_unmatched_route_not_the_path(client):
    http, reader = client
    response = http.get("/definitely-not-a-route/abc-123")
    assert response.status_code == 404

    points = _data_points(reader)
    attributes = points[otel_metrics.HTTP_SERVER_DURATION_METRIC][0].attributes
    assert attributes["route"] == "/unmatched"
    assert attributes["status_class"] == "4xx"
    # The concrete path never becomes a label.
    assert "abc-123" not in str(attributes)


def test_an_unhandled_500_still_returns_a_correlated_response_and_a_5xx(client):
    http, reader = client

    @main.app.get("/__test__/metrics-boom")
    async def _boom():
        raise RuntimeError("simulated failure below the router")

    response = http.get("/__test__/metrics-boom")
    assert response.status_code == 500
    # The response is opaque on purpose, so the request id is the only way back to
    # the traceback; the boundary must not cost it.
    assert response.headers.get("x-request-id")

    points = _data_points(reader)
    attributes = points[otel_metrics.HTTP_SERVER_DURATION_METRIC][0].attributes
    assert attributes["route"] == "/__test__/metrics-boom"
    assert attributes["status_class"] == "5xx"
    assert points[otel_metrics.HTTP_SERVER_ERRORS_METRIC][0].value == 1


def test_a_recorder_that_raises_does_not_fail_the_request(client, monkeypatch):
    http, _reader = client
    assert otel_metrics._duration is not None

    class _Exploding:
        def record(self, *args, **kwargs):
            raise RuntimeError("recorder is broken")

    monkeypatch.setattr(otel_metrics, "_duration", _Exploding())
    response = http.get("/health/live")
    assert response.status_code == 200
