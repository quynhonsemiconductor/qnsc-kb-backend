"""Celery task metrics recorded from the worker's signal handlers.

The signal wiring itself runs in the forked worker child (prefork), so it is
exercised in the deployment, not here. These tests pin the recording contract:
one run per finished task, a failure only for a non-SUCCESS state, and a no-op
when telemetry was never configured (local tests, no endpoint).
"""

from __future__ import annotations

from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from src.core import otel_metrics


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


def test_success_records_a_run_and_no_failure():
    reader = InMemoryMetricReader()
    otel_metrics.reset_metrics_for_testing()
    try:
        assert otel_metrics.configure_metrics(metric_reader=reader) is True
        otel_metrics.record_task_finished("sync_drive", "SUCCESS", 1250.0)

        points = _data_points(reader)
        runs = points[otel_metrics.CELERY_TASK_RUNS_METRIC]
        assert sum(point.value for point in runs) == 1
        assert runs[0].attributes["task"] == "sync_drive"
        assert runs[0].attributes["state"] == "SUCCESS"
        assert otel_metrics.CELERY_TASK_FAILURES_METRIC not in points
        assert points[otel_metrics.CELERY_TASK_DURATION_METRIC][0].count == 1
    finally:
        otel_metrics.reset_metrics_for_testing()


def test_a_non_success_state_counts_as_a_failure():
    reader = InMemoryMetricReader()
    otel_metrics.reset_metrics_for_testing()
    try:
        assert otel_metrics.configure_metrics(metric_reader=reader) is True
        otel_metrics.record_task_finished("index_article", "failure", 30.0)

        points = _data_points(reader)
        failures = points[otel_metrics.CELERY_TASK_FAILURES_METRIC]
        assert sum(point.value for point in failures) == 1
        assert failures[0].attributes["state"] == "FAILURE"
    finally:
        otel_metrics.reset_metrics_for_testing()


def test_without_a_configured_provider_nothing_is_recorded():
    otel_metrics.reset_metrics_for_testing()
    otel_metrics.record_task_finished("sync_drive", "SUCCESS", 1.0)
    assert otel_metrics._task_runs is None
