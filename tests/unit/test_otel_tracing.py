"""Trace export goes to the signal path, and the OTEL_ENABLED switch covers it."""

from __future__ import annotations

import pytest

from src.core import tracing
from src.core.config import settings


class _SpyExporter:
    instances: list["_SpyExporter"] = []

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint
        self.exported = 0
        _SpyExporter.instances.append(self)

    def export(self, spans) -> int:
        self.exported += len(list(spans))
        return 0

    def force_flush(self, *args, **kwargs) -> bool:
        return True

    def shutdown(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _spy(monkeypatch):
    _SpyExporter.instances = []
    monkeypatch.setattr(tracing, "OTLPSpanExporter", _SpyExporter)
    tracing._provider = None
    yield
    if tracing._provider is not None:
        tracing.shutdown_tracing()
    tracing._provider = None


def test_traces_are_posted_to_the_signal_path(monkeypatch):
    monkeypatch.delenv("OTEL_ENABLED", raising=False)
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4318")

    tracing.configure_tracing()

    assert [exporter.endpoint for exporter in _SpyExporter.instances] == [
        "http://alloy:4318/v1/traces"
    ]


def test_otlp_enabled_false_silences_traces_too(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4318")

    tracing.configure_tracing()

    assert _SpyExporter.instances == []
    assert tracing._provider is None


def test_no_endpoint_means_no_provider(monkeypatch):
    monkeypatch.delenv("OTEL_ENABLED", raising=False)
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", None)

    tracing.configure_tracing()

    assert _SpyExporter.instances == []
    assert tracing._provider is None


def test_shutdown_flushes_and_stops_the_provider(monkeypatch):
    monkeypatch.delenv("OTEL_ENABLED", raising=False)
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4318")
    tracing.configure_tracing()

    calls: list[str] = []
    provider = tracing._provider

    class _Recorder:
        def force_flush(self, *args, **kwargs):
            calls.append("flush")
            return True

        def shutdown(self):
            calls.append("shutdown")

    tracing._provider = _Recorder()
    tracing.shutdown_tracing()

    assert calls == ["flush", "shutdown"]
    # The real provider still gets stopped by the fixture teardown, not leaked.
    tracing._provider = provider
