from __future__ import annotations

from typing import Any

from src.core.otel import resource_attributes, signal_endpoint, telemetry_disabled

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:  # Keep local/unit environments lightweight.
    trace = None  # type: ignore[assignment]  # OpenTelemetry absent: trace stays None


_provider: Any = None


def configure_tracing() -> None:
    global _provider
    if _provider is not None:
        # Idempotent: the worker calls it from worker_process_init and again from
        # task_prerun for a solo pool; a second provider would duplicate every span.
        return
    endpoint = signal_endpoint("traces")
    # The same switch the metrics half honours: OTEL_ENABLED=false silences ALL
    # telemetry, not only the metrics that had no gate before.
    if trace is None or endpoint is None or telemetry_disabled():
        return
    provider = TracerProvider(resource=Resource.create(resource_attributes()))
    # The full /v1/traces path is built in signal_endpoint(): the exporter does not
    # append it to an explicit endpoint, so the base URL alone posted to the root.
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _provider = provider


def get_tracer():
    if trace is None:
        return None
    return trace.get_tracer("qnsc-kb")


def shutdown_tracing() -> None:
    """Flush buffered spans and stop the exporter. A no-op when never configured."""
    provider = _provider
    if provider is None:
        return
    try:
        provider.force_flush()
    finally:
        provider.shutdown()
