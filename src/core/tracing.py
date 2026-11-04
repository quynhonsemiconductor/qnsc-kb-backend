from __future__ import annotations

from src.core.otel import resource_attributes, signal_endpoint

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:  # Keep local/unit environments lightweight.
    trace = None  # type: ignore[assignment]  # OpenTelemetry absent: trace stays None


def configure_tracing() -> None:
    endpoint = signal_endpoint("traces")
    if trace is None or endpoint is None:
        return
    provider = TracerProvider(resource=Resource.create(resource_attributes()))
    # The full /v1/traces path is built in signal_endpoint(): the exporter does not
    # append it to an explicit endpoint, so the base URL alone posted to the root.
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer():
    if trace is None:
        return None
    return trace.get_tracer("qnsc-kb")
