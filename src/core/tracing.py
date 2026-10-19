from __future__ import annotations

from src.core.config import settings

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:  # Keep local/unit environments lightweight.
    trace = None


def configure_tracing() -> None:
    if trace is None or not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.PROJECT_NAME})
    )
    exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer():
    if trace is None:
        return None
    return trace.get_tracer("qnsc-kb")
