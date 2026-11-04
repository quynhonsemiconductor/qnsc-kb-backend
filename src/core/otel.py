"""Shared OpenTelemetry wiring: endpoint shape, resource attributes, identity.

Deliberately dependency-light and importable without the OpenTelemetry SDK: the
tracing and metrics modules each guard their own imports, so a local checkout
without the exporter installed still imports the app.

The values here mirror what the chart injects and what rova/opshub (Node) publish,
because Alloy turns the resource into the Prometheus `job` label as
``<service.namespace>/<service.name>`` and the environment into the
``deployment_environment_name`` label the Adaptive Metrics rules keep. A differing
spelling is not a cosmetic difference: it moves the series into another job, or
loses the environment split entirely.
"""

from __future__ import annotations

import os

from src.core.config import settings

#: The product label under which every kb series appears (``job="kb/kb-api"``).
#: The chart also sets `service.namespace=kb` through OTEL_RESOURCE_ATTRIBUTES,
#: but that variable is not read by Resource.create() unless the env detector is
#: installed, so it is named here as well.
SERVICE_NAMESPACE = "kb"

#: Values of OTEL_ENABLED that mean "off". Unset means on, because kb has never
#: carried the variable (the endpoint has always been the switch, as it is for
#: tracing). An explicit false is still honoured so one deployment can silence
#: telemetry without removing the endpoint.
_DISABLED_VALUES = {"false", "0", "no", "off"}


def otlp_endpoint() -> str | None:
    """The OTLP base endpoint, or None when telemetry is not configured."""
    endpoint = (settings.OTEL_EXPORTER_OTLP_ENDPOINT or "").strip()
    return endpoint or None


def signal_endpoint(signal: str) -> str | None:
    """Full URL for one OTLP/HTTP signal (``traces``, ``metrics``, ``logs``).

    The Python exporters do NOT append the signal path when the endpoint is passed
    explicitly: ``OTLPSpanExporter(endpoint="http://host:4318")`` posts to the root,
    which the Alloy OTLP receiver does not serve (it routes ``/v1/traces`` and
    ``/v1/metrics``). Building the full path here is what makes the explicit
    argument equivalent to what the SDK would derive from the environment.
    """
    base = otlp_endpoint()
    if base is None:
        return None
    trimmed = base.rstrip("/")
    # Tolerate a caller that already supplied the full signal URL (a local run
    # following the OTLP spec's *_ENDPOINT form) instead of doubling the path.
    if trimmed.endswith(f"/v1/{signal}"):
        return trimmed
    return f"{trimmed}/v1/{signal}"


def telemetry_disabled() -> bool:
    """True when OTEL_ENABLED is explicitly false. Unset is enabled."""
    return (os.environ.get("OTEL_ENABLED") or "").strip().lower() in _DISABLED_VALUES


def deployment_environment() -> str:
    """The deployment identity the environment label is built from.

    DEPLOYMENT_ENV first, then the old-semconv ``deployment.environment`` the chart
    puts in OTEL_RESOURCE_ATTRIBUTES, else ``unknown``. NEVER ENVIRONMENT: kb pins
    ENVIRONMENT=production in every environment so validate_production() runs, so
    that value would label dev telemetry as production and destroy the split.
    """
    explicit = (os.environ.get("DEPLOYMENT_ENV") or "").strip()
    if explicit:
        return explicit
    attributes = os.environ.get("OTEL_RESOURCE_ATTRIBUTES") or ""
    for pair in attributes.split(","):
        key, separator, value = pair.partition("=")
        if separator and key.strip() == "deployment.environment" and value.strip():
            return value.strip()
    return "unknown"


def service_name() -> str:
    """The service name the chart injects, falling back to the project name."""
    injected = (os.environ.get("OTEL_SERVICE_NAME") or "").strip()
    return injected or settings.PROJECT_NAME


def resource_attributes() -> dict[str, str]:
    """Resource attributes shared by traces and metrics."""
    return {
        "service.name": service_name(),
        "service.namespace": SERVICE_NAMESPACE,
        "deployment.environment.name": deployment_environment(),
    }
