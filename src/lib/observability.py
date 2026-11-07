import logging

import structlog

from src.lib import log_redaction
from src.lib.log_redaction import structlog_processor


def add_trace_context(logger, method_name, event_dict):
    """Add the active span's trace_id/span_id (lowercase hex) for Loki→Tempo.

    A no-op when OpenTelemetry is absent or no span is active; an invalid/zero
    context is never emitted, because a bogus id breaks the derived-field link.
    """
    try:
        from src.core.tracing import trace
    except ImportError:  # pragma: no cover - tracing imports config only
        return event_dict
    if trace is None:
        return event_dict
    context = trace.get_current_span().get_span_context()
    if context is not None and context.is_valid:
        event_dict.setdefault("trace_id", format(context.trace_id, "032x"))
        event_dict.setdefault("span_id", format(context.span_id, "016x"))
    return event_dict


def setup_logging():
    """Emit structured JSON logs to stdout for log aggregation.

    Keeps stdlib logging (uvicorn/sqlalchemy/celery route their records
    through structlog) in sync so container logs are uniformly parseable.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_trace_context,
            # LAST before the renderer: after contextvars are merged and the traceback is text, so both are
            # redacted. Placed first, it never saw either.
            structlog_processor,
            # Compact separators, like rova/opshub (pino): `"trace_id":"<hex>"` with no space.
            # Grafana Cloud's provisioned Loki derived field (read-only, not ours to edit)
            # matches `trace_id"?[:=]"?(\w+)`, so a space after the colon broke log → trace links.
            structlog.processors.JSONRenderer(separators=(",", ":")),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # httpx/httpcore to WARNING, and every stdlib record redacted too (see log_redaction.py).
    log_redaction.install()
