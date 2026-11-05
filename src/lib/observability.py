import logging

import structlog

from src.lib import log_redaction
from src.lib.log_redaction import structlog_processor

#: Platform probe paths: these create no spans and no access-log lines. Matched on the
#: whole path with the query string removed, never as a prefix -- `/readyzfoo` is
#: a normal route. The list is the contract's, verbatim; the legacy `/health/*`
#: aliases stay observed on purpose (the deploy pipeline polls them).
PROBE_PATHS = (
    "/livez",
    "/readyz",
    "/v1/readyz",
    "/healthz",
    "/v1/healthz",
    "/favicon.ico",
)


def is_probe_path(path: str) -> bool:
    """True for the platform probe paths, query string removed, whole-path match."""
    return path.split("?", 1)[0] in PROBE_PATHS


class ProbeAccessFilter(logging.Filter):
    """Silence `uvicorn.access` lines for probe paths (audit gap B4).

    uvicorn's access line is `'%s - "%s %s HTTP/%s" %d'`, so the path sits in
    `record.args[2]` as `"/readyz HTTP/1.1"`-style. An unknown record shape
    passes through: this filter only ever withholds lines, never rewrites them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        candidate = str(args[2]) if len(args) > 2 else ""
        return not is_probe_path(candidate)


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
            # LAST before the renderer: after contextvars are merged and the traceback is text, so both are
            # redacted. Placed first, it never saw either.
            structlog_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # httpx/httpcore to WARNING, and every stdlib record redacted too (see log_redaction.py).
    log_redaction.install()
