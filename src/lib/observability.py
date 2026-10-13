import logging

import structlog

from src.lib import log_redaction
from src.lib.log_redaction import structlog_processor


def setup_logging():
    """Emit structured JSON logs to stdout for log aggregation.

    Keeps stdlib logging (uvicorn/sqlalchemy/celery route their records
    through structlog) in sync so container logs are uniformly parseable.
    """
    structlog.configure(
        processors=[
            structlog_processor,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # httpx/httpcore to WARNING, and every stdlib record redacted too (see log_redaction.py).
    log_redaction.install()
