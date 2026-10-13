"""Keep bearer credentials out of the logs.

Why this exists. The worker logged full SharePoint download URLs, `tempauth=...` included, from httpx's
INFO line `HTTP Request: GET https://...`. A SharePoint/OneDrive pre-authenticated download URL carries its
authorisation in the query string: anyone who can read the log line can download that file for about 24 hours,
no login needed. Logs are broader-readable than documents, so a log line is a document leak.

Two layers, because neither is enough alone:

1. `httpx` and `httpcore` are set to WARNING. Their INFO lines exist to log every request URL and this
   application never needs them; that removes the known source.
2. Anything that DOES get logged is redacted, whoever logs it: stdlib records (a LogRecord factory, so it
   applies to every logger and handler including Celery's own) and structlog events (a processor). The
   second layer is the one that catches the next library, and the `str(exc)` of an httpx error, which names
   the URL it failed on.

Redaction is by parameter NAME in a URL query string (and a `Bearer` token), not by trying to recognise
secrets in general. A value is replaced; the name stays so a log is still debuggable.
"""
from __future__ import annotations

import logging
import re
from typing import Any

REDACTED = "[REDACTED]"

# Names that carry a credential. Matched as a substring of the parameter name, case-insensitively, so
# `X-Amz-Signature`, `access_token`, `client_secret` and `tempauth` are all covered without listing each.
_SENSITIVE = r"[\w.\-]*(?:tempauth|auth|token|secret|signature|password|passwd|credential|apikey|api_key|sig|code|key|sas)[\w.\-]*"
_QUERY_PARAM = re.compile(rf"(?i)([?&;]{_SENSITIVE}=)([^&\s'\"<>#]+)")
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]{12,}")

QUIET_LOGGERS = ("httpx", "httpcore")


def redact(text: str) -> str:
    if "=" not in text and "earer" not in text:
        return text
    return _BEARER.sub(rf"\1{REDACTED}", _QUERY_PARAM.sub(rf"\1{REDACTED}", text))


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        redacted = [_redact_value(v) for v in value]
        return type(value)(redacted) if isinstance(value, tuple) else redacted
    if isinstance(value, BaseException):
        # An exception's text is what gets logged (`error=str(exc)`); a URL in it is as exposed as one in a message.
        return redact(str(value))
    return value


def structlog_processor(_logger: Any, _method: str, event_dict: dict) -> dict:
    """A structlog processor: redact every string in the event, however deeply nested."""
    return {key: _redact_value(value) for key, value in event_dict.items()}


_installed = False


def install() -> None:
    """Idempotent. Call once at process start, in every process that logs (api, worker, beat)."""
    global _installed
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    if _installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        # Only pay for records that can contain a URL or token.
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = _redact_value(record.args)
        return record

    logging.setLogRecordFactory(factory)

    import structlog

    config = structlog.get_config()
    processors = list(config.get("processors", []))
    if structlog_processor not in processors:
        # First, so everything after it (including the renderer) only ever sees redacted values.
        structlog.configure(processors=[structlog_processor, *processors])
    _installed = True
