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

# A query parameter is redacted by its WHOLE NAME, not by a substring of it. Substring matching redacted
# `keyword=`, `author=` and `status_code=` (they contain `key`, `auth`, `code`), which made logs harder to read
# and taught people to distrust the redaction.
#
# A name is sensitive when, lower-cased and split into words on `-`, `_`, `.` and camelCase:
#   * it is exactly one of _WHOLE_NAMES (`code`, `key`, `sig`, `tempauth`, ...), or
#   * its LAST word is one of _SECRET_WORDS (`access_token`, `X-Amz-Signature`, `client_secret`, `refreshToken`), or
#   * it ends in `key` after a word that makes it a credential (`api_key`, `X-Api-Key`, `subscription-key`).
_WHOLE_NAMES = frozenset({
    "tempauth", "auth", "authorization", "token", "sig", "signature", "secret", "password", "passwd", "pwd",
    "code", "key", "apikey", "sas", "credential", "credentials", "assertion", "jwt", "bearer",
})
_SECRET_WORDS = frozenset({"token", "secret", "signature", "password", "passwd", "credential", "assertion", "apikey"})
_KEY_PREFIXES = frozenset({"api", "access", "secret", "private", "subscription", "signing", "encryption", "x"})

_QUERY_PARAM = re.compile(r"([?&;])([\w.\-]+)=([^&\s'\"<>#]+)")
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]{12,}")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

QUIET_LOGGERS = ("httpx", "httpcore")


def _sensitive_name(name: str) -> bool:
    words = [w.lower() for w in re.split(r"[-_.]", _CAMEL.sub("-", name)) if w]
    if not words:
        return False
    if "".join(words) in _WHOLE_NAMES:
        return True
    if words[-1] in _SECRET_WORDS:
        return True
    return len(words) > 1 and words[-1] == "key" and words[-2] in _KEY_PREFIXES


def _redact_param(match: "re.Match[str]") -> str:
    sep, name, value = match.groups()
    return f"{sep}{name}={REDACTED}" if _sensitive_name(name) else match.group(0)


def redact(text: str) -> str:
    if "=" not in text and "earer" not in text:
        return text
    return _BEARER.sub(rf"\1{REDACTED}", _QUERY_PARAM.sub(_redact_param, text))


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
        # The FORMATTED message is what gets written, so that is what is redacted. Redacting `msg` and the str
        # args separately missed everything that only becomes text on formatting (an `httpx.URL` argument, an
        # exception object). Args are consumed here, so the record carries the finished, redacted message.
        try:
            message = record.getMessage()
        except Exception:  # a malformed format string is the caller's bug; do not turn it into a lost record
            message = None
        if message is not None:
            record.msg, record.args = redact(message), None
        elif isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.exc_info and record.exc_info[0] is not None and not record.exc_text:
            # Tracebacks are rendered later by the Formatter and name the URLs of failed requests. Render once
            # here, redacted; the Formatter uses `exc_text` when it is set.
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        return record

    logging.setLogRecordFactory(factory)

    import structlog

    config = structlog.get_config()
    processors = list(config.get("processors", []))
    if structlog_processor not in processors and processors:
        # Just BEFORE the renderer: after everything that adds text (contextvars, stack info, rendered
        # exceptions), so what is redacted is what is written. Ahead of the renderer, redaction placed first
        # never saw bound contextvars or a formatted traceback. `format_exc_info` is added in front of it so an
        # exception is already text by then (the default ConsoleRenderer would render it itself, too late).
        renderer = processors.pop()
        if structlog.processors.format_exc_info not in processors:
            processors.append(structlog.processors.format_exc_info)
        processors.extend([structlog_processor, renderer])
        structlog.configure(processors=processors)
    _installed = True
