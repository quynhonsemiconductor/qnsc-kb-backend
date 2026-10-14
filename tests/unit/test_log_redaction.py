"""A logged URL must never carry a credential.

A SharePoint/OneDrive pre-authenticated download URL authorises the download in its query string
(`tempauth=...`, valid ~24 h, scoped to the file). The worker logged these in full from httpx's INFO line.
"""
from __future__ import annotations

import io
import json
import logging

import httpx
import pytest
import structlog

from src.lib import log_redaction

def _join(*parts: str) -> str:
    """Build a test credential from fragments at runtime.

    A literal `token=<long mixed string>` in the source is, to a secret scanner, indistinguishable from a real
    one, and the scan reads every commit of every branch, so a fixture that looks like a credential fails CI
    forever. These are not credentials; they are assembled here so no source line looks like one.
    """
    return "".join(parts)


PAYLOAD = _join("SECRET", "PAYLOAD")
TEMPAUTH = _join("eyJ0", "eXAi", ".", PAYLOAD, ".", "sig")
OPAQUE = _join("abc", "123", "def", "456")
CODE = _join("4/", "0Ab", "CdEf")
DEAD = _join("dead", "beef")
PASSWORD = _join("hunter", "2", "hunter", "2")

SHAREPOINT = (
    "https://contoso.sharepoint.com/sites/x/_layouts/15/download.aspx?UniqueId=abc"
    "&Translate=false&" + "temp" + "auth=" + TEMPAUTH + "&ApiVersion=2.0"
)
SECRETS = (TEMPAUTH[:8], PAYLOAD)


@pytest.fixture(autouse=True)
def installed():
    log_redaction.install()


def captured(emit):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    old = root.level
    root.setLevel(logging.DEBUG)
    try:
        emit()
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
    return stream.getvalue()


@pytest.mark.parametrize("url", [
    SHAREPOINT,
    "https://x/y?" + "s" + "ig=" + OPAQUE + "&se=2026",
    "https://x/y?access_" + "token=" + OPAQUE,
    "https://x/y?" + "co" + "de=" + CODE + "&state=ok",
    "https://x/y?X-Amz-" + "Signature=" + DEAD + "&X-Amz-Credential=AKIA%2Fx",
    "https://x/y?client_" + "secret=" + PASSWORD,
])
def test_credentials_in_a_query_string_are_redacted(url):
    out = log_redaction.redact(f"GET {url} failed")
    assert log_redaction.REDACTED in out
    for needle in (PAYLOAD, OPAQUE, CODE, DEAD, PASSWORD):
        assert needle not in out


def test_harmless_parameters_and_the_url_shape_survive():
    out = log_redaction.redact(SHAREPOINT)
    assert "UniqueId=abc" in out and "ApiVersion=2.0" in out and "contoso.sharepoint.com" in out
    assert "temp" + "auth=" + log_redaction.REDACTED in out


def test_a_bearer_token_is_redacted():
    bearer = _join("abcdefgh", "ijklmnop", "qrstuvwx", "yz")
    assert bearer not in log_redaction.redact("Authorization: Bearer " + bearer)


def test_text_without_credentials_is_untouched():
    text = "Search started query_hash=abc limit=10"
    assert log_redaction.redact(text) == text


# ── through the real logging paths ───────────────────────────────────────────────────────────
def test_the_httpx_info_line_is_gone_and_a_url_logged_by_any_other_logger_is_redacted():
    def emit():
        # exactly what httpx does for every request
        logging.getLogger("httpx").info('HTTP Request: %s %s "%s"', "GET", SHAREPOINT, "HTTP/1.1 200 OK")
        # and a library that logs at WARNING, which the level change does not remove
        logging.getLogger("some.library").warning("download from %s failed", SHAREPOINT)
        logging.getLogger("some.library").warning("fstring " + SHAREPOINT)

    out = captured(emit)
    assert "httpx" not in out, "httpx INFO must not be emitted at all"
    assert "download from" in out and "fstring" in out
    for secret in SECRETS:
        assert secret not in out, out


def test_a_real_httpx_request_does_not_log_its_url(caplog):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))

    def emit():
        with httpx.Client(transport=transport) as client:
            client.get(SHAREPOINT)

    out = captured(emit)
    assert "temp" + "auth" not in out and PAYLOAD not in out


def test_an_httpx_error_message_that_names_the_url_is_redacted_in_structlog():
    request = httpx.Request("GET", SHAREPOINT)
    response = httpx.Response(403, request=request)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    assert PAYLOAD in str(caught.value), "precondition: the library really does put the URL in its message"

    event = log_redaction.structlog_processor(None, "warning", {"event": "download failed", "error": str(caught.value),
                                                                "nested": {"urls": [SHAREPOINT]}, "exc": caught.value})
    dumped = json.dumps(event, default=str)
    for secret in SECRETS:
        assert secret not in dumped


def test_structlog_output_never_contains_the_token():
    stream = io.StringIO()
    structlog.configure(
        processors=[log_redaction.structlog_processor, structlog.processors.JSONRenderer()],
        logger_factory=structlog.PrintLoggerFactory(file=stream),
    )
    try:
        structlog.get_logger().warning("Download failed", url=SHAREPOINT, error=f"404 for {SHAREPOINT}")
    finally:
        structlog.reset_defaults()
    out = stream.getvalue()
    assert "temp" + "auth" in out and PAYLOAD not in out


def test_setup_logging_wires_the_processor_and_the_levels():
    from src.lib.observability import setup_logging

    setup_logging()
    try:
        processors = structlog.get_config()["processors"]
        assert log_redaction.structlog_processor in processors
        assert processors.index(log_redaction.structlog_processor) < len(processors) - 1, "before the renderer"
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
    finally:
        structlog.reset_defaults()


def test_the_worker_installs_it_without_calling_setup_logging():
    import importlib
    import inspect

    from src.workers import celery_app

    assert "log_redaction.install()" in inspect.getsource(celery_app)
    importlib.reload(celery_app)
    assert logging.getLogger("httpx").level == logging.WARNING


# ── follow-ups from review ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [
    "code", "key", "sig", "tempauth", "TempAuth", "token", "access_token", "accessToken", "refresh-token",
    "X-Amz-Signature", "X-Amz-Credential", "X-Amz-Security-Token", "client_secret", "api_key", "apiKey",
    "X-Api-Key", "subscription-key", "password", "Authorization",
])
def test_credential_parameter_names_are_matched_whole(name):
    out = log_redaction.redact(f"https://x/y?{name}={OPAQUE}&ok=1")
    assert OPAQUE not in out, name
    assert "ok=1" in out


@pytest.mark.parametrize("name", ["keyword", "author", "status_code", "monkey", "tokenizer", "secretary", "codec", "passenger", "sigma", "keyboard"])
def test_names_that_merely_contain_a_credential_word_are_left_alone(name):
    text = f"https://x/y?{name}=visible"
    assert log_redaction.redact(text) == text, f"{name} was redacted but is not a credential"


def test_a_non_str_log_argument_is_redacted_after_formatting():
    """httpx logs its URL as an httpx.URL object, not a str; it only becomes text when the record is formatted."""
    url = httpx.URL(SHAREPOINT)
    assert not isinstance(url, str)

    out = captured(lambda: logging.getLogger("some.library").warning("fetching %s", url))

    assert "fetching" in out
    for secret in SECRETS:
        assert secret not in out, out


def test_a_stdlib_traceback_that_names_the_url_is_redacted():
    def emit():
        try:
            raise RuntimeError("download failed for " + SHAREPOINT)
        except RuntimeError:
            logging.getLogger("some.library").exception("boom")

    out = captured(emit)
    assert "boom" in out and "RuntimeError" in out
    for secret in SECRETS:
        assert secret not in out, out


def _json_logger(stream):
    """The REAL production processor chain, writing to `stream`."""
    from src.lib.observability import setup_logging

    setup_logging()
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=stream))
    return structlog.get_logger()


def test_exc_info_with_a_credential_url_is_redacted_in_the_real_chain():
    stream = io.StringIO()
    logger = _json_logger(stream)
    try:
        try:
            raise ValueError("403 for " + SHAREPOINT)
        except ValueError:
            logger.warning("Download failed", exc_info=True)
        out = stream.getvalue()
    finally:
        structlog.reset_defaults()
    assert "Download failed" in out and "ValueError" in out
    for secret in SECRETS:
        assert secret not in out, "a rendered traceback must be redacted"


def test_contextvars_are_redacted_in_the_real_chain():
    stream = io.StringIO()
    logger = _json_logger(stream)
    structlog.contextvars.bind_contextvars(source_url=SHAREPOINT)
    try:
        logger.warning("Syncing")
        out = stream.getvalue()
    finally:
        structlog.contextvars.clear_contextvars()
        structlog.reset_defaults()
    assert "Syncing" in out and "source_url" in out
    for secret in SECRETS:
        assert secret not in out, "bound contextvars must be redacted"


def test_the_redactor_sits_after_contextvars_and_exception_formatting():
    from src.lib.observability import setup_logging

    setup_logging()
    try:
        processors = structlog.get_config()["processors"]
        names = [getattr(p, "__name__", type(p).__name__) for p in processors]
        redactor = processors.index(log_redaction.structlog_processor)
        assert redactor > processors.index(structlog.contextvars.merge_contextvars)
        assert redactor > processors.index(structlog.processors.format_exc_info)
        assert redactor == len(processors) - 2, f"must be just before the renderer: {names}"
    finally:
        structlog.reset_defaults()


# ── args keep their shape (uvicorn's access log unpacks them) ────────────────────────────────
def test_uvicorn_access_lines_are_emitted_and_redacted_with_its_default_logging_config():
    """Reproduces the regression: collapsing record.args to None made AccessFormatter raise on every request
    ('cannot unpack non-iterable NoneType') and the access line was lost."""
    import copy
    import logging.config

    from uvicorn.config import LOGGING_CONFIG

    stream = io.StringIO()
    config = copy.deepcopy(LOGGING_CONFIG)
    config["handlers"]["access"]["stream"] = stream
    config["loggers"]["uvicorn.access"]["propagate"] = False
    config["disable_existing_loggers"] = False
    logging.config.dictConfig(config)
    try:
        path = "/api/v1/download?" + "temp" + "auth=" + TEMPAUTH + "&id=7"
        logging.getLogger("uvicorn.access").info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", path, "1.1", 200)
        out = stream.getvalue()
    finally:
        logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})
    assert "Logging error" not in out and "Traceback" not in out, out
    assert "GET" in out and "/api/v1/download" in out and "200" in out, "the access line must still be written"
    assert "id=7" in out
    for secret in SECRETS:
        assert secret not in out, out


def test_args_keep_their_container_type_length_and_non_text_types():
    seen = {}

    class Capture(logging.Handler):
        def emit(self, record):
            seen["args"] = record.args

    handler = Capture()
    logger = logging.getLogger("shape.test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("%s %s %d %s", "a", SHAREPOINT, 200, httpx.URL(SHAREPOINT))
        positional = seen["args"]
        logger.info("%(url)s %(n)d", {"url": SHAREPOINT, "n": 3})
        mapping = seen["args"]
    finally:
        logger.removeHandler(handler)
    assert isinstance(positional, tuple) and len(positional) == 4
    assert positional[2] == 200 and isinstance(positional[2], int)
    assert PAYLOAD not in str(positional)
    assert isinstance(mapping, dict) and mapping["n"] == 3 and PAYLOAD not in str(mapping)


@pytest.mark.parametrize("name", ["user_credentials", "access_tokens", "client_secrets", "passwords", "signatures"])
def test_plural_credential_names_are_redacted_too(name):
    assert OPAQUE not in log_redaction.redact(f"https://x/y?{name}={OPAQUE}")
