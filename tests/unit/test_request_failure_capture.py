"""A 500 must record WHY it happened, not just that it did.

The middleware always caught every unhandled exception and logged it, but `record_request_metric`
persisted only the status code — so diagnosing a production failure meant reading CloudWatch,
which needs an AWS role switch. The exception was formatted and then discarded one line later.

These tests pin the properties that make the failure diagnosable without AWS:

* the exception type and message reach the row
* a chained `raise ... from ...` cause survives, since that is usually the real fault
* the detail is capped, because a traceback is unbounded and this runs on the request path
* a SUCCESSFUL request stores nothing, so the table does not grow for traffic nobody asks about
* a storage failure cannot mask the exception it was trying to record
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap

from src.api import main as api_main
from src.api.main import ERROR_DETAIL_MAX_CHARS, record_request_metric
from src.models.ops import ApiRequestMetric


class _FakeSession:
    """Captures what would be written, and can be told to fail on commit."""

    instances: list = []
    committed = False
    fail_on_commit = False

    def __init__(self):
        type(self).instances = []
        type(self).committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def add(self, item):
        type(self).instances.append(item)

    async def commit(self):
        if type(self).fail_on_commit:
            raise RuntimeError("database is unreachable")
        type(self).committed = True


def _record(exc=None, status_code=500, monkeypatch=None):
    """Run record_request_metric against a fake session and return the row it added."""
    monkeypatch.setattr(api_main, "SessionLocal", _FakeSession)
    monkeypatch.setattr(api_main, "record_request", lambda *args, **kwargs: None)
    asyncio.run(
        record_request_metric(
            "req-123", "POST", "/upload-source", status_code, 91.5, exc
        )
    )
    rows = [item for item in _FakeSession.instances if isinstance(item, ApiRequestMetric)]
    return rows[0] if rows else None


def _chained_exception() -> BaseException:
    """A real nested exception, so the traceback under test is not synthetic."""

    def inner():
        raise ValueError("PaddleOCR failed to load model weights")

    try:
        try:
            inner()
        except ValueError as cause:
            raise RuntimeError("upload-source extraction failed") from cause
    except RuntimeError as exc:
        return exc
    raise AssertionError("unreachable")


def test_a_failure_records_the_exception_type(monkeypatch):
    row = _record(_chained_exception(), monkeypatch=monkeypatch)

    assert row.error_type == "RuntimeError"


def test_a_failure_records_the_message_first_so_it_survives_truncation(monkeypatch):
    """Message before traceback: a deep traceback must not push the message out."""
    row = _record(_chained_exception(), monkeypatch=monkeypatch)

    assert row.error_detail.startswith("upload-source extraction failed")


def test_the_chained_cause_is_kept_because_it_is_usually_the_real_fault(monkeypatch):
    row = _record(_chained_exception(), monkeypatch=monkeypatch)

    assert "ValueError" in row.error_detail
    assert "PaddleOCR failed to load model weights" in row.error_detail


def test_the_traceback_names_the_code_that_raised(monkeypatch):
    row = _record(_chained_exception(), monkeypatch=monkeypatch)

    assert "Traceback" in row.error_detail
    assert "test_request_failure_capture.py" in row.error_detail


def test_the_detail_is_capped_because_a_traceback_is_unbounded(monkeypatch):
    """This runs on the request path; an unbounded write there is a latency bug."""
    try:
        raise RuntimeError("x" * (ERROR_DETAIL_MAX_CHARS * 3))
    except RuntimeError as exc:
        row = _record(exc, monkeypatch=monkeypatch)

    assert len(row.error_detail) == ERROR_DETAIL_MAX_CHARS


def test_a_successful_request_stores_no_error_columns(monkeypatch):
    """Most traffic succeeds; those rows must not carry empty strings or noise."""
    row = _record(None, status_code=200, monkeypatch=monkeypatch)

    assert row.status_code == 200
    assert row.error_type is None
    assert row.error_detail is None


def test_the_metric_row_still_carries_the_request_id_users_are_shown(monkeypatch):
    """X-Request-ID is returned to the browser, so it is how a report is looked up."""
    row = _record(_chained_exception(), monkeypatch=monkeypatch)

    assert row.request_id == "req-123"
    assert row.path == "/upload-source"


def test_a_storage_failure_cannot_mask_the_exception_being_recorded(monkeypatch):
    """The except block must not shadow `exc`, or the real error is lost twice over."""
    _FakeSession.fail_on_commit = True
    try:
        # Must not raise: failing to persist a metric cannot break the request path.
        row = _record(_chained_exception(), monkeypatch=monkeypatch)
    finally:
        _FakeSession.fail_on_commit = False

    assert row.error_type == "RuntimeError"
    source = inspect.getsource(record_request_metric)
    assert "except Exception as persist_error" in source, (
        "reusing `exc` here would overwrite the exception under record"
    )


def test_the_middleware_passes_the_exception_only_on_the_failure_path():
    """Two call sites, and only one should carry `exc`.

    The success path records a normal response and has no exception to pass; the
    `except` path is the one that must forward it, or the columns exist and are never
    populated.

    Parsed with `ast` rather than string-split, because the argument list contains a
    nested call — `_metric_path_for(request)` — so splitting on ")" truncates the call
    text before reaching the last argument and fails against perfectly correct code.
    """
    source = textwrap.dedent(inspect.getsource(api_main.request_logging_middleware))
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "record_request_metric"
    ]

    assert len(calls) == 2, f"expected a success and a failure call site, got {len(calls)}"

    arg_counts = sorted(len(call.args) for call in calls)
    assert arg_counts == [5, 6], f"one call must add the exception argument: {arg_counts}"

    failure_call = max(calls, key=lambda call: len(call.args))
    assert getattr(failure_call.args[-1], "id", None) == "exc", (
        "the failure path's last argument must be the caught exception"
    )


def test_the_failures_endpoint_requires_global_governance_read():
    from src.api.routers.governance import get_request_failures

    source = inspect.getsource(get_request_failures)

    assert 'require_permission("governance.read", scope="global")' in source


def test_the_failures_endpoint_defaults_to_server_errors_only():
    """A default of 400 would bury real crashes under routine validation rejections."""
    from src.api.routers.governance import get_request_failures

    default = inspect.signature(get_request_failures).parameters["min_status"].default

    assert getattr(default, "default", default) == 500
