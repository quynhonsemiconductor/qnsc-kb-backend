"""Worker tracing: a span per task, and trace_id/span_id on the JSON log record."""

from __future__ import annotations

from types import SimpleNamespace

from src.workers import celery_app


class _Context:
    is_valid = True
    trace_id = 0x0123456789ABCDEF0123456789ABCDEF
    span_id = 0x0123456789ABCDEF


class _Span:
    def get_span_context(self):
        return _Context()


def test_the_log_processor_adds_lowercase_hex_ids(monkeypatch):
    from src.lib.observability import add_trace_context

    monkeypatch.setattr(
        "src.core.tracing.trace",
        SimpleNamespace(get_current_span=lambda: _Span()),
    )
    event = add_trace_context(None, "info", {"event": "x"})
    assert event["trace_id"] == "0123456789abcdef0123456789abcdef"
    assert event["span_id"] == "0123456789abcdef"


def test_an_invalid_span_adds_nothing(monkeypatch):
    from src.lib.observability import add_trace_context

    class Invalid:
        is_valid = False

    monkeypatch.setattr(
        "src.core.tracing.trace",
        SimpleNamespace(
            get_current_span=lambda: SimpleNamespace(get_span_context=lambda: Invalid())
        ),
    )
    event = add_trace_context(None, "info", {"event": "x"})
    assert "trace_id" not in event and "span_id" not in event


def test_a_task_opens_and_closes_one_span(monkeypatch):
    class ContextManager:
        entered = False
        exited = False
        recorded = None
        status = None

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_exc):
            self.exited = True
            return False

        def record_exception(self, exception):
            self.recorded = exception

        def set_status(self, status):
            self.status = status

    context = ContextManager()

    class Tracer:
        def start_as_current_span(self, _name: str):
            return context

    monkeypatch.setattr(celery_app, "get_tracer", lambda: Tracer())
    monkeypatch.setattr(celery_app, "configure_tracing", lambda: None)
    monkeypatch.setattr(celery_app, "setup_logging", lambda: None)

    celery_app._task_started(
        task_id="t1", task=SimpleNamespace(name="replay_outbox_task")
    )
    assert context.entered and not context.exited
    celery_app._task_finished(
        task_id="t1", state="SUCCESS", sender=SimpleNamespace(name="replay_outbox_task")
    )
    assert context.exited and "t1" not in celery_app._task_spans


def test_task_failure_marks_the_span_error_before_postrun(monkeypatch):
    class Span:
        recorded = None
        status = None

        def record_exception(self, exception):
            self.recorded = exception

        def set_status(self, status):
            self.status = status

    class ContextManager:
        exited = False

        def __enter__(self):
            return span

        def __exit__(self, *_exc):
            self.exited = True
            return False

    span = Span()
    context = ContextManager()

    class Tracer:
        def start_as_current_span(self, _name: str):
            return context

    monkeypatch.setattr(celery_app, "get_tracer", lambda: Tracer())
    monkeypatch.setattr(celery_app, "configure_tracing", lambda: None)
    monkeypatch.setattr(celery_app, "setup_logging", lambda: None)
    monkeypatch.setattr(
        "src.core.tracing.trace",
        SimpleNamespace(
            Status=lambda code, message: (code, message),
            StatusCode=SimpleNamespace(ERROR="ERROR"),
        ),
    )

    celery_app._task_started(task_id="t2", task=SimpleNamespace(name="failing"))
    celery_app._task_failed(task_id="t2", exception=ValueError("boom"))
    assert isinstance(span.recorded, ValueError)
    assert span.status == ("ERROR", "boom")
    celery_app._task_finished(
        task_id="t2", state="FAILURE", sender=SimpleNamespace(name="failing")
    )
    assert context.exited and "t2" not in celery_app._task_spans
