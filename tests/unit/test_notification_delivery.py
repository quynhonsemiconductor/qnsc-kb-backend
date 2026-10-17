"""deliver_notification_queue: bounded, backed-off, starvation-proof email delivery.

The database is faked at the session boundary (rows are plain objects; the SELECT is captured and compiled so its
WHERE / ORDER BY / LIMIT are asserted as SQL), the sender is scripted. What is under test is the loop's decisions:
which failures are permanent, when a row is dead, what stops a batch, and when a row may be tried again.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import structlog
from sqlalchemy.dialects import postgresql

from src.services import email
from src.workers import tasks


class FakeSession:
    def __init__(self, rows):
        self.rows, self.statements, self.committed = rows, [], False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: list(self.rows)))

    async def commit(self):
        self.committed = True


class ScriptedSender(email.EmailSender):
    """Outcome per recipient: None = delivered, an exception = raised."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = outcomes, []

    async def send(self, *, to, subject, text, html=None, mailbox=None):
        self.calls.append(to)
        outcome = self.outcomes.get(to)
        if outcome:
            raise outcome


def row(to, *, attempts=0):
    return SimpleNamespace(id=uuid.uuid4(), type="email", status="pending", attempts=attempts, last_error=None,
                           sent_at=None, next_attempt_at=None, payload={"to": to, "subject": "s", "text": "t"})


@pytest.fixture
def run(monkeypatch):
    def go(rows, outcomes):
        session, sender = FakeSession(rows), ScriptedSender(outcomes)
        monkeypatch.setattr(tasks, "SessionLocal", lambda: session)

        async def no_context(*a, **k):
            return None

        monkeypatch.setattr(tasks, "set_database_context", no_context)
        monkeypatch.setattr(email, "get_email_sender", lambda: sender)
        monkeypatch.setattr(tasks.settings, "MAIL_GRAPH_CLIENT_ID", "client", raising=False)
        with structlog.testing.capture_logs() as logs:
            tasks.deliver_notification_queue()
        return session, sender, logs

    return go


def err(status, *, stage="send", retry_after=None):
    return email.MailSendError(f"HTTP {status}", status=status, stage=stage, retry_after=retry_after)


def test_a_delivered_row_is_sent_and_a_transient_failure_backs_off(run):
    ok, flaky = row("ok@x"), row("flaky@x")
    session, sender, _ = run([ok, flaky], {"flaky@x": err(500)})

    assert ok.status == "sent" and ok.sent_at and ok.next_attempt_at is None and ok.attempts == 1
    assert flaky.status == "failed" and flaky.attempts == 1 and "HTTP 500" in flaky.last_error
    wait = (flaky.next_attempt_at - datetime.utcnow()).total_seconds()
    assert 25 <= wait <= 31  # first backoff: 30 s
    assert session.committed


@pytest.mark.parametrize("status", [400, 403, 404])
def test_a_permanent_failure_is_dead_at_once_and_alarmed(run, status):
    doomed = row("d@x")
    _, _, logs = run([doomed], {"d@x": err(status)})

    assert doomed.status == "dead" and doomed.attempts == 1 and doomed.next_attempt_at is None
    assert any(e["event"] == "Notification marked dead" and e["log_level"] == "error" and e["permanent"] for e in logs)


def test_the_attempt_cap_turns_a_transient_failure_into_a_dead_row(run):
    last = row("l@x", attempts=tasks.MAIL_MAX_ATTEMPTS - 1)
    _, _, logs = run([last], {"l@x": err(500)})
    assert last.status == "dead" and last.attempts == tasks.MAIL_MAX_ATTEMPTS
    assert any(e["event"] == "Notification marked dead" and not e["permanent"] for e in logs)


def test_backoff_doubles_and_is_capped():
    assert [tasks.mail_backoff_seconds(n) for n in (1, 2, 3, 4)] == [30, 60, 120, 240]
    assert tasks.mail_backoff_seconds(50) == tasks.MAIL_BACKOFF_CAP_SECONDS


def test_throttling_stops_the_batch_and_honours_retry_after(run):
    first, second, third = row("a@x"), row("b@x"), row("c@x")
    _, sender, _ = run([first, second, third], {"a@x": err(429, retry_after=600)})

    assert sender.calls == ["a@x"]  # the other two were not tried
    assert (first.next_attempt_at - datetime.utcnow()).total_seconds() > 590  # Retry-After beat the 30 s backoff
    assert second.attempts == third.attempts == 0 and second.status == third.status == "pending"


def test_a_token_endpoint_failure_stops_the_batch_without_spending_attempts_on_the_rest(run):
    rows = [row(f"{n}@x") for n in range(5)]
    _, sender, _ = run(rows, {"0@x": err(401, stage="token")})

    assert sender.calls == ["0@x"]
    assert [r.attempts for r in rows] == [1, 0, 0, 0, 0]
    assert rows[0].status == "failed"  # not dead: the credential may be fixed and mail resumes


def test_a_permanent_failure_does_not_stop_the_batch(run):
    bad, good = row("bad@x"), row("good@x")
    _, sender, _ = run([bad, good], {"bad@x": err(403)})
    assert sender.calls == ["bad@x", "good@x"] and good.status == "sent" and bad.status == "dead"


def test_the_select_leaves_backing_off_and_dead_rows_out_and_orders_new_mail_first(run):
    session, _, _ = run([], {})
    sql = str(session.statements[0].compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "notification_queue.status IN ('pending', 'failed')" in sql  # 'dead' and 'sent' are never selected
    assert "notification_queue.next_attempt_at IS NULL OR notification_queue.next_attempt_at <=" in sql
    assert "ORDER BY notification_queue.next_attempt_at ASC NULLS FIRST, notification_queue.created_at" in sql
    assert "LIMIT 100" in sql and "FOR UPDATE SKIP LOCKED" in sql


def test_unconfigured_mail_delivers_nothing_and_says_so_every_pass(monkeypatch):
    sent = ScriptedSender({})
    session = FakeSession([row("a@x")])
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    monkeypatch.setattr(tasks.settings, "MAIL_GRAPH_CLIENT_ID", "", raising=False)
    monkeypatch.setattr(email, "get_email_sender", lambda: email.MicrosoftGraphEmailSender())
    with structlog.testing.capture_logs() as logs:
        tasks.deliver_notification_queue()
    assert session.statements == []  # not even a SELECT: no row spends an attempt
    assert any(e["event"].startswith("Mail is not configured") and e["log_level"] == "error" for e in logs)
    assert sent.calls == []


def test_a_provider_that_must_not_run_here_is_an_error_not_a_crash(monkeypatch):
    def refuse():
        raise RuntimeError("EMAIL_PROVIDER=console is for local development only")

    session = FakeSession([row("a@x")])
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    monkeypatch.setattr(email, "get_email_sender", refuse)
    with structlog.testing.capture_logs() as logs:
        tasks.deliver_notification_queue()
    assert session.statements == [] and any(e["event"] == "Mail is not configured" for e in logs)
