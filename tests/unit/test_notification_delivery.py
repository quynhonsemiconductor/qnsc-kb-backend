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
    """``rows`` is what the SELECT returns (the first 100 deliverable rows); ``others`` are deliverable rows beyond
    that batch. An UPDATE on notification_queue is compiled to Postgres SQL, recorded, and APPLIED to every row it
    would match (its WHERE is asserted separately), so a test can say which rows ended up deferred."""

    def __init__(self, rows, others=()):
        self.rows, self.others = rows, list(others)
        self.statements, self.updates, self.committed = [], [], False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement):
        self.statements.append(statement)
        if statement.is_update:
            compiled = statement.compile(dialect=postgresql.dialect())
            self.updates.append(str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))
            moments = sorted(v for v in compiled.params.values() if isinstance(v, datetime))
            earliest_default, until = moments[0], moments[-1]  # coalesce(col, now) and the deferral target
            for r in [*self.rows, *self.others]:
                if r.type == "email" and r.status in ("pending", "failed"):
                    r.next_attempt_at = max(r.next_attempt_at or earliest_default, until)
            return SimpleNamespace()
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


def row(to, *, attempts=0, age=timedelta(minutes=5)):
    return SimpleNamespace(id=uuid.uuid4(), type="email", status="pending", attempts=attempts, last_error=None,
                           sent_at=None, next_attempt_at=None, created_at=datetime.utcnow() - age,
                           payload={"to": to, "subject": "s", "text": "t"})


@pytest.fixture
def run(monkeypatch):
    def go(rows, outcomes, others=()):
        session, sender = FakeSession(rows, others), ScriptedSender(outcomes)
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


@pytest.mark.parametrize("status", [400, 413])
def test_a_refused_message_is_dead_at_once_and_alarmed(run, status):
    doomed = row("d@x")
    _, _, logs = run([doomed], {"d@x": err(status)})

    assert doomed.status == "dead" and doomed.attempts == 1 and doomed.next_attempt_at is None
    assert any(e["event"] == "Notification marked dead" and e["log_level"] == "error" and e["permanent"] for e in logs)


def test_an_outage_longer_than_an_hour_does_not_kill_mail_but_age_does(run):
    """The old rule (8 attempts) made every queued message dead after about 64 minutes of any outage."""
    young_but_many_attempts = row("a@x", attempts=20, age=timedelta(hours=2))
    _, _, _ = run([young_but_many_attempts], {"a@x": err(500)})
    assert young_but_many_attempts.status == "failed" and young_but_many_attempts.next_attempt_at is not None

    stale = row("s@x", age=tasks.MAIL_MAX_AGE + timedelta(minutes=1))
    _, _, logs = run([stale], {"s@x": err(500)})
    assert stale.status == "dead" and any(e["event"] == "Notification marked dead" and e["too_old"] for e in logs)


def test_backoff_doubles_and_is_capped():
    assert [tasks.mail_backoff_seconds(n) for n in (1, 2, 3, 4)] == [30, 60, 120, 240]
    assert tasks.mail_backoff_seconds(50) == tasks.MAIL_BACKOFF_CAP_SECONDS


def test_throttling_stops_the_batch_honours_retry_after_and_charges_no_attempt(run):
    first, second, third = row("a@x"), row("b@x"), row("c@x")
    _, sender, _ = run([first, second, third], {"a@x": err(429, retry_after=600)})

    assert sender.calls == ["a@x"]  # the other two were not tried
    assert (first.next_attempt_at - datetime.utcnow()).total_seconds() > 590  # Retry-After beat the 30 s backoff
    assert first.attempts == 0 and "HTTP 429" in first.last_error  # an outage is not the row's fault
    assert second.attempts == third.attempts == 0 and second.status == third.status == "pending"


def test_a_token_endpoint_failure_stops_the_batch_without_spending_any_attempt(run):
    rows = [row(f"{n}@x") for n in range(5)]
    _, sender, _ = run(rows, {"0@x": err(401, stage="token")})

    assert sender.calls == ["0@x"]
    assert [r.attempts for r in rows] == [0, 0, 0, 0, 0]
    assert rows[0].status == "pending" and rows[0].next_attempt_at is not None  # not dead: mail resumes when fixed


def test_a_long_outage_cannot_walk_every_row_to_dead(run):
    """Each pass tries one row and stops; 50 passes of a token outage leave every row alive with 0 attempts."""
    rows = [row(f"{n}@x") for n in range(5)]
    for _ in range(50):
        for r in rows:
            r.next_attempt_at = None  # as if the backoff had elapsed
        run(rows, {f"{n}@x": err(401, stage="token") for n in range(5)})
    assert all(r.status in {"pending", "failed"} and r.attempts == 0 for r in rows)


@pytest.mark.parametrize("status", [403, 404])
def test_a_refused_sender_stops_the_batch_keeps_the_rows_and_alarms(run, status):
    """Scope missing or propagating (up to ~2 h), mailbox missing: a SENDER problem, so no row may die for it."""
    rows = [row(f"{n}@x") for n in range(100)]
    _, sender, logs = run(rows, {"0@x": err(status)})

    assert sender.calls == ["0@x"]
    assert all(r.status == "pending" and r.attempts == 0 for r in rows)
    assert not any(e["event"] == "Notification marked dead" for e in logs)
    assert 14 * 60 <= (rows[0].next_attempt_at - datetime.utcnow()).total_seconds() <= 15 * 60 + 5
    assert any(e["event"] == "Mail sender refused" and e["log_level"] == "error" and e["status"] == status for e in logs)


def test_a_refused_message_does_not_stop_the_batch(run):
    bad, good = row("bad@x"), row("good@x")
    _, sender, _ = run([bad, good], {"bad@x": err(400)})
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


# ── a stopped batch defers EVERY deliverable row, not only the one that failed ──────────────────
@pytest.mark.parametrize("failure,wait_at_least", [
    (err(429, retry_after=600), 590),
    (err(403), 14 * 60),
    (err(401, stage="token"), 25),
])
def test_a_stopped_batch_defers_every_email_row_including_those_beyond_the_batch(run, failure, wait_at_least):
    """Before: only the failing row got next_attempt_at; the other 99 stayed NULL and were hit again 30 s later,
    forever, through the Retry-After / 15-minute wait that was meant to stop exactly that."""
    batch = [row(f"{n}@x") for n in range(100)]
    beyond = [row(f"b{n}@x") for n in range(50)]  # deliverable, but past LIMIT 100
    session, sender, _ = run(batch, {"0@x": failure}, others=beyond)

    assert sender.calls == ["0@x"]  # the batch stopped
    now = datetime.utcnow()
    for r in [*batch, *beyond]:
        assert r.next_attempt_at is not None, r.payload["to"]
        assert (r.next_attempt_at - now).total_seconds() >= wait_at_least - 5, r.payload["to"]
    assert all(r.attempts == 0 for r in [*batch, *beyond])  # attempts untouched
    assert len(session.updates) == 1  # one UPDATE, not a row-by-row loop


def test_the_deferral_never_shortens_a_longer_existing_wait_and_skips_dead_and_other_types(run):
    longer = row("l@x")
    longer.next_attempt_at = datetime.utcnow() + timedelta(hours=2)
    dead, sent, slack = row("d@x"), row("s@x"), row("k@x")
    dead.status, sent.status, slack.type = "dead", "sent", "slack"
    first = row("a@x")
    session, _, _ = run([first, longer], {"a@x": err(429, retry_after=60)}, others=[dead, sent, slack])

    assert (longer.next_attempt_at - datetime.utcnow()).total_seconds() > 2 * 3600 - 5  # greatest(), not overwrite
    assert dead.next_attempt_at is None and sent.next_attempt_at is None and slack.next_attempt_at is None
    sql = session.updates[0]
    assert "greatest(coalesce(notification_queue.next_attempt_at" in sql
    assert "notification_queue.type = 'email'" in sql and "notification_queue.status IN ('pending', 'failed')" in sql
    assert "attempts" not in sql  # attempts are not touched
    # the WHERE is exactly these two conditions: no id, no created_at, nothing that narrows it to a subset
    assert sql.strip().endswith("WHERE notification_queue.type = 'email' AND notification_queue.status IN ('pending', 'failed')")


def test_a_successful_or_message_level_pass_runs_no_deferral(run):
    ok, bad = row("ok@x"), row("bad@x")
    session, _, _ = run([ok, bad], {"bad@x": err(400)})
    assert session.updates == []
