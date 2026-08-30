"""A document that arrives through a connector must get the AI reading view too.

Connector drafts were created `restructure_status="lossless_ready"` with
`restructured_body_md` set to the raw extracted text. Nothing ever asked for the AI pass:
`dispatch_restructure_pending_draft` was called only from the upload endpoints, so a
document that came in through a connector kept its unformatted extraction unless a
reviewer noticed and pressed "Retry AI format" on it. For a 241-file drive that is not a
workflow, and the difference is invisible until someone opens the draft.

Two things here are easy to get wrong and are pinned below.

The dispatch happens AFTER the caller commits. The worker loads the draft by id, so a
task queued inside the transaction can be picked up before the row exists and find
nothing. And an item that rolls back must not leave anything queued at all.

A broker that is down must not cost the document. The draft keeps the lossless view, says
so, and stays retryable -- it is not left claiming to be queued forever.
"""
from __future__ import annotations

import asyncio
import types
import uuid

import pytest

from src.domain.cloud_sync import _dispatch_queued_formats, _queue_draft_formatting


class _Draft:
    def __init__(self, created_by=None):
        self.id = uuid.uuid4()
        self.created_by = created_by
        self.restructure_status = "lossless_ready"
        self.restructure_error = "stale error from a previous revision"


class _Connector:
    company_domain = "qnsc.vn"


class _DB:
    """Records what the failure path writes, without a database."""

    def __init__(self):
        self.statements = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)

    async def commit(self):
        self.commits += 1


@pytest.fixture
def dispatcher(monkeypatch):
    """Stub src.workers.tasks: importing it for real needs celery, which CI has and a
    developer machine does not."""
    calls: list[tuple] = []
    state = {"result": True, "raises": None}

    def dispatch_restructure_pending_draft(draft_id, company_domain, user_id):
        calls.append((draft_id, company_domain, user_id))
        if state["raises"] is not None:
            raise state["raises"]
        return state["result"]

    module = types.ModuleType("src.workers.tasks")
    module.dispatch_restructure_pending_draft = dispatch_restructure_pending_draft
    monkeypatch.setitem(__import__("sys").modules, "src.workers.tasks", module)
    return types.SimpleNamespace(calls=calls, state=state)


def test_a_connector_draft_is_marked_for_formatting():
    draft = _Draft(created_by=uuid.uuid4())
    queue: list = []

    _queue_draft_formatting(draft, _Connector(), queue)

    assert draft.restructure_status == "queued"
    # A previous revision's error must not stay on a draft that is about to be redone.
    assert draft.restructure_error is None
    assert queue == [(str(draft.id), "qnsc.vn", str(draft.created_by))]


def test_a_draft_with_no_creator_keeps_the_old_behaviour():
    """The worker checks the AI feature flag against a user. With no creator there is no
    user, the pass would be skipped anyway, and claiming "queued" would be a lie."""
    draft = _Draft(created_by=None)
    queue: list = []

    _queue_draft_formatting(draft, _Connector(), queue)

    assert draft.restructure_status == "lossless_ready"
    assert queue == []


def test_a_caller_that_passes_no_queue_is_left_alone():
    """The parameter is optional, and an absent queue means the caller has nowhere to
    dispatch from -- marking the draft queued would strand it."""
    draft = _Draft(created_by=uuid.uuid4())
    _queue_draft_formatting(draft, _Connector(), None)
    assert draft.restructure_status == "lossless_ready"


def test_every_queued_draft_is_dispatched_once(dispatcher):
    db = _DB()
    queue = [("d1", "qnsc.vn", "u1"), ("d2", "qnsc.vn", "u1")]

    asyncio.run(_dispatch_queued_formats(db, queue))

    assert dispatcher.calls == [("d1", "qnsc.vn", "u1"), ("d2", "qnsc.vn", "u1")]
    # Cleared, or the next committed item re-queues everything before it.
    assert queue == []
    assert db.statements == [] and db.commits == 0


def test_nothing_queued_touches_neither_broker_nor_database(dispatcher):
    db = _DB()
    asyncio.run(_dispatch_queued_formats(db, []))
    assert dispatcher.calls == [] and db.commits == 0


def test_a_broker_that_refuses_leaves_the_draft_retryable(dispatcher):
    """Returning False is how dispatch reports a missing broker."""
    dispatcher.state["result"] = False
    db = _DB()
    queue = [(str(uuid.uuid4()), "qnsc.vn", "u1")]

    asyncio.run(_dispatch_queued_formats(db, queue))

    assert db.commits == 1, "the fallback status must be persisted"
    assert queue == []


def test_a_broker_that_raises_does_not_fail_the_sync(dispatcher):
    """This runs inside the item loop. An exception here would be counted as a failed
    document, and 25 of those abandon the whole scope."""
    dispatcher.state["raises"] = RuntimeError("no broker")
    db = _DB()
    queue = [(str(uuid.uuid4()), "qnsc.vn", "u1")]

    asyncio.run(_dispatch_queued_formats(db, queue))

    assert db.commits == 1
    assert queue == []


def test_one_bad_dispatch_does_not_strand_the_others(dispatcher):
    """The loop must keep going: the second draft is not punished for the first."""
    good, bad = str(uuid.uuid4()), str(uuid.uuid4())
    seen: list[str] = []

    def flaky(draft_id, company_domain, user_id):
        seen.append(draft_id)
        return draft_id != bad

    __import__("sys").modules["src.workers.tasks"].dispatch_restructure_pending_draft = flaky
    db = _DB()

    asyncio.run(_dispatch_queued_formats(db, [(bad, "qnsc.vn", "u"), (good, "qnsc.vn", "u")]))

    assert seen == [bad, good]
    assert db.commits == 1
