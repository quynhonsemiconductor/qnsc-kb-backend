"""A department an Admin opted out of review publishes its drafts without a person.

The decision logic lives in `GovernanceService.approve_draft`, which is covered elsewhere;
what is pinned here is what this sweep adds around it: which drafts it offers to that call,
who it acts as, what it never touches, and that one refusal does not stop the rest.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.domain import department_auto_publish
from src.domain.governance import GovernanceService
from src.repositories.user import UserRepository
from src.workers import tasks

ACTOR = uuid.uuid4()


class _Result:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return self

    def all(self):
        return self.values


class _Db:
    """Answers the sweep's two queries in order: departments, then drafts."""

    def __init__(self, departments, drafts):
        self.answers = [departments, drafts]
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, _statement):
        return _Result(self.answers.pop(0))

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def _department(name="Marketing", by=ACTOR):
    return SimpleNamespace(name=name, auto_publish_by=by)


def _draft(dept="Marketing", sensitivity=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        dept=dept,
        content_metadata={"sensitivity": sensitivity} if sensitivity else {},
    )


@pytest.fixture
def approvals(monkeypatch):
    calls = []
    refuse = set()

    async def approve(self, user, draft_id, **kwargs):
        calls.append((user.id, draft_id, kwargs.get("review_note")))
        if draft_id in refuse:
            raise HTTPException(status_code=409, detail="update_confirmation_required")
        return SimpleNamespace(id=uuid.uuid4())

    async def get_by_id(self, user_id, viewer=None):
        return SimpleNamespace(id=user_id, active=True)

    monkeypatch.setattr(GovernanceService, "approve_draft", approve)
    monkeypatch.setattr(UserRepository, "get_by_id", get_by_id)
    return SimpleNamespace(calls=calls, refuse=refuse)


def _run(db):
    return asyncio.run(department_auto_publish.run(db, "acme.test"))


def test_a_draft_is_approved_as_the_admin_who_opted_the_department_out(approvals):
    draft = _draft()
    db = _Db([_department()], [draft])

    summary = _run(db)

    assert summary["approved"] == 1
    assert approvals.calls == [
        (
            ACTOR,
            draft.id,
            "Auto-published: department Marketing does not require review.",
        )
    ]
    assert [row.action for row in db.added] == ["department_auto_publish"]
    assert db.added[0].user_id == ACTOR and db.added[0].target_id == str(draft.id)


def test_no_department_without_review_means_no_work(approvals):
    db = _Db([], [_draft()])

    assert _run(db) == {"approved": 0, "left_for_review": 0, "skipped": 0}
    assert approvals.calls == []
    assert len(db.answers) == 1  # the drafts were never even queried


def test_a_sensitive_draft_is_never_auto_published(approvals):
    db = _Db(
        [_department()],
        [_draft(sensitivity="restricted"), _draft(sensitivity="confidential")],
    )

    summary = _run(db)

    assert summary == {"approved": 0, "left_for_review": 0, "skipped": 2}
    assert approvals.calls == []


def test_a_draft_the_approval_checks_refuse_stays_for_a_person_and_the_rest_go_on(
    approvals,
):
    blocked, fine = _draft(), _draft()
    approvals.refuse.add(blocked.id)
    db = _Db([_department()], [blocked, fine])

    summary = _run(db)

    assert summary == {"approved": 1, "left_for_review": 1, "skipped": 0}
    assert db.rollbacks == 1
    assert [row.target_id for row in db.added] == [str(fine.id)]


def test_nothing_is_published_when_the_admin_who_opted_out_is_gone(
    monkeypatch, approvals
):
    async def missing(self, user_id, viewer=None):
        return None

    monkeypatch.setattr(UserRepository, "get_by_id", missing)
    db = _Db([_department()], [_draft()])

    assert _run(db)["left_for_review"] == 1
    assert approvals.calls == []


def test_a_deactivated_admin_does_not_publish(monkeypatch, approvals):
    async def inactive(self, user_id, viewer=None):
        return SimpleNamespace(id=user_id, active=False)

    monkeypatch.setattr(UserRepository, "get_by_id", inactive)
    db = _Db([_department()], [_draft()])

    assert _run(db)["left_for_review"] == 1
    assert approvals.calls == []


def test_each_draft_is_approved_as_the_admin_of_its_own_department(approvals):
    other = uuid.uuid4()
    first, second = _draft("Marketing"), _draft("Sales")
    db = _Db(
        [_department("Marketing", ACTOR), _department("Sales", other)], [first, second]
    )

    _run(db)

    assert [(call[0], call[1]) for call in approvals.calls] == [
        (ACTOR, first.id),
        (other, second.id),
    ]


class _Session:
    def __init__(self, domains):
        self.domains = domains

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def execute(self, _statement):
        return _Result(self.domains)


async def _fake_context(*_args, **_kwargs):
    return None


def test_the_beat_task_sweeps_each_company_that_has_such_a_department(monkeypatch):
    seen = []

    async def fake_run(db, company_domain, **_kwargs):
        seen.append(company_domain)
        return {}

    monkeypatch.setattr(
        tasks, "SessionLocal", lambda: _Session(["acme.test", "qnsc.vn"])
    )
    monkeypatch.setattr(tasks, "set_database_context", _fake_context)
    monkeypatch.setattr(department_auto_publish, "run", fake_run)

    asyncio.run(tasks._run_department_auto_publish_sweep())

    assert seen == ["acme.test", "qnsc.vn"]


def test_a_failing_sweep_does_not_skip_the_other(monkeypatch):
    ran = []

    async def broken():
        ran.append("agent")
        raise RuntimeError("boom")

    async def auto():
        ran.append("auto")

    monkeypatch.setattr(tasks, "_run_approval_agent_sweep", broken)
    monkeypatch.setattr(tasks, "_run_department_auto_publish_sweep", auto)

    asyncio.run(tasks._run_review_sweeps())

    assert ran == ["agent", "auto"]


def test_only_an_admin_or_ceo_may_let_a_department_skip_review():
    from src.api.routers.auth import _may_waive_review

    assert _may_waive_review(SimpleNamespace(role="Admin", roles=[]))
    assert _may_waive_review(SimpleNamespace(role="CEO", roles=[]))
    assert not _may_waive_review(SimpleNamespace(role="Employee", roles=[]))
    # holding user.manage (which can edit a department) is not enough on its own
    assert not _may_waive_review(SimpleNamespace(role="Manager", roles=[]))
