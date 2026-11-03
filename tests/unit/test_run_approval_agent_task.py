"""The beat task that gives the approval agent an automatic trigger.

Before this, approval_agent.run() (src/domain/approval_agent.py) and its API
(POST /governance/approval-agent/run) were both reachable, but nothing ever called them
except a human -- a correctly-scoped, authority-granted rule still left every new draft
in the human queue until somebody remembered to run it again. This only tests the
wiring: which domains get swept and that each is run for real, not dry-run. The agent's
own decision logic is covered in test_approval_agent.py.
"""
from __future__ import annotations

import asyncio

from src.domain import approval_agent
from src.workers import tasks


class _Result:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return self

    def all(self):
        return self.values


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


def test_sweeps_every_domain_with_an_active_rule(monkeypatch):
    calls = []

    async def fake_run(db, company_domain, *, dry_run=False):
        calls.append((company_domain, dry_run))
        return {"evaluated": 0}

    monkeypatch.setattr(tasks, "SessionLocal", lambda: _Session(["acme.test", "qnsc.vn"]))
    monkeypatch.setattr(tasks, "set_database_context", _fake_context)
    monkeypatch.setattr(approval_agent, "run", fake_run)

    asyncio.run(tasks._run_approval_agent_sweep())

    assert calls == [("acme.test", False), ("qnsc.vn", False)]


def test_no_active_rules_means_no_runs(monkeypatch):
    calls = []

    async def fake_run(db, company_domain, *, dry_run=False):
        calls.append(company_domain)
        return {"evaluated": 0}

    monkeypatch.setattr(tasks, "SessionLocal", lambda: _Session([]))
    monkeypatch.setattr(tasks, "set_database_context", _fake_context)
    monkeypatch.setattr(approval_agent, "run", fake_run)

    asyncio.run(tasks._run_approval_agent_sweep())

    assert calls == []
