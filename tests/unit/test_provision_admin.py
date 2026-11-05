"""The bootstrap Admin script sets its own RLS context, and can be SSO-only."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from src.models.rbac import Role

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "provision_admin", ROOT / "scripts" / "provision_admin.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSession:
    """Enough of an AsyncSession for provision(): one user lookup, one Role lookup."""

    def __init__(self, user: object | None, role: object) -> None:
        self._rows = [user, role]
        self.added: list[object] = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def scalar(self, _statement):
        return self._rows.pop(0)

    def add(self, value: object) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1


def test_provision_sets_the_global_admin_context_before_the_write(monkeypatch):
    module = _module()
    session = _FakeSession(user=None, role=Role(name="Admin", company_domain=None))
    contexts: list[tuple[object, str | None, dict[str, object]]] = []

    async def record_context(db, domain, **kwargs):
        contexts.append((db, domain, kwargs))

    async def no_rbac(_db):
        return None

    monkeypatch.setenv("ADMIN_EMAIL", "Ops.Admin@QNSC.vn")
    monkeypatch.setenv("ADMIN_NAME", "Ops Admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "a-long-enough-password")
    monkeypatch.setattr(module, "SessionLocal", lambda: session)
    monkeypatch.setattr(module, "set_database_context", record_context)
    monkeypatch.setattr(module, "bootstrap_rbac", no_rbac)

    asyncio.run(module.provision())

    assert len(contexts) == 1
    db, domain, flags = contexts[0]
    assert db is session
    assert domain == "qnsc.vn"
    # The bypass the identity policies look for, plus the capability flags an Admin
    # request carries; without global_admin the users INSERT is refused under kb_app.
    assert flags == {
        "global_admin": True,
        "global_article_access": True,
        "global_identity_access": True,
        "global_connector_access": True,
        "global_governance_access": True,
    }
    assert session.added, "a missing user must be inserted"
    assert session.commits == 1


def test_no_password_is_sso_only_and_ignores_a_supplied_password(monkeypatch):
    module = _module()
    session = _FakeSession(user=None, role=Role(name="Admin", company_domain=None))
    hashed: list[str] = []

    async def no_context(*_args, **_kwargs):
        return None

    async def no_rbac(_db):
        return None

    monkeypatch.setenv("ADMIN_EMAIL", "sso-admin@qnsc.vn")
    monkeypatch.setenv("ADMIN_PASSWORD", "the-password-from-the-environment")
    monkeypatch.setattr(module, "SessionLocal", lambda: session)
    monkeypatch.setattr(module, "set_database_context", no_context)
    monkeypatch.setattr(module, "bootstrap_rbac", no_rbac)
    monkeypatch.setattr(
        module, "get_password_hash", lambda value: hashed.append(value) or value
    )

    asyncio.run(module.provision(no_password=True))

    assert len(hashed) == 1
    assert hashed[0] != "the-password-from-the-environment"
    assert len(hashed[0]) >= 32


def test_without_the_flag_a_short_or_missing_password_is_refused(monkeypatch):
    module = _module()
    monkeypatch.setenv("ADMIN_EMAIL", "admin@qnsc.vn")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(module, "SessionLocal", lambda: _FakeSession(None, object()))

    with pytest.raises(SystemExit, match="ADMIN_PASSWORD"):
        asyncio.run(module.provision())
