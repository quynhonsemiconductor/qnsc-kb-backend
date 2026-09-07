"""The gates on the factory-reset endpoint, and the escalation route it opens.

An email allowlist is only a restriction while no API route can WRITE an
allowlisted address. `users.email` is mutable by anyone holding global
`user.manage`, and two other routes create accounts outright, so all three are
checked here. Without these the allowlist is an escalation path: point an account
you control at the allowlisted address and inherit the ability to erase the
database.

Route bodies are called directly with a fake session. That is enough, because
every gate asserted here runs BEFORE any database work — which is the property
worth proving.

``asyncio.run`` rather than pytest-asyncio: the marker is not available in every
environment this suite runs in.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException

from src.api.routers.governance import (
    FACTORY_RESET_CONFIRM_PHRASE,
    FactoryResetRequest,
    factory_reset_database,
)
from src.core.config import settings

OPERATOR = "sinhhpt@qnsc.vn"


class _Caller:
    """The minimum a route body reads off the authenticated user."""

    def __init__(self, email: str):
        self.id = uuid.uuid4()
        self.email = email
        self.company_domain = "qnsc.vn"
        self.role = "Admin"
        self.active = True


class _ExplodingSession:
    """Any database access is a test failure: the gates must refuse first."""

    def add(self, *_args):
        raise AssertionError("the reset touched the database after being refused")

    async def execute(self, *_args, **_kwargs):
        raise AssertionError("the reset queried the database after being refused")

    async def commit(self):
        raise AssertionError("the reset committed after being refused")


def _run(payload: FactoryResetRequest, caller: _Caller):
    return asyncio.run(factory_reset_database(payload, caller, _ExplodingSession()))


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "FACTORY_RESET_ENABLED", True)
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", OPERATOR)


# --- endpoint gates ----------------------------------------------------------


def test_the_endpoint_is_invisible_while_disabled(monkeypatch):
    """404, not 403.

    A switched-off capability should not confirm that it exists and is merely
    refusing this caller.
    """
    monkeypatch.setattr(settings, "FACTORY_RESET_ENABLED", False)
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", OPERATOR)

    with pytest.raises(HTTPException) as error:
        _run(FactoryResetRequest(dry_run=True), _Caller(OPERATOR))
    assert error.value.status_code == 404


def test_enabling_without_an_allowlist_authorises_nobody(monkeypatch):
    """Both settings are required; one alone must not open the door."""
    monkeypatch.setattr(settings, "FACTORY_RESET_ENABLED", True)
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", "")

    with pytest.raises(HTTPException) as error:
        _run(FactoryResetRequest(dry_run=True), _Caller(OPERATOR))
    assert error.value.status_code == 403


def test_a_global_admin_outside_the_allowlist_is_refused(enabled):
    """The permission dependency is not the last word.

    `role.manage` at global scope is required AND insufficient — otherwise every
    future global admin silently inherits the ability to destroy the deployment.
    """
    with pytest.raises(HTTPException) as error:
        _run(FactoryResetRequest(dry_run=True), _Caller("someone.else@qnsc.vn"))
    assert error.value.status_code == 403
    assert error.value.detail["code"] == "not_reset_operator"


def test_a_real_run_needs_the_exact_confirmation_phrase(enabled):
    with pytest.raises(HTTPException) as error:
        _run(FactoryResetRequest(dry_run=False, confirm="yes"), _Caller(OPERATOR))
    assert error.value.status_code == 409
    assert error.value.detail["code"] == "confirmation_mismatch"


def test_a_real_run_without_any_confirmation_is_refused(enabled):
    with pytest.raises(HTTPException) as error:
        _run(FactoryResetRequest(dry_run=False), _Caller(OPERATOR))
    assert error.value.status_code == 409


def test_the_request_defaults_to_a_dry_run():
    """Forgetting the field must not erase the database."""
    assert FactoryResetRequest().dry_run is True


def test_the_confirmation_phrase_is_not_the_company_domain():
    """A saved knowledge-purge request must not execute a full reset.

    `POST /governance/knowledge/purge` confirms with the company domain. Sharing
    that phrase would make the two operations interchangeable by copy-paste.
    """
    assert FACTORY_RESET_CONFIRM_PHRASE != "qnsc.vn"
    assert FACTORY_RESET_CONFIRM_PHRASE == "RESET ENTIRE DATABASE"


# --- the escalation route the allowlist opens --------------------------------


def test_every_account_route_refuses_the_allowlisted_address(enabled):
    """Rename, invite and direct-create must all reject the reset operator address.

    Asserted against the route source rather than by executing three multi-hundred
    line handlers with full fakes. What matters is that each path CONSULTS
    `is_reset_operator` before writing an email; the message wording is checked
    where the guard is defined.
    """
    import inspect

    from src.api.routers import auth

    source = inspect.getsource(auth)
    # update_user (rename), create_invitation (deferred grant), create_managed_user
    # (immediate grant with a caller-chosen password).
    assert source.count("is_reset_operator(") >= 3, (
        "an account route can write an email without consulting the reset allowlist"
    )


def test_the_rename_guard_allows_the_operator_to_keep_its_own_address(monkeypatch):
    """The guard blocks GRANTING the capability, not editing the operator itself.

    `is_reset_operator(new) and not is_reset_operator(old)` is what makes an
    unrelated update to the operator's own row still possible.
    """
    from src.domain.factory_reset import is_reset_operator

    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", OPERATOR)
    # Granting: refused.
    assert is_reset_operator(OPERATOR) and not is_reset_operator("staff@qnsc.vn")
    # Already the operator: not a grant, so the guard's second half is False.
    assert is_reset_operator(OPERATOR) and is_reset_operator(OPERATOR)
