"""The startup bootstrap that makes a fresh deployment reachable.

Two halves, and the second is the one that matters: creating a global administrator
automatically is only acceptable because production cannot come up with the password that
is written down in this repository.

Settings is constructed with `_env_file=None` throughout — a developer's own .env is
present in a working checkout and would otherwise decide what these assert.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.config import DEVELOPMENT_DEFAULT_PASSWORD, Settings
from src.core.security import verify_password
from src.domain import admin_bootstrap
from src.models.rbac import Role


def _production_settings(**overrides) -> Settings:
    """A Settings that passes validate_production, before the overrides under test."""
    base = dict(
        _env_file=None,
        ENVIRONMENT="production",
        SECRET_KEY="a" * 32,
        DATA_ENCRYPTION_KEY="d" * 32,
        AUTO_CREATE_SCHEMA=False,
        ALLOW_SELF_REGISTRATION=False,
        ENABLE_API_DOCS=False,
        MALWARE_SCAN_ENABLED=True,
        MALWARE_SCANNER_HOST="clamav",
        FRONTEND_URL="https://kb.example.com",
        CORS_ORIGINS="https://kb.example.com",
        # Production insists on a complete Entra registration, so a settings object that
        # is valid apart from the bootstrap credentials has to carry one.
        MICROSOFT_CLIENT_ID="client-id",
        MICROSOFT_CLIENT_SECRET="client-secret",
        MICROSOFT_TENANT_ID="11111111-2222-4333-8444-555555555555",
        MICROSOFT_REDIRECT_URI="https://kb.example.com/api/v1/connectors/oauth/callback",
        MICROSOFT_LOGIN_REDIRECT_URI="https://kb.example.com/api/v1/auth/entra/callback",
        GOOGLE_REDIRECT_URI=None,
        SOURCE_STORAGE_BACKEND="r2",
        SOURCE_STORAGE_BUCKET="private-kb",
        R2_ACCOUNT_ID="account-id",
        R2_ACCESS_KEY_ID="access-key",
        R2_SECRET_ACCESS_KEY="secret-key",
        S3_ENDPOINT_URL="",
        BOOTSTRAP_ADMIN_PASSWORD="a-real-deployment-password",
        # Outbound mail is part of what production insists on (EMAIL_PROVIDER defaults to graph).
        MICROSOFT_GRAPH_SENDER="noreply-kb@qnsc.vn",
        MAIL_GRAPH_CLIENT_ID="mail-client-id",
        MAIL_GRAPH_CERT_PEM="certificate",
        MAIL_GRAPH_PRIVATE_KEY_PEM="private-key",
    )
    base.update(overrides)
    return Settings(**base)


# ── The production guard ─────────────────────────────────────────────────────

def test_production_refuses_to_start_with_the_repositorys_default_password():
    settings = _production_settings(BOOTSTRAP_ADMIN_PASSWORD=DEVELOPMENT_DEFAULT_PASSWORD)

    with pytest.raises(RuntimeError, match="BOOTSTRAP_ADMIN_PASSWORD"):
        settings.validate_production()


def test_production_refuses_a_short_bootstrap_password():
    settings = _production_settings(BOOTSTRAP_ADMIN_PASSWORD="short")

    with pytest.raises(RuntimeError, match="at least 12 characters"):
        settings.validate_production()


def test_production_refuses_a_bootstrap_address_that_is_not_an_email():
    settings = _production_settings(BOOTSTRAP_ADMIN_EMAIL="admin")

    with pytest.raises(RuntimeError, match="BOOTSTRAP_ADMIN_EMAIL"):
        settings.validate_production()


def test_production_accepts_an_explicitly_configured_bootstrap_admin():
    _production_settings().validate_production()


def test_disabling_the_bootstrap_leaves_its_credentials_unchecked():
    # The default password is still in place, but nothing will read it.
    settings = _production_settings(
        BOOTSTRAP_ADMIN_ENABLED=False,
        BOOTSTRAP_ADMIN_PASSWORD=DEVELOPMENT_DEFAULT_PASSWORD,
    )

    settings.validate_production()


def test_the_default_password_is_usable_outside_production():
    settings = Settings(_env_file=None, ENVIRONMENT="development")

    assert settings.BOOTSTRAP_ADMIN_ENABLED
    assert settings.BOOTSTRAP_ADMIN_PASSWORD == DEVELOPMENT_DEFAULT_PASSWORD
    settings.validate_production()  # a no-op outside production


# ── Creating the account ─────────────────────────────────────────────────────

class _FakeSession:
    """Enough AsyncSession to drive ensure_bootstrap_admin.

    `scalar` answers from a queue, in the order the function issues its two queries:
    first "does a global administrator exist", then "fetch the global Admin role".
    """

    def __init__(self, *scalar_results):
        self._scalars = list(scalar_results)
        self.added = []
        self.committed = False

    async def scalar(self, *_args, **_kwargs):
        return self._scalars.pop(0)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        self.committed = True

    async def rollback(self):
        pass


def test_an_empty_deployment_gets_a_global_administrator(monkeypatch):
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_ENABLED", True)
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_EMAIL", "Admin@QNSC.vn")
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_NAME", "Admin")
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_PASSWORD", "a-real-password")
    admin_role = Role(name="Admin", company_domain=None, active=True)
    db = _FakeSession(None, admin_role)

    user = asyncio.run(admin_bootstrap.ensure_bootstrap_admin(db))

    assert user is not None
    assert db.committed
    # Normalised, because login looks the address up in lower case.
    assert user.email == "admin@qnsc.vn"
    assert user.company_domain == "qnsc.vn"
    assert user.active is True
    # The role RELATIONSHIP is what AuthorizationService.is_global_administrator reads;
    # the scalar column alone would leave an admin that cannot administer anything.
    assert user.roles == [admin_role]
    assert user.role == "Admin"


def test_the_password_is_stored_hashed(monkeypatch):
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_ENABLED", True)
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_PASSWORD", "a-real-password")
    db = _FakeSession(None, Role(name="Admin", company_domain=None, active=True))

    user = asyncio.run(admin_bootstrap.ensure_bootstrap_admin(db))

    assert user.password_hash != "a-real-password"
    assert verify_password("a-real-password", user.password_hash)


def test_an_existing_global_administrator_is_never_replaced(monkeypatch):
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_ENABLED", True)
    # A deployment that already has one — including one whose seeded admin was
    # deliberately renamed or re-passworded. Restarting must not undo that.
    db = _FakeSession("an-existing-user-id")

    assert asyncio.run(admin_bootstrap.ensure_bootstrap_admin(db)) is None
    assert db.added == []
    assert not db.committed


def test_the_bootstrap_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_ENABLED", False)
    db = _FakeSession()

    assert asyncio.run(admin_bootstrap.ensure_bootstrap_admin(db)) is None
    assert db.added == []


def test_a_missing_global_admin_role_creates_nobody(monkeypatch):
    # bootstrap_rbac creates that role unconditionally, so this means it did not run.
    # Creating the user anyway would leave an account with no administrative rights that
    # then suppresses every later attempt, because the existence check looks for the role.
    monkeypatch.setattr(admin_bootstrap.settings, "BOOTSTRAP_ADMIN_ENABLED", True)
    db = _FakeSession(None, None)

    assert asyncio.run(admin_bootstrap.ensure_bootstrap_admin(db)) is None
    assert not db.committed
