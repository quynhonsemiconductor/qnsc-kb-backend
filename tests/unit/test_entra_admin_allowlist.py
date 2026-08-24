"""There is no config-driven Entra administrator allowlist any more.

Entra sign-in used to auto-provision anyone from ENTRA_AUTO_PROVISION_DOMAIN as Staff and
promote the addresses named in ENTRA_ADMIN_EMAILS to Admin. That path is deliberately
gone: /api/v1/auth/entra/callback refuses any account without a matching, unexpired
invitation and takes the role FROM that invitation, which is a named administrator's
explicit decision rather than a value in the deployment environment.

This test is what stops the allowlist coming back by accident. Two authorities over the
same role decision — an invitation and an environment variable — is precisely the shape
that silently overrules the admin UI, and it was ENTRA_ADMIN_EMAILS that won last time.
"""
from __future__ import annotations

import inspect

from src.api.routers import auth
from src.core.config import Settings


def test_no_admin_allowlist_setting_exists():
    """A setting nothing reads is worse than no setting: operators trust it."""
    assert "ENTRA_ADMIN_EMAILS" not in Settings.model_fields


def test_the_provisioned_role_comes_from_the_invitation():
    source = inspect.getsource(auth.entra_callback)
    assert "role=invitation.role" in source
    assert "ENTRA_ADMIN_EMAILS" not in source


def test_an_uninvited_account_is_refused_rather_than_provisioned():
    """Domain membership alone must not create an account."""
    source = inspect.getsource(auth.entra_callback)
    assert "has no active QNSC invitation" in source
    assert "ENTRA_AUTO_PROVISION_DOMAIN" not in source
