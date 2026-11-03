"""Entra sign-in and the connector consent are pinned to ONE tenant, and fail closed when it is not.

An audit found the app registration is signInAudience=AzureADMultipleOrgs. The registration decides who Entra will
let reach us; THIS code decides who we accept. Before this change MICROSOFT_TENANT_ID defaulted to "common" and
`_validate_tenant_claims` skipped the tenant comparison for it, so with that default a validly signed token from any
tenant verified. Every test here is a case where a foreign or unpinned configuration used to pass or to produce a
"common" URL.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from src.api.routers import auth as auth_router
from src.core.config import Settings, settings
from src.domain import connector_adapters, entra_auth

OURS = "dc0f2078-ac28-4ff2-b21a-d4b28df32361"
FOREIGN = "99999999-8888-4777-8666-555555555555"
CLIENT = "dbd99dbb-d20e-4076-8f8b-75c15e733414"
NONCE = "the-nonce"
KID = "k1"
UNPINNED = ["", "   ", "common", "COMMON", "organizations", "consumers"]


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(autouse=True)
def _settings(monkeypatch, signing_key):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", OURS)
    monkeypatch.setattr(settings, "MICROSOFT_CLIENT_ID", CLIENT)
    monkeypatch.setattr(settings, "MICROSOFT_CLIENT_SECRET", "secret")
    monkeypatch.setattr(settings, "MICROSOFT_LOGIN_REDIRECT_URI", "https://api.test/api/v1/auth/entra/callback")
    entry = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    entry["kid"] = KID
    entry.pop("alg", None)

    class _Response:
        status_code = 200

        def json(self):
            return {"keys": [entry]}

    class _Client:
        urls: list[str] = []

        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            _Client.urls.append(url)
            return _Response()

    _Client.urls = []
    monkeypatch.setattr(entra_auth.httpx, "AsyncClient", _Client)
    return _Client


def token(key, *, tid=OURS, iss_tenant=None, user="someone@qnsc.vn") -> str:
    now = datetime.now(timezone.utc)
    claims = {"iss": f"https://login.microsoftonline.com/{iss_tenant or tid}/v2.0", "tid": tid, "aud": CLIENT,
              "oid": "oid-1", "preferred_username": user, "nonce": NONCE, "exp": now + timedelta(minutes=10), "iat": now}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


# ── the pin itself ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", UNPINNED)
def test_blank_and_multi_tenant_authorities_are_not_a_pin(monkeypatch, value):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    assert settings.microsoft_tenant is None


def test_a_guid_is_the_pin_and_the_default_is_not_common():
    assert Settings(_env_file=None, MICROSOFT_TENANT_ID=f" {OURS} ").microsoft_tenant == OURS
    assert Settings(_env_file=None).MICROSOFT_TENANT_ID == "", "the default must be empty, never common"
    assert Settings(_env_file=None).microsoft_tenant is None


# ── sign-in: token verification ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_our_tenant_token_verifies_case_insensitively(signing_key, monkeypatch):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", OURS.upper())
    claims = await entra_auth.verify_id_token(token(signing_key), NONCE)
    assert claims["email"] == "someone@qnsc.vn"


@pytest.mark.asyncio
async def test_a_validly_signed_token_from_another_tenant_is_rejected(signing_key):
    with pytest.raises(ValueError, match="tenant is invalid"):
        await entra_auth.verify_id_token(token(signing_key, tid=FOREIGN, user="x@othercorp.example"), NONCE)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", UNPINNED)
async def test_with_no_pinned_tenant_NOTHING_verifies_not_even_ours(signing_key, monkeypatch, value):
    """The old behaviour: tenant 'common' accepted every tenant. Now an unpinned config accepts no one."""
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    for tid in (OURS, FOREIGN):
        with pytest.raises(ValueError, match="MICROSOFT_TENANT_ID"):
            await entra_auth.verify_id_token(token(signing_key, tid=tid), NONCE)


@pytest.mark.asyncio
async def test_the_issuer_must_be_ours_even_when_tid_is_ours(signing_key):
    with pytest.raises(ValueError, match="issuer"):
        await entra_auth.verify_id_token(token(signing_key, tid=OURS, iss_tenant=FOREIGN), NONCE)


@pytest.mark.asyncio
async def test_a_token_with_no_tenant_claim_is_rejected(signing_key):
    now = datetime.now(timezone.utc)
    bare = jwt.encode({"iss": f"https://login.microsoftonline.com/{OURS}/v2.0", "aud": CLIENT, "oid": "o", "nonce": NONCE,
                       "preferred_username": "a@qnsc.vn", "exp": now + timedelta(minutes=5)}, signing_key, algorithm="RS256", headers={"kid": KID})
    with pytest.raises(ValueError, match="no tenant claim"):
        await entra_auth.verify_id_token(bare, NONCE)


# ── sign-in: authority URLs and availability ─────────────────────────────────────────────────
def test_the_authorize_url_is_on_our_tenant():
    assert f"/{OURS}/oauth2/v2.0/authorize?" in entra_auth.authorization_url("s", "n")


@pytest.mark.parametrize("value", UNPINNED)
def test_there_is_no_authorize_url_without_a_pin(monkeypatch, value):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    with pytest.raises(ValueError, match="MICROSOFT_TENANT_ID"):
        entra_auth.authorization_url("s", "n")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", UNPINNED)
async def test_the_code_exchange_never_goes_to_a_multi_tenant_authority(monkeypatch, value):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    with pytest.raises(ValueError, match="MICROSOFT_TENANT_ID"):
        await entra_auth.exchange_code("code")


@pytest.mark.asyncio
async def test_signing_keys_are_fetched_from_our_tenant(signing_key, _settings):
    await entra_auth.verify_id_token(token(signing_key), NONCE)
    assert _settings.urls == [f"https://login.microsoftonline.com/{OURS}/discovery/v2.0/keys"]


@pytest.mark.parametrize("value", UNPINNED)
def test_sign_in_is_not_configured_without_a_pin(monkeypatch, value):
    assert entra_auth.configured() is True
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    assert entra_auth.configured() is False


# ── what an external-tenant user gets, end to end through the callback ──────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize("authority", [OURS, "common"])
async def test_an_external_tenant_user_gets_401_and_nothing_is_created(signing_key, monkeypatch, authority):
    """Both with the pin (rejected as a foreign tenant) and with the OLD default 'common' (now not configured at all):
    the callback never reaches the database, so no user, no identity link, no audit row, no session."""
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", authority)
    state = jwt.encode({"type": "entra_login", "nonce": NONCE, "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
                       settings.SECRET_KEY, algorithm="HS256")
    db = MagicMock()
    db.execute, db.commit, db.add = AsyncMock(), AsyncMock(), MagicMock()

    async def exchange(code):
        return {"id_token": token(signing_key, tid=FOREIGN, user="attacker@othercorp.example")}

    with patch.object(auth_router.entra_auth, "exchange_code", exchange), patch.object(auth_router, "set_database_context", AsyncMock()) as ctx:
        with pytest.raises(HTTPException) as caught:
            await auth_router.entra_callback(request=MagicMock(), code="c", state=state, db=db)

    assert caught.value.status_code in (401, 503)
    assert db.execute.await_count == 0 and db.add.call_count == 0 and db.commit.await_count == 0 and ctx.await_count == 0


# ── the connector's delegated consent ────────────────────────────────────────────────────────
def _adapter():
    return connector_adapters.MicrosoftGraphAdapter.__new__(connector_adapters.MicrosoftGraphAdapter)


def test_the_connector_consent_url_is_on_our_tenant():
    url = _adapter().oauth_url("state")
    assert url.startswith(f"https://login.microsoftonline.com/{OURS}/oauth2/v2.0/authorize?")


@pytest.mark.parametrize("value", UNPINNED)
def test_the_connector_never_builds_a_multi_tenant_consent_url(monkeypatch, value):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    with pytest.raises(connector_adapters.ConnectorProviderError) as caught:
        _adapter().oauth_url("state")
    assert caught.value.code == "tenant_not_pinned" and caught.value.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize("value", UNPINNED)
async def test_the_connector_code_exchange_refuses_a_multi_tenant_authority(monkeypatch, value):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    with pytest.raises(connector_adapters.ConnectorProviderError) as caught:
        await _adapter().exchange_code("code")
    assert caught.value.code == "tenant_not_pinned"


# ── the pin must be GUID-shaped ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", ["qnsc.vn", "contoso.onmicrosoft.com", "tenant-id", "not-a-guid", f"{OURS}x", OURS[:-1],
                                   f"{OURS}/evil", "dc0f2078ac284ff2b21ad4b28df32361"])
def test_anything_that_is_not_a_guid_is_not_a_pin(monkeypatch, value):
    """A domain name is a valid Entra authority but can never equal a token's tid; a typo must not become a URL."""
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", value)
    assert settings.microsoft_tenant is None
    assert entra_auth.configured() is False


@pytest.mark.parametrize("value", [OURS, OURS.upper(), "00000000-0000-0000-0000-000000000000", "12345678-1234-1234-1234-123456789abc"])
def test_any_guid_shape_is_a_pin_with_no_version_or_variant_restriction(monkeypatch, value):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", f" {value} ")
    assert settings.microsoft_tenant == value


def test_production_accepts_the_same_shape_and_refuses_the_rest():
    from tests.unit.test_admin_bootstrap import _production_settings

    _production_settings(MICROSOFT_TENANT_ID="00000000-0000-0000-0000-000000000000").validate_production()
    with pytest.raises(RuntimeError, match="tenant GUID"):
        _production_settings(MICROSOFT_TENANT_ID="qnsc.vn").validate_production()
