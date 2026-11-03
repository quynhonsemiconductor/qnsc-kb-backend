"""Entra ID-token verification must survive the python-jose → PyJWT migration.

This is the one place the swap was not mechanical. python-jose offered
`jwt.get_unverified_claims()` and `jwk.construct()`; PyJWT has neither, so the JWKS entry
is now turned into a key with `jwt.PyJWK.from_dict()` and the pre-signature claims come
from `jwt.decode(..., options={"verify_signature": False})`.

PyJWT also differs on audience: it RAISES `InvalidAudienceError` when a token carries
`aud` and no `audience` argument is passed, where jose simply skipped the check. Every
Entra id_token carries `aud`, so a missed argument here would have failed closed — but
only at real sign-in, against a live tenant. Hence this test.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.core.config import settings
from src.domain import entra_auth

TENANT = "dc0f2078-ac28-4ff2-b21a-d4b28df32361"
CLIENT_ID = "dbd99dbb-d20e-4076-8f8b-75c15e733414"
NONCE = "the-expected-nonce"
KID = "test-signing-key"


@pytest.fixture
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks(key: rsa.RSAPrivateKey) -> dict:
    entry = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    entry["kid"] = KID
    # Entra omits the optional `alg` member, which is why the code passes the
    # header-validated algorithm to PyJWK explicitly.
    entry.pop("alg", None)
    return {"keys": [entry]}


def _id_token(key: rsa.RSAPrivateKey, **overrides) -> str:
    claims = {
        "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
        "tid": TENANT,
        "aud": CLIENT_ID,
        "oid": "user-object-id",
        "preferred_username": "Someone@QNSC.vn",
        "nonce": NONCE,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
        "iat": datetime.now(timezone.utc),
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


@pytest.fixture(autouse=True)
def _entra_settings(monkeypatch):
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", TENANT)
    monkeypatch.setattr(settings, "MICROSOFT_CLIENT_ID", CLIENT_ID)


@pytest.fixture(autouse=True)
def _stub_jwks(monkeypatch, signing_key):
    """Serve the JWKS locally — no network, and no dependency on a live tenant."""

    class _Response:
        status_code = 200

        def json(self):
            return _jwks(signing_key)

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            return _Response()

    monkeypatch.setattr(entra_auth.httpx, "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_a_valid_id_token_verifies_and_maps_the_user(signing_key):
    claims = await entra_auth.verify_id_token(_id_token(signing_key), NONCE)

    assert claims["subject"] == "user-object-id"
    assert claims["email"] == "someone@qnsc.vn", "the address is normalised to lower case"


@pytest.mark.asyncio
async def test_a_token_for_another_audience_is_rejected(signing_key):
    """The check PyJWT enforces by raising, rather than skipping as jose did."""
    token = _id_token(signing_key, aud="some-other-application")

    with pytest.raises(jwt.InvalidAudienceError):
        await entra_auth.verify_id_token(token, NONCE)


@pytest.mark.asyncio
async def test_a_replayed_nonce_is_rejected(signing_key):
    with pytest.raises(ValueError, match="nonce"):
        await entra_auth.verify_id_token(_id_token(signing_key), "a-different-nonce")


@pytest.mark.asyncio
async def test_an_expired_token_is_rejected(signing_key):
    token = _id_token(signing_key, exp=datetime.now(timezone.utc) - timedelta(minutes=1))

    with pytest.raises(jwt.ExpiredSignatureError):
        await entra_auth.verify_id_token(token, NONCE)


@pytest.mark.asyncio
async def test_a_token_from_another_tenant_is_rejected(signing_key):
    """Tenant validation moved AFTER signature verification — prove it still bites.

    It used to run on claims read before the signature was checked. Nothing required
    that: the JWKS is fetched from the CONFIGURED tenant, so the only pre-verification
    reads needed are `kid` and `alg` from the header.
    """
    other = "11111111-2222-3333-4444-555555555555"
    token = _id_token(
        signing_key, tid=other, iss=f"https://login.microsoftonline.com/{other}/v2.0"
    )

    with pytest.raises(ValueError, match="tenant"):
        await entra_auth.verify_id_token(token, NONCE)


@pytest.mark.asyncio
async def test_a_token_with_a_forged_issuer_is_rejected(signing_key):
    token = _id_token(signing_key, iss="https://login.microsoftonline.com/evil/v2.0")

    with pytest.raises(ValueError, match="issuer"):
        await entra_auth.verify_id_token(token, NONCE)


@pytest.mark.asyncio
async def test_a_token_signed_by_a_different_key_is_rejected(signing_key):
    """Signature verification is real: the JWKS key must be the one that signed it."""
    impostor = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _id_token(impostor)

    with pytest.raises(jwt.InvalidSignatureError):
        await entra_auth.verify_id_token(token, NONCE)
