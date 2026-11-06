"""Cloudflare Access JWT verification: the claims and audience checks MCP-PLAN §5 names."""

from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError

from src.domain.access_jwt import (
    AccessJwtConfig,
    AccessJwtError,
    AccessJwtVerifier,
)

TEAM = "https://qnsc.cloudflareaccess.com"
AUDIENCE = "mcp-access-aud-tag"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJwkClient:
    def __init__(
        self,
        *,
        key=None,
        error: Exception | None = None,
        unknown_kids: set[str] | None = None,
    ) -> None:
        self._key = key if key is not None else KEY.public_key()
        self._error = error
        self._unknown_kids = unknown_kids or set()
        self.lookups: list[str] = []

    def get_signing_key(self, kid: str):
        self.lookups.append(kid)
        if self._error is not None:
            raise self._error
        if kid in self._unknown_kids:
            raise PyJWKClientError(
                f'Unable to find a signing key that matches: "{kid}"'
            )
        return SimpleNamespace(key=self._key)


def verifier(*, allow_email_fallback: bool = True, **overrides) -> AccessJwtVerifier:
    config = AccessJwtConfig(
        team_domain=overrides.pop("team_domain", TEAM),
        audience=overrides.pop("audience", AUDIENCE),
        allow_email_fallback=allow_email_fallback,
        **overrides,
    )
    return AccessJwtVerifier(config, jwk_client=FakeJwkClient())


def access_token(*, remove: tuple[str, ...] = (), **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": TEAM,
        "aud": AUDIENCE,
        "sub": "access-sub-1",
        "email": "An.Nguyen@qnsc.vn",
        "iat": now - 10,
        "nbf": now - 10,
        "exp": now + 300,
    }
    claims.update(overrides)
    for key in remove:
        claims.pop(key, None)
    return jwt.encode(claims, KEY, algorithm="RS256", headers={"kid": "test-key"})


def rejection(token: str, *, v: AccessJwtVerifier | None = None) -> str:
    with pytest.raises(AccessJwtError) as caught:
        (v or verifier()).verify(token)
    return caught.value.code


def test_a_valid_token_maps_its_principal_and_email():
    identity = verifier().verify(access_token())
    assert identity.sub == "access-sub-1"
    assert identity.email == "An.Nguyen@qnsc.vn"
    assert identity.oid is None


def test_email_fallback_is_opt_in():
    assert rejection(access_token(), v=verifier(allow_email_fallback=False)) == (
        "missing_principal"
    )


def test_an_oid_claim_is_preferred_when_present():
    identity = verifier(allow_email_fallback=False).verify(
        access_token(oid="entra-oid")
    )
    assert identity.oid == "entra-oid" and identity.email == "An.Nguyen@qnsc.vn"


def test_a_nested_oid_claim_path_is_honoured():
    identity = verifier(oid_claim_path="custom.oid").verify(
        access_token(custom={"oid": "nested-oid"})
    )
    assert identity.oid == "nested-oid"


def test_an_oid_only_token_works_without_the_email_fallback():
    identity = verifier(allow_email_fallback=False).verify(
        access_token(oid="entra-oid", email=None)
    )
    assert identity.oid == "entra-oid" and identity.email is None


def test_wrong_audience_is_rejected():
    assert rejection(access_token(aud="another-app")) == "bad_audience"


def test_wrong_issuer_is_rejected():
    assert rejection(access_token(iss="https://someone-else.cloudflareaccess.com")) == (
        "bad_issuer"
    )


def test_an_audience_list_accepts_any_configured_tag():
    identity = verifier(audience=[AUDIENCE, "second-tag"]).verify(access_token())
    assert identity.sub == "access-sub-1"


def test_expired_and_not_yet_valid_tokens_are_rejected():
    now = int(time.time())
    assert rejection(access_token(exp=now - 120)) == "expired"
    assert rejection(access_token(nbf=now + 120)) == "not_yet_valid"


def test_a_missing_required_claim_is_malformed():
    assert rejection(access_token(remove=("exp",))) == "malformed"
    assert rejection(access_token(remove=("aud",))) == "malformed"


def test_an_empty_subject_is_missing_principal():
    assert rejection(access_token(sub="")) == "missing_principal"


def test_a_non_rs256_token_is_rejected_before_key_lookup():
    token = jwt.encode(
        {"iss": TEAM, "aud": AUDIENCE, "sub": "s"},
        "a" * 32,
        algorithm="HS256",
    )
    assert rejection(token) == "wrong_algorithm"


def test_a_malformed_token_is_rejected():
    assert rejection("not-a-jwt") == "malformed"


def test_a_jwks_outage_is_an_availability_error_not_a_bad_token():
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=FakeJwkClient(error=PyJWKClientConnectionError("down")),
    )
    assert rejection(access_token(), v=v) == "jwks_unavailable"


def test_configuration_is_pinned():
    with pytest.raises(AccessJwtError):
        AccessJwtVerifier(
            AccessJwtConfig(team_domain="http://qnsc.example", audience=AUDIENCE)
        )
    with pytest.raises(AccessJwtError):
        AccessJwtVerifier(AccessJwtConfig(team_domain=TEAM, audience=""))
    with pytest.raises(AccessJwtError):
        AccessJwtVerifier(
            AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, oid_claim_path="")
        )


def test_an_unknown_kid_is_negative_cached_before_a_second_lookup():
    client = FakeJwkClient(unknown_kids={"test-key"})
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=client,
    )
    token = access_token()
    for _ in range(3):
        assert rejection(token, v=v) == "invalid_signature"
    # One lookup, then the negative cache answers: a repeated bogus kid cannot drive the
    # JWKS client (and so cannot drive an outbound refresh) again inside the TTL.
    assert client.lookups == ["test-key"]


def test_the_negative_cache_is_bounded():
    client = FakeJwkClient(unknown_kids={f"kid-{i}" for i in range(20)})
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=client,
    )
    for i in range(20):
        token = access_token()
        # Re-sign with a distinct unknown kid.
        token = jwt.encode(
            jwt.decode(
                token,
                KEY.public_key(),
                algorithms=["RS256"],
                audience=AUDIENCE,
                issuer=TEAM,
            ),
            KEY,
            algorithm="RS256",
            headers={"kid": f"kid-{i}"},
        )
        assert rejection(token, v=v) == "invalid_signature"
    from src.domain import access_jwt as module

    assert len(v._unknown_kids) <= module.MAX_NEGATIVE_KID_ENTRIES


def test_the_jwks_cooldown_is_at_least_a_minute(monkeypatch):
    captured: dict[str, object] = {}

    class Client:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("src.domain.access_jwt.PyJWKClient", Client)
    AccessJwtVerifier(AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE))
    assert captured["cooldown_duration"] >= 60
