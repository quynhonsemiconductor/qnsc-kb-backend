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
    """A tiny JWKS: a set of kids, a refresh that may publish a rotated key, and counters."""

    def __init__(
        self,
        *,
        keys: dict[str, object] | None = None,
        error: Exception | None = None,
        rotate_to: str | None = None,
        rotate_key=None,
    ) -> None:
        self._keys = dict(keys or {"test-key": KEY.public_key()})
        self._error = error
        self._rotate_to = rotate_to
        self._rotate_key = rotate_key or KEY.public_key()
        self.fetches = 0
        self.refreshes = 0

    def get_signing_keys(self, refresh: bool = False):
        self.fetches += 1
        if self._error is not None:
            raise self._error
        if refresh:
            self.refreshes += 1
            if self._rotate_to is not None:
                self._keys[self._rotate_to] = self._rotate_key
        return [SimpleNamespace(key=key, key_id=kid) for kid, key in self._keys.items()]


def verifier(*, allow_email_fallback: bool = True, **overrides) -> AccessJwtVerifier:
    config = AccessJwtConfig(
        team_domain=overrides.pop("team_domain", TEAM),
        audience=overrides.pop("audience", AUDIENCE),
        allow_email_fallback=allow_email_fallback,
        **overrides,
    )
    return AccessJwtVerifier(config, jwk_client=FakeJwkClient())


def access_token(
    *, kid: str = "test-key", remove: tuple[str, ...] = (), **overrides
) -> str:
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
    return jwt.encode(claims, KEY, algorithm="RS256", headers={"kid": kid})


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


def test_an_unknown_kid_is_negative_cached_before_a_second_refresh():
    client = FakeJwkClient(keys={"other-key": KEY.public_key()})
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=client,
    )
    token = access_token()
    for _ in range(3):
        assert rejection(token, v=v) == "invalid_signature"
    # Cached set, one forced refresh, then the negative cache answers: a repeated bogus
    # kid cannot drive the JWKS client again inside the TTL.
    assert client.fetches == 2 and client.refreshes == 1
    assert v._unknown_kids


def test_a_real_key_rotation_refreshes_once_and_then_verifies():
    client = FakeJwkClient(
        keys={"old-key": KEY.public_key()},
        rotate_to="new-key",
        rotate_key=KEY.public_key(),
    )
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, allow_email_fallback=True),
        jwk_client=client,
    )
    identity = v.verify(access_token(kid="new-key"))
    assert identity.sub == "access-sub-1"
    assert (
        client.refreshes == 1
    ), "a new Cloudflare kid must trigger exactly one refresh"
    # The rotated key is now in the cached set: no second refresh.
    assert v.verify(access_token(kid="new-key")).sub == "access-sub-1"
    assert client.refreshes == 1


def test_different_unknown_kids_share_one_refresh_per_cooldown():
    client = FakeJwkClient(keys={"other-key": KEY.public_key()})
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=client,
    )
    for kid in ("bogus-a", "bogus-b", "bogus-c"):
        assert rejection(access_token(kid=kid), v=v) == "invalid_signature"
    assert client.refreshes == 1


def test_the_negative_cache_is_bounded():
    client = FakeJwkClient(keys={"other-key": KEY.public_key()})
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=client,
    )
    for i in range(20):
        assert rejection(access_token(kid=f"kid-{i}"), v=v) == "invalid_signature"
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
