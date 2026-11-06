"""Cloudflare Access JWT verification: the claims and audience checks MCP-PLAN §5 names."""

from __future__ import annotations

import threading
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
    """A tiny JWKS: a set of kids, an optional rotation on a later fetch, and counters."""

    def __init__(
        self,
        *,
        keys: dict[str, object] | None = None,
        error: Exception | None = None,
        rotate_to: str | None = None,
        rotate_key=None,
        rotate_after: int = 1,
    ) -> None:
        self._keys = dict(keys or {"test-key": KEY.public_key()})
        self._error = error
        self._rotate_to = rotate_to
        self._rotate_key = rotate_key or KEY.public_key()
        self._rotate_after = rotate_after
        self._lock = threading.Lock()
        self.fetches = 0

    def get_signing_keys(self, refresh: bool = False):
        with self._lock:
            self.fetches += 1
            fetch_number = self.fetches
            if self._error is not None:
                raise self._error
            if self._rotate_to is not None and fetch_number >= self._rotate_after:
                self._keys[self._rotate_to] = self._rotate_key
            return [
                SimpleNamespace(key=key, key_id=kid) for kid, key in self._keys.items()
            ]


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


def test_an_unknown_kid_is_refused_without_a_second_fetch():
    client = FakeJwkClient(keys={"other-key": KEY.public_key()})
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=client,
    )
    token = access_token()
    for _ in range(3):
        assert rejection(token, v=v) == "invalid_signature"
    # Initial fetch + one forced refresh, then the cooldown answers: no fetch per request.
    assert client.fetches == 2


def test_a_real_key_rotation_refreshes_once_and_then_verifies():
    client = FakeJwkClient(
        keys={"old-key": KEY.public_key()},
        rotate_to="new-key",
        rotate_key=KEY.public_key(),
        rotate_after=2,
    )
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, allow_email_fallback=True),
        jwk_client=client,
    )
    identity = v.verify(access_token(kid="new-key"))
    assert identity.sub == "access-sub-1"
    assert client.fetches == 2, "a new Cloudflare kid must trigger one refresh"
    # The rotated key is now in the cached set: no second fetch.
    assert v.verify(access_token(kid="new-key")).sub == "access-sub-1"
    assert client.fetches == 2


def test_a_bogus_kid_does_not_stop_a_later_rotation_from_verifying():
    client = FakeJwkClient(
        keys={"old-key": KEY.public_key()},
        rotate_to="new-key",
        rotate_key=KEY.public_key(),
        rotate_after=2,
    )
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, allow_email_fallback=True),
        jwk_client=client,
    )
    assert rejection(access_token(kid="bogus"), v=v) == "invalid_signature"
    assert v.verify(access_token(kid="new-key")).sub == "access-sub-1"
    assert client.fetches == 2


def test_a_concurrent_rotation_verifies_on_both_threads():
    client = FakeJwkClient(
        keys={"old-key": KEY.public_key()},
        rotate_to="new-key",
        rotate_key=KEY.public_key(),
        rotate_after=2,
    )
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, allow_email_fallback=True),
        jwk_client=client,
    )
    token = access_token(kid="new-key")
    barrier = threading.Barrier(2)
    results: list[str] = []

    def verify() -> None:
        barrier.wait()
        results.append(v.verify(token).sub)

    threads = [threading.Thread(target=verify) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == ["access-sub-1", "access-sub-1"]
    assert client.fetches == 2


def test_a_jwks_outage_fetches_once_per_cooldown():
    client = FakeJwkClient(error=PyJWKClientConnectionError("down"))
    v = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE),
        jwk_client=client,
    )
    for _ in range(5):
        assert rejection(access_token(), v=v) == "jwks_unavailable"
    assert client.fetches == 1


def test_the_cache_and_cooldown_are_validated():
    with pytest.raises(AccessJwtError):
        AccessJwtVerifier(
            AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, jwks_cache_seconds=0)
        )
    with pytest.raises(AccessJwtError):
        AccessJwtVerifier(
            AccessJwtConfig(
                team_domain=TEAM, audience=AUDIENCE, jwks_cooldown_seconds=-1
            )
        )
