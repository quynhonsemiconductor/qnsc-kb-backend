"""Two refresh tokens minted for the same user must never be identical.

refresh_sessions.token_hash carries a UNIQUE index, and rotation revokes the old row and
inserts a new one in the same request. The claims used to be {exp, sub, type, av}, and
`exp` has one-second resolution — so two tokens minted for one user inside the same
second were byte-identical, the insert raised UniqueViolationError, and /auth/refresh
returned 500.

It is not a narrow race: signing in and refreshing immediately hits it, as does a second
browser tab refreshing at the same moment. Revoked rows keep their hash, so a collision
outlives the session it replaced.
"""
from __future__ import annotations

import hashlib

import jwt

from src.core.config import settings
from src.core.security import create_refresh_token


def _hash(token: str) -> str:
    """What _store_refresh_session writes into the unique column."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def test_consecutive_tokens_for_one_user_differ():
    first = create_refresh_token("someone@qnsc.vn", auth_version=0)
    second = create_refresh_token("someone@qnsc.vn", auth_version=0)

    assert first != second
    assert _hash(first) != _hash(second), (
        "identical hashes violate the unique index on refresh_sessions.token_hash and "
        "surface as a 500 from /auth/refresh"
    )


def test_a_burst_of_tokens_is_all_distinct():
    # Same user, same second, no sleeps — the exact shape of the production failure.
    hashes = {_hash(create_refresh_token("someone@qnsc.vn")) for _ in range(50)}

    assert len(hashes) == 50


def test_the_claims_the_refresh_endpoint_reads_are_unchanged():
    token = create_refresh_token("someone@qnsc.vn", auth_version=7)
    claims = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])

    assert claims["type"] == "refresh"
    assert claims["sub"] == "someone@qnsc.vn"
    assert claims["av"] == 7
    assert claims["jti"], "the uniqueness nonce must be present"
