"""Reading a display label must never fail a connector authorization.

Clicking "Authorize source" opened a provider tab, came straight back to the app, and the
connector was still unauthorized — with nothing shown to say why.

The callback read the account label out of the provider's id_token with
`jwt.get_unverified_claims`. That function belongs to **python-jose**; this project uses
**PyJWT**, which has no such attribute, so the call raised AttributeError. The handler
beside it caught `jwt.PyJWTError`, which does not cover AttributeError, so the exception
escaped the callback, hit the blanket `except Exception` in `oauth_callback_entry`, and
redirected to /admin/connectors?oauth=error.

Microsoft's requested scope includes `openid`, so an id_token is ALWAYS returned and that
branch always ran. SharePoint authorization failed 100% of the time, and looked exactly
like the provider declining.

The label is decoration. These tests pin that it can never again take the flow down with
it, whatever a provider puts in the field.
"""
from __future__ import annotations

import jwt
import pytest

from src.api.routers.connectors import _subject_from_id_token


def _id_token(claims: dict) -> str:
    return jwt.encode(claims, "irrelevant-the-signature-is-not-checked", algorithm="HS256")


def test_the_subject_is_read_from_the_token():
    token = _id_token({"sub": "user@qnsc.vn", "aud": "some-client"})
    assert _subject_from_id_token(token, "Bearer") == "user@qnsc.vn"


def test_the_library_actually_in_use_provides_this_api():
    """The regression itself: PyJWT has no get_unverified_claims, python-jose does."""
    assert not hasattr(jwt, "get_unverified_claims")
    assert hasattr(jwt, "decode")


def test_a_signature_we_cannot_verify_is_still_read():
    """The token arrives from the provider's token endpoint over TLS, in exchange for our
    client secret. Verifying it would mean fetching provider JWKS to read a label."""
    token = jwt.encode({"sub": "signed-by-someone-else"}, "a-different-key", algorithm="HS256")
    assert _subject_from_id_token(token, "Bearer") == "signed-by-someone-else"


@pytest.mark.parametrize(
    "value",
    [None, "", "not-a-jwt", "a.b.c", "eyJhbGciOiJIUzI1NiJ9.@@@.@@@"],
)
def test_an_unusable_token_falls_back_instead_of_raising(value):
    """Every one of these used to abort the authorization."""
    assert _subject_from_id_token(value, "Bearer") == "Bearer"


def test_a_token_without_a_subject_falls_back():
    assert _subject_from_id_token(_id_token({"name": "no sub here"}), "Bearer") == "Bearer"


def test_an_empty_subject_falls_back():
    assert _subject_from_id_token(_id_token({"sub": ""}), "Bearer") == "Bearer"
