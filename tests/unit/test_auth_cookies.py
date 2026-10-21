"""The auth cookie pair: `__Host-`-prefixed and `Secure` in every deployed env.

Handlers are called directly (no TestClient), the way `test_password_auth_flows.py`
does it; the `Set-Cookie`/`Delete-Cookie` headers are inspected raw because
`http.cookies` silently merges cookies of the same name set on different paths.
"""

from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response

from src.api.deps import get_current_user
from src.api.routers import auth
from src.core.config import settings


def _set_cookie_headers(response: Response) -> list[str]:
    return [
        value.decode() for key, value in response.headers.raw if key == b"set-cookie"
    ]


def test_deployed_environment_sets_host_locked_secure_cookies(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    response = Response()
    auth._set_auth_cookies(response, "access-value", "refresh-value")

    cookies = {
        name: value
        for value in _set_cookie_headers(response)
        for name in value.split("=", 1)[:1]
    }
    assert "__Host-access_token" in cookies and "__Host-refresh_token" in cookies
    issued = [raw for raw in _set_cookie_headers(response) if "Max-Age=0" not in raw]
    assert len(issued) == 2
    for raw in issued:
        assert "Secure" in raw and "HttpOnly" in raw and "SameSite=lax" in raw
        assert "Path=/" in raw
        assert "Domain=" not in raw
    # The refresh cookie moved from the narrower /api/v1/auth path: __Host- requires Path=/.
    # (The legacy expiry entry for the old path is asserted separately below.)


def test_local_development_sets_plain_insecure_cookies(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    response = Response()
    auth._set_auth_cookies(response, "access-value", "refresh-value")

    raw = " ".join(_set_cookie_headers(response))
    assert "access_token=" in raw and "refresh_token=" in raw
    assert "__Host-" not in raw
    assert "Secure" not in raw


def test_clear_removes_every_generation_and_old_refresh_path(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    response = Response()
    auth._clear_auth_cookies(response)

    raw = " ".join(_set_cookie_headers(response))
    # The names this environment sets, the pre-rename plain names, and the refresh
    # cookie's old /api/v1/auth path — all three must receive an explicit expiry.
    for name in (
        "__Host-access_token",
        "__Host-refresh_token",
        "access_token",
        "refresh_token",
    ):
        assert f"{name}=" in raw
    assert "Path=/api/v1/auth" in raw


def test_deployed_clear_still_expires_plain_cookies(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    response = Response()
    auth._clear_auth_cookies(response)
    raw = " ".join(_set_cookie_headers(response))
    assert 'access_token=""' in raw and 'refresh_token=""' in raw


@pytest.mark.asyncio
async def test_reader_accepts_host_prefixed_cookie(monkeypatch):
    """Drive get_current_user with the `__Host-` cookie only."""
    monkeypatch.setattr(settings, "SECRET_KEY", "test-key")

    def fake_decode(token, key, algorithms=None):
        return {"type": "access", "sub": "user@corp.vn", "av": 0}

    monkeypatch.setattr("src.api.deps.jwt.decode", fake_decode)

    async def no_context(*_args, **_kwargs):
        return None

    monkeypatch.setattr("src.api.deps.set_database_context", no_context)

    class _Repo:
        def __init__(self, _db):
            pass

        async def get_by_email(self, _email):
            return SimpleNamespace(
                active=True,
                auth_version=0,
                roles=["viewer"],
                company_domain="corp.vn",
                id="u1",
            )

    monkeypatch.setattr("src.api.deps.UserRepository", _Repo)
    monkeypatch.setattr(
        "src.api.deps.AuthorizationService",
        SimpleNamespace(
            is_global_administrator=lambda _u: False,
            has_global_article_access=lambda _u: False,
            has_global_identity_management=lambda _u: False,
            has_global_connector_management=lambda _u: False,
            has_global_governance_access=lambda _u: False,
        ),
    )

    class _Request:
        method = "GET"
        headers: dict = {}

    user = await get_current_user(
        request=_Request(),
        db=None,
        token=None,
        access_token_cookie=None,
        host_access_token_cookie="host-cookie-value",
    )
    assert user.id == "u1"


def test_deployed_issuance_also_expires_the_legacy_names(monkeypatch):
    """Deployed, a fresh login/refresh kills the pre-rename plain cookies too — the
    refresh one at both its paths — so a stale plain value never outlives the rename."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    response = Response()
    auth._set_auth_cookies(response, "a", "r")

    raw = " ".join(_set_cookie_headers(response))
    assert 'access_token=""' in raw and 'refresh_token=""' in raw
    assert raw.count("Path=/api/v1/auth") == 1
    # The real cookies are still issued: expiry entries must not shadow them.
    assert "__Host-access_token=a" in raw and "__Host-refresh_token=r" in raw


def test_local_issuance_does_not_expire_its_own_plain_cookies(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    response = Response()
    auth._set_auth_cookies(response, "a", "r")

    raw = " ".join(_set_cookie_headers(response))
    assert 'access_token=""' not in raw
    assert "Path=/api/v1/auth" not in raw
    assert "access_token=a" in raw and "refresh_token=r" in raw


def test_reader_prefers_the_host_locked_cookie(monkeypatch):
    """A request carrying BOTH generations authenticates from the __Host- one."""
    monkeypatch.setattr(settings, "SECRET_KEY", "test-key")
    seen: list[str] = []

    def fake_decode(token, key, algorithms=None):
        seen.append(token)
        return {"type": "access", "sub": "user@corp.vn", "av": 0}

    monkeypatch.setattr("src.api.deps.jwt.decode", fake_decode)

    async def no_context(*_args, **_kwargs):
        return None

    monkeypatch.setattr("src.api.deps.set_database_context", no_context)

    class _Repo:
        def __init__(self, _db):
            pass

        async def get_by_email(self, _email):
            return SimpleNamespace(
                active=True,
                auth_version=0,
                roles=["viewer"],
                company_domain="corp.vn",
                id="u1",
            )

    monkeypatch.setattr("src.api.deps.UserRepository", _Repo)
    monkeypatch.setattr(
        "src.api.deps.AuthorizationService",
        SimpleNamespace(
            is_global_administrator=lambda _u: False,
            has_global_article_access=lambda _u: False,
            has_global_identity_management=lambda _u: False,
            has_global_connector_management=lambda _u: False,
            has_global_governance_access=lambda _u: False,
        ),
    )

    class _Request:
        method = "GET"
        headers: dict = {}

    asyncio.run(
        get_current_user(
            request=_Request(),
            db=None,
            token=None,
            access_token_cookie="plain-value",
            host_access_token_cookie="host-value",
        )
    )
    assert seen == ["host-value"]


def test_refresh_prefers_the_host_locked_cookie(monkeypatch):
    """`/refresh` resolves the body-less cookie request from the __Host- cookie first."""
    seen: list[str] = []

    def claims(value):
        seen.append(value)
        raise ValueError(value)

    monkeypatch.setattr(auth, "_refresh_claims_from_token", claims)

    class _Request:
        method = "POST"
        headers: dict = {}

    with pytest.raises(ValueError):
        asyncio.run(
            auth.refresh_token(
                request=_Request(),
                response=Response(),
                req=None,
                refresh_token_cookie="plain-value",
                host_refresh_token_cookie="host-value",
                db=None,
            )
        )
    assert seen == ["host-value"]


def test_deployed_reader_refuses_a_plain_only_cookie(monkeypatch):
    """Deployed, a plain cookie alone is not a session: a sibling *.qnsc.vn host can
    plant one, and that is exactly what __Host- prevents. Without the fallback the
    request gets 401, not a session."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    request = SimpleNamespace(method="GET", headers={})
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            get_current_user(
                request=request,
                db=None,
                token=None,
                access_token_cookie="planted-value",
                host_access_token_cookie=None,
            )
        )
    assert excinfo.value.status_code == 401


def test_local_reader_still_accepts_the_plain_cookie(monkeypatch):
    """Development has no __Host- generation; the plain cookie must keep working."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "SECRET_KEY", "test-key")
    monkeypatch.setattr(
        "src.api.deps.jwt.decode",
        lambda token, key, algorithms=None: {
            "type": "access",
            "sub": "user@corp.vn",
            "av": 0,
        },
    )

    async def no_context(*_args, **_kwargs):
        return None

    monkeypatch.setattr("src.api.deps.set_database_context", no_context)

    class _Repo:
        def __init__(self, _db):
            pass

        async def get_by_email(self, _email):
            return SimpleNamespace(
                active=True,
                auth_version=0,
                roles=["viewer"],
                company_domain="corp.vn",
                id="u1",
            )

    monkeypatch.setattr("src.api.deps.UserRepository", _Repo)
    monkeypatch.setattr(
        "src.api.deps.AuthorizationService",
        SimpleNamespace(
            is_global_administrator=lambda _u: False,
            has_global_article_access=lambda _u: False,
            has_global_identity_management=lambda _u: False,
            has_global_connector_management=lambda _u: False,
            has_global_governance_access=lambda _u: False,
        ),
    )

    user = asyncio.run(
        get_current_user(
            request=SimpleNamespace(method="GET", headers={}),
            db=None,
            token=None,
            access_token_cookie="plain-value",
            host_access_token_cookie=None,
        )
    )
    assert user.id == "u1"


def test_deployed_refresh_refuses_a_plain_only_cookie(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    request = SimpleNamespace(method="POST", headers={})
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            auth.refresh_token(
                request=request,
                response=Response(),
                req=None,
                refresh_token_cookie="planted-value",
                host_refresh_token_cookie=None,
                db=None,
            )
        )
    assert excinfo.value.status_code == 401
    assert "Refresh token is required" in excinfo.value.detail
