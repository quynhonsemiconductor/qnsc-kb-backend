"""The Access branch over real ASGI: RS256 signature, audience, issuer, user mapping.

Only the JWKS client is faked; the verifier, the route and the ASGI transport are real.
The last tests use the production app so the global middleware chain is exercised too.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI

from src.api.routers import mcp
from src.core.config import settings
from src.core.rate_limit import InMemoryRateLimiter
from src.domain.access_jwt import AccessJwtConfig, AccessJwtVerifier

TEAM = "https://qnsc.cloudflareaccess.com"
AUDIENCE = "mcp-access-aud"
KID = "test-key"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
USER_ID = uuid.uuid4()


class FakeJwkClient:
    def get_signing_keys(self, refresh: bool = False):
        return [SimpleNamespace(key=KEY.public_key(), key_id=KID)]


def token(
    *, audience=AUDIENCE, issuer=TEAM, sub="access-sub", email="an@qnsc.vn"
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": sub,
            "email": email,
            "iat": now - 10,
            "nbf": now - 10,
            "exp": now + 300,
        },
        KEY,
        algorithm="RS256",
        headers={"kid": KID},
    )


def user(**overrides):
    values = {
        "id": USER_ID,
        "email": "an@qnsc.vn",
        "company_domain": "qnsc.vn",
        "active": True,
        "auth_version": 1,
        "access_subject": "access-sub",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def install(monkeypatch, *, found=user()):
    verifier = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, allow_email_fallback=True),
        jwk_client=FakeJwkClient(),
    )

    @asynccontextmanager
    async def fake_session():
        yield object()

    class Repo:
        def __init__(self, db):
            self.db = db

        async def get_by_email(self, email):
            return found

    async def fake_helper(db, candidate):
        return candidate

    async def fake_context(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp, "access_verifier", lambda: verifier)
    monkeypatch.setattr(mcp, "SessionLocal", fake_session)
    monkeypatch.setattr(mcp, "UserRepository", Repo)
    monkeypatch.setattr(mcp, "active_user_with_context", fake_helper)
    monkeypatch.setattr(mcp, "set_database_context", fake_context)
    monkeypatch.setattr(mcp, "_limiter", InMemoryRateLimiter(limit=1000))
    monkeypatch.setattr(mcp, "_preauth_limiter", InMemoryRateLimiter(limit=1000))


def post(app, value: str, body: dict | None = None, *, content: bytes | None = None):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            headers = {mcp.ACCESS_JWT_HEADER: value}
            if content is not None:
                return await c.post("/mcp", content=content, headers=headers)
            return await c.post(
                "/mcp",
                json=body or {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                headers=headers,
            )

    return asyncio.run(go())


def router_app() -> FastAPI:
    app = FastAPI()
    app.include_router(mcp.router, prefix="/mcp")
    return app


def test_a_real_rs256_token_authenticates_and_lists_only_get_tools(monkeypatch):
    install(monkeypatch)
    app = router_app()
    response = post(app, token())
    assert response.status_code == 200, response.text

    listing = post(
        app, token(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    ).json()["result"]["tools"]
    names = {tool["name"] for tool in listing}
    assert "search_knowledge" in names and "get_article" in names
    assert "ask_knowledge_base" not in names
    assert not {name for name in names if name in {"create_article", "update_article"}}


def test_a_wrong_audience_or_issuer_is_401(monkeypatch):
    install(monkeypatch)
    app = router_app()
    assert post(app, token(audience="another-app")).status_code == 401
    assert (
        post(app, token(issuer="https://other.cloudflareaccess.com")).status_code == 401
    )


def test_an_unknown_user_is_403(monkeypatch):
    install(monkeypatch, found=None)
    assert post(router_app(), token()).status_code == 403


def test_the_production_middleware_covers_mcp(monkeypatch):
    install(monkeypatch)
    from src.api.main import app

    response = post(app, token())
    assert response.status_code == 200, response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store, private"
    assert response.headers["x-request-id"]

    if settings.cors_origin_list:
        origin = settings.cors_origin_list[0]

        async def cors():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                    headers={
                        mcp.ACCESS_JWT_HEADER: token(),
                        "Origin": origin,
                    },
                )

        assert asyncio.run(cors()).headers["access-control-allow-origin"] == origin


def test_an_oversized_body_is_413_on_mcp(monkeypatch):
    install(monkeypatch)
    from src.api.main import app

    monkeypatch.setattr(mcp, "MAX_BODY_BYTES", 100)
    response = post(app, token(), content=b"{" + b"x" * 200 + b"}")
    assert response.status_code == 413
