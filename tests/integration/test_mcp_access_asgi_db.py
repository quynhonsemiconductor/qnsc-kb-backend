"""DB-backed ASGI proof for the Access principal: only the JWKS client is faked.

The verifier, the route, `SessionLocal`, `UserRepository`, `active_user_with_context` and
`set_database_context` are the production implementations, running against the migrated
PostgreSQL test database. The four cases are the ones a Cloudflare Access caller can hit:
active (200, read-only), unknown (403), inactive (403) and a `sub` that does not match the
account's first binding (403).
"""

from __future__ import annotations

import os
import time
import uuid
from types import SimpleNamespace

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api import deps
from src.api.deps import set_database_context
from src.api.routers import mcp
from src.models.user import User
from src.domain.access_jwt import AccessJwtConfig, AccessJwtVerifier

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs the migrated PostgreSQL test database",
)

TEAM = "https://qnsc.cloudflareaccess.com"
AUDIENCE = "mcp-access-aud"
KID = "round3-key"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
RUN = uuid.uuid4().hex[:10]


class FakeJwkClient:
    """The one fake: the key set. Every other layer is the production code."""

    def get_signing_keys(self, refresh: bool = False):
        return [SimpleNamespace(key=KEY.public_key(), key_id=KID)]


def token(email: str, sub: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": TEAM,
            "aud": AUDIENCE,
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


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine(os.environ["DATABASE_URL"])
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield maker
    finally:
        await engine.dispose()


async def _add(db, email: str, *, active: bool, subject: str | None) -> uuid.UUID:
    user_id = uuid.uuid4()
    db.add(
        User(
            id=user_id,
            email=email,
            name="Round 3",
            password_hash="x",
            company_domain="qnsc.vn",
            role="Staff",
            active=active,
            auth_version=0,
            access_subject=subject,
        )
    )
    return user_id


def app() -> FastAPI:
    api = FastAPI()
    api.include_router(mcp.router, prefix="/mcp")
    return api


def _post(api: FastAPI, value: str, body: dict):
    async def go():
        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post(
                "/mcp", json=body, headers={mcp.ACCESS_JWT_HEADER: value}
            )

    return go()


@pytest.mark.asyncio
async def test_the_real_gate_and_database_decide_every_access_case(
    factory, monkeypatch
):
    active_email = f"round3-active-{RUN}@qnsc.vn"
    inactive_email = f"round3-inactive-{RUN}@qnsc.vn"
    mismatch_email = f"round3-mismatch-{RUN}@qnsc.vn"
    active_subject = f"access-sub-{RUN}"
    inactive_subject = f"inactive-sub-{RUN}"
    other_subject = f"other-sub-{RUN}"

    async with factory() as db:
        await set_database_context(db, None, True)
        active_id = await _add(db, active_email, active=True, subject=active_subject)
        inactive_id = await _add(
            db, inactive_email, active=True, subject=inactive_subject
        )
        inactive_row = await db.get(User, inactive_id)
        inactive_row.active = False
        mismatch_id = await _add(db, mismatch_email, active=True, subject=other_subject)
        await db.commit()

    # Only the JWKS client is faked; the verifier itself is real.
    verifier = AccessJwtVerifier(
        AccessJwtConfig(team_domain=TEAM, audience=AUDIENCE, allow_email_fallback=True),
        jwk_client=FakeJwkClient(),
    )
    monkeypatch.setattr(mcp, "access_verifier", lambda: verifier)
    api = app()

    try:

        async def case(email: str, sub: str) -> httpx.Response:
            return await _post(
                api,
                token(email, sub),
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            )

        response = await case(active_email, active_subject)
        assert response.status_code == 200, response.text

        listing = await _post(
            api,
            token(active_email, active_subject),
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        names = {tool["name"] for tool in listing.json()["result"]["tools"]}
        assert "search_knowledge" in names and "get_article" in names
        assert "ask_knowledge_base" not in names
        assert not {"create_article", "update_article"} & names

        assert (
            await case(f"round3-unknown-{RUN}@qnsc.vn", active_subject)
        ).status_code == 403
        assert (await case(inactive_email, inactive_subject)).status_code == 403
        assert (await case(mismatch_email, active_subject)).status_code == 403
    finally:
        async with factory() as db:
            await set_database_context(db, None, True)
            for user_id in (active_id, inactive_id, mismatch_id):
                row = await db.get(User, user_id)
                if row is not None:
                    await db.delete(row)
            await db.commit()
        # The deps engine is process-wide and this test's loop is about to close; do not
        # leave a pooled connection bound to it for a later test.
        await deps.engine.dispose()
