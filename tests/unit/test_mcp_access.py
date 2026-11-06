"""Cloudflare Access principals for MCP: mapping, RLS context and the read-only gate."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import jwt
import pytest
from fastapi import FastAPI, Request

from src.api import deps
from src.api.routers import mcp
from src.core.config import settings
from src.core.rate_limit import InMemoryRateLimiter
from src.domain.access_jwt import AccessIdentity, AccessJwtError

USER_ID = uuid.uuid4()
STAFF_ROLE = SimpleNamespace(
    active=True, name="Staff", company_domain="qnsc.vn", permissions=[]
)


def identity(email: str | None = "An.Nguyen@qnsc.vn") -> AccessIdentity:
    return AccessIdentity(sub="access-sub", email=email, oid=None)


def user(**overrides):
    values = {
        "id": USER_ID,
        "email": "an.nguyen@qnsc.vn",
        "company_domain": "qnsc.vn",
        "active": True,
        "auth_version": 4,
        "roles": [STAFF_ROLE],
        "role": "Staff",
        "access_subject": "access-sub",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# -- resolving the Access identity to a KB user ------------------------------------


def test_access_identity_is_mapped_through_the_shared_rest_gate(monkeypatch):
    contexts: list[tuple] = []
    looked_up: list[str] = []
    helper_calls: list[object] = []

    class Repo:
        def __init__(self, db):
            self.db = db

        async def get_by_email(self, email):
            looked_up.append(email)
            return user()

    async def fake_context(db, *args):
        contexts.append((db, *args))

    async def fake_helper(db, candidate):
        helper_calls.append(candidate)
        return candidate

    monkeypatch.setattr(mcp, "UserRepository", Repo)
    monkeypatch.setattr(mcp, "set_database_context", fake_context)
    monkeypatch.setattr(mcp, "active_user_with_context", fake_helper)

    db = object()
    principal = asyncio.run(mcp.resolve_access_principal(db, identity()))

    # The domain context is set before the identity lookup; the REST gate/context is the
    # shared helper, not a second implementation of "usable account".
    assert contexts == [(db, "qnsc.vn")]
    assert looked_up == ["an.nguyen@qnsc.vn"]
    assert helper_calls == [user()]
    assert principal == mcp.AccessPrincipal(
        user_id=USER_ID, email="an.nguyen@qnsc.vn", auth_version=4
    )
    assert principal.can_write is False


def test_the_shared_gate_is_the_rest_one(monkeypatch):
    contexts: list[tuple] = []

    async def fake_context(db, *args):
        contexts.append((db, *args))

    monkeypatch.setattr(deps, "set_database_context", fake_context)

    db = object()
    prepared = asyncio.run(deps.active_user_with_context(db, user()))
    assert prepared is not None and prepared.id == USER_ID
    assert contexts == [
        (db, "qnsc.vn", False, str(USER_ID), False, False, False, False)
    ]

    assert asyncio.run(deps.active_user_with_context(db, None)) is None
    assert asyncio.run(deps.active_user_with_context(db, user(active=False))) is None


def test_an_unknown_or_inactive_user_is_refused(monkeypatch):
    contexts: list[tuple] = []

    class Repo:
        def __init__(self, db):
            self.db = db

        async def get_by_email(self, email):
            return None

    async def fake_context(db, *args):
        contexts.append((db, *args))

    monkeypatch.setattr(mcp, "UserRepository", Repo)
    monkeypatch.setattr(mcp, "set_database_context", fake_context)
    monkeypatch.setattr(mcp, "active_user_with_context", deps.active_user_with_context)
    monkeypatch.setattr(deps, "set_database_context", fake_context)

    db = object()
    with pytest.raises(mcp.AccessDeniedError):
        asyncio.run(mcp.resolve_access_principal(db, identity()))
    assert contexts == [(db, "qnsc.vn")]

    class InactiveRepo(Repo):
        async def get_by_email(self, email):
            return user(active=False)

    monkeypatch.setattr(mcp, "UserRepository", InactiveRepo)
    with pytest.raises(mcp.AccessDeniedError):
        asyncio.run(mcp.resolve_access_principal(object(), identity()))


def test_the_first_access_subject_is_bound_and_a_different_one_is_refused(monkeypatch):
    class FakeDB:
        def __init__(self):
            self.executed: list[object] = []
            self.commits = 0

        async def execute(self, statement, *args):
            self.executed.append(statement)
            return SimpleNamespace(rowcount=1)

        async def commit(self):
            self.commits += 1

        async def rollback(self):
            pass

    reads = {"count": 0}

    class Repo:
        def __init__(self, db):
            self.db = db

        async def get_by_email(self, email):
            reads["count"] += 1
            # First read unbound, the read after the binding write carries the subject.
            return user(access_subject=None if reads["count"] == 1 else "access-sub")

    async def fake_helper(db, candidate):
        return candidate

    monkeypatch.setattr(mcp, "UserRepository", Repo)
    monkeypatch.setattr(mcp, "active_user_with_context", fake_helper)

    async def fake_context(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp, "set_database_context", fake_context)
    monkeypatch.setattr(deps, "set_database_context", fake_context)

    db = FakeDB()
    principal = asyncio.run(mcp.resolve_access_principal(db, identity()))
    assert principal.user_id == USER_ID and db.commits == 1 and len(db.executed) == 1

    # A verified token for the SAME email but a different Access identity is refused: the
    # mailbox was reassigned, so the account does not follow it.
    class MismatchedRepo:
        def __init__(self, db):
            self.db = db

        async def get_by_email(self, email):
            return user(access_subject="someone-else")

    monkeypatch.setattr(mcp, "UserRepository", MismatchedRepo)
    with pytest.raises(mcp.AccessDeniedError):
        asyncio.run(mcp.resolve_access_principal(FakeDB(), identity()))


def test_an_identity_without_an_email_never_touches_the_database(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("no database for an identity with no email")

    monkeypatch.setattr(mcp, "UserRepository", boom)
    monkeypatch.setattr(mcp, "set_database_context", boom)

    with pytest.raises(mcp.AccessDeniedError):
        asyncio.run(mcp.resolve_access_principal(object(), identity(email=None)))
    with pytest.raises(mcp.AccessDeniedError):
        asyncio.run(mcp.resolve_access_principal(object(), identity(email="no-at")))


# -- the HTTP path ------------------------------------------------------------------


class Harness:
    def __init__(self, monkeypatch, *, principal=None, verify_error=None, deny=False):
        self.calls = []
        self.principal = principal or mcp.AccessPrincipal(
            user_id=USER_ID, email="an.nguyen@qnsc.vn", auth_version=4
        )
        self.identities = []
        self.deny = deny

        @asynccontextmanager
        async def fake_session():
            yield object()

        def fake_verify(token):
            if verify_error is not None:
                raise verify_error
            return identity()

        async def fake_resolve(db, who):
            self.identities.append(who)
            if deny:
                raise mcp.AccessDeniedError()
            return self.principal

        async def fake_run_call(request, who, call):
            self.calls.append((who, call))
            return 200, {}

        monkeypatch.setattr(mcp, "SessionLocal", fake_session)
        monkeypatch.setattr(mcp, "verify_access_token", fake_verify)
        monkeypatch.setattr(mcp, "resolve_access_principal", fake_resolve)
        monkeypatch.setattr(mcp, "run_call", fake_run_call)
        monkeypatch.setattr(mcp, "_limiter", InMemoryRateLimiter(limit=1000))
        monkeypatch.setattr(mcp, "_preauth_limiter", InMemoryRateLimiter(limit=1000))
        self.app = FastAPI()
        self.app.include_router(mcp.router, prefix="/mcp")

    def rpc(self, method, params=None, *, headers=None, token=None):
        body = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            body["params"] = params

        async def go():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                request_headers = dict(headers or {})
                if token is not None:
                    request_headers["Authorization"] = f"Bearer {token}"
                return await c.post("/mcp", json=body, headers=request_headers)

        return asyncio.run(go())


def access_headers(value="access-jwt") -> dict[str, str]:
    return {mcp.ACCESS_JWT_HEADER: value}


def test_the_edge_header_authenticates_and_lists_only_read_tools(monkeypatch):
    h = Harness(monkeypatch)
    response = h.rpc("tools/list", headers=access_headers())
    assert response.status_code == 200, response.text
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert "search_knowledge" in names and "get_article" in names
    assert not {"create_article", "update_article", "record_lesson"} & names
    # ask POSTs to /ai/ask and records a conversation, so it is not offered to Access.
    assert "ask_knowledge_base" not in names
    assert h.identities and h.identities[0].sub == "access-sub"


def test_a_jwt_shaped_bearer_is_accepted_but_an_opaque_one_is_not(monkeypatch):
    h = Harness(monkeypatch)
    jwt_like = ".".join(["eyJ" + "h" * 17, "a", "b"])
    assert h.rpc("ping", token=jwt_like).status_code == 200
    assert h.identities
    assert h.rpc("ping", token="opaque-access-token").status_code == 401
    assert len(h.identities) == 1


def test_a_bad_access_token_is_401_and_a_jwks_outage_is_503(monkeypatch):
    bad = Harness(
        monkeypatch,
        verify_error=AccessJwtError("invalid_signature", "nope"),
    )
    response = bad.rpc("ping", headers=access_headers())
    assert response.status_code == 401 and bad.identities == []

    down = Harness(
        monkeypatch,
        verify_error=AccessJwtError("jwks_unavailable", "down"),
    )
    assert down.rpc("ping", headers=access_headers()).status_code == 503

    unconfigured = Harness(
        monkeypatch,
        verify_error=AccessJwtError("not_configured", "off"),
    )
    assert unconfigured.rpc("ping", headers=access_headers()).status_code == 503


def test_an_unverified_employee_is_403_and_no_tool_runs(monkeypatch):
    h = Harness(monkeypatch, deny=True)
    response = h.rpc(
        "tools/call",
        {"name": "search_knowledge", "arguments": {"query": "vpn"}},
        headers=access_headers(),
    )
    assert response.status_code == 403 and h.calls == []


def test_access_cannot_call_a_write_tool_even_by_name(monkeypatch):
    h = Harness(monkeypatch)
    response = h.rpc(
        "tools/call",
        {"name": "create_article", "arguments": {"title": "t", "body_md": "b"}},
        headers=access_headers(),
    )
    result = response.json()["result"]
    assert result["isError"] is True and "read-only" in result["content"][0]["text"]
    assert h.calls == []


def test_the_access_budget_is_per_person(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(mcp, "_limiter", InMemoryRateLimiter(limit=2))
    statuses = [h.rpc("ping", headers=access_headers()).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    assert h.rpc("ping", headers=access_headers()).headers["retry-after"]


# -- the in-process ASGI call -------------------------------------------------------


def test_run_call_acts_as_the_access_user_without_a_socket_or_the_edge_header():
    app = FastAPI()

    @app.get(f"{settings.API_V1_STR}/probe")
    async def probe(request: Request):
        return {
            "authorization": request.headers.get("authorization"),
            "access_header": request.headers.get(mcp.ACCESS_JWT_HEADER),
        }

    req = SimpleNamespace()
    req.app = app
    req.state = SimpleNamespace(request_id="rid-1")
    who = mcp.AccessPrincipal(
        user_id=USER_ID, email="an.nguyen@qnsc.vn", auth_version=4
    )
    status, payload = asyncio.run(mcp.run_call(req, who, mcp.Call("GET", "/probe")))

    assert status == 200 and payload["access_header"] is None
    token = payload["authorization"].removeprefix("Bearer ")
    claims = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    assert claims["sub"] == "an.nguyen@qnsc.vn" and claims["av"] == 4
    assert claims["type"] == "access"


def test_access_cannot_call_a_post_backed_read_tool_even_by_name(monkeypatch):
    h = Harness(monkeypatch)
    response = h.rpc(
        "tools/call",
        {"name": "ask_knowledge_base", "arguments": {"question": "what is x?"}},
        headers=access_headers(),
    )
    result = response.json()["result"]
    assert result["isError"] is True and "read-only" in result["content"][0]["text"]
    assert h.calls == []


def test_the_pre_auth_budget_runs_before_the_jwks_verifier(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(mcp, "_preauth_limiter", InMemoryRateLimiter(limit=2))
    statuses = [h.rpc("ping", headers=access_headers()).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    # The third request never reached the verifier (or the per-person budget).
    assert len(h.identities) == 2
