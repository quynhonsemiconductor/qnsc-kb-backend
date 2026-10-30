"""MCP server: protocol, scope, argument safety, and that tools map onto real endpoints."""

import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import jwt
import pytest
from fastapi import FastAPI, Request

from src.api.routers import access_tokens as access_tokens_router
from src.api.routers import ai, articles, auth, graph, mcp, mcp_config, meta, search
from src.core.config import settings
from src.domain import access_tokens, mcp_tools
from src.domain.access_tokens import TokenPrincipal
from src.domain.mcp_tools import Call

ARTICLE_ID = str(uuid.uuid4())


def principal(scope: str = "read") -> TokenPrincipal:
    return TokenPrincipal(
        token_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        email="an@qnsc.vn",
        auth_version=3,
        scope=scope,
    )


class Harness:
    """A bare app serving the MCP router, with authentication and REST calls faked."""

    def __init__(self, monkeypatch, scope: str = "read", token_ok: bool = True):
        self.calls: list[Call] = []
        self.reply = (200, {})
        self.principal = principal(scope)

        @asynccontextmanager
        async def fake_session():
            yield object()

        async def fake_context(*args, **kwargs):
            return None

        async def fake_auth(db, raw):
            return self.principal if token_ok and raw == "kbpat_good" else None

        async def fake_run_call(request, who, call):
            self.calls.append(call)
            return self.reply

        monkeypatch.setattr(mcp, "SessionLocal", fake_session)
        monkeypatch.setattr(mcp, "set_database_context", fake_context)
        monkeypatch.setattr(mcp, "authenticate_token", fake_auth)
        monkeypatch.setattr(mcp, "run_call", fake_run_call)
        monkeypatch.setattr(mcp, "_limiter", mcp.InMemoryRateLimiter(limit=1000))
        self.app = FastAPI()
        self.app.include_router(mcp.router, prefix="/mcp")

    def post(self, body=None, *, token="kbpat_good", headers=None, content=None):
        async def go():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                h = dict(headers or {})
                if token:
                    h["Authorization"] = f"Bearer {token}"
                if content is not None:
                    return await c.post("/mcp", content=content, headers=h)
                return await c.post("/mcp", json=body, headers=h)

        return asyncio.run(go())

    def rpc(self, method, params=None, **kw):
        body = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            body["params"] = params
        return self.post(body, **kw)

    def result(self, method, params=None):
        response = self.rpc(method, params)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "error" not in data, data
        return data["result"]


@pytest.fixture
def h(monkeypatch):
    return Harness(monkeypatch)


# -- protocol -----------------------------------------------------------------------


def test_initialize_negotiates_the_protocol_version(h):
    ok = h.result("initialize", {"protocolVersion": "2025-03-26"})
    assert ok["protocolVersion"] == "2025-03-26"
    assert ok["serverInfo"]["name"] == "qnsc-kb"
    assert set(ok["capabilities"]) == {"tools", "resources", "prompts"}
    unknown = h.result("initialize", {"protocolVersion": "1999-01-01"})
    assert unknown["protocolVersion"] == mcp.SUPPORTED_VERSIONS[0]


def test_ping_and_unknown_method(h):
    assert h.result("ping") == {}
    data = h.rpc("nope/nothing").json()
    assert data["error"]["code"] == mcp.METHOD_NOT_FOUND


def test_notifications_get_202_and_no_body(h):
    r = h.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status_code == 202 and r.content == b""


def test_batches_answer_only_requests(h):
    r = h.post(
        [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 7, "method": "ping"},
        ]
    )
    assert r.status_code == 200
    assert r.json() == [{"jsonrpc": "2.0", "id": 7, "result": {}}]
    assert h.post([]).status_code == 400
    too_many = [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(21)]
    assert h.post(too_many).status_code == 400


def test_garbage_is_a_parse_or_request_error(h):
    r = h.post(content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == mcp.PARSE_ERROR
    r = h.post({"id": 1, "method": "ping"})
    assert r.json()["error"]["code"] == mcp.INVALID_REQUEST


def test_get_and_delete_are_405(h):
    async def go(method):
        transport = httpx.ASGITransport(app=h.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.request(method, "/mcp")

    for method in ("GET", "DELETE"):
        r = asyncio.run(go(method))
        assert r.status_code == 405 and r.headers["allow"] == "POST"


# -- authentication -----------------------------------------------------------------


def test_missing_or_bad_token_is_401_with_challenge(monkeypatch):
    h = Harness(monkeypatch)
    for token in (None, "kbpat_wrong", "not-a-token"):
        r = h.rpc("ping", token=token)
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Bearer")
    assert h.calls == []


def test_a_bearer_that_is_not_token_shaped_never_reaches_the_database(monkeypatch):
    h = Harness(monkeypatch)

    def boom():
        raise AssertionError("no database connection for a junk bearer")

    monkeypatch.setattr(mcp, "SessionLocal", boom)
    # Built from parts so no credential-shaped literal sits in the source (secret scanners
    # flag `token="eyJ..."`), while the value is still a real JWT-looking bearer.
    jwt_like = ".".join(["eyJ" + "h" * 17, "a", "b"])
    assert h.rpc("ping", token=jwt_like).status_code == 401


def test_foreign_origin_is_refused_but_no_origin_is_fine(h):
    assert h.rpc("ping", headers={"Origin": "https://evil.example"}).status_code == 403
    assert h.rpc("ping").status_code == 200
    ours = settings.cors_origin_list[0]
    assert h.rpc("ping", headers={"Origin": ours}).status_code == 200


def test_rate_limit_is_per_token(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(mcp, "_limiter", mcp.InMemoryRateLimiter(limit=2))
    assert [h.rpc("ping").status_code for _ in range(3)] == [200, 200, 429]
    assert h.rpc("ping").headers["retry-after"]


def test_disabled_server_is_404(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(settings, "MCP_ENABLED", False)
    assert h.rpc("ping").status_code == 404


# -- tools --------------------------------------------------------------------------


def test_tool_catalogue_is_well_formed():
    names = [t.name for t in mcp_tools.TOOLS]
    assert len(names) == len(set(names))
    for tool in mcp_tools.TOOLS:
        definition = tool.definition()
        assert set(tool.required) <= set(tool.properties), tool.name
        assert definition["annotations"]["readOnlyHint"] is (not tool.write)
        assert definition["annotations"]["destructiveHint"] is False
        assert re.fullmatch(r"[a-z_]{3,40}", tool.name)


def test_nothing_destructive_or_administrative_is_offered():
    banned = (
        "delete",
        "restore",
        "approve",
        "reject",
        "reindex",
        "role",
        "user",
        "connector",
    )
    for tool in mcp_tools.TOOLS:
        assert not any(word in tool.name for word in banned), tool.name
        assert tool.build is not None
    for tool in mcp_tools.TOOLS:
        for sample in ("DELETE",):
            call = _sample_call(tool)
            assert call.method != sample, tool.name


def test_read_tokens_never_see_write_tools(monkeypatch):
    read = Harness(monkeypatch, "read")
    names = {t["name"] for t in read.result("tools/list")["tools"]}
    assert "search_knowledge" in names and "create_article" not in names
    write = Harness(monkeypatch, "write")
    names = {t["name"] for t in write.result("tools/list")["tools"]}
    assert {"create_article", "update_article", "request_article_edit"} <= names


def test_a_read_token_cannot_call_a_write_tool_even_by_name(h):
    result = h.result(
        "tools/call",
        {"name": "create_article", "arguments": {"title": "t", "body_md": "b"}},
    )
    assert result["isError"] is True and "read-only" in result["content"][0]["text"]
    assert h.calls == []


def test_write_tool_runs_with_a_write_token(monkeypatch):
    h = Harness(monkeypatch, "write")
    h.reply = (201, {"id": ARTICLE_ID, "status": "pending_review", "message": "ok"})
    result = h.result(
        "tools/call",
        {
            "name": "create_article",
            "arguments": {"title": "T", "body_md": "B", "tags": ["x"]},
        },
    )
    assert result["isError"] is False
    (call,) = h.calls
    assert (call.method, call.path) == ("POST", "/articles/")
    assert call.body["title"] == "T" and call.body["visibility"] == "department"


def test_search_forwards_filters_and_clamps_limit(h):
    h.reply = (200, {"results": []})
    h.result(
        "tools/call",
        {
            "name": "search_knowledge",
            "arguments": {"query": "vpn", "limit": 999, "dept": "IT"},
        },
    )
    (call,) = h.calls
    assert call.path == "/search"
    assert call.params == {"q": "vpn", "dept": "IT", "limit": 20}


@pytest.mark.parametrize(
    "arguments",
    [
        {"article_id": "../admin/users"},
        {"article_id": "not-a-uuid"},
        {"article_id": ARTICLE_ID + "/versions"},
        {"article_id": 5},
        {},
    ],
)
def test_ids_are_validated_so_a_tool_cannot_reach_another_path(h, arguments):
    result = h.result("tools/call", {"name": "get_article", "arguments": arguments})
    assert result["isError"] is True
    assert h.calls == []


def test_missing_required_and_wrong_types_are_tool_errors_not_crashes(h):
    for name, args in (
        ("search_knowledge", {}),
        ("search_knowledge", {"query": 5}),
        ("search_knowledge", {"query": "x", "limit": "many"}),
        ("ask_knowledge_base", {"question": "x"}),
        ("ask_knowledge_base", {"question": "ok?", "language": "fr"}),
        ("update_article", {"article_id": ARTICLE_ID}),
        ("rate_answer", {"log_id": ARTICLE_ID, "rating": 5}),
    ):
        tool = mcp_tools.TOOLS_BY_NAME[name]
        if tool.write:
            continue
        r = h.result("tools/call", {"name": name, "arguments": args})
        assert r["isError"] is True, (name, args)


def test_unknown_tool_is_a_protocol_error(h):
    data = h.rpc("tools/call", {"name": "drop_database", "arguments": {}}).json()
    assert data["error"]["code"] == mcp.INVALID_PARAMS


def test_rest_errors_become_tool_errors_with_the_reason(h):
    h.reply = (403, {"detail": "Not authorized to view this article"})
    r = h.result(
        "tools/call", {"name": "get_article", "arguments": {"article_id": ARTICLE_ID}}
    )
    assert r["isError"] is True
    assert (
        "403" in r["content"][0]["text"] and "Not authorized" in r["content"][0]["text"]
    )


def test_article_results_are_trimmed_and_long_bodies_capped(h, monkeypatch):
    monkeypatch.setattr(settings, "MCP_MAX_RESULT_CHARS", 1000)
    h.reply = (
        200,
        {
            "id": ARTICLE_ID,
            "title": "T",
            "body_md": "x" * 5000,
            "owner_id": "secret",
            "explicit_user_ids": ["u"],
        },
    )
    r = h.result(
        "tools/call", {"name": "get_article", "arguments": {"article_id": ARTICLE_ID}}
    )
    text = r["content"][0]["text"]
    assert "owner_id" not in text and "explicit_user_ids" not in text
    assert "truncated" in text and len(text) < 2000


def test_ask_returns_only_the_answer_fields(h):
    h.reply = (
        200,
        {
            "answer": "A",
            "citations": [1],
            "conversation_id": "c",
            "debug": {"secret": 1},
        },
    )
    r = h.result(
        "tools/call",
        {"name": "ask_knowledge_base", "arguments": {"question": "what is x?"}},
    )
    assert (
        "debug" not in r["content"][0]["text"]
        and '"answer": "A"' in r["content"][0]["text"]
    )
    (call,) = h.calls
    assert call.path == "/ai/ask" and call.body["language"] == "vi"
    assert "confirm_edit" not in call.body and "edit_instruction" not in call.body


def test_search_hits_carry_one_passage_not_the_parent_and_child_copies(h):
    long_text = "x" * 5000
    h.reply = (
        200,
        [
            {
                "chunk_id": "c1",
                "article_id": ARTICLE_ID,
                "title": "Doc",
                "dept": "IT",
                "heading": "H",
                "page_number": 3,
                "chunk_text": long_text,
                "parent_text": "PARENT",
                "child_texts": ["CHILD"] * 10,
                "owner_email": "a@b.c",
                "score": 1.2,
                "source_url": "/api/v1/articles/x/source?page=3",
                "evidence": {"kind": "page_visual", "label": "L", "id": "e1"},
            }
        ],
    )
    r = h.result(
        "tools/call", {"name": "search_knowledge", "arguments": {"query": "vpn"}}
    )
    text = r["content"][0]["text"]
    assert "PARENT" not in text and "CHILD" not in text and "owner_email" not in text
    (hit,) = json.loads(text)
    assert hit["article_id"] == ARTICLE_ID and hit["page_number"] == 3
    assert hit["evidence"] == {"kind": "page_visual", "label": "L"}
    assert len(hit["text"]) < 1600 and hit["text"].endswith("[...]")


def test_ask_citations_are_slim_and_the_answer_is_not_repeated(h):
    h.reply = (
        200,
        {
            "answer": "A [C1]",
            "answer_grounded": "A [C1]",
            "answer_extended": "",
            "conversation_id": "c",
            "log_id": "l",
            "citations": [
                {
                    "source_id": "C1",
                    "article_id": ARTICLE_ID,
                    "title": "Doc",
                    "page_number": 2,
                    "excerpt": "e" * 3000,
                    "highlight_text": "H",
                    "highlight_texts": ["H"] * 9,
                    "child_chunk_ids": ["1", "2"],
                }
            ],
        },
    )
    r = h.result(
        "tools/call",
        {"name": "ask_knowledge_base", "arguments": {"question": "what is x?"}},
    )
    data = json.loads(r["content"][0]["text"])
    assert "answer_grounded" not in data and "answer_extended" not in data
    (citation,) = data["citations"]
    assert citation["source_id"] == "C1" and citation["page_number"] == 2
    assert "highlight_texts" not in citation and len(citation["excerpt"]) < 700
    assert data["log_id"] == "l" and data["conversation_id"] == "c"


def test_graph_tools_exist_only_when_the_graph_is_enabled(monkeypatch):
    graph_tools = {"find_entities", "get_entity_neighbors"}
    monkeypatch.setattr(mcp_tools.settings, "KNOWLEDGE_GRAPH_ENABLED", False)
    off = Harness(monkeypatch)
    names = {t["name"] for t in off.result("tools/list")["tools"]}
    assert not graph_tools & names
    result = off.result("tools/call", {"name": "find_entities", "arguments": {}})
    assert (
        result["isError"] is True and "knowledge graph" in result["content"][0]["text"]
    )
    assert off.calls == []
    monkeypatch.setattr(mcp_tools.settings, "KNOWLEDGE_GRAPH_ENABLED", True)
    on = Harness(monkeypatch)
    names = {t["name"] for t in on.result("tools/list")["tools"]}
    assert graph_tools <= names


def test_instructions_tell_a_new_client_how_to_work(h):
    text = h.result("initialize", {"protocolVersion": "2025-06-18"})["instructions"]
    for needle in (
        "ask_knowledge_base",
        "search_knowledge",
        "get_article",
        "citations",
        "review",
    ):
        assert needle in text


# -- resources and prompts ----------------------------------------------------------


def test_resources_list_read_and_templates(h):
    h.reply = (
        200,
        [
            {
                "id": ARTICLE_ID,
                "title": "VPN",
                "dept": "IT",
                "type": "SOP",
                "status": "published",
            }
        ],
    )
    listing = h.result("resources/list")["resources"]
    assert listing == [
        {
            "uri": f"kb://article/{ARTICLE_ID}",
            "name": "VPN",
            "description": "IT · SOP · published",
            "mimeType": "text/markdown",
        }
    ]
    h.reply = (
        200,
        {"id": ARTICLE_ID, "title": "VPN", "body_md": "Steps", "version": 2},
    )
    contents = h.result("resources/read", {"uri": f"kb://article/{ARTICLE_ID}"})[
        "contents"
    ]
    assert contents[0]["text"].startswith("# VPN") and "Steps" in contents[0]["text"]
    templates = h.result("resources/templates/list")["resourceTemplates"]
    assert templates[0]["uriTemplate"] == "kb://article/{id}"


def test_resource_errors(h):
    for uri in ("kb://article/../x", "file:///etc/passwd", 5):
        data = h.rpc("resources/read", {"uri": uri}).json()
        assert data["error"]["code"] == mcp.RESOURCE_NOT_FOUND, uri
    h.reply = (404, {"detail": "Article not found"})
    data = h.rpc("resources/read", {"uri": f"kb://article/{ARTICLE_ID}"}).json()
    assert data["error"]["code"] == mcp.RESOURCE_NOT_FOUND


def test_prompts(h):
    names = {p["name"] for p in h.result("prompts/list")["prompts"]}
    assert {
        "explore_knowledge_base",
        "summarize_article",
        "answer_with_sources",
        "compare_articles",
        "find_knowledge_gaps",
    } <= names
    got = h.result(
        "prompts/get",
        {"name": "summarize_article", "arguments": {"article_id": ARTICLE_ID}},
    )
    assert ARTICLE_ID in got["messages"][0]["content"]["text"]
    tour = h.result("prompts/get", {"name": "explore_knowledge_base"})
    assert "list_articles" in tour["messages"][0]["content"]["text"]
    for params in (
        {"name": "summarize_article", "arguments": {}},
        {"name": "summarize_article", "arguments": {"article_id": "x"}},
        {"name": "nope"},
    ):
        assert (
            h.rpc("prompts/get", params).json()["error"]["code"] == mcp.INVALID_PARAMS
        )


# -- acting as the owner ------------------------------------------------------------


def test_run_call_acts_as_the_owner_through_the_application(monkeypatch):
    app = FastAPI()

    @app.get(f"{settings.API_V1_STR}/probe")
    async def probe(request: Request):
        return {
            "authorization": request.headers.get("authorization"),
            "q": request.query_params.get("q"),
        }

    class Req:
        pass

    req = Req()
    req.app = app
    req.state = type("S", (), {"request_id": "rid-1"})()
    who = principal()
    status, payload = asyncio.run(
        mcp.run_call(req, who, Call("GET", "/probe", params={"q": "z"}))
    )
    assert status == 200 and payload["q"] == "z"
    token = payload["authorization"].removeprefix("Bearer ")
    claims = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    assert (
        claims["sub"] == "an@qnsc.vn"
        and claims["type"] == "access"
        and claims["av"] == 3
    )
    lifetime = datetime.fromtimestamp(claims["exp"], timezone.utc) - datetime.now(
        timezone.utc
    )
    assert timedelta(0) < lifetime <= timedelta(minutes=5)


def test_run_call_returns_error_bodies_instead_of_raising():
    app = FastAPI()
    req = type("R", (), {})()
    req.app = app
    req.state = type("S", (), {})()
    status, payload = asyncio.run(
        mcp.run_call(req, principal(), Call("GET", "/missing"))
    )
    assert status == 404 and "detail" in payload


# -- every tool maps onto a route that exists ---------------------------------------

ROUTERS = {
    "/articles": articles.router,
    "/search": search.router,
    "/ai": ai.router,
    "/auth": auth.router,
    "/meta": meta.router,
    "/graph": graph.router,
    "/mcp-config": mcp_config.router,
}
SAMPLE = {
    "query": "q",
    "question": "what is x?",
    "article_id": ARTICLE_ID,
    "entity_id": ARTICLE_ID,
    "version": 1,
    "title": "T",
    "body_md": "B",
    "request_text": "please fix",
    "log_id": ARTICLE_ID,
    "rating": 1,
}


def _sample_call(tool: mcp_tools.Tool) -> Call:
    return tool.build({k: v for k, v in SAMPLE.items() if k in tool.properties})


def _route_exists(call: Call) -> bool:
    for prefix, router in ROUTERS.items():
        if not call.path.startswith(prefix):
            continue
        rest = call.path[len(prefix) :] or ""
        for route in router.routes:
            pattern = re.sub(r"\{[^}]+\}", "[^/]+", route.path)
            if call.method in route.methods and re.fullmatch(pattern, rest):
                return True
    return False


@pytest.mark.parametrize("tool", mcp_tools.TOOLS, ids=lambda t: t.name)
def test_every_tool_targets_an_existing_endpoint(tool):
    call = _sample_call(tool)
    assert _route_exists(call), f"{tool.name} -> {call.method} {call.path}"


def test_routes_are_mounted():
    main = open("src/api/main.py", encoding="utf-8").read()
    assert 'mcp.router, prefix=f"{settings.API_V1_STR}/mcp"' in main
    assert "/auth/me/access-tokens" in main


# -- tokens -------------------------------------------------------------------------


def test_token_format_and_hashing():
    raw = access_tokens.new_token()
    assert raw.startswith("kbpat_") and len(raw) > 40
    assert access_tokens.new_token() != raw
    assert (
        len(access_tokens.hash_token(raw)) == 64
        and access_tokens.hash_token(raw) != raw
    )
    assert access_tokens.looks_like_token(raw)
    for bad in (None, "", "Bearer x", "eyJhbGciOi.jwt.like", "kbpat_" + "x" * 200):
        assert not access_tokens.looks_like_token(bad)


def test_authenticate_rejects_non_tokens_without_touching_the_database():
    class Boom:
        async def execute(self, *a, **k):
            raise AssertionError("database must not be queried")

    assert (
        asyncio.run(access_tokens.authenticate_token(Boom(), "eyJ.jwt.value")) is None
    )
    assert asyncio.run(access_tokens.authenticate_token(Boom(), None)) is None


def test_scope_rules():
    assert principal("write").can_write and not principal("read").can_write


def test_token_management_routes_need_a_signed_in_user():
    paths = {
        (r.path, tuple(sorted(r.methods))) for r in access_tokens_router.router.routes
    }
    assert ("", ("GET",)) in paths and ("", ("POST",)) in paths
    assert ("/{token_id}", ("DELETE",)) in paths
    for route in access_tokens_router.router.routes:
        deps = [d.call.__name__ for d in route.dependant.dependencies]
        assert "get_current_user" in deps, route.path
