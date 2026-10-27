"""MCP server (Model Context Protocol, Streamable HTTP, stateless).

`POST {API_V1_STR}/mcp` speaks JSON-RPC 2.0. A client authenticates with a personal access
token (`Authorization: Bearer kbpat_...`) created under Account > Access tokens.

How a request is authorised: the token identifies its owner, we mint that owner a
five-minute access JWT, and every tool call is made against this same application through
an in-process ASGI transport. So the endpoint a tool maps onto performs exactly the checks
it performs for the web app (RBAC, department scope, ACLs, row-level security, per-user AI
rate limit, audit). There is no second implementation of "who may read this" to drift.

Stateless on purpose: no sessions, no server-initiated messages, no SSE. A client that
tries GET gets 405, which the spec allows. That keeps it correct behind several tasks.
"""

import json
from datetime import timedelta
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from src.api.deps import SessionLocal, set_database_context
from src.core.config import settings
from src.core.rate_limit import InMemoryRateLimiter
from src.core.security import create_access_token
from src.domain import mcp_tools
from src.domain.access_tokens import (
    TokenPrincipal,
    authenticate_token,
    looks_like_token,
)
from src.domain.mcp_tools import Call, ToolInputError
from src.version import __version__

logger = structlog.get_logger()
router = APIRouter()

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_BODY_BYTES = 2_500_000
MAX_BATCH = 20
CALL_TIMEOUT_SECONDS = 120.0
INTERNAL_BASE = "http://kb.internal"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002

INSTRUCTIONS = (
    "Company knowledge base. Use ask_knowledge_base for questions that need a cited "
    "answer, search_knowledge to locate passages, and get_article to read a whole "
    "document. Everything returned is already limited to what the signed-in user may "
    "see; if something is missing, they do not have access to it, so do not try to "
    "work around it. Changes you propose go through the normal review workflow."
)

_limiter = InMemoryRateLimiter(
    limit=settings.MCP_RATE_LIMIT_PER_MINUTE, window_seconds=60
)


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# -- calling the application as the token's owner -----------------------------------


async def run_call(request: Request, principal: TokenPrincipal, call: Call):
    """Execute one REST call as the token's owner and return `(status, json_payload)`."""
    token = create_access_token(
        principal.email, timedelta(minutes=5), principal.auth_version
    )
    headers = {"Authorization": f"Bearer {token}"}
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        headers["X-Request-ID"] = str(request_id)
    transport = httpx.ASGITransport(app=request.app, client=("127.0.0.1", 0))
    async with httpx.AsyncClient(
        transport=transport,
        base_url=INTERNAL_BASE,
        headers=headers,
        timeout=CALL_TIMEOUT_SECONDS,
    ) as client:
        response = await client.request(
            call.method,
            f"{settings.API_V1_STR}{call.path}",
            params=call.params,
            json=call.body,
        )
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text[:500]}
    return response.status_code, payload


# -- method handlers ----------------------------------------------------------------


class Context:
    def __init__(self, principal: TokenPrincipal, request: Request) -> None:
        self.principal = principal
        self.request = request

    async def call(self, call: Call):
        return await run_call(self.request, self.principal, call)


def _negotiated_version(requested: Any) -> str:
    return requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]


async def _initialize(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocolVersion": _negotiated_version(params.get("protocolVersion")),
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {
            "name": "qnsc-kb",
            "title": "QNSC Knowledge Base",
            "version": __version__,
        },
        "instructions": INSTRUCTIONS,
    }


async def _tools_list(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    tools = mcp_tools.visible_tools(ctx.principal.can_write)
    return {"tools": [tool.definition() for tool in tools]}


async def _tools_call(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    tool = mcp_tools.TOOLS_BY_NAME.get(name) if isinstance(name, str) else None
    if tool is None:
        raise JsonRpcError(INVALID_PARAMS, f"Unknown tool: {name!r}")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise JsonRpcError(INVALID_PARAMS, "'arguments' must be an object")
    if tool.write and not ctx.principal.can_write:
        return mcp_tools.error_result(
            "This access token is read-only. Create a token with write scope to use "
            f"{tool.name}."
        )
    logger.info(
        "MCP tool call",
        tool=tool.name,
        user_id=str(ctx.principal.user_id),
        token_id=str(ctx.principal.token_id),
    )
    try:
        call = tool.build(arguments)
        status, payload = await ctx.call(call)
    except ToolInputError as exc:
        return mcp_tools.error_result(f"Invalid arguments: {exc}")
    except httpx.TimeoutException:
        return mcp_tools.error_result("The knowledge base took too long to answer.")
    return mcp_tools.format_result(tool, status, payload)


async def _resources_list(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    status, payload = await ctx.call(
        Call("GET", "/articles/", params={"limit": 50, "offset": 0})
    )
    items = payload if status < 400 and isinstance(payload, list) else []
    return {
        "resources": [
            {
                "uri": mcp_tools.article_uri(str(item["id"])),
                "name": str(item.get("title") or item["id"]),
                "description": " · ".join(
                    str(item[k]) for k in ("dept", "type", "status") if item.get(k)
                ),
                "mimeType": "text/markdown",
            }
            for item in items
            if isinstance(item, dict) and item.get("id")
        ]
    }


async def _resource_templates(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    return {"resourceTemplates": mcp_tools.resource_templates()}


async def _resources_read(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    uri = params.get("uri")
    try:
        article_id = mcp_tools.article_id_from_uri(uri)
    except ToolInputError as exc:
        raise JsonRpcError(RESOURCE_NOT_FOUND, str(exc)) from None
    status, payload = await ctx.call(Call("GET", f"/articles/{article_id}"))
    if status >= 400 or not isinstance(payload, dict):
        raise JsonRpcError(RESOURCE_NOT_FOUND, f"Resource not available: {uri}")
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": "text/markdown",
                "text": mcp_tools.article_markdown(payload),
            }
        ]
    }


async def _prompts_list(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    return {"prompts": [p.definition() for p in mcp_tools.PROMPTS]}


async def _prompts_get(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    arguments = params.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        raise JsonRpcError(INVALID_PARAMS, "'arguments' must be an object")
    try:
        return mcp_tools.render_prompt(params.get("name"), arguments)
    except ToolInputError as exc:
        raise JsonRpcError(INVALID_PARAMS, str(exc)) from None


async def _ping(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    return {}


METHODS = {
    "initialize": _initialize,
    "ping": _ping,
    "tools/list": _tools_list,
    "tools/call": _tools_call,
    "resources/list": _resources_list,
    "resources/templates/list": _resource_templates,
    "resources/read": _resources_read,
    "prompts/list": _prompts_list,
    "prompts/get": _prompts_get,
}


def _error(message_id: Any, code: int, text: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": text},
    }


async def handle_message(ctx: Context, message: Any) -> dict[str, Any] | None:
    """One JSON-RPC message in, its response out (None for notifications and replies)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, INVALID_REQUEST, "Not a JSON-RPC 2.0 message")
    method = message.get("method")
    message_id = message.get("id")
    if method is None:
        # A response to something we never sent (this server sends no requests).
        return None
    if not isinstance(method, str):
        return _error(message_id, INVALID_REQUEST, "'method' must be a string")
    if "id" not in message:
        return None  # notification: notifications/initialized, notifications/cancelled
    handler = METHODS.get(method)
    if handler is None:
        return _error(message_id, METHOD_NOT_FOUND, f"Method not found: {method}")
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return _error(message_id, INVALID_PARAMS, "'params' must be an object")
    try:
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": await handler(ctx, params),
        }
    except JsonRpcError as exc:
        return _error(message_id, exc.code, exc.message)
    except Exception:
        # The detail stays in the server log; a client should not receive internals.
        logger.exception("MCP request failed", method=method)
        return _error(message_id, INTERNAL_ERROR, "Internal error")


# -- transport ----------------------------------------------------------------------


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {
            "error": "invalid_token",
            "error_description": "Missing or invalid access token",
        },
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer realm="qnsc-kb"'},
    )


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


def _check_origin(request: Request) -> None:
    """Refuse browser requests from origins we do not serve (DNS-rebinding defence).

    MCP clients are not browsers and send no Origin. A browser page that does send one
    must be one of ours; otherwise a hostile page could drive a token held in a tab.
    """
    origin = request.headers.get("origin")
    if origin and origin not in settings.cors_origin_list:
        raise HTTPException(status_code=403, detail="Origin not allowed")


@router.post("")
async def mcp_endpoint(request: Request) -> Response:
    if not settings.MCP_ENABLED:
        raise HTTPException(status_code=404, detail="MCP is not enabled")
    _check_origin(request)
    raw = _bearer(request)
    # Anything that is not shaped like a token is refused before a connection is taken.
    if raw is None or not looks_like_token(raw):
        return _unauthorized()
    async with SessionLocal() as db:
        # The token is found by its hash before its owner's tenant is known, like a
        # refresh token; nothing else runs in this global context.
        await set_database_context(db, None, True)
        principal = await authenticate_token(db, raw)
    if principal is None:
        return _unauthorized()

    allowed, retry_after = _limiter.allow(str(principal.token_id))
    if not allowed:
        return JSONResponse(
            {"detail": "Too many MCP requests. Slow down."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return JSONResponse(_error(None, INVALID_REQUEST, "Request too large"), 413)
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(_error(None, INVALID_REQUEST, "Request too large"), 413)
    try:
        parsed = json.loads(body)
    except ValueError:
        return JSONResponse(_error(None, PARSE_ERROR, "Parse error"), status_code=400)

    ctx = Context(principal, request)
    if isinstance(parsed, list):
        if not parsed or len(parsed) > MAX_BATCH:
            return JSONResponse(
                _error(
                    None, INVALID_REQUEST, f"Batch must have 1-{MAX_BATCH} messages"
                ),
                status_code=400,
            )
        results = [r for r in [await handle_message(ctx, m) for m in parsed] if r]
        if not results:
            return Response(status_code=202)
        return JSONResponse(results)

    result = await handle_message(ctx, parsed)
    if result is None:
        return Response(status_code=202)
    return JSONResponse(result)


# GET (server-initiated stream) and DELETE (end session) are optional in the spec and this
# server has neither. Not registering them is deliberate: the framework then answers 405
# with `Allow: POST`, and no anonymous-reachable route exists beyond the authenticated POST.
