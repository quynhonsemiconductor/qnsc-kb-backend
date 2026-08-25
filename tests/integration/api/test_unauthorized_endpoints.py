import asyncio
import re
import uuid

import httpx

from src.api import deps
from src.api.main import app
from src.models.user import User


PUBLIC_ROUTE_FRAGMENTS = (
    "/openapi.json",
    "/auth/login",
    "/auth/refresh",
    "/auth/logout",
    "/auth/oidc/config",
    "/auth/entra/login",
    "/auth/entra/callback",
    "/oauth/callback",
    "/connectors/webhooks/",
)


def _walk_routes(routes, prefix: str = ""):
    """Yield ``(full_path, methods)`` for every endpoint, however it was mounted.

    FastAPI 0.141 / Starlette 1.6 stopped flattening `include_router()` into
    `app.routes`. Each include is now a `_IncludedRouter` wrapper that carries NO `path`
    attribute, holds the sub-routes on `original_router.routes` with paths relative to
    the mount, and keeps the prefix on `include_context.prefix`.

    Reading `route.path` off the top level therefore found only the four routes declared
    directly on the app, and every `/api/v1/*` endpoint vanished. That is why the caller
    asserts the inventory is non-empty: this is a security regression test, and an empty
    inventory would have made it pass while checking nothing.
    """
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            context = getattr(route, "include_context", None)
            yield from _walk_routes(
                included.routes, prefix + (getattr(context, "prefix", "") or "")
            )
            continue
        path = getattr(route, "path", None)
        if path is None:
            continue
        yield prefix + path, set(getattr(route, "methods", set()) or set())


def _anonymous_path(path: str) -> str:
    """Replace FastAPI path parameters with safe syntactic examples."""
    values = {
        "company_domain": "acme.test",
        "version_num": "1",
        "key": "test-flag",
        "external_group_id": "external-group",
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return values.get(name, str(uuid.UUID(int=0)))

    return re.sub(r"{([^}]+)}", replace, path)


def test_api_data_endpoints_reject_anonymous_requests(monkeypatch):
    """R2 regression: private data routes never return partial anonymous data."""

    async def fake_db():
        yield object()

    async def no_metric(*_args, **_kwargs):
        return None

    monkeypatch.setattr("src.api.main.record_request_metric", no_metric)
    app.dependency_overrides[deps.get_db] = fake_db
    draft_id = uuid.uuid4()
    article_id = uuid.uuid4()
    connector_id = uuid.uuid4()
    try:
        requests = [
            ("GET", "/api/v1/auth/users", None),
            ("GET", "/api/v1/articles/", None),
            ("GET", "/api/v1/articles/source-uploads", None),
            ("POST", "/api/v1/articles/auto-tags", {"article_ids": [str(article_id)]}),
            (
                "POST",
                "/api/v1/articles/tags/confirm",
                {"items": [{"article_id": str(article_id), "tags": []}]},
            ),
            (
                "POST",
                "/api/v1/articles/source-uploads",
                {
                    "filename": "policy.txt",
                    "source_hash": "a" * 64,
                    "content_length": 10,
                },
            ),
            ("POST", f"/api/v1/articles/source-uploads/{draft_id}/complete", None),
            ("GET", "/api/v1/search?q=policy", None),
            ("GET", "/api/v1/ai/conversations", None),
            ("GET", "/api/v1/governance/pending-drafts", None),
            ("GET", f"/api/v1/governance/pending-drafts/{draft_id}/candidates", None),
            (
                "POST",
                f"/api/v1/governance/pending-drafts/{draft_id}/candidates/operation",
                {
                    "operation": "rename",
                    "candidate_id": str(uuid.uuid4()),
                    "title": "Nope",
                },
            ),
            (
                "POST",
                f"/api/v1/governance/pending-drafts/{draft_id}/candidates/commit",
                None,
            ),
            ("GET", "/api/v1/governance/audit-log", None),
            ("GET", "/api/v1/governance/health-metrics", None),
            ("POST", "/api/v1/governance/index/reprocess", {"article_ids": []}),
            ("GET", f"/api/v1/governance/index/reprocess/{draft_id}", None),
            ("GET", "/api/v1/connectors", None),
            ("GET", f"/api/v1/connectors/{connector_id}/jobs", None),
            ("GET", f"/api/v1/connectors/{connector_id}/acl-principals", None),
            ("GET", "/api/v1/knowledge/home", None),
            ("GET", "/api/v1/knowledge/browse", None),
            ("GET", "/api/v1/knowledge/sources", None),
            ("POST", "/api/v1/knowledge/content-requests", {"query": "missing policy"}),
            ("POST", "/api/v1/knowledge/role-preview", {"role": "Staff"}),
            ("GET", "/api/v1/meta/tags", None),
            ("GET", "/api/v1/meta/glossary", None),
            ("GET", "/api/v1/meta/groups", None),
            ("PUT", f"/api/v1/auth/groups/{uuid.uuid4()}/members", {"user_ids": []}),
            ("GET", "/api/v1/notifications", None),
            ("POST", f"/api/v1/notifications/{uuid.uuid4()}/read", None),
            (
                "POST",
                f"/api/v1/interactions/articles/{article_id}/comments",
                {"text": "Nope"},
            ),
            ("GET", f"/api/v1/interactions/articles/{article_id}/comments", None),
            ("DELETE", f"/api/v1/interactions/comments/{uuid.uuid4()}", None),
            ("POST", f"/api/v1/interactions/articles/{article_id}/votes", {"value": 1}),
            ("GET", f"/api/v1/interactions/articles/{article_id}/votes", None),
            ("GET", f"/api/v1/interactions/articles/{article_id}/user-vote", None),
            ("POST", f"/api/v1/interactions/articles/{article_id}/bookmark", None),
            ("DELETE", f"/api/v1/interactions/articles/{article_id}/bookmark", None),
            ("GET", "/api/v1/interactions/bookmarks", None),
            ("GET", "/api/v1/admin/llm/config", None),
        ]

        async def run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                results = []
                for method, path, body in requests:
                    response = await client.request(method, path, json=body)
                    results.append((method, path, response.status_code))
                return results

        results = asyncio.run(run())
        assert all(status == 401 for _, _, status in results), results
    finally:
        app.dependency_overrides.pop(deps.get_db, None)


def test_every_private_api_route_rejects_anonymous_requests(monkeypatch):
    """R2 contract: every current private API method must fail closed at auth."""

    async def fake_db():
        yield object()

    async def no_metric(*_args, **_kwargs):
        return None

    monkeypatch.setattr("src.api.main.record_request_metric", no_metric)
    app.dependency_overrides[deps.get_db] = fake_db
    route_cases = []
    try:
        for path, methods in _walk_routes(app.routes):
            if not path.startswith("/api/v1") or any(
                fragment in path for fragment in PUBLIC_ROUTE_FRAGMENTS
            ):
                continue
            for method in sorted(methods & {"GET", "POST", "PUT", "PATCH", "DELETE"}):
                route_cases.append((method, _anonymous_path(path)))

        async def run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                results = []
                for method, path in route_cases:
                    response = await client.request(method, path)
                    results.append((method, path, response.status_code))
                return results

        results = asyncio.run(run())
        assert route_cases, "The private API route inventory must not be empty"
        assert all(status == 401 for _, _, status in results), results
    finally:
        app.dependency_overrides.pop(deps.get_db, None)


def test_every_non_public_api_route_declares_auth_dependency():
    """R2 contract: adding a data route without auth must fail CI."""
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/v1") or any(
            fragment in path for fragment in PUBLIC_ROUTE_FRAGMENTS
        ):
            continue
        dependency_names = {
            getattr(item.call, "__name__", "")
            for item in getattr(route, "dependant", None).dependencies
        }
        assert dependency_names & {"get_current_user", "permission_checker"}, (
            sorted(getattr(route, "methods", set())),
            path,
        )


def test_authenticated_user_without_article_access_cannot_read_version_routes(
    monkeypatch,
):
    """R2 regression: version endpoints return no data outside Article scope."""

    class EmptyResult:
        def scalar_one_or_none(self):
            return None

        def scalars(self):
            return self

        def all(self):
            return []

    class EmptyDB:
        async def execute(self, _statement):
            return EmptyResult()

    async def fake_db():
        yield EmptyDB()

    unauthorized = User(
        id=uuid.uuid4(),
        email="unauthorized@acme.test",
        name="Unauthorized",
        role="Unassigned",
        company_domain="acme.test",
    )

    async def no_metric(*_args, **_kwargs):
        return None

    monkeypatch.setattr("src.api.main.record_request_metric", no_metric)
    app.dependency_overrides[deps.get_db] = fake_db
    app.dependency_overrides[deps.get_current_user] = lambda: unauthorized
    article_id = uuid.uuid4()
    try:

        async def run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                history = await client.get(f"/api/v1/articles/{article_id}/versions")
                version = await client.get(f"/api/v1/articles/{article_id}/versions/1")
                return history, version

        history, version = asyncio.run(run())
        assert history.status_code == 404
        assert version.status_code == 404
    finally:
        app.dependency_overrides.pop(deps.get_db, None)
        app.dependency_overrides.pop(deps.get_current_user, None)
