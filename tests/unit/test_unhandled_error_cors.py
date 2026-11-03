"""An unhandled 500 must still reach the browser, or a server bug reads as an outage.

A source upload failed in the browser with a bare ``Network Error`` — no status, no
CORS headers. It was diagnosed twice as infrastructure (the Cloudflare tunnel's memory
limit, then the ClamAV sidecar) and was neither. The API had returned a real HTTP 500,
and the browser threw the response away before any JavaScript could see it:

Starlette turns an escaped exception into a 500 inside ``ServerErrorMiddleware``, which
wraps every user middleware. That response therefore never travels back out through
``CORSMiddleware``, so it carries no ``access-control-allow-origin``; a cross-origin
caller's browser discards it and axios reports ``Network Error`` with no status. The
failure is then indistinguishable from the API being unreachable, which is exactly how
two rounds of investigation went to the wrong layer.

The fix is ordering, and ordering is easy to reintroduce a bug into: ``add_middleware``
inserts at the FRONT, so the last middleware registered is the OUTERMOST. The boundary
must be registered BEFORE ``CORSMiddleware`` to sit inside it. Swapping those two lines
restores the original bug while every other test still passes — hence this file.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from src.api import main

ORIGIN = "https://kb.example.test"


@pytest.fixture(scope="module")
def client_and_origin():
    app = main.app
    # Use an origin the deployed configuration actually allows, so the assertions test
    # the middleware rather than a rejected Origin header.
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    origin = cors.kwargs["allow_origins"][0]

    @app.post("/__test__/unhandled")
    async def _unhandled():
        raise RuntimeError("simulated failure below the router")

    @app.post("/__test__/unhandled-non-ascii")
    async def _unhandled_non_ascii():
        # The traceback rendered by the boundary's own logging must not be able to
        # defeat the boundary: on a non-UTF-8 stream this raises UnicodeEncodeError
        # while logging, which previously escaped and restored the CORS-less 500.
        raise RuntimeError("Tài liệu — em dash and Vietnamese diacritics")

    @app.post("/__test__/handled")
    async def _handled():
        raise HTTPException(status_code=422, detail="handled")

    return TestClient(app, raise_server_exceptions=False), origin


def test_the_boundary_sits_inside_cors():
    """Ordering is the whole fix, so assert it directly rather than only via behaviour."""
    names = [
        m.kwargs.get("dispatch").__name__
        if m.kwargs.get("dispatch") is not None
        else m.cls.__name__
        for m in main.app.user_middleware
    ]
    assert "unhandled_error_boundary" in names, names
    assert "CORSMiddleware" in names, names
    # user_middleware is ordered outermost -> innermost. CORS must wrap the boundary,
    # otherwise it cannot add headers to the response the boundary returns.
    assert names.index("CORSMiddleware") < names.index("unhandled_error_boundary"), names


def test_an_unhandled_error_returns_500_with_cors_headers(client_and_origin):
    client, origin = client_and_origin
    response = client.post("/__test__/unhandled", headers={"Origin": origin})
    assert response.status_code == 500
    # The assertion that matters: without this header the browser discards the response
    # and the caller sees "Network Error" instead of a 500 it can report.
    assert response.headers.get("access-control-allow-origin") == origin


def test_the_error_body_does_not_leak_internals(client_and_origin):
    client, origin = client_and_origin
    response = client.post("/__test__/unhandled", headers={"Origin": origin})
    assert response.json() == {"detail": "Internal server error"}
    assert "simulated failure" not in response.text


def test_the_500_is_correlatable_with_its_log_line(client_and_origin):
    """The body is deliberately opaque, so the request id is the only way back to the
    traceback in CloudWatch."""
    client, origin = client_and_origin
    response = client.post("/__test__/unhandled", headers={"Origin": origin})
    assert response.headers.get("x-request-id")


def test_logging_failure_cannot_defeat_the_boundary(client_and_origin):
    client, origin = client_and_origin
    response = client.post("/__test__/unhandled-non-ascii", headers={"Origin": origin})
    assert response.status_code == 500
    assert response.headers.get("access-control-allow-origin") == origin


def test_a_handled_error_is_unaffected(client_and_origin):
    """HTTPException is turned into a response further in and must not reach the
    boundary, so its status and body survive untouched."""
    client, origin = client_and_origin
    response = client.post("/__test__/handled", headers={"Origin": origin})
    assert response.status_code == 422
    assert response.json() == {"detail": "handled"}
    assert response.headers.get("access-control-allow-origin") == origin
