"""The cited source must be viewable in our own review iframe — and only there.

The preview was blocked by two headers pulling against each other:

* the global middleware ASSIGNED ``X-Frame-Options: DENY`` to every response, overwriting
  whatever the endpoint had chosen, and DENY blocks an iframe even same-origin;
* the source response set a bare ``sandbox``, under which a browser's built-in PDF viewer
  renders nothing, because that viewer is itself scripted.

Relaxing either one carelessly is how an untrusted upload gets to read the signed-in
user's session, so most of what follows pins the part that must NOT be relaxed.
"""
from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from src.api import main
from src.api.routers.articles import SOURCE_VIEW_SECURITY_HEADERS

CSP = SOURCE_VIEW_SECURITY_HEADERS["Content-Security-Policy"]


def test_the_source_may_be_framed_by_our_own_ui():
    assert SOURCE_VIEW_SECURITY_HEADERS["X-Frame-Options"] == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in CSP


def test_the_document_is_still_sandboxed():
    assert "sandbox" in CSP


def test_scripts_are_allowed_because_the_pdf_viewer_needs_them():
    assert "sandbox allow-scripts" in CSP


def test_the_sandbox_never_grants_same_origin():
    """allow-scripts WITH allow-same-origin lets an uploaded file escape the sandbox.

    That combination would give a hostile HTML upload our origin, and with it the
    signed-in user's session. It is the one thing that must never appear here.
    """
    assert "allow-same-origin" not in CSP


def test_the_document_cannot_load_or_reach_anything():
    """Even if it can script, it has no channel to exfiltrate what it sees."""
    assert "default-src 'none'" in CSP
    assert "base-uri 'none'" in CSP
    assert "form-action 'none'" in CSP


def test_the_response_is_never_content_sniffed():
    assert SOURCE_VIEW_SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"


def test_the_middleware_defaults_rather_than_overwrites():
    """Assignment here is what made the endpoint's own choice unreachable."""
    source = inspect.getsource(main)

    assert 'setdefault("X-Frame-Options", "DENY")' in source
    assert 'headers["X-Frame-Options"] = "DENY"' not in source


def test_every_other_response_is_still_denied_framing():
    with TestClient(main.app) as client:
        response = client.get("/health/live")

    assert response.headers["X-Frame-Options"] == "DENY"


def test_an_unauthorized_source_request_is_not_framable():
    """The relaxation must ride on the endpoint, not on the path."""
    with TestClient(main.app) as client:
        response = client.get(
            "/api/v1/articles/00000000-0000-0000-0000-000000000000/source"
        )

    assert response.status_code in {401, 403}
    assert response.headers["X-Frame-Options"] == "DENY"
