"""The two controls that used to fail open: the ENVIRONMENT default and cookie CSRF.

Both were well-written checks pointed at the wrong condition. ``validate_production``
covered everything it needed to and then returned immediately because the default
ENVIRONMENT was ``development``, so an unset variable — a missed task-definition entry, a
platform that drops empty values — booted a public deployment with the SECRET_KEY that is
committed to this repository. ``_reject_cross_site_auth_request`` was correct and was
wired to the /auth routes only, while ``get_current_user`` accepted the httpOnly cookie on
every route, leaving every other mutating endpoint CSRF-able.

Everything here exercises the real callables rather than matching source text: a check
that reads the file cannot tell whether the check is reached.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api import main
from src.api.deps import _reject_cross_site_cookie_request, get_current_user
from src.core.config import Settings

ALLOWED_ORIGIN = "https://kb.example.com"
FOREIGN_ORIGIN = "https://evil.example.com"


@pytest.fixture
def allow_listed_origin(monkeypatch) -> str:
    """Point the shared allow-list at one explicit HTTPS origin."""
    from src.api import deps

    monkeypatch.setattr(deps.settings, "CORS_ORIGINS", ALLOWED_ORIGIN)
    return ALLOWED_ORIGIN


def _request(method: str, **headers: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "path": "/api/v1/articles",
            "headers": [
                (name.encode("latin-1"), value.encode("latin-1"))
                for name, value in headers.items()
            ],
        }
    )


# ── The default ──────────────────────────────────────────────────────────────


def test_an_unset_environment_is_production(monkeypatch):
    """The flip itself. The suite exports ENVIRONMENT, so remove it first."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    assert Settings(_env_file=None).ENVIRONMENT == "production"


def test_the_committed_secret_key_cannot_boot_by_default(monkeypatch):
    """What the permissive default allowed: the repository's own signing key, in public."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    settings = Settings(_env_file=None)

    assert settings.SECRET_KEY == "super-secret-key-change-in-production"
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        settings.validate_production()


def test_development_still_has_to_be_asked_for(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    Settings(_env_file=None, ENVIRONMENT="development").validate_production()


# ── Cookie CSRF ──────────────────────────────────────────────────────────────


def test_an_unsafe_method_from_a_foreign_origin_is_rejected(allow_listed_origin):
    with pytest.raises(HTTPException) as raised:
        _reject_cross_site_cookie_request(_request("POST", origin=FOREIGN_ORIGIN))

    assert raised.value.status_code == 403


def test_an_unsafe_method_from_an_allowed_origin_is_accepted(allow_listed_origin):
    _reject_cross_site_cookie_request(_request("POST", origin=allow_listed_origin))


def test_reads_are_never_blocked(allow_listed_origin):
    """A cross-origin GET cannot mutate anything, and blocking it would break embeds."""
    for method in ("GET", "HEAD", "OPTIONS"):
        _reject_cross_site_cookie_request(_request(method, origin=FOREIGN_ORIGIN))


def test_referer_is_the_fallback_when_origin_is_suppressed(allow_listed_origin):
    """Some browsers and privacy settings omit Origin on same-site requests."""
    _reject_cross_site_cookie_request(
        _request("POST", referer=f"{allow_listed_origin}/articles/new")
    )
    with pytest.raises(HTTPException) as raised:
        _reject_cross_site_cookie_request(
            _request("POST", referer=f"{FOREIGN_ORIGIN}/attack.html")
        )

    assert raised.value.status_code == 403


def test_an_unparseable_referer_is_rejected_rather_than_crashing(allow_listed_origin):
    """`urlparse` accepts a bad authority; reading `.port` is what raises ValueError.

    An uncaught one here would surface as a 500 on a normal write, so it is pinned.
    """
    with pytest.raises(HTTPException) as raised:
        _reject_cross_site_cookie_request(
            _request("POST", referer="https://kb.example.com:notaport/articles")
        )

    assert raised.value.status_code == 403


def test_a_referer_without_a_host_is_rejected(allow_listed_origin):
    with pytest.raises(HTTPException) as raised:
        _reject_cross_site_cookie_request(_request("POST", referer="about:blank"))

    assert raised.value.status_code == 403


def test_a_client_that_sends_neither_header_stays_supported(allow_listed_origin):
    """Scripts and CLI clients send no Origin and no Referer; browsers send at least one."""
    _reject_cross_site_cookie_request(_request("POST"))


def test_a_cookie_post_from_a_foreign_origin_is_rejected_before_the_token_is_read(
    allow_listed_origin,
):
    """Through the dependency, which is where the cookie is actually accepted.

    Driven with ``asyncio.run``, the way test_password_auth_flows.py drives its handlers:
    pytest-asyncio is a dev dependency that is not installed in every environment this
    suite runs in, and a coroutine returned to pytest is silently never awaited.
    """
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            get_current_user(
                _request("POST", origin=FOREIGN_ORIGIN),
                db=None,
                token=None,
                access_token_cookie="any-cookie-value",
            )
        )

    assert raised.value.status_code == 403


def test_a_bearer_post_is_unaffected_by_the_origin(allow_listed_origin):
    """A browser cannot set Authorization cross-origin without a preflight we must allow.

    The token is deliberately invalid: reaching a 401 proves the request got past the
    cross-site gate and into signature verification, which is the whole distinction.
    """
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            get_current_user(
                _request("POST", origin=FOREIGN_ORIGIN),
                db=None,
                token="not-a-real-jwt",
                access_token_cookie=None,
            )
        )

    assert raised.value.status_code == 401


# ── /metrics ─────────────────────────────────────────────────────────────────


def _permission_dependency_of(endpoint) -> object:
    parameter = inspect.signature(endpoint).parameters["current_user"]
    return parameter.default.dependency


def test_metrics_requires_a_governance_permission():
    """It published every route template, request count and latency total to anyone."""
    checker = _permission_dependency_of(main.metrics)
    reader = SimpleNamespace(roles=[], role="Staff", company_domain="qnsc.vn", id=None)

    with pytest.raises(HTTPException) as raised:
        checker(current_user=reader)

    assert raised.value.status_code == 403


def test_metrics_admits_a_governance_reader():
    checker = _permission_dependency_of(main.metrics)
    admin = SimpleNamespace(roles=[], role="Admin", company_domain="qnsc.vn", id=None)

    assert checker(current_user=admin) is admin
