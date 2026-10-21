"""/readyz is a pure alias of /health/ready — same handler behaviour, one name.

The route table is asserted, not a served request: standing the app up needs the
local stack (see test_source_preview_headers.py), and the property under test is
wiring, not transport.
"""

import asyncio

import pytest
from fastapi import HTTPException

import src.api.main as main


def _get_routes(path: str):
    return [
        route
        for route in main.app.routes
        if getattr(route, "path", None) == path
        and "GET" in getattr(route, "methods", set())
    ]


def test_readyz_route_exists():
    assert len(_get_routes("/readyz")) == 1


def test_readyz_delegates_to_health_ready(monkeypatch):
    sentinel = {"status": "ready", "via": "alias"}

    async def fake_ready():
        return sentinel

    monkeypatch.setattr(main, "health_ready", fake_ready)
    assert asyncio.run(main.readyz()) is sentinel


def test_readyz_propagates_the_503_warmup_gate(monkeypatch):
    async def warming_up():
        raise HTTPException(status_code=503, detail="not ready")

    monkeypatch.setattr(main, "health_ready", warming_up)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(main.readyz())
    assert excinfo.value.status_code == 503
