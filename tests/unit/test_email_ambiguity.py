"""A case-insensitive email collision is always a refusal, never a 500."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import deps
from src.api.deps import get_current_user
from src.core.config import settings
from src.domain import auth as auth_domain
from src.repositories.user import AmbiguousEmailError


class _AmbiguousRepo:
    def __init__(self, *_args, **_kwargs):
        pass

    async def get_by_email(self, _email):
        raise AmbiguousEmailError("two rows collide")


@pytest.mark.asyncio
async def test_eventual_rest_caller_gets_401(monkeypatch):
    monkeypatch.setattr(settings, "SECRET_KEY", "test-key")
    monkeypatch.setattr(
        deps.jwt,
        "decode",
        lambda *a, **k: {"type": "access", "sub": "user@corp.vn", "av": 0},
    )

    async def no_context(*_args, **_kwargs):
        return None

    monkeypatch.setattr(deps, "set_database_context", no_context)
    monkeypatch.setattr(deps, "UserRepository", _AmbiguousRepo)

    class _Request:
        method = "GET"
        headers: dict = {}

    with pytest.raises(HTTPException) as caught:
        await get_current_user(
            request=_Request(),
            db=None,
            token=None,
            access_token_cookie=None,
            host_access_token_cookie="host-cookie-value",
        )
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_password_login_gets_401(monkeypatch):
    service = auth_domain.AuthService(_AmbiguousRepo())
    with pytest.raises(HTTPException) as caught:
        await service.authenticate_user("user@corp.vn", "password")
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_registration_gets_400(monkeypatch):
    service = auth_domain.AuthService(_AmbiguousRepo())
    with pytest.raises(HTTPException) as caught:
        await service.register_user(
            email="user@corp.vn",
            name="User",
            password="Password123!",
        )
    assert caught.value.status_code == 400
