"""What a signed-in user can do with their own account: name, sessions, recent activity.

Login, logout, forgot/reset and change-password live in `auth.py`. These are the rest of the
self-service surface, kept in their own module because they all act on the CALLER and never
take a user id: there is nothing here an administrator can point at someone else, and no
permission to check beyond being signed in.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db
from src.api.routers.auth import (
    _clear_auth_cookies,
    _refresh_hash,
    _reject_cross_site_auth_request,
    _set_auth_cookies,
    _store_refresh_session,
    _user_response,
)
from src.core.config import settings
from src.domain.auth import AuthService
from src.domain.device import describe_device
from src.models.governance import AuditLog
from src.models.sessions import RefreshSession
from src.models.user import User
from src.repositories.audit import AuditRepository
from src.repositories.user import UserRepository

router = APIRouter()

# What the activity list shows. Everything else in the audit log is about documents and
# administration, which is not the user's own account history.
ACTIVITY_ACTIONS = (
    "login",
    "logout",
    "password_change",
    "password_reset",
    "profile_update",
    "session_revoke",
    "sessions_revoke_others",
    "invitation_accept",
    "entra_identity_link",
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class ProfileUpdate(BaseModel):
    name: str


def _clean_name(raw: str) -> str:
    """A display name: single-spaced, no control characters, 1-255 characters."""
    name = " ".join(_CONTROL_CHARACTERS.sub(" ", raw).split())
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    if len(name) > 255:
        raise HTTPException(status_code=422, detail="Name is too long")
    return name


def _utc(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return moment.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _current_refresh_value(
    plain: str | None, host: str | None, body: str | None = None
) -> str | None:
    """The caller's refresh token, read the way /auth/refresh and /auth/logout read it."""
    return body or host or (plain if settings.is_development_environment else None)


async def _live_sessions(db: AsyncSession, user_id: uuid.UUID) -> list[RefreshSession]:
    rows = (
        (
            await db.execute(
                select(RefreshSession)
                .where(
                    RefreshSession.user_id == user_id,
                    RefreshSession.revoked_at.is_(None),
                    RefreshSession.expires_at > datetime.utcnow(),
                )
                .order_by(RefreshSession.last_used_at.desc().nullslast())
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


def _session_view(item: RefreshSession, current_hash: str | None) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "current": current_hash is not None and item.token_hash == current_hash,
        "signed_in_at": _utc(item.signed_in_at or item.created_at),
        "last_used_at": _utc(item.last_used_at or item.created_at),
        "ip_address": item.ip_address,
        **describe_device(item.user_agent),
    }


@router.patch("")
async def update_profile(
    request: Request,
    payload: ProfileUpdate,
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Change the caller's own display name. Email, role and departments are admin-managed."""
    _reject_cross_site_auth_request(request)
    name = _clean_name(payload.name)
    user = (
        await db.execute(select(User).where(User.id == current_user.id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")
    previous = user.name
    if name != previous:
        user.name = name
        await AuditRepository(db).record(
            user.id,
            "profile_update",
            "user",
            str(user.id),
            detail={"before": {"name": previous}, "after": {"name": name}},
            commit=False,
        )
        await db.commit()
    reloaded = await UserRepository(db).get_by_id(user.id)
    return _user_response(reloaded or user)


@router.get("/sessions")
async def list_sessions(
    refresh_token_cookie: str | None = Cookie(default=None, alias="refresh_token"),
    host_refresh_token_cookie: str | None = Cookie(
        default=None, alias="__Host-refresh_token"
    ),
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The caller's signed-in devices, most recently used first."""
    value = _current_refresh_value(refresh_token_cookie, host_refresh_token_cookie)
    current_hash = _refresh_hash(value) if value else None
    sessions = await _live_sessions(db, current_user.id)
    return [_session_view(item, current_hash) for item in sessions]


@router.delete("/sessions/{session_id}")
async def revoke_session(
    request: Request,
    response: Response,
    session_id: uuid.UUID,
    refresh_token_cookie: str | None = Cookie(default=None, alias="refresh_token"),
    host_refresh_token_cookie: str | None = Cookie(
        default=None, alias="__Host-refresh_token"
    ),
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Sign one device out. Another user's session id is simply not found.

    The device's short-lived access token keeps working until it expires (at most
    ACCESS_TOKEN_EXPIRE_MINUTES); what ends immediately is its ability to renew. "Sign out
    everywhere else" bumps the account's auth version and so ends access tokens too.
    """
    _reject_cross_site_auth_request(request)
    item = (
        await db.execute(
            select(RefreshSession).where(
                RefreshSession.id == session_id,
                RefreshSession.user_id == current_user.id,
                RefreshSession.revoked_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Session not found")
    value = _current_refresh_value(refresh_token_cookie, host_refresh_token_cookie)
    was_current = value is not None and item.token_hash == _refresh_hash(value)
    item.revoked_at = datetime.utcnow()
    await AuditRepository(db).record(
        current_user.id,
        "session_revoke",
        "user",
        str(current_user.id),
        detail={
            "ip": item.ip_address,
            "user_agent": item.user_agent,
            "current": was_current,
        },
        commit=False,
    )
    await db.commit()
    if was_current:
        _clear_auth_cookies(response)
    return {"status": "revoked", "current": was_current}


@router.post("/sessions/revoke-others")
async def revoke_other_sessions(
    request: Request,
    response: Response,
    refresh_token_cookie: str | None = Cookie(default=None, alias="refresh_token"),
    host_refresh_token_cookie: str | None = Cookie(
        default=None, alias="__Host-refresh_token"
    ),
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Sign out every device but this one, including their access tokens.

    Bumping `auth_version` is what cuts the other devices' access tokens off at once, and
    it would cut this caller's too, so this session is replaced with fresh tokens exactly
    as a password change does. If the current session cannot be identified nothing is
    revoked: ending sessions without knowing which one is "this one" could sign the caller
    out of the very browser they are using.
    """
    _reject_cross_site_auth_request(request)
    value = _current_refresh_value(refresh_token_cookie, host_refresh_token_cookie)
    if not value:
        raise HTTPException(
            status_code=400, detail="The current session could not be identified"
        )
    current_hash = _refresh_hash(value)
    sessions = await _live_sessions(db, current_user.id)
    current = next((item for item in sessions if item.token_hash == current_hash), None)
    if current is None:
        raise HTTPException(
            status_code=400, detail="The current session could not be identified"
        )
    others = [item for item in sessions if item.id != current.id]

    user = (
        await db.execute(select(User).where(User.id == current_user.id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")
    now = datetime.utcnow()
    user.auth_version += 1
    await db.execute(
        update(RefreshSession)
        .where(
            RefreshSession.user_id == user.id,
            RefreshSession.revoked_at.is_(None),
        )
        .values(revoked_at=now)
    )
    await db.flush()
    reloaded = await UserRepository(db).get_by_id(user.id)
    user = reloaded or user
    service = AuthService(UserRepository(db))
    access = service.create_token(user)
    refresh = service.create_refresh_token(user)
    await _store_refresh_session(db, user, refresh, request, previous=current)
    await AuditRepository(db).record(
        user.id,
        "sessions_revoke_others",
        "user",
        str(user.id),
        detail={"revoked": len(others)},
        commit=False,
    )
    await db.commit()
    _set_auth_cookies(response, access, refresh)
    return {"status": "revoked", "revoked": len(others)}


@router.get("/activity")
async def recent_activity(
    limit: int = 20,
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The caller's recent sign-ins, sign-outs and security changes, newest first."""
    rows = (
        (
            await db.execute(
                select(AuditLog)
                .where(
                    AuditLog.user_id == current_user.id,
                    AuditLog.action.in_(ACTIVITY_ACTIONS),
                )
                .order_by(AuditLog.created_at.desc())
                .limit(max(1, min(limit, 50)))
            )
        )
        .scalars()
        .all()
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        detail = row.detail_json or {}
        out.append(
            {
                "id": str(row.id),
                "action": row.action,
                "outcome": row.outcome,
                "at": _utc(row.created_at),
                "ip_address": detail.get("ip"),
                "method": detail.get("method"),
                **describe_device(detail.get("user_agent") or None),
            }
        )
    return out
