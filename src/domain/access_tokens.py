"""Personal access tokens: the credential an MCP client presents.

A token acts as its owner. Nothing here grants anything: `authenticate_token` only answers
"which live user is this", and the request that follows is authorised by that user's own
roles and ACL like any other.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.models.access_token import PersonalAccessToken
from src.models.user import User

TOKEN_PREFIX = "kbpat_"
SCOPES = ("read", "write")
# last_used_at is a hint for the owner, not an audit log; writing it on every call would
# turn each MCP request into a database write.
LAST_USED_RESOLUTION = timedelta(minutes=5)


@dataclass(frozen=True)
class TokenPrincipal:
    token_id: uuid.UUID
    user_id: uuid.UUID
    email: str
    auth_version: int
    scope: str

    @property
    def can_write(self) -> bool:
        return self.scope == "write"


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def looks_like_token(raw: str | None) -> bool:
    return isinstance(raw, str) and raw.startswith(TOKEN_PREFIX) and len(raw) <= 128


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def token_view(item: PersonalAccessToken, now: datetime | None = None) -> dict:
    now = now or datetime.utcnow()
    return {
        "id": str(item.id),
        "name": item.name,
        "prefix": item.token_prefix,
        "scope": item.scope,
        "created_at": item.created_at,
        "expires_at": item.expires_at,
        "last_used_at": item.last_used_at,
        "revoked": item.revoked_at is not None,
        "expired": item.expires_at <= now,
    }


async def list_tokens(
    db: AsyncSession, user_id: uuid.UUID
) -> list[PersonalAccessToken]:
    rows = await db.execute(
        select(PersonalAccessToken)
        .where(PersonalAccessToken.user_id == user_id)
        .order_by(PersonalAccessToken.created_at.desc())
    )
    return list(rows.scalars().all())


async def count_live_tokens(db: AsyncSession, user_id: uuid.UUID) -> int:
    now = datetime.utcnow()
    return sum(
        1
        for item in await list_tokens(db, user_id)
        if item.revoked_at is None and item.expires_at > now
    )


async def create_token(
    db: AsyncSession,
    user: User,
    *,
    name: str,
    scope: str,
    expires_in_days: int,
) -> tuple[PersonalAccessToken, str]:
    """Persist a new token and return it with the raw value, which is never stored."""
    if scope not in SCOPES:
        raise ValueError("scope must be 'read' or 'write'")
    days = max(1, min(expires_in_days, settings.MCP_TOKEN_MAX_DAYS))
    raw = new_token()
    item = PersonalAccessToken(
        user_id=user.id,
        name=name.strip(),
        token_prefix=raw[: len(TOKEN_PREFIX) + 4],
        token_hash=hash_token(raw),
        scope=scope,
        expires_at=datetime.utcnow() + timedelta(days=days),
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item, raw


async def revoke_token(
    db: AsyncSession, user_id: uuid.UUID, token_id: uuid.UUID
) -> PersonalAccessToken | None:
    item = (
        await db.execute(
            select(PersonalAccessToken).where(
                PersonalAccessToken.id == token_id,
                PersonalAccessToken.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if item is None:
        return None
    if item.revoked_at is None:
        item.revoked_at = datetime.utcnow()
        await db.commit()
    return item


async def authenticate_token(
    db: AsyncSession, raw: str | None
) -> TokenPrincipal | None:
    """The principal for a live token, or None for anything else.

    The caller must have put `db` in the global-admin context first: the token is looked
    up by hash before its owner's tenant is known, the same way a refresh token is.
    Unknown, revoked, expired and deactivated-owner all look identical to the caller.
    """
    if not looks_like_token(raw):
        return None
    now = datetime.utcnow()
    row = (
        await db.execute(
            select(PersonalAccessToken, User)
            .join(User, User.id == PersonalAccessToken.user_id)
            .where(
                PersonalAccessToken.token_hash == hash_token(raw or ""),
                PersonalAccessToken.revoked_at.is_(None),
                PersonalAccessToken.expires_at > now,
            )
        )
    ).first()
    if row is None:
        return None
    item, user = row
    if not user.active:
        return None
    if item.last_used_at is None or now - item.last_used_at > LAST_USED_RESOLUTION:
        await db.execute(
            update(PersonalAccessToken)
            .where(PersonalAccessToken.id == item.id)
            .values(last_used_at=now)
        )
        await db.commit()
    return TokenPrincipal(
        token_id=item.id,
        user_id=user.id,
        email=user.email.strip().lower(),
        auth_version=user.auth_version,
        scope=item.scope,
    )
