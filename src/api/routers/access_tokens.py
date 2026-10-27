import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db
from src.core.config import settings
from src.domain import access_tokens
from src.models import User
from src.repositories.audit import AuditRepository

router = APIRouter()


class AccessTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scope: Literal["read", "write"] = "read"
    expires_in_days: int = Field(default=90, ge=1, le=3650)


def _require_enabled() -> None:
    if not settings.MCP_ENABLED:
        raise HTTPException(status_code=404, detail="MCP access is not enabled")


@router.get("")
async def list_access_tokens(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    _require_enabled()
    items = await access_tokens.list_tokens(db, current_user.id)
    return {
        "items": [access_tokens.token_view(item) for item in items],
        "max_days": settings.MCP_TOKEN_MAX_DAYS,
        "max_tokens": settings.MCP_TOKENS_PER_USER,
        "endpoint_path": f"{settings.API_V1_STR}/mcp",
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_access_token(
    payload: AccessTokenCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a token. The raw value is in this response and nowhere else, ever."""
    _require_enabled()
    if (
        await access_tokens.count_live_tokens(db, current_user.id)
        >= settings.MCP_TOKENS_PER_USER
    ):
        raise HTTPException(
            status_code=409,
            detail="Too many active tokens. Revoke one you no longer use first.",
        )
    item, raw = await access_tokens.create_token(
        db,
        current_user,
        name=payload.name,
        scope=payload.scope,
        expires_in_days=payload.expires_in_days,
    )
    await AuditRepository(db).record(
        current_user.id,
        "access_token_create",
        "access_token",
        str(item.id),
        detail={"name": item.name, "scope": item.scope},
    )
    return {**access_tokens.token_view(item), "token": raw}


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_access_token(
    token_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    _require_enabled()
    item = await access_tokens.revoke_token(db, current_user.id, token_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Token not found")
    await AuditRepository(db).record(
        current_user.id, "access_token_revoke", "access_token", str(item.id)
    )
