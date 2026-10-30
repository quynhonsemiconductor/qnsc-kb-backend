"""The team's standing instructions for AI clients (see src/domain/mcp_config.py).

Read by any signed-in member (the MCP tool `get_project_config` calls this as its token's
owner); written only by an Admin or CEO. Every edit adds a version, and an editor must say
which version they started from, so two admins cannot silently overwrite each other.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db
from src.domain.mcp_config import (
    MAX_CHARS,
    can_edit_config,
    config_payload,
    digest,
)
from src.models.mcp_config import McpConfig, McpConfigVersion
from src.repositories.audit import AuditRepository

router = APIRouter()
NAME = "default"


class ConfigUpdate(BaseModel):
    content_md: str = Field(max_length=MAX_CHARS)
    #: The version this edit started from (0 when nothing is configured yet).
    expected_version: int = Field(ge=0)


async def _current(db: AsyncSession, company_domain: str, lock: bool = False):
    stmt = select(McpConfig).where(
        McpConfig.company_domain == company_domain, McpConfig.name == NAME
    )
    if lock:
        stmt = stmt.with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()


@router.get("")
async def get_config(
    known_version: int | None = Query(default=None, ge=0),
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    row = await _current(db, current_user.company_domain)
    version = row.version if row else 0
    if known_version is not None and known_version == version:
        return {"up_to_date": True, "configured": row is not None, "version": version}
    return config_payload(row)


@router.put("")
async def update_config(
    payload: ConfigUpdate,
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if not can_edit_config(current_user):
        raise HTTPException(
            status_code=403,
            detail="Only an Admin or CEO can change the team configuration",
        )
    content = payload.content_md.strip()
    if not content:
        raise HTTPException(status_code=422, detail="The configuration cannot be empty")
    row = await _current(db, current_user.company_domain, lock=True)
    current_version = row.version if row else 0
    if payload.expected_version != current_version:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "version_conflict",
                "message": "The configuration changed since you opened it; reload and merge.",
                "current_version": current_version,
            },
        )
    if row is not None and row.content_md == content:
        return config_payload(row)
    if row is None:
        row = McpConfig(
            company_domain=current_user.company_domain,
            name=NAME,
            version=1,
            content_md=content,
            updated_by=current_user.id,
        )
        db.add(row)
        await db.flush()
    else:
        row.version = current_version + 1
        row.content_md = content
        row.updated_by = current_user.id
    db.add(
        McpConfigVersion(
            config_id=row.id,
            version=row.version,
            content_md=content,
            updated_by=current_user.id,
        )
    )
    await db.commit()
    await db.refresh(row)
    await AuditRepository(db).record(
        current_user.id,
        "mcp_config_update",
        "mcp_config",
        str(row.id),
        detail={"version": row.version, "sha256": digest(content)[:12]},
    )
    return config_payload(row)


@router.get("/history")
async def config_history(
    current_user: Any = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    if not can_edit_config(current_user):
        raise HTTPException(
            status_code=403,
            detail="Only an Admin or CEO can read the configuration history",
        )
    row = await _current(db, current_user.company_domain)
    if row is None:
        return []
    versions = (
        (
            await db.execute(
                select(McpConfigVersion)
                .where(McpConfigVersion.config_id == row.id)
                .order_by(McpConfigVersion.version.desc())
                .limit(30)
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "version": item.version,
            "content_md": item.content_md,
            "updated_by": item.updated_by,
            "created_at": item.created_at,
        }
        for item in versions
    ]
