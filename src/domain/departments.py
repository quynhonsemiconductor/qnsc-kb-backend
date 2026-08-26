"""Tenant-scoped department validation and canonicalization."""
from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import Department


def normalize_department_name(name: str | None) -> str | None:
    value = (name or "").strip()
    return value or None


async def resolve_active_department(
    db: AsyncSession,
    company_domain: str,
    name: str | None,
    *,
    required: bool = True,
) -> Department | None:
    canonical = normalize_department_name(name)
    if canonical is None:
        if required:
            raise HTTPException(status_code=422, detail="A department is required")
        return None
    department = await db.scalar(
        select(Department).where(
            Department.company_domain == company_domain,
            Department.active.is_(True),
            Department.kind == "org",
            func.lower(Department.name) == canonical.lower(),
        )
    )
    if department is None:
        raise HTTPException(status_code=422, detail="Department does not exist or is inactive")
    return department


async def resolve_active_departments(
    db: AsyncSession,
    company_domain: str,
    department_ids: list,
    *,
    required: bool = True,
) -> list[Department]:
    """Resolve a unique, tenant-scoped set of active departments."""
    unique_ids = list(dict.fromkeys(department_ids))
    if not unique_ids:
        if required:
            raise HTTPException(status_code=422, detail="At least one department is required")
        return []
    departments = list((await db.execute(select(Department).where(
        Department.id.in_(unique_ids),
        Department.company_domain == company_domain,
        Department.active.is_(True),
    ))).scalars().all())
    if len(departments) != len(unique_ids):
        raise HTTPException(status_code=422, detail="Every department must be active and belong to the user's company")
    by_id = {department.id: department for department in departments}
    return [by_id[department_id] for department_id in unique_ids]


async def lock_company_access_groups(db: AsyncSession, company_domain: str) -> None:
    """Serialize bit-position allocation for one tenant on PostgreSQL."""
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:domain, 0))"),
        {"domain": company_domain},
    )
