"""Seed deterministic development users for the permission test matrix.

The password is supplied through ``SEED_TEST_PASSWORD`` so this script never
creates a credential in source control. It is intended for development/UAT
databases only, not for production provisioning.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, insert, select

from src.api.deps import SessionLocal
from src.core.security import get_password_hash
from src.domain.rbac import bootstrap_rbac
from src.models.rbac import Role, user_roles
from src.models.user import Department, User, user_departments


SEED_USERS = (
    ("kb-admin", "Admin", "Engineering"),
    ("kb-ceo", "CEO", "Engineering"),
    ("kb-reviewer", "Reviewer", "Engineering"),
    ("kb-staff", "Staff", "Finance"),
)


def seed_user_definitions(company_domain: str) -> list[dict[str, str]]:
    return [
        {
            "email": f"{local_part}@{company_domain}",
            "name": f"QNSC {role}",
            "role": role,
            "department": department,
        }
        for local_part, role, department in SEED_USERS
    ]


async def seed() -> None:
    company_domain = os.environ.get("SEED_COMPANY_DOMAIN", "acme.test").strip().lower()
    password = os.environ.get("SEED_TEST_PASSWORD", "").strip()
    if not password:
        raise SystemExit("SEED_TEST_PASSWORD must be set; refusing to create seeded accounts without an explicit password")
    if "@" in company_domain or not company_domain:
        raise SystemExit("SEED_COMPANY_DOMAIN must be a bare email domain")

    async with SessionLocal() as db:
        for department_name in {item["department"] for item in seed_user_definitions(company_domain)}:
            department = await db.scalar(select(Department).where(
                Department.company_domain == company_domain,
                Department.name == department_name,
            ))
            if department is None:
                department = Department(company_domain=company_domain, name=department_name, active=True)
                db.add(department)
        await db.flush()

        # ``bootstrap_rbac`` discovers company domains from users. Create the
        # deterministic identities first so it provisions this tenant's
        # system roles before the association rows are written below.
        for definition in seed_user_definitions(company_domain):
            user = await db.scalar(select(User).where(User.email == definition["email"]))
            if user is None:
                db.add(User(
                    email=definition["email"],
                    name=definition["name"],
                    password_hash=get_password_hash(password),
                    company_domain=company_domain,
                    role=definition["role"],
                    active=True,
                ))
            else:
                user.company_domain = company_domain
                user.role = definition["role"]
                user.active = True
        await db.flush()
        await bootstrap_rbac(db)

        # Company-scoped rows ONLY. bootstrap_rbac now creates an Admin role per domain
        # alongside the global one, so including `company_domain IS NULL` here would put
        # two rows under the key "Admin" and attach whichever the query returned last —
        # half the time the global role, which bypasses tenant RLS. Seed identities are
        # tenants; the global Admin role has one legitimate producer, admin_bootstrap.
        roles = {
            role.name: role
            for role in (await db.execute(
                select(Role).where(Role.company_domain == company_domain)
            )).scalars().all()
        }
        departments = {
            department.name: department
            for department in (await db.execute(
                select(Department).where(Department.company_domain == company_domain)
            )).scalars().all()
        }
        for definition in seed_user_definitions(company_domain):
            user = await db.scalar(
                select(User)
                .where(User.email == definition["email"])
            )
            if user is None:
                user = User(
                    email=definition["email"],
                    name=definition["name"],
                    password_hash=get_password_hash(password),
                    company_domain=company_domain,
                    role=definition["role"],
                    active=True,
                )
                db.add(user)
                await db.flush()
            else:
                user.name = definition["name"]
                user.password_hash = get_password_hash(password)
                user.auth_version += 1
                user.company_domain = company_domain
                user.role = definition["role"]
                user.active = True
            department = departments[definition["department"]]
            user.dept = department.name
            role = roles.get(definition["role"])
            if role is None:
                raise RuntimeError(f"RBAC bootstrap did not create role {definition['role']}")
            await db.execute(delete(user_departments).where(user_departments.c.user_id == user.id))
            await db.execute(insert(user_departments).values(user_id=user.id, department_id=department.id))
            await db.execute(delete(user_roles).where(user_roles.c.user_id == user.id))
            await db.execute(insert(user_roles).values(user_id=user.id, role_id=role.id))
        await db.commit()
    print(f"Seeded {len(SEED_USERS)} users for {company_domain}: " + ", ".join(item[1] for item in SEED_USERS))


if __name__ == "__main__":
    asyncio.run(seed())
