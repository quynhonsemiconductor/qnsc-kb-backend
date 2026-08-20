"""Idempotently provision a global QNSC administrator for SSO bootstrap.

Usage:
    ADMIN_EMAIL=... ADMIN_PASSWORD=... poetry run python scripts/provision_admin.py

The account receives the system Admin role — it bypasses tenant RLS and can
read every company's data, so the password must always be supplied through
ADMIN_PASSWORD (no repo-committed default). The script deliberately resets the
password and re-grants Admin on every run; point it at the intended account.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.api.deps import SessionLocal
from src.core.security import get_password_hash
from src.domain.rbac import bootstrap_rbac
from src.models.rbac import Role
from src.models.user import User


async def provision() -> None:
    email = os.environ.get("ADMIN_EMAIL", "admin@qnsc.vn").strip().lower()
    if not email or "@" not in email:
        raise SystemExit("ADMIN_EMAIL must be a valid email address")
    company_domain = email.rsplit("@", 1)[1]
    display_name = os.environ.get("ADMIN_NAME", "Admin").strip() or "Admin"
    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    if len(password) < 12:
        raise SystemExit(
            "ADMIN_PASSWORD must be supplied and at least 12 characters long; "
            "this account bypasses tenant isolation."
        )

    async with SessionLocal() as db:
        user = await db.scalar(
            select(User)
            .where(User.email == email)
            .options(selectinload(User.roles))
        )
        if user is None:
            user = User(
                email=email,
                name=display_name,
                password_hash=get_password_hash(password),
                company_domain=company_domain,
                role="Admin",
                active=True,
            )
            db.add(user)
            await db.flush()
        else:
            user.name = display_name
            user.password_hash = get_password_hash(password)
            user.auth_version += 1
            user.company_domain = company_domain
            user.role = "Admin"
            user.active = True

        await bootstrap_rbac(db)
        admin_role = await db.scalar(
            select(Role)
            .where(Role.name == "Admin", Role.company_domain.is_(None))
            .options(selectinload(Role.permissions))
        )
        if admin_role is None:
            raise RuntimeError("Global Admin role was not created by RBAC bootstrap")
        user.roles = [admin_role]
        await db.commit()
        print(f"Provisioned active global Admin: {email} ({user.id})")


if __name__ == "__main__":
    asyncio.run(provision())
