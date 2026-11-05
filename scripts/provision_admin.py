"""Idempotently provision a global QNSC administrator for SSO bootstrap.

Usage:
    ADMIN_EMAIL=... ADMIN_PASSWORD=... poetry run python scripts/provision_admin.py
    ADMIN_EMAIL=... poetry run python scripts/provision_admin.py --no-password

The account receives the system Admin role — it bypasses tenant RLS and can
read every company's data, so the password must always be supplied through
ADMIN_PASSWORD (no repo-committed default). The script deliberately resets the
password and re-grants Admin on every run; point it at the intended account.

`--no-password` is for an SSO-only bootstrap (prod admins sign in with Entra):
a random password is generated and discarded, so the account has no usable
password. A supplied ADMIN_PASSWORD is ignored in that mode.

The session sets the same tenant context an Admin request gets before it reads
or writes: the identity RLS policies FORCE row-level security, and under the
application role (`kb_app`) an INSERT into `users` without it is refused.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.api.deps import SessionLocal, set_database_context
from src.core.security import get_password_hash
from src.domain.rbac import bootstrap_rbac
from src.models.rbac import Role
from src.models.user import User


async def provision(*, no_password: bool = False) -> None:
    email = os.environ.get("ADMIN_EMAIL", "admin@qnsc.vn").strip().lower()
    if not email or "@" not in email:
        raise SystemExit("ADMIN_EMAIL must be a valid email address")
    company_domain = email.rsplit("@", 1)[1]
    display_name = os.environ.get("ADMIN_NAME", "Admin").strip() or "Admin"
    if no_password:
        # SSO-only: there must be no usable password, and the value never leaves this call.
        password = secrets.token_urlsafe(32)
    else:
        password = os.environ.get("ADMIN_PASSWORD", "").strip()
        if len(password) < 12:
            raise SystemExit(
                "ADMIN_PASSWORD must be supplied and at least 12 characters long "
                "(or pass --no-password for an SSO-only account); "
                "this account bypasses tenant isolation."
            )

    async with SessionLocal() as db:
        # The same context an Admin request gets. Under FORCEd identity RLS an INSERT into
        # `users` is otherwise refused for kb_app: "new row violates row-level security
        # policy". Set before the first read, so the existing-account lookup is visible too.
        await set_database_context(
            db,
            company_domain,
            global_admin=True,
            global_article_access=True,
            global_identity_access=True,
            global_connector_access=True,
            global_governance_access=True,
        )

        user = await db.scalar(
            select(User).where(User.email == email).options(selectinload(User.roles))
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
        mode = " (SSO-only, no usable password)" if no_password else ""
        print(f"Provisioned active global Admin: {email} ({user.id}){mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-password",
        action="store_true",
        help="SSO-only account: generate a random password, discard it, and ignore ADMIN_PASSWORD",
    )
    args = parser.parse_args()
    asyncio.run(provision(no_password=args.no_password))


if __name__ == "__main__":
    main()
