"""Create the first global administrator.

A freshly migrated database has no users, and there is no way in through the API to fix
that: the only unauthenticated route is POST /login, POST /users requires the
`user.manage` permission, and there is no signup route. Without this script a deployed
environment is unreachable by anyone.

Run it as a ONE-OFF task using the migrator task definition, which already carries the
master database credential and the application code:

    aws ecs run-task \\
      --cluster qnsc-kb-develop \\
      --task-definition qnsc-kb-develop-migrator \\
      --launch-type FARGATE \\
      --network-configuration "awsvpcConfiguration={subnets=[...],securityGroups=[...]}" \\
      --overrides '{"containerOverrides":[{"name":"migrator",
                    "command":["python","scripts/create_admin.py"],
                    "environment":[{"name":"ADMIN_EMAIL","value":"..."},
                                   {"name":"ADMIN_NAME","value":"..."},
                                   {"name":"ADMIN_PASSWORD","value":"..."}]}]}'

Idempotent: an existing address is reported and left alone, so a re-run after a partial
failure is safe.
"""
from __future__ import annotations

import asyncio
import os
import sys

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from src.core.config import settings
from src.domain.auth import AuthService
from src.domain.rbac import bootstrap_rbac
from src.models.rbac import Role
from src.repositories.user import UserRepository


async def main() -> int:
    email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
    name = (os.getenv("ADMIN_NAME") or "").strip()
    password = os.getenv("ADMIN_PASSWORD") or ""

    if not email or not name or not password:
        print("ADMIN_EMAIL, ADMIN_NAME and ADMIN_PASSWORD are all required", file=sys.stderr)
        return 1

    # The MASTER credential, not the application role: this writes identity rows before
    # any application role has been granted anything, and it needs to set the RLS bypass
    # below.
    url = settings.MIGRATION_DATABASE_URL or settings.DATABASE_URL
    engine = create_async_engine(url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with Session() as db:
            # ENABLE_RLS is on and the identity policies FORCE row-level security, which
            # applies to the table owner too. Every policy admits
            # `current_setting('app.global_admin', true) = 'true'`, which is the
            # documented bypass — without it the inserts below are silently filtered
            # rather than rejected, and the script would report success having written
            # nothing.
            await db.execute(text("SET app.global_admin = 'true'"))

            repo = UserRepository(db)
            if await repo.get_by_email(email):
                print(f"user {email} already exists — nothing to do")
                return 0

            service = AuthService(repo)
            user = await service.register_user(
                email=email,
                name=name,
                password=password,
                role="Admin",
                # Without this the role is silently downgraded to Staff — register_user
                # ignores a privileged role unless the caller is explicitly allowed to
                # grant one.
                allow_privileged_role=True,
            )

            # Creates the role catalogue, including the global Admin role. The BACKFILL in
            # bootstrap_rbac deliberately hands a company-scoped Admin role to any user
            # that has a company_domain — that is what keeps an ordinary account from
            # becoming a cross-tenant administrator by acquiring the role string "Admin" —
            # so the global grant this script exists to make has to be stated here.
            await bootstrap_rbac(db)
            admin_role = await db.scalar(
                select(Role)
                .where(Role.name == "Admin", Role.company_domain.is_(None))
                .options(selectinload(Role.permissions))
            )
            if admin_role is None:
                print(
                    "global Admin role is missing after the RBAC bootstrap; no administrator created",
                    file=sys.stderr,
                )
                await db.rollback()
                return 1
            # AuthorizationService.is_global_administrator looks for exactly this role: the
            # one with company_domain = NULL. A company-scoped Admin is not the same thing
            # and cannot administer another tenant.
            user = await repo.get_by_id(user.id)
            user.roles = [admin_role]
            await db.commit()

            print(f"created global administrator {email} (id={user.id})")
            return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
