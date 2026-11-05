"""provision_admin.py runs as the application role under FORCEd identity RLS.

The production failure this pins: `src.api.deps.SessionLocal` opened before any tenant
context was set, and the `users` RLS policy refused the INSERT under kb_app. The script
must set the same Admin context a request gets BEFORE it reads or writes.

Skipped unless the configured database is PostgreSQL, is reachable as a NON-superuser
application role, and has RLS enabled on `users` — a superuser bypasses every policy and
would make this test pass without proving anything.
"""

from __future__ import annotations

import importlib.util
import os
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", "")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not DATABASE_URL.startswith("postgresql"),
        reason="needs a real PostgreSQL database with RLS and an application role",
    ),
]


def _provision_module():
    spec = importlib.util.spec_from_file_location(
        "provision_admin", ROOT / "scripts" / "provision_admin.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_provision_succeeds_as_the_application_role_with_rls_on(monkeypatch):
    engine = create_async_engine(DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.connect() as conn:
            role = await conn.scalar(sa.text("SELECT current_user"))
            superuser = await conn.scalar(
                sa.text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
            )
            rls = await conn.scalar(
                sa.text(
                    "SELECT relrowsecurity AND relforcerowsecurity FROM pg_class "
                    "WHERE oid = to_regclass('public.users')"
                )
            )
        if superuser or not rls:
            pytest.skip(
                f"database role {role!r} bypasses RLS (superuser={superuser}, forced_rls={rls})"
            )

        module = _provision_module()
        monkeypatch.setattr(module, "SessionLocal", factory)
        email = f"provision-admin-test-{uuid.uuid4().hex[:12]}@qnsc.vn"
        # The flag must ignore this value: an SSO-only account gets a discarded random one.
        supplied = "password-that-must-not-be-used"
        monkeypatch.setenv("ADMIN_EMAIL", email)
        monkeypatch.setenv("ADMIN_NAME", "Provision Test")
        monkeypatch.setenv("ADMIN_PASSWORD", supplied)

        try:
            await module.provision(no_password=True)

            async with factory() as db:
                await module.set_database_context(db, "qnsc.vn", global_admin=True)
                row = (
                    await db.execute(
                        sa.text(
                            "SELECT password_hash, role FROM users WHERE email = :email"
                        ),
                        {"email": email},
                    )
                ).first()
            assert row is not None, "the insert was refused or rolled back"
            password_hash, role = row
            assert password_hash and password_hash != supplied
            assert role == "Admin"
        finally:
            async with factory() as db:
                await module.set_database_context(db, "qnsc.vn", global_admin=True)
                await db.execute(
                    sa.text("DELETE FROM users WHERE email = :email"), {"email": email}
                )
                await db.commit()
    finally:
        await engine.dispose()
