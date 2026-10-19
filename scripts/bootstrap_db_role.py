"""Ensure the least-privilege application database role exists.

Migration ``20260802_05_tenant_rls`` grants schema, table and sequence privileges to
``APP_DATABASE_ROLE`` but never creates it. Under Compose that gap is filled by
``docker/init-db.sh``, which Postgres runs as a container init hook. A managed
database (RDS) has no such hook, so the role has to be created by something that runs
before the migrations — this script, from the migrator task's entrypoint.

Connects with ``MIGRATION_DATABASE_URL`` (the master credential), falling back to
``DATABASE_URL`` exactly as ``migrations/env.py`` does. Creating a role requires
CREATEROLE, which the application role deliberately does not hold.

No-ops when either ``APP_DATABASE_ROLE`` or ``APP_DATABASE_PASSWORD`` is unset, so a
deployment that has not adopted the least-privilege role still migrates normally.

The password is re-applied on every run, not only at creation. Rotating the secret
therefore takes effect on the next deploy without a manual ``ALTER ROLE`` — otherwise
the application would keep the new password while the role kept the old one, and the
failure would surface as an authentication error minutes later in a different task.
"""

from __future__ import annotations

import asyncio
import os
import sys


def _quote_ident(value: str) -> str:
    """Quote an SQL identifier. Doubling embedded quotes is the escape."""
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    """Quote an SQL string literal. Doubling embedded quotes is the escape."""
    return "'" + value.replace("'", "''") + "'"


def _asyncpg_dsn(url: str) -> str:
    """Strip the SQLAlchemy driver marker: asyncpg takes a plain libpq URL."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1).replace(
        "postgresql+psycopg2://", "postgresql://", 1
    )


async def _ensure_role(dsn: str, role: str, password: str) -> None:
    import asyncpg

    connection = await asyncpg.connect(dsn)
    try:
        exists = await connection.fetchval(
            "SELECT 1 FROM pg_roles WHERE rolname = $1", role
        )
        ident = _quote_ident(role)
        secret = _quote_literal(password)
        if exists:
            # DDL takes no bind parameters, hence the manual quoting above.
            await connection.execute(f"ALTER ROLE {ident} WITH LOGIN PASSWORD {secret}")
            print(f"bootstrap_db_role: role {role!r} already exists, password synced")
        else:
            await connection.execute(f"CREATE ROLE {ident} LOGIN PASSWORD {secret}")
            print(f"bootstrap_db_role: created role {role!r}")
    finally:
        await connection.close()


def main() -> int:
    role = (os.getenv("APP_DATABASE_ROLE") or "").strip()
    password = os.getenv("APP_DATABASE_PASSWORD") or ""
    if not role or not password:
        print(
            "bootstrap_db_role: APP_DATABASE_ROLE / APP_DATABASE_PASSWORD unset — "
            "skipping (the application connects as the master user)"
        )
        return 0

    # Resolved through Settings, NOT os.getenv, because a deployed migrator is handed
    # connection PARTS (DATABASE_HOST + MIGRATION_DATABASE_USER/PASSWORD) rather than a
    # URL — a password arrives as its own injected secret and cannot be interpolated into
    # the middle of a URL at task start. Settings.model_post_init composes the URL from
    # whichever form is present, so reading the environment directly here would work
    # locally and fail in every deployed environment.
    from src.core.config import settings

    url = settings.MIGRATION_DATABASE_URL or settings.DATABASE_URL
    if not url:
        print(
            "bootstrap_db_role: no database connection configured — set "
            "MIGRATION_DATABASE_URL, or DATABASE_HOST with MIGRATION_DATABASE_USER and "
            "MIGRATION_DATABASE_PASSWORD",
            file=sys.stderr,
        )
        return 1

    asyncio.run(_ensure_role(_asyncpg_dsn(url), role, password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
