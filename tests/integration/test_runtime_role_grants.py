"""Prove the runtime role's grants on a fresh database, with the real two roles.

The on-prem failure this test exists for is ownership versus access: CNPG creates
``kb_app`` and ``kb_migrator``, ``kb_migrator`` owns the database, Alembic runs as
``kb_migrator``, and ``kb_app`` receives nothing -- so the API dies with
``permission denied for table permissions``.

This test builds a throwaway database owned by ``kb_migrator``, runs the whole
Alembic chain as that role, and checks both directions:

  * ``kb_app`` can SELECT/INSERT/UPDATE/DELETE every existing table and use every
    sequence, and a table created AFTER the migration is reachable the same way;
  * ``kb_app`` cannot create DDL, and the DEFAULT PRIVILEGES recorded for
    ``kb_migrator`` hold exactly the two least-privilege object sets.

Requires a PostgreSQL superuser connection so roles and a database can be
created. Uses ``MIGRATION_DATABASE_URL``/``DATABASE_URL`` and skips when neither
points at PostgreSQL or the connection is not a superuser.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import URL, make_url

ROOT = Path(__file__).resolve().parents[2]
APP_ROLE = "kb_app"
MIGRATOR_ROLE = "kb_migrator"
ROLE_PASSWORD = "runtime-grants-test"
_ROLE_PASSWORD_SQL = "'" + ROLE_PASSWORD.replace("'", "''") + "'"


def _configured_url() -> str:
    return os.getenv("MIGRATION_DATABASE_URL") or os.getenv("DATABASE_URL") or ""


def _admin_url() -> URL | None:
    raw = _configured_url()
    if not raw:
        return None
    try:
        url = make_url(raw)
    except Exception:
        return None
    if url.get_backend_name() != "postgresql" or not url.host:
        return None
    # The maintenance database always exists; the configured one may be the app's.
    return url.set(drivername="postgresql+psycopg2", database="postgres")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sync_url(url: URL, database: str) -> URL:
    return URL.create(
        drivername="postgresql+psycopg2",
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        database=database,
    )


def _async_url(url: URL, database: str, role: str, password: str) -> str:
    return URL.create(
        drivername="postgresql+asyncpg",
        username=role,
        password=password,
        host=url.host,
        port=url.port,
        database=database,
    ).render_as_string(hide_password=False)


def _ensure_role(conn, role: str) -> bool:
    exists = bool(
        conn.execute(
            sa.text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}
        ).scalar()
    )
    if exists:
        conn.execute(
            sa.text(
                f"ALTER ROLE {_quote(role)} WITH LOGIN PASSWORD {_ROLE_PASSWORD_SQL}"
            )
        )
    else:
        conn.execute(
            sa.text(
                f"CREATE ROLE {_quote(role)} WITH LOGIN PASSWORD {_ROLE_PASSWORD_SQL}"
            )
        )
    return not exists


@pytest.mark.skipif(_admin_url() is None, reason="no PostgreSQL admin URL configured")
def test_runtime_role_has_dml_and_no_ddl_on_a_fresh_database() -> None:
    admin_url = _admin_url()
    assert admin_url is not None
    database = f"kb_grants_{uuid.uuid4().hex[:12]}"
    created_roles: list[str] = []
    admin_engine = sa.create_engine(admin_url, isolation_level="AUTOCOMMIT")
    app_engine = None
    migrator_engine = None

    try:
        with admin_engine.connect() as conn:
            is_superuser = bool(
                conn.execute(
                    sa.text(
                        "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
                    )
                ).scalar()
            )
            if not is_superuser:
                pytest.skip("role creation requires a PostgreSQL superuser")

            if _ensure_role(conn, APP_ROLE):
                created_roles.append(APP_ROLE)
            if _ensure_role(conn, MIGRATOR_ROLE):
                created_roles.append(MIGRATOR_ROLE)
            conn.execute(
                sa.text(
                    f"CREATE DATABASE {_quote(database)} OWNER "
                    f"{_quote(MIGRATOR_ROLE)}"
                )
            )

        # CNPG's Database resource installs these as the operator (superuser) before
        # the migrator connects; vector is not a trusted extension, so the migrator
        # cannot create it. Everything else in the schema is built by kb_migrator.
        fresh_admin_engine = sa.create_engine(_sync_url(admin_url, database))
        with fresh_admin_engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))

        environment = os.environ.copy()
        environment.update(
            {
                "MIGRATION_DATABASE_URL": _async_url(
                    admin_url, database, MIGRATOR_ROLE, ROLE_PASSWORD
                ),
                "APP_DATABASE_ROLE": APP_ROLE,
                # This test is about object privileges, not RLS: leave policies off
                # so a DML check does not need an app.company_domain setting.
                "ENABLE_RLS": "false",
                "ENVIRONMENT": "development",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "migrations/alembic.ini",
                "upgrade",
                "head",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            "migrations failed as kb_migrator:\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )

        migrator_engine = sa.create_engine(
            _sync_url(admin_url, database).set(
                username=MIGRATOR_ROLE, password=ROLE_PASSWORD
            )
        )
        app_engine = sa.create_engine(
            _sync_url(admin_url, database).set(
                username=APP_ROLE, password=ROLE_PASSWORD
            )
        )

        try:
            with fresh_admin_engine.connect() as conn:
                tables = set(
                    conn.execute(
                        sa.text(
                            "SELECT c.relname FROM pg_class c "
                            "JOIN pg_namespace n ON n.oid = c.relnamespace "
                            "WHERE n.nspname = 'public' "
                            "AND c.relkind IN ('r', 'p', 'v', 'm', 'f')"
                        )
                    ).scalars()
                )
                # Tie the assertion to the outage that motivated this revision.
                assert "permissions" in tables

                missing_table_grants = conn.execute(sa.text("""
                        SELECT c.relname
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = 'public'
                          AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                          AND NOT (
                              has_table_privilege('kb_app', c.oid, 'SELECT')
                              AND has_table_privilege('kb_app', c.oid, 'INSERT')
                              AND has_table_privilege('kb_app', c.oid, 'UPDATE')
                              AND has_table_privilege('kb_app', c.oid, 'DELETE')
                          )
                        ORDER BY c.relname
                        """)).scalars().all()
                assert missing_table_grants == []

                missing_sequence_grants = conn.execute(sa.text("""
                        SELECT c.relname
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = 'public'
                          AND c.relkind = 'S'
                          AND NOT (
                              has_sequence_privilege('kb_app', c.oid, 'USAGE')
                              AND has_sequence_privilege('kb_app', c.oid, 'SELECT')
                          )
                        ORDER BY c.relname
                        """)).scalars().all()
                assert missing_sequence_grants == []

                assert not conn.execute(
                    sa.text("SELECT has_schema_privilege('kb_app', 'public', 'CREATE')")
                ).scalar_one()

                default_privileges = conn.execute(sa.text("""
                        SELECT d.defaclobjtype, a.privilege_type
                        FROM pg_default_acl d
                        JOIN pg_namespace n ON n.oid = d.defaclnamespace
                        JOIN pg_roles grantor ON grantor.oid = d.defaclrole
                        CROSS JOIN LATERAL aclexplode(d.defaclacl) a
                        JOIN pg_roles grantee ON grantee.oid = a.grantee
                        WHERE n.nspname = 'public'
                          AND grantor.rolname = 'kb_migrator'
                          AND grantee.rolname = 'kb_app'
                        """)).all()

            table_defaults = {
                privilege for kind, privilege in default_privileges if kind == "r"
            }
            sequence_defaults = {
                privilege for kind, privilege in default_privileges if kind == "S"
            }
            assert table_defaults == {"SELECT", "INSERT", "UPDATE", "DELETE"}
            assert sequence_defaults == {"USAGE", "SELECT"}

            # DEFAULT PRIVILEGES apply at CREATE TABLE time. A table created after the
            # migration must behave exactly like the ones it granted explicitly.
            with migrator_engine.begin() as conn:
                conn.execute(
                    sa.text(
                        "CREATE TABLE runtime_grant_probe "
                        "(id serial PRIMARY KEY, note text NOT NULL)"
                    )
                )

            with app_engine.begin() as conn:
                conn.execute(
                    sa.text(
                        "INSERT INTO runtime_grant_probe (note) VALUES ('app-inserted')"
                    )
                )
                conn.execute(
                    sa.text(
                        "UPDATE runtime_grant_probe SET note = 'app-updated' "
                        "WHERE note = 'app-inserted'"
                    )
                )
                assert (
                    conn.execute(
                        sa.text(
                            "SELECT note FROM runtime_grant_probe WHERE note = 'app-updated'"
                        )
                    ).scalar_one()
                    == "app-updated"
                )
                conn.execute(
                    sa.text(
                        "DELETE FROM runtime_grant_probe WHERE note = 'app-updated'"
                    )
                )

            with app_engine.connect() as conn:
                with pytest.raises(sa.exc.ProgrammingError) as excinfo:
                    conn.execute(
                        sa.text("CREATE TABLE public.runtime_grant_ddl (id int)")
                    )
                assert getattr(excinfo.value.orig, "pgcode", None) == "42501"
                conn.rollback()

        finally:
            if app_engine is not None:
                app_engine.dispose()
            if migrator_engine is not None:
                migrator_engine.dispose()
            fresh_admin_engine.dispose()

    finally:
        with admin_engine.connect() as conn:
            conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :database AND pid <> pg_backend_pid()"
                ),
                {"database": database},
            )
            conn.execute(sa.text(f"DROP DATABASE IF EXISTS {_quote(database)}"))
            for role in created_roles:
                try:
                    conn.execute(sa.text(f"DROP ROLE IF EXISTS {_quote(role)}"))
                except sa.exc.DBAPIError:
                    # The role may already be in use by another database on this
                    # cluster; the grants under test lived only in `database`.
                    pass
        admin_engine.dispose()
