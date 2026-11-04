"""Grant the runtime role least-privilege DML access to every migrated object

On-prem CNPG creates ``kb_app`` and ``kb_migrator`` as managed roles and makes
``kb_migrator`` the owner of the ``kb`` database. Alembic runs as ``kb_migrator``,
so every table it creates is owned by that role -- and ownership is not access.
``kb_app`` therefore started with no privileges at all and the API failed on its
first query with ``permission denied for table permissions``.

This revision re-states the runtime grant set once, after the whole schema has
been built:

  * USAGE on the ``public`` schema;
  * SELECT, INSERT, UPDATE and DELETE on every existing table;
  * USAGE and SELECT on every existing sequence;
  * the same privileges as DEFAULT PRIVILEGES for tables and sequences created
    later by the role running this migration (``kb_migrator`` on-prem). Without
    the default privileges every future migration would reintroduce the outage.

No DDL is granted: ``kb_app`` must not be able to CREATE, ALTER or DROP anything.

The runtime role is ``APP_DATABASE_ROLE`` when that is set (the AWS task
definition and CI both set it), otherwise ``kb_app`` -- the on-prem chart does
not pass the variable, but CNPG always creates the role. A configured role that
does not exist is an error; an unconfigured environment with no ``kb_app`` is a
local ``create_all`` database and is skipped, matching ``20260802_05``.

Idempotent: GRANT and ALTER DEFAULT PRIVILEGES replace the same entries, so
re-running the revision changes nothing.
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "20261104_92"
down_revision = "20261102_91"
branch_labels = None
depends_on = None

DEFAULT_RUNTIME_ROLE = "kb_app"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _role_exists(bind, role: str) -> bool:
    return bool(
        bind.execute(
            sa.text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}
        ).scalar()
    )


def _runtime_role(bind) -> str | None:
    configured = (os.getenv("APP_DATABASE_ROLE") or "").strip()
    if configured:
        if not _role_exists(bind, configured):
            raise RuntimeError(
                f"APP_DATABASE_ROLE={configured!r} does not exist: the runtime role must be "
                "created before migrations run (scripts/bootstrap_db_role.py on AWS, CNPG "
                "managed roles on-prem)"
            )
        return configured
    if _role_exists(bind, DEFAULT_RUNTIME_ROLE):
        return DEFAULT_RUNTIME_ROLE
    return None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    role = _runtime_role(bind)
    if role is None:
        return

    runtime = _quote_ident(role)
    # The grantor of DEFAULT PRIVILEGES must be the role that creates future
    # tables, i.e. the role running this migration. Naming the configured
    # runtime role here would be wrong (it creates nothing) and naming
    # kb_migrator explicitly would fail on the AWS path, where the master owns
    # the schema instead. current_user is kb_migrator on-prem.
    owner = _quote_ident(bind.execute(sa.text("SELECT current_user")).scalar_one())

    op.execute(f"GRANT USAGE ON SCHEMA public TO {runtime}")
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        f"TO {runtime}"
    )
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {runtime}")

    op.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} IN SCHEMA public "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {runtime}"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} IN SCHEMA public "
        f"GRANT USAGE, SELECT ON SEQUENCES TO {runtime}"
    )


def downgrade() -> None:
    # No downgrade. The grants may predate this revision (20260802_05 granted the
    # same privileges when RLS was enabled), so revoking them here could take away
    # access this migration did not add. Revoke by hand if a deployment ever needs
    # to roll the runtime role back.
    pass
