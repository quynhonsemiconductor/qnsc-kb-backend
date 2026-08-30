"""Every mapped Connector column must actually exist in the database.

Cloud connector polling was dead. Every beat tick, `schedule_cloud_connector_syncs`
raised:

    asyncpg.exceptions.UndefinedColumnError:
    column connectors.sync_interval_minutes does not exist

`sync_interval_minutes` was declared on the ORM model and read by NOTHING — one
declaration, zero uses across src, tests and migrations. Its comment said it was
"retained for compatibility with pre-Alembic connector rows".

That is the trap. SQLAlchemy SELECTs every mapped column whether anyone reads it or not,
so a column kept for the benefit of old databases broke every database Alembic actually
built: no migration creates it. `20260802_08_connector_sync` patches the oauth columns
onto the pre-existing table with ADD COLUMN IF NOT EXISTS and skips this one.

Removing the attribute fixes both directions. A legacy database keeps its column —
unmapped, it is never referenced again — and an Alembic-built one stops asking for a
column that was never going to be there.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.models.ops import Connector

REPO = Path(__file__).parents[2]
MIGRATIONS = REPO / "migrations" / "versions"

#: Columns the pre-Alembic schema already carried, which therefore appear in no
#: migration. Everything outside this set must be created by one.
PRE_ALEMBIC_COLUMNS = {
    "id",
    "created_at",
    "updated_at",
    "name",
    "system",
    "status",
    "last_sync",
    "config_json",
    "company_domain",
    "created_by",
}


def _migration_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in MIGRATIONS.glob("*.py")
    )


def test_the_phantom_column_is_gone():
    """The specific column that killed connector polling."""
    assert "sync_interval_minutes" not in Connector.__table__.columns


def test_no_mapped_column_is_missing_from_the_migrations():
    """The general rule it broke: a mapped column with no migration is a query that
    fails at runtime, not a schema that quietly lags."""
    text = _migration_text()
    missing = [
        column.name
        for column in Connector.__table__.columns
        if column.name not in PRE_ALEMBIC_COLUMNS
        and not re.search(rf"\bconnectors?\b[^\n]*\b{re.escape(column.name)}\b", text)
        and column.name not in text
    ]
    assert not missing, (
        f"Connector maps {missing}, which no migration creates. SQLAlchemy SELECTs every "
        "mapped column, so this fails at runtime on any Alembic-built database."
    )
