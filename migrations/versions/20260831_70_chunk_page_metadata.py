"""page_number on both chunk tables, and the chunk_metadata table

These three schema objects were declared on the models and never migrated:
`parent_chunks.page_number`, `article_chunks.page_number` and the whole
`chunk_metadata` table (models/chunk.py). They looked correct because the baseline
20260802_00 builds the schema with `Base.metadata.create_all` against the LIVE models, so
a freshly built database has whatever the models currently say and `alembic check` sees no
drift. A database migrated FORWARD from before those model edits has none of them.

That is not a cosmetic gap. SQLAlchemy SELECTs every mapped column, so on such a database
the first chunk read fails with

    UndefinedColumnError: column article_chunks.page_number does not exist

which takes down search, citations and re-indexing at once -- the same outage
connectors.sync_interval_minutes already caused, documented at models/ops.py:17-27.

THE CLASS OF BUG, not just this instance: because the baseline runs create_all, a new
mapped column needs no migration for CI or a fresh deploy to pass, so drift against
already-deployed databases is invisible to `alembic check`. The baseline cannot be
rewritten -- every deployed database has it stamped -- so the rule has to be held by hand:
a new model column or table needs an explicit migration here, guarded, or it exists only
on databases that were built rather than migrated.

Guarded like 20260830_68 and 20260830_67: create_all may already have made all of this, so
every statement is conditional and re-running is a no-op.

Revision ID: 20260831_70
Revises: 20260831_69
Create Date: 2026-08-31
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260831_70"
down_revision = "20260831_69"
branch_labels = None
depends_on = None

METADATA_TABLE = "chunk_metadata"
# SQLAlchemy's default name for the `index=True` on ChunkMetadata.key. Spelled out so the
# guarded create below and a create_all-built database agree on one name.
METADATA_KEY_INDEX = "ix_chunk_metadata_key"


def upgrade() -> None:
    # Nullable with no default: page_number is unknown for anything indexed from
    # body_md rather than from extracted page text, and NULL is what the retrieval
    # formatter already falls back on.
    op.execute("ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")
    op.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")

    inspector = inspect(op.get_bind())
    if METADATA_TABLE not in set(inspector.get_table_names()):
        op.create_table(
            METADATA_TABLE,
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column(
                "chunk_id",
                sa.UUID(),
                sa.ForeignKey("article_chunks.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("key", sa.String(length=100), nullable=False),
            sa.Column("value", sa.String(length=255), nullable=False),
        )
    # Outside the table guard: a database built by create_all has the table but the index
    # name is the one thing worth asserting, since retrieval filters metadata by key.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {METADATA_KEY_INDEX} ON {METADATA_TABLE} (key)"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {METADATA_KEY_INDEX}")
    inspector = inspect(op.get_bind())
    if METADATA_TABLE in set(inspector.get_table_names()):
        op.drop_table(METADATA_TABLE)
    op.execute("ALTER TABLE article_chunks DROP COLUMN IF EXISTS page_number")
    op.execute("ALTER TABLE parent_chunks DROP COLUMN IF EXISTS page_number")
