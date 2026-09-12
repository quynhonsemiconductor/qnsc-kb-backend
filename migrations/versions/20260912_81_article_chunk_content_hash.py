"""add content_hash to article_chunks for skip-if-unchanged re-embedding

Every re-index (an edit, a connector re-sync, a chunking/embedding-version bump) wiped and
rebuilt EVERY chunk of an article, even the ones whose text had not actually changed --
paying for a fresh embed call on content the corpus already had a perfectly good vector
for. `content_hash` (sha256 of the child chunk's own `chunk_text`, computed in
indexing.py) lets a re-index recognise that case and reuse the existing embedding instead.

Nullable and NOT backfilled: a chunk written before this column existed has no way to
retroactively prove what its hash "would have been" without re-reading the same text this
migration has no business doing, and a NULL hash simply never matches on the next
re-index -- degrading to "re-embed everything," which is the behavior every chunk already
had. The cache warms up naturally as articles are edited/re-indexed going forward.

Guarded the same way every column-add migration here is (20260910_74, 20260912_80): the
baseline revision builds tables from model metadata, so a fresh database may already have
this column.

Revision ID: 20260912_81
Revises: 20260912_80
Create Date: 2026-09-12
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260912_81"
down_revision = "20260912_80"
branch_labels = None
depends_on = None

TABLE = "article_chunks"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "content_hash" not in columns:
        op.add_column(TABLE, sa.Column("content_hash", sa.String(length=64), nullable=True))
        op.execute(
            f"CREATE INDEX IF NOT EXISTS ix_article_chunks_content_hash "
            f"ON {TABLE} (content_hash)"
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "content_hash" in columns:
        op.execute(f"DROP INDEX IF EXISTS ix_article_chunks_content_hash")
        op.drop_column(TABLE, "content_hash")
