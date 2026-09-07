"""baseline schema for fresh and legacy databases

The original project created its schema at application startup. This
migration captures the current SQLAlchemy metadata so a new production
database can be initialized by Alembic alone.

`create_all` reads the LIVE models, which makes this baseline a moving target:
it builds whatever the models say TODAY, while the migrations after it replay
history. When something is removed from the models, every later migration that
still names it breaks on a from-scratch chain -- even though a forward-migrated
database is perfectly fine, because there the object was created before it was
removed from the models.

That is what `access_groups`, `user_groups`, `article_access` and the two
`access_group_bitmap` columns are doing below. 20260901_71 retired them in
favour of Department, so `create_all` no longer emits them, but the migrations
between here and there still index them, INSERT into them, write to them and
attach RLS policies to them. They are therefore created explicitly, exactly as
the models used to declare them, and dropped again by 20260901_71 at the point
history actually removes them.

Anything deleted from the models in future needs the same treatment: move its
DDL here rather than deleting it outright. A forward-migrated database will NOT
show the failure, so this path has to be exercised from an EMPTY database --
which is what a fresh production deploy and the CI test job both do.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from src.models import Base

revision = "20260802_00"
down_revision = None
branch_labels = None
depends_on = None


def _create_retired_audience_objects() -> None:
    """Recreate the pre-Department audience schema the later chain expects."""
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    if "access_groups" not in tables:
        op.create_table(
            "access_groups",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("company_domain", sa.String(length=255), nullable=False, server_default="local"),
            sa.Column("bitmask_position", sa.Integer(), nullable=False),
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_access_groups_company_domain "
            "ON access_groups (company_domain)"
        )
        # 20260806_12 and 20260806_14 replace these with composite tenant-scoped
        # indexes using IF NOT EXISTS / IF EXISTS, so creating them here keeps that
        # pair a no-op rather than an error.
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_access_groups_company_name "
            "ON access_groups (company_domain, name)"
        )

    if "user_groups" not in tables:
        op.create_table(
            "user_groups",
            sa.Column(
                "user_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "group_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("access_groups.id", ondelete="CASCADE"),
                primary_key=True,
            ),
        )

    if "article_access" not in tables:
        op.create_table(
            "article_access",
            sa.Column(
                "article_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("articles.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "group_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("access_groups.id", ondelete="CASCADE"),
                primary_key=True,
            ),
        )

    # Same reason as the tables above: these columns existed only because the
    # models declared them, and 20260807_32 writes to the article_chunks one.
    # 20260901_71 drops both again where history actually removes them.
    op.execute(
        "ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS "
        "access_group_bitmap BIGINT NOT NULL DEFAULT 1"
    )
    op.execute(
        "ALTER TABLE ai_cache ADD COLUMN IF NOT EXISTS "
        "access_group_bitmap BIGINT NOT NULL DEFAULT 0"
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, checkfirst=True)
    _create_retired_audience_objects()


def downgrade() -> None:
    # The baseline is intentionally not destructive. Individual migrations
    # own their reversible changes; dropping the entire application schema
    # from a production downgrade is unsafe.
    pass
