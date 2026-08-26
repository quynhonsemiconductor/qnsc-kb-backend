"""Add provenance, governed tags, follows, conflicts and eval-set metadata."""

from alembic import op
import sqlalchemy as sa
import os

revision = "20260816_56"
down_revision = "20260816_55"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _has_column(table: str, column: str) -> bool:
    return any(item["name"] == column for item in sa.inspect(op.get_bind()).get_columns(table))


def upgrade() -> None:
    if not _has_column("articles", "source_changed"):
        op.add_column("articles", sa.Column("source_changed", sa.Boolean(), nullable=False, server_default=sa.false()))
        op.add_column("articles", sa.Column("source_changed_at", sa.DateTime(), nullable=True))
        op.add_column("articles", sa.Column("source_previous_hash", sa.String(length=64), nullable=True))
        op.alter_column("articles", "source_changed", server_default=None)
    if _has_table("draft_candidates") and not _has_column("draft_candidates", "source_position"):
        op.add_column("draft_candidates", sa.Column("source_position", sa.JSON(), nullable=True))

    if not _has_table("tag_catalog"):
        op.create_table(
            "tag_catalog",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("company_domain", sa.String(length=255), nullable=False),
            sa.Column("tag", sa.String(length=80), nullable=False),
            sa.Column("normalized_tag", sa.String(length=80), nullable=False),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("deprecated_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.UniqueConstraint("company_domain", "normalized_tag", name="uq_tag_catalog_company_normalized"),
        )
        op.create_index("ix_tag_catalog_company_active", "tag_catalog", ["company_domain", "active"])

    if not _has_table("article_followers"):
        op.create_table(
            "article_followers",
            sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("article_id", sa.Uuid(), sa.ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )

    if not _has_table("conflict_records"):
        op.create_table(
            "conflict_records",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("company_domain", sa.String(length=255), nullable=False),
            sa.Column("fact", sa.String(length=255), nullable=False),
            sa.Column("article_ids", sa.JSON(), nullable=False),
            sa.Column("evidence", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(length=30), nullable=False, server_default="open"),
            sa.Column("resolved_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("resolved_at", sa.DateTime(), nullable=True),
            sa.Column("resolution_note", sa.Text(), nullable=True),
        )
        op.create_index("ix_conflicts_company_status", "conflict_records", ["company_domain", "status"])

    if not _has_table("eval_sets"):
        op.create_table(
            "eval_sets",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("company_domain", sa.String(length=255), nullable=False),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("version", sa.String(length=50), nullable=False),
            sa.Column("environment", sa.String(length=50), nullable=False, server_default="uat"),
            sa.Column("status", sa.String(length=30), nullable=False, server_default="draft"),
            sa.Column("approved_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("approved_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("company_domain", "name", "version", name="uq_eval_set_company_name_version"),
        )

    if _has_table("eval_questions") and not _has_column("eval_questions", "eval_set_id"):
        op.add_column("eval_questions", sa.Column("eval_set_id", sa.Uuid(), sa.ForeignKey("eval_sets.id", ondelete="SET NULL"), nullable=True))
        op.create_index("ix_eval_questions_eval_set_id", "eval_questions", ["eval_set_id"])
    if _has_table("eval_runs") and not _has_column("eval_runs", "latency_ms"):
        op.add_column("eval_runs", sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"))
        op.alter_column("eval_runs", "latency_ms", server_default=None)

    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        for table, policy in (("tag_catalog", "tenant_tag_catalog"), ("conflict_records", "tenant_conflict_records"), ("eval_sets", "tenant_eval_sets")):
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY {policy} ON {table} USING "
                "(current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true)) "
                "WITH CHECK (current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true))"
            )


def downgrade() -> None:
    if _has_table("eval_questions") and _has_column("eval_questions", "eval_set_id"):
        op.drop_index("ix_eval_questions_eval_set_id", table_name="eval_questions")
        op.drop_column("eval_questions", "eval_set_id")
    if _has_table("eval_runs") and _has_column("eval_runs", "latency_ms"):
        op.drop_column("eval_runs", "latency_ms")
    for table, policy in (("eval_sets", "tenant_eval_sets"), ("conflict_records", "tenant_conflict_records"), ("tag_catalog", "tenant_tag_catalog")):
        op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
    for table in ("eval_sets", "conflict_records", "article_followers", "tag_catalog"):
        if _has_table(table):
            op.drop_table(table)
    for column in ("source_previous_hash", "source_changed_at", "source_changed"):
        if _has_column("articles", column):
            op.drop_column("articles", column)
    if _has_table("draft_candidates") and _has_column("draft_candidates", "source_position"):
        op.drop_column("draft_candidates", "source_position")
