"""add risk_tiers/version to approval_rules and the approval_rule_versions history table

Two additions to the approval agent's scoping, neither changing any existing rule's
behavior on its own:

- `risk_tiers` on approval_rules: an additional optional scoping filter (None means
  "any", same convention every other scoping column here already uses), matched against
  src/domain/approval_agent.py::compute_draft_risk_tier. A rule written before this
  column existed keeps its exact old scope.
- `version` on approval_rules, plus the approval_rule_versions history table: rules
  mutate in place, so there was no way to answer "what did this rule actually say when it
  fired on that document last month?" once it had since been edited. Every create/update
  in src/api/routers/governance.py now snapshots the rule's full state at that version.

Guarded the same way 20260910_74 (tag_catalog.parent_id) and 20260829_65
(approval_rules itself) are: the baseline revision builds tables from model metadata, so
a fresh database may already have these; an unguarded ADD COLUMN/CREATE TABLE fails on
one of the two directions depending which ran first.

Revision ID: 20260912_80
Revises: 20260912_79
Create Date: 2026-09-12
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision = "20260912_80"
down_revision = "20260912_79"
branch_labels = None
depends_on = None

RULES_TABLE = "approval_rules"
VERSIONS_TABLE = "approval_rule_versions"


def upgrade() -> None:
    inspector = inspect(op.get_bind())

    if RULES_TABLE in set(inspector.get_table_names()):
        columns = {column["name"] for column in inspector.get_columns(RULES_TABLE)}
        if "risk_tiers" not in columns:
            op.add_column(RULES_TABLE, sa.Column("risk_tiers", sa.JSON(), nullable=True))
        if "version" not in columns:
            op.add_column(
                RULES_TABLE,
                sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
            )

    if VERSIONS_TABLE not in set(inspector.get_table_names()):
        op.create_table(
            VERSIONS_TABLE,
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("rule_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=150), nullable=False),
            sa.Column("instruction", sa.Text(), nullable=False),
            sa.Column("active", sa.Boolean(), nullable=False),
            sa.Column("priority", sa.Integer(), nullable=False),
            sa.Column("connector_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("dept", sa.String(length=100), nullable=True),
            sa.Column("file_extensions", sa.JSON(), nullable=True),
            sa.Column("max_similarity_score", sa.Float(), nullable=True),
            sa.Column("risk_tiers", sa.JSON(), nullable=True),
            sa.Column("can_approve", sa.Boolean(), nullable=False),
            sa.Column("can_reject", sa.Boolean(), nullable=False),
            sa.Column("changed_by", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
            ),
            sa.Column(
                "updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
            ),
            sa.ForeignKeyConstraint(["rule_id"], ["approval_rules.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["changed_by"], ["users.id"], ondelete="SET NULL"),
            sa.UniqueConstraint("rule_id", "version", name="uq_approval_rule_version"),
        )
        op.execute(
            f"CREATE INDEX IF NOT EXISTS ix_approval_rule_versions_rule "
            f"ON {VERSIONS_TABLE} (rule_id, version)"
        )

    # Backfill: every rule that predates this column gets exactly one version row (its
    # current state, as version 1) so the history is never empty for an existing rule --
    # an approval_rule with zero approval_rule_versions rows would otherwise look like a
    # data-integrity gap rather than "created before versioning existed".
    op.execute(
        f"""
        INSERT INTO {VERSIONS_TABLE}
            (id, rule_id, version, name, instruction, active, priority, connector_id,
             dept, file_extensions, max_similarity_score, risk_tiers, can_approve,
             can_reject, changed_by, created_at, updated_at)
        SELECT gen_random_uuid(), r.id, r.version, r.name, r.instruction, r.active,
               r.priority, r.connector_id, r.dept, r.file_extensions,
               r.max_similarity_score, r.risk_tiers, r.can_approve, r.can_reject,
               r.created_by, r.created_at, r.updated_at
        FROM {RULES_TABLE} r
        WHERE NOT EXISTS (
            SELECT 1 FROM {VERSIONS_TABLE} v
            WHERE v.rule_id = r.id AND v.version = r.version
        )
        """
    )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if VERSIONS_TABLE in set(inspector.get_table_names()):
        op.execute(f"DROP TABLE IF EXISTS {VERSIONS_TABLE}")
    if RULES_TABLE in set(inspector.get_table_names()):
        columns = {column["name"] for column in inspector.get_columns(RULES_TABLE)}
        if "version" in columns:
            op.drop_column(RULES_TABLE, "version")
        if "risk_tiers" in columns:
            op.drop_column(RULES_TABLE, "risk_tiers")
