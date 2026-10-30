"""the team's standing instructions for MCP clients, versioned

An admin writes one text per company; each AI client keeps a copy in its CLAUDE.md and
compares the version it holds with the server's. `mcp_config_versions` keeps every text.

Guarded like the other additive migrations: the baseline builds tables from the model
metadata, so a fresh database already has them.

Revision ID: 20261030_89
Revises: 20261030_88
Create Date: 2026-10-30

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261030_89"
down_revision = "20261030_88"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(inspect(op.get_bind()).get_table_names())
    if "mcp_configs" not in tables:
        op.create_table(
            "mcp_configs",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("company_domain", sa.String(length=255), nullable=False),
            sa.Column("name", sa.String(length=50), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("content_md", sa.Text(), nullable=False),
            sa.Column(
                "updated_by",
                sa.UUID(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "company_domain", "name", name="uq_mcp_configs_company_name"
            ),
        )
        op.create_index(
            "ix_mcp_configs_company_domain", "mcp_configs", ["company_domain"]
        )
    if "mcp_config_versions" not in tables:
        op.create_table(
            "mcp_config_versions",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column(
                "config_id",
                sa.UUID(),
                sa.ForeignKey("mcp_configs.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("content_md", sa.Text(), nullable=False),
            sa.Column(
                "updated_by",
                sa.UUID(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "config_id", "version", name="uq_mcp_config_versions_version"
            ),
        )
        op.create_index(
            "ix_mcp_config_versions_config_id", "mcp_config_versions", ["config_id"]
        )


def downgrade() -> None:
    tables = set(inspect(op.get_bind()).get_table_names())
    if "mcp_config_versions" in tables:
        op.drop_table("mcp_config_versions")
    if "mcp_configs" in tables:
        op.drop_table("mcp_configs")
