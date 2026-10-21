"""Add workspace LLM provider configuration."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

revision = "20260803_10"
down_revision = "20260802_09"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if inspect(op.get_bind()).has_table("llm_provider_configs"):
        return
    op.create_table(
        "llm_provider_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "config_key", sa.String(50), nullable=False, server_default="workspace"
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("provider", sa.String(30), nullable=False, server_default="openai"),
        sa.Column("model", sa.String(150), nullable=False),
        sa.Column("base_url", sa.String(500), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.UniqueConstraint("config_key", name="uq_llm_provider_configs_key"),
    )


def downgrade() -> None:
    op.drop_table("llm_provider_configs")
