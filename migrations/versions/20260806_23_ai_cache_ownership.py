"""make generated-answer cache entries user-owned

Revision ID: 20260806_23
Revises: 20260806_22
Create Date: 2026-08-06
"""

import os

from alembic import op

revision = "20260806_23"
down_revision = "20260806_22"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE ai_cache ADD COLUMN IF NOT EXISTS owner_user_id UUID REFERENCES users(id) ON DELETE CASCADE"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_ai_cache_owner_user_id ON ai_cache (owner_user_id)"
    )
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    admin = "current_setting('app.global_admin', true) = 'true'"
    owner = f"{admin} OR owner_user_id::text = NULLIF(current_setting('app.user_id', true), '')"
    op.execute("ALTER TABLE ai_cache ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE ai_cache FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY owner_ai_cache ON ai_cache USING ({owner})")


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS owner_ai_cache ON ai_cache")
    op.execute("ALTER TABLE ai_cache DISABLE ROW LEVEL SECURITY")
    op.execute("DROP INDEX IF EXISTS ix_ai_cache_owner_user_id")
    op.execute("ALTER TABLE ai_cache DROP COLUMN IF EXISTS owner_user_id")
