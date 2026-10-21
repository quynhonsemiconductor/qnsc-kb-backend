"""add exact authorization namespace to AI cache"""

from alembic import op

revision = "20260802_07"
down_revision = "20260802_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE ai_cache ADD COLUMN IF NOT EXISTS authorization_fingerprint VARCHAR(64)"
    )
    # Existing entries cannot be safely proven to belong to an authorization
    # context. Expire them instead of allowing a legacy cache hit.
    op.execute(
        "UPDATE ai_cache SET authorization_fingerprint = 'legacy-expired', expires_at = now() WHERE authorization_fingerprint IS NULL"
    )
    op.execute(
        "ALTER TABLE ai_cache ALTER COLUMN authorization_fingerprint SET DEFAULT 'legacy-expired'"
    )
    op.execute(
        "ALTER TABLE ai_cache ALTER COLUMN authorization_fingerprint SET NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_ai_cache_authorization_fingerprint ON ai_cache (authorization_fingerprint)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_ai_cache_authorization_fingerprint")
    op.execute("ALTER TABLE ai_cache DROP COLUMN IF EXISTS authorization_fingerprint")
