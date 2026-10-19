"""allow tenant-scoped notification delivery under RLS

Revision ID: 20260806_22
Revises: 20260806_21
Create Date: 2026-08-06
"""

import os

from alembic import op

revision = "20260806_22"
down_revision = "20260806_21"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    admin = "current_setting('app.global_admin', true) = 'true'"
    recipient = f"{admin} OR recipient_user_id::text = NULLIF(current_setting('app.user_id', true), '')"
    same_tenant_recipient = f"{admin} OR EXISTS (SELECT 1 FROM users u WHERE u.id = recipient_user_id AND u.company_domain = current_setting('app.company_domain', true))"
    op.execute("DROP POLICY IF EXISTS recipient_notifications ON notification_queue")
    op.execute(
        f"CREATE POLICY recipient_notifications_read ON notification_queue FOR SELECT USING ({recipient})"
    )
    op.execute(
        f"CREATE POLICY recipient_notifications_update ON notification_queue FOR UPDATE USING ({recipient}) WITH CHECK ({recipient})"
    )
    op.execute(
        f"CREATE POLICY recipient_notifications_delete ON notification_queue FOR DELETE USING ({recipient})"
    )
    op.execute(
        f"CREATE POLICY tenant_notification_delivery ON notification_queue FOR INSERT WITH CHECK ({same_tenant_recipient})"
    )


def downgrade() -> None:
    for name in (
        "tenant_notification_delivery",
        "recipient_notifications_delete",
        "recipient_notifications_update",
        "recipient_notifications_read",
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON notification_queue")
    admin = "current_setting('app.global_admin', true) = 'true'"
    recipient = f"{admin} OR recipient_user_id::text = NULLIF(current_setting('app.user_id', true), '')"
    op.execute(
        f"CREATE POLICY recipient_notifications ON notification_queue USING ({recipient})"
    )
