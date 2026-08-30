"""record what the provider calls each ACL principal

The ACL screen could only ever show a GUID, because that is all that was stored. An
administrator was asked to decide which internal access group
`0429160a-7320-49fd-8af1-15347fa66627` corresponds to, for fifty-seven of them, before
any document from that source could be approved. That is not a decision anyone can make.

Nullable, and left null for rows already written: the name is display only, no
permission decision reads it, and the next sync of each document repopulates it. Backfill
would mean re-querying the provider for every historical snapshot to improve a label.

Guarded like 20260820_63 -- the baseline builds tables from the model metadata, so a
fresh database already has this column.

Revision ID: 20260829_66
Revises: 20260829_65
Create Date: 2026-08-29

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260829_66"
down_revision = "20260829_65"
branch_labels = None
depends_on = None

TABLE = "external_acl_principals"


def upgrade() -> None:
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns(TABLE)}
    if "principal_name" not in columns:
        op.add_column(TABLE, sa.Column("principal_name", sa.String(length=255), nullable=True))


def downgrade() -> None:
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns(TABLE)}
    if "principal_name" in columns:
        op.drop_column(TABLE, "principal_name")
