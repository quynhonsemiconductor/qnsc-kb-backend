"""Let any provider ACL principal be mapped to a department

`external_group_mappings` could only ever express "this provider GROUP means this
department". The sync path collected ids only from `group`/`siteGroup`
(`cloud_sync._save_permissions`), so a row keyed on any other principal id was
never read back, and `unmapped_principal_ids` was computed without consulting the
mapping table at all. A SharePoint `user`, `link`, `domain`, `application`,
`device` or `unknown` principal therefore had NO route to resolution: it sat in
the blocking set forever and `governance.approve_draft` refused the draft with
`external_acl_mapping_required`. That is what an operator sees as a permanently
unresolved count on the ACL panel.

The table already stored an opaque 512-char id keyed only on
`(connector_id, external_group_id)`, so it physically accepted any principal id.
What it could not do was say WHICH KIND of principal a row meant — a provider
group and a provider user sharing an id collapsed to one row with unrecoverable
intent. `principal_type` fixes that and joins the unique key.

Renames, because the columns no longer hold group-specific values:
    external_group_id   -> principal_id
    external_group_name -> principal_name

The TABLE keeps its name deliberately. Its RLS policy `tenant_external_group_mappings`
is created by name in 20260807_28 and re-declared in `src/api/main.py`; renaming the
table would mean reissuing that policy for no behavioural gain.

Existing rows are all groups by construction, so `principal_type` backfills to
`'group'` and every current mapping keeps its exact meaning.

Fail-closed is unchanged: `unmapped_*` still forces `source_restricts` in
`_provider_acl_intersection`, and `internal_acl_snapshot` is still written once,
so a provider ACL can still only ever NARROW the internal policy. Mapping a
principal is now an explicit, audited widening decision
(`AuditLog(action="connector_permission_mapping")`) that requires the principal to
have actually been observed on that connector.

Revision ID: 20260901_72
Revises: 20260901_71
Create Date: 2026-09-01
"""

from alembic import op
from sqlalchemy import inspect


revision = "20260901_72"
down_revision = "20260901_71"
branch_labels = None
depends_on = None

TABLE = "external_group_mappings"
UNIQUE = "uq_external_group_mapping"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    if "principal_type" not in columns:
        op.execute(
            f"ALTER TABLE {TABLE} ADD COLUMN principal_type VARCHAR(30) "
            "NOT NULL DEFAULT 'group'"
        )
    if "external_group_id" in columns and "principal_id" not in columns:
        op.execute(f"ALTER TABLE {TABLE} RENAME COLUMN external_group_id TO principal_id")
    if "external_group_name" in columns and "principal_name" not in columns:
        op.execute(
            f"ALTER TABLE {TABLE} RENAME COLUMN external_group_name TO principal_name"
        )

    # The old key was (connector_id, external_group_id). Two principals of different
    # kinds may legitimately share an id, so the type joins the key.
    op.execute(f"ALTER TABLE {TABLE} DROP CONSTRAINT IF EXISTS {UNIQUE}")
    op.execute(
        f"ALTER TABLE {TABLE} ADD CONSTRAINT {UNIQUE} "
        "UNIQUE (connector_id, principal_type, principal_id)"
    )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    # Rows for principals that are not groups cannot be represented by the old
    # group-only shape. Dropping them is the honest reversal: leaving them behind
    # under a column named external_group_id would assert they are groups, and the
    # pre-change sync path would read them as such.
    if "principal_type" in columns:
        op.execute(f"DELETE FROM {TABLE} WHERE principal_type <> 'group'")

    if "principal_id" in columns and "external_group_id" not in columns:
        op.execute(f"ALTER TABLE {TABLE} RENAME COLUMN principal_id TO external_group_id")
    if "principal_name" in columns and "external_group_name" not in columns:
        op.execute(
            f"ALTER TABLE {TABLE} RENAME COLUMN principal_name TO external_group_name"
        )
    op.execute(f"ALTER TABLE {TABLE} DROP COLUMN IF EXISTS principal_type")

    op.execute(f"ALTER TABLE {TABLE} DROP CONSTRAINT IF EXISTS {UNIQUE}")
    op.execute(
        f"ALTER TABLE {TABLE} ADD CONSTRAINT {UNIQUE} "
        "UNIQUE (connector_id, external_group_id)"
    )
