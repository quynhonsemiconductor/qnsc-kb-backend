"""Retire AccessGroup: Department is the only audience

20260816_57 already copied every `access_groups` row into `departments`, every
`user_groups` row into `user_departments`, and every `article_access` row into
`article_departments`, then deliberately kept the legacy tables "for
connector/FK compatibility during rollout". That rollout is finished here.

Two facts make this a constraint swap rather than a data migration:

1. The backfill reused the SOURCE ids (`SELECT id, ... FROM access_groups`), so
   every `external_group_mappings.access_group_id` value is ALREADY a valid
   `departments.id`. Renaming the column and repointing the foreign key
   preserves every existing connector mapping.
2. `Department.kind` is dropped. The distinction it encoded (org departments
   route approvals; access-kind rows were read audiences) is gone by decision:
   one flat department list is both the org unit and the read audience.
   `resolve_active_department` no longer filters on kind, so every active
   department is selectable as an article's primary department.

The bitmask layer goes with it. It was already write-only before this change:
`repositories/chunk.py` removed the `access_group_bitmap & user_bitmask` gate
from retrieval (it imposed a 62-group ceiling and was a second, divergent
permission algorithm), leaving the column written by indexing and read by
nothing. `ai_cache.access_group_bitmap` was likewise superseded by
`authorization_fingerprint`, which is the actual cache key.

Existing `ai_cache` rows go cold: `authorization_fingerprint` now hashes
department membership instead of group bitmask positions. That is a cache miss,
not a correctness problem, and the rows expire on their own; they are deleted
here so the miss happens once rather than being re-checked per request.

Guarded throughout, like 20260831_70 and 20260830_68: the 20260802_00 baseline
builds the schema with `create_all` against the live models, so a freshly built
database already lacks these objects while a forward-migrated one still has
them. Every statement is conditional and re-running is a no-op.

Revision ID: 20260901_71
Revises: 20260831_70
Create Date: 2026-09-01
"""

from alembic import op
from sqlalchemy import inspect


revision = "20260901_71"
down_revision = "20260831_70"
branch_labels = None
depends_on = None

MAPPINGS_TABLE = "external_group_mappings"
DEPARTMENT_FK = "fk_external_group_mappings_department_id"
LEGACY_TABLES = ("article_access", "user_groups", "access_groups")
# RLS policies attached to tables that no longer exist. Dropping the table drops
# its policies, but access_groups policies were also (re)declared against
# articles-scope migrations, so the names are removed explicitly first.
LEGACY_POLICIES = (
    ("article_access", "tenant_article_access"),
    ("user_groups", "tenant_user_groups"),
    ("access_groups", "tenant_access_groups"),
)


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    for table, policy in LEGACY_POLICIES:
        if table in tables:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")

    # --- Connector group mappings: rename in place, then repoint the FK.
    if MAPPINGS_TABLE in tables:
        columns = {column["name"] for column in inspector.get_columns(MAPPINGS_TABLE)}
        if "access_group_id" in columns and "department_id" not in columns:
            # Drop the old FK by whatever name PostgreSQL gave it, before the
            # rename, so the referenced table can be dropped below.
            for constraint in inspector.get_foreign_keys(MAPPINGS_TABLE):
                if constraint["constrained_columns"] == ["access_group_id"]:
                    if constraint.get("name"):
                        op.execute(
                            f"ALTER TABLE {MAPPINGS_TABLE} "
                            f"DROP CONSTRAINT IF EXISTS {constraint['name']}"
                        )
            op.execute(
                f"ALTER TABLE {MAPPINGS_TABLE} "
                "RENAME COLUMN access_group_id TO department_id"
            )
        # A mapping whose id is not a live department can no longer be
        # represented, and must not silently widen access. 20260816_57 copied
        # every group id into departments, so this removes only rows whose
        # group was deleted afterwards.
        op.execute(
            f"DELETE FROM {MAPPINGS_TABLE} WHERE department_id NOT IN "
            "(SELECT id FROM departments)"
        )
        op.execute(
            f"ALTER TABLE {MAPPINGS_TABLE} DROP CONSTRAINT IF EXISTS {DEPARTMENT_FK}"
        )
        op.execute(
            f"ALTER TABLE {MAPPINGS_TABLE} ADD CONSTRAINT {DEPARTMENT_FK} "
            "FOREIGN KEY (department_id) REFERENCES departments (id) ON DELETE CASCADE"
        )

    # --- Legacy audience tables. article_access/user_groups first: both carry a
    # FK to access_groups.
    op.execute("DROP INDEX IF EXISTS ix_article_access_group_id")
    for table in LEGACY_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    # --- Department.kind: one flat list, no org/access split.
    op.execute("ALTER TABLE departments DROP COLUMN IF EXISTS kind")

    # --- The bitmask layer, already unread by every query path.
    op.execute("ALTER TABLE article_chunks DROP COLUMN IF EXISTS access_group_bitmap")
    op.execute("ALTER TABLE ai_cache DROP COLUMN IF EXISTS access_group_bitmap")
    # ix_article_chunks_permission_lookup is deliberately KEPT: it indexes
    # (article_id, department_id, visibility), not the bitmask.
    # authorization_fingerprint now hashes department membership, so every stored
    # answer is keyed under a namespace no live request will ask for again.
    op.execute("DELETE FROM ai_cache")


def downgrade() -> None:
    # The legacy tables are NOT recreated. Their data lives in departments,
    # user_departments and article_departments, which this migration does not
    # touch; rebuilding empty access_groups/user_groups/article_access tables
    # would present an authoritative-looking but empty ACL, which fails OPEN for
    # any code path that treats "no groups" as unrestricted. Restoring the old
    # model is a restore-from-backup operation, not a schema step.
    op.execute(
        "ALTER TABLE departments ADD COLUMN IF NOT EXISTS kind "
        "VARCHAR(10) NOT NULL DEFAULT 'org'"
    )
    op.execute(
        "ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS access_group_bitmap "
        "BIGINT NOT NULL DEFAULT 1"
    )
    op.execute(
        "ALTER TABLE ai_cache ADD COLUMN IF NOT EXISTS access_group_bitmap "
        "BIGINT NOT NULL DEFAULT 0"
    )

    inspector = inspect(op.get_bind())
    if MAPPINGS_TABLE in set(inspector.get_table_names()):
        columns = {column["name"] for column in inspector.get_columns(MAPPINGS_TABLE)}
        if "department_id" in columns and "access_group_id" not in columns:
            op.execute(
                f"ALTER TABLE {MAPPINGS_TABLE} DROP CONSTRAINT IF EXISTS {DEPARTMENT_FK}"
            )
            op.execute(
                f"ALTER TABLE {MAPPINGS_TABLE} "
                "RENAME COLUMN department_id TO access_group_id"
            )
