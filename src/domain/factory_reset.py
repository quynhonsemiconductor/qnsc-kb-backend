"""Return a deployment to a just-released state, keeping only identity.

WHAT THIS IS NOT. ``kb_purge`` already erases one tenant's CONTENT and keeps
connectors, audit history, tag vocabulary, feature flags and the LLM provider
configuration. This is the harder operation: **every** table except identity, for
**every** tenant. Different keep-set, different blast radius, so it is a separate
module rather than a flag on the purge.

KEPT: users, roles, permissions, role_permissions, user_roles, departments,
department_managers, user_departments, external_identities.

Everything else goes — including audit_logs, connectors (and their OAuth grants),
the LLM provider config, feature flags and refresh sessions. An operator will have
to reconnect every source and re-enter the provider API key afterwards. That is
what "like new project just release" means, and it is worth stating plainly
because it cannot be undone.

WHY DELETE AND NOT TRUNCATE. ``TRUNCATE`` needs the TRUNCATE privilege, and
migration ``20260802_05_tenant_rls`` grants the application role only
SELECT/INSERT/UPDATE/DELETE. `TRUNCATE ... CASCADE` would work as the migration
superuser and fail in production with a permission error — the worst possible
place to discover the difference. ``TRUNCATE`` also silently ignores RLS, whereas
DELETE respects it, which is why the caller must hold a global-admin context.

WHY REVERSE METADATA ORDER. ``Base.metadata.sorted_tables`` is dependency order,
parents first. Deleting in reverse removes children before parents, so a
RESTRICT/NO ACTION foreign key cannot abort the run. Most edges here are CASCADE,
but not all, and relying on that would make this fragile against the next
migration.

THE EXHAUSTIVENESS GUARD IS THE POINT. The purge list is DERIVED — every mapped
table that is not explicitly kept. A future migration therefore gets erased by
default rather than silently surviving a "factory reset" and leaving a half-reset
database that looks clean. ``KEPT_TABLES`` is verified against the live metadata,
so deleting a model without updating this module fails loudly instead of quietly
widening the keep-set.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.domain.source_storage import delete_source
from src.models.article import DocumentSource
from src.models.base import Base
from src.models.connectors import DocumentVersion
from src.models.governance import PendingDraft

logger = structlog.get_logger()


#: Identity and authorization. Everything not named here is deleted.
#:
#: Closed under foreign keys BY CONSTRUCTION, and asserted by
#: `assert_keep_set_is_closed`: no kept table may reference a purged one, or the
#: reset would either abort on a RESTRICT edge or null out identity data.
KEPT_TABLES: frozenset[str] = frozenset(
    {
        "users",
        "roles",
        "permissions",
        "role_permissions",
        "user_roles",
        "departments",
        "department_managers",
        "user_departments",
        # The Entra/Google subject links. Dropping these would silently convert
        # every SSO account into one that can no longer sign in, while the user
        # row still looked intact.
        "external_identities",
    }
)

#: Never touched: Alembic's own bookkeeping. It is not in `Base.metadata` at all,
#: so it cannot be reached by the derived purge list; named here for the reader
#: who wonders whether a reset also rewinds migrations. It does not.
UNMANAGED_TABLES: frozenset[str] = frozenset({"alembic_version"})


class FactoryResetError(RuntimeError):
    """The reset cannot run safely. Raised before anything is deleted."""


@dataclass
class ResetCounts:
    """Rows removed per table, plus the object-storage work list."""

    rows: dict[str, int] = field(default_factory=dict)
    #: Collected BEFORE any delete: afterwards the keys are unrecoverable.
    storage_keys: list[str] = field(default_factory=list)
    storage_objects: int = 0
    storage_deleted: int = 0
    storage_failures: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    def as_dict(self) -> dict[str, object]:
        # `storage_keys` is deliberately absent: it is a work list, not a result,
        # and it names private object paths.
        return {
            "tables_cleared": sum(1 for count in self.rows.values() if count),
            "total_rows": self.total_rows,
            "rows_by_table": {
                table: count for table, count in sorted(self.rows.items()) if count
            },
            "storage_objects": self.storage_objects,
            "storage_deleted": self.storage_deleted,
            "storage_failures": self.storage_failures,
        }


def allowed_reset_emails() -> frozenset[str]:
    """The accounts permitted to run a factory reset, lowercased.

    Read from settings on every call rather than captured at import: a deployment
    revokes an operator by editing the environment and restarting, and a cached
    value would keep a removed address live for the life of the process.
    """

    raw = settings.FACTORY_RESET_ALLOWED_EMAILS or ""
    return frozenset(value.strip().lower() for value in raw.split(",") if value.strip())


def is_reset_operator(email: str | None) -> bool:
    """Whether this address is allowlisted for a factory reset.

    Used by the endpoint AND by the user-update path, which must refuse to rename
    an account INTO this allowlist. ``users.email`` is mutable by anyone holding
    global ``user.manage`` (routers/auth.py), so without that second check the
    allowlist is a privilege-escalation route rather than a restriction: rename
    yourself to the allowlisted address and the reset becomes available.
    """

    if not email:
        return False
    return email.strip().lower() in allowed_reset_emails()


def assert_keep_set_is_closed() -> None:
    """Fail loudly if the keep-set no longer matches the schema.

    Two failures matter, and both are silent without this check:

    * A name in ``KEPT_TABLES`` that no longer exists means a model was renamed
      and the reset would now delete the table the operator meant to preserve.
    * A kept table with a foreign key into a purged table means the reset either
      aborts halfway on a RESTRICT edge or nulls out identity data on SET NULL.
    """

    tables = {table.name for table in Base.metadata.sorted_tables}
    unknown = KEPT_TABLES - tables
    if unknown:
        raise FactoryResetError(
            "KEPT_TABLES names tables that are not mapped: " + ", ".join(sorted(unknown))
        )
    purged = tables - KEPT_TABLES
    dangling = [
        f"{table.name}.{fk.parent.name} -> {fk.column.table.name}"
        for table in Base.metadata.sorted_tables
        if table.name in KEPT_TABLES
        for fk in table.foreign_keys
        if fk.column.table.name in purged
    ]
    if dangling:
        raise FactoryResetError(
            "A kept table references a purged table: " + ", ".join(sorted(dangling))
        )


def purge_order() -> list[str]:
    """Tables to clear, children before parents.

    Derived from the metadata rather than listed by hand, so a table added by a
    later migration is erased by default instead of quietly surviving a reset.
    """

    assert_keep_set_is_closed()
    return [
        table.name
        for table in reversed(Base.metadata.sorted_tables)
        if table.name not in KEPT_TABLES and table.name not in UNMANAGED_TABLES
    ]


async def _all_storage_keys(db: AsyncSession) -> list[str]:
    """Every R2 object key any tenant references, read BEFORE the deletes.

    Unscoped on purpose: a factory reset spans every tenant, so the per-tenant
    joins ``kb_purge`` uses would leave another tenant's objects behind while
    their rows were deleted — downloadable content with nothing left pointing at
    it. Reading the keys up front is the only chance to learn them.
    """

    keys: set[str] = set()
    for column in (
        DocumentSource.storage_key,
        PendingDraft.storage_key,
        DocumentVersion.storage_key,
    ):
        rows = await db.execute(select(column).where(column.is_not(None)))
        keys.update(key for key in rows.scalars().all() if key)
    return sorted(keys)


async def factory_reset(db: AsyncSession, *, dry_run: bool = True) -> ResetCounts:
    """Clear every non-identity table, in one transaction.

    One transaction so the 30-second background workers observe the whole reset or
    none of it. The caller commits and then calls `delete_reset_objects`: object
    storage has no rollback, so destroying the R2 objects before the commit is
    exactly what makes a failed commit unrecoverable.

    A dry run counts what a real run would delete and writes nothing. Defaulting
    to a dry run matches ``kb_purge`` and the approval agent: the destructive path
    has to be asked for.

    Requires a global-admin RLS context. Every table here FORCEs row security, so
    without it the deletes are silently FILTERED rather than refused, and the run
    reports success having removed a fraction of the rows.
    """

    counts = ResetCounts()
    tables = purge_order()
    keys = await _all_storage_keys(db)
    counts.storage_keys = [] if dry_run else keys
    counts.storage_objects = len(keys)

    metadata_tables = Base.metadata.tables
    for name in tables:
        table = metadata_tables[name]
        if dry_run:
            counts.rows[name] = (
                await db.scalar(select(func.count()).select_from(table)) or 0
            )
            continue
        result = await db.execute(delete(table))
        counts.rows[name] = result.rowcount or 0

    if dry_run:
        logger.info("Factory reset dry run", **counts.as_dict())
        return counts

    logger.warning("Factory reset executed", **counts.as_dict())
    return counts


async def delete_reset_objects(counts: ResetCounts) -> None:
    """Delete the R2 objects the reset orphaned. Call AFTER the commit.

    Raises nothing. By the time this runs the row deletions are durable, so the
    reset has already succeeded from the caller's point of view; letting a storage
    failure escape would return a 500 that implies nothing happened and invite the
    operator to run it again. Failures are logged and left to the daily orphan
    sweep, which now finds those keys genuinely unreferenced. Same reasoning as
    ``kb_purge.delete_purged_objects``.
    """

    if not counts.storage_keys:
        return

    for key in counts.storage_keys:
        try:
            await asyncio.to_thread(delete_source, key)
            counts.storage_deleted += 1
        except Exception as exc:  # noqa: BLE001 - best effort by contract
            counts.storage_failures.append(key)
            logger.warning(
                "Factory reset could not delete source object",
                key=key,
                error=str(exc),
            )
