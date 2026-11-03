"""Timestamp columns must be filled Python-side, not read back with RETURNING.

When a column's only default is a server default, the ORM has to fetch the generated
value, so every insert becomes `INSERT ... RETURNING created_at, updated_at`. Under
row-level security a RETURNING clause applies the table's SELECT policy to the new row
in addition to its INSERT policy. Several tables here are deliberately writable by more
people than can read them — notification_queue may be written for any user in the
company (tenant_notification_delivery) but read only by its recipient
(recipient_notifications_read) — so RETURNING turns a legitimate write into "new row
violates row-level security policy for table ...", an error that names the table but
gives no hint that the RETURNING is what failed.

That shipped: approving a draft (a reviewer notifying the author) returned 500, while
assigning an approver worked, because the latter runs as a global admin who can read
every notification. A Python-side default sends the value in the INSERT, so no RETURNING
is emitted and write permission alone is enough.

The check is on the mixin rather than on compiled SQL because `Table.insert()` compiles
without RETURNING at Core level — the clause is added by the ORM's persistence layer for
server-generated columns, which a unit test cannot observe without a live connection.
"""
from __future__ import annotations

import pytest

from src.models.base import TimestampMixin


@pytest.mark.parametrize("column_name", ["created_at", "updated_at"])
def test_timestamp_columns_have_python_side_defaults(column_name: str) -> None:
    column = TimestampMixin.__dict__[column_name].column
    assert column.default is not None, (
        f"TimestampMixin.{column_name} has no Python-side default, so the ORM will read "
        "it back with RETURNING — which applies the SELECT policy to the inserted row "
        "and breaks every write-for-another-user path under RLS"
    )


def test_updated_at_refreshes_python_side() -> None:
    """`onupdate=func.now()` would put RETURNING back on UPDATE statements."""
    column = TimestampMixin.__dict__["updated_at"].column
    assert column.onupdate is not None and not column.onupdate.is_server_default, (
        "updated_at must refresh Python-side; a server-side onupdate makes every UPDATE "
        "emit RETURNING and hit the same RLS problem as INSERT"
    )
