"""api_request_metrics(created_at): index the time window every investigate query filters on

Every read of the request log -- /governance/investigate/*, the health dashboard's latency and error panels,
/governance/request-failures with a window -- starts with `created_at >= now() - window`. The table only has a
partial index on 5xx rows and one on request_id, so a window over all requests sequentially scanned the whole log.

Measured on 2.5M rows, the table in insertion (= created_at) order as an append-only log is:
  DB time, 24 h window:           103 ms (parallel seq scan) -> 12 ms (index only scan)
  endpoint, 1 h / 24 h / 168 h:   552 / 654 / 1596 ms        -> 247 / 394 / 1594 ms
The 168 h window covers most of the table, so the planner (rightly) keeps scanning it: it does not get slower,
and does not get faster. 54 MB of index on a 568 MB table; built in about a second.

CONCURRENTLY: the table takes a write per API request while the API runs. A failed CONCURRENTLY build leaves an
INVALID index behind, which is listed, costs writes and is never used, so "already exists" must not mean "done".

Revision ID: 20261102_91
Revises: 20261030_90
Create Date: 2026-11-02
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261102_91"
down_revision = "20261030_90"
branch_labels = None
depends_on = None

TABLE = "api_request_metrics"
INDEX = "ix_api_request_metrics_created_at"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if TABLE not in set(inspector.get_table_names()):
        return
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if INDEX in indexes and _index_is_invalid(bind):
        with op.get_context().autocommit_block():
            op.drop_index(INDEX, table_name=TABLE, postgresql_concurrently=True)
        indexes.discard(INDEX)
    if INDEX not in indexes:
        with op.get_context().autocommit_block():
            op.create_index(INDEX, TABLE, ["created_at"], postgresql_concurrently=True)


def _index_is_invalid(bind) -> bool:
    """True when the index exists in pg_index with indisvalid = false (PostgreSQL only; other dialects have none)."""
    if bind.dialect.name != "postgresql":
        return False
    valid = bind.execute(
        sa.text(
            "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = :name"
        ),
        {"name": INDEX},
    ).scalar()
    return valid is False


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    if INDEX in {index["name"] for index in inspector.get_indexes(TABLE)}:
        with op.get_context().autocommit_block():
            op.drop_index(INDEX, table_name=TABLE, postgresql_concurrently=True)
