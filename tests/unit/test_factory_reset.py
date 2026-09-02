"""Guards for the irreversible factory reset.

Three things are being defended, and every one of them fails silently:

1. The KEEP-SET. If a kept table stops being kept, the reset deletes the accounts
   that were meant to survive and the operator finds out by being locked out.
2. The DERIVED purge list. A table added by a later migration must be erased by
   default, not quietly survive a "factory reset" that claims to be complete.
3. The AUTHORIZATION. `users.email` is mutable, so an email allowlist is only a
   restriction while no API route can write an allowlisted address.

``asyncio.run`` rather than pytest-asyncio: the marker is not available in every
environment this suite runs in.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.config import settings
from src.domain.factory_reset import (
    KEPT_TABLES,
    FactoryResetError,
    ResetCounts,
    allowed_reset_emails,
    assert_keep_set_is_closed,
    delete_reset_objects,
    is_reset_operator,
    purge_order,
)
from src.models.base import Base


# --- keep-set integrity ------------------------------------------------------


def test_the_keep_set_matches_the_live_schema():
    """A renamed or removed model must break this, not the operator's accounts."""
    assert_keep_set_is_closed()


def test_identity_tables_are_never_in_the_purge_list():
    order = purge_order()
    assert not set(order) & KEPT_TABLES
    # The four that matter most: without these the survivors cannot sign in or be
    # authorised, which is the entire point of keeping anything.
    for table in ("users", "roles", "user_roles", "external_identities"):
        assert table not in order


def test_content_and_operational_tables_are_purged():
    """Named explicitly, so silently dropping one from the reset is a test failure.

    `audit_logs` is included deliberately and differs from `kb_purge`, which keeps
    it: a factory reset is a factory reset. That is why the endpoint writes its own
    audit row AFTER the deletes.
    """
    order = set(purge_order())
    for table in (
        "articles",
        "article_chunks",
        "pending_drafts",
        "connectors",
        "external_documents",
        "sync_cursors",
        "audit_logs",
        "refresh_sessions",
        "feature_flags",
        "llm_provider_configs",
        "ingestion_fingerprints",
    ):
        assert table in order, f"{table} must be cleared by a factory reset"


def test_every_mapped_table_is_either_kept_or_purged():
    """The exhaustiveness property. A new migration cannot land in neither bucket.

    This is what makes the purge list DERIVED rather than a hand-maintained list
    that drifts: a table added later is erased by default, and a table that must
    survive has to be named in KEPT_TABLES on purpose.
    """
    mapped = {table.name for table in Base.metadata.sorted_tables}
    assert mapped == set(purge_order()) | KEPT_TABLES


def test_alembic_version_is_never_touched():
    """A reset returns the data to day one, NOT the schema to an earlier revision."""
    assert "alembic_version" not in purge_order()
    assert "alembic_version" not in {t.name for t in Base.metadata.sorted_tables}


def test_children_are_deleted_before_their_parents():
    """Reverse dependency order, so a RESTRICT/NO ACTION FK cannot abort the run.

    Verified against the real foreign keys rather than by asserting a fixed list:
    for every FK between two purged tables, the referencing table must be deleted
    first.
    """
    order = purge_order()
    position = {name: index for index, name in enumerate(order)}
    for table in Base.metadata.sorted_tables:
        if table.name not in position:
            continue
        for fk in table.foreign_keys:
            target = fk.column.table.name
            if target == table.name or target not in position:
                continue
            assert position[table.name] < position[target], (
                f"{table.name} references {target} and must be deleted first"
            )


def test_a_kept_table_referencing_purged_data_is_rejected(monkeypatch):
    """The failure mode the guard exists for.

    A kept table with an FK into a purged table either aborts the reset halfway or
    nulls out identity data. `articles` is purged, and `article_tags` references
    it, so pretending to keep `article_tags` must be refused rather than produce a
    half-reset database.
    """
    monkeypatch.setattr(
        "src.domain.factory_reset.KEPT_TABLES",
        KEPT_TABLES | {"article_tags"},
    )
    with pytest.raises(FactoryResetError, match="references a purged table"):
        assert_keep_set_is_closed()


def test_a_stale_kept_table_name_is_rejected(monkeypatch):
    """A renamed model must fail loudly instead of widening the purge set."""
    monkeypatch.setattr(
        "src.domain.factory_reset.KEPT_TABLES",
        KEPT_TABLES | {"users_old_name"},
    )
    with pytest.raises(FactoryResetError, match="not mapped"):
        assert_keep_set_is_closed()


# --- authorization -----------------------------------------------------------


def test_no_operator_is_authorised_by_default(monkeypatch):
    """Fail closed. An unconfigured deployment must not carry a loaded gun."""
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", "")
    assert allowed_reset_emails() == frozenset()
    assert is_reset_operator("sinhhpt@qnsc.vn") is False


def test_the_configured_operator_is_matched_case_insensitively(monkeypatch):
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", "sinhhpt@qnsc.vn")
    assert is_reset_operator("sinhhpt@qnsc.vn") is True
    # Postgres stores what it was given; the routes lowercase, but a legacy row or a
    # hand-edited value must not sidestep the check on capitalisation alone.
    assert is_reset_operator("SinhHPT@QNSC.VN") is True
    assert is_reset_operator("  sinhhpt@qnsc.vn  ") is True


def test_nobody_else_is_authorised(monkeypatch):
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", "sinhhpt@qnsc.vn")
    assert is_reset_operator("attacker@qnsc.vn") is False
    # Not a prefix or substring match: a similar-looking address is a different one.
    assert is_reset_operator("sinhhpt@qnsc.vn.evil.test") is False
    assert is_reset_operator("xsinhhpt@qnsc.vn") is False
    assert is_reset_operator(None) is False
    assert is_reset_operator("") is False


def test_the_allowlist_is_re_read_rather_than_cached(monkeypatch):
    """Revoking an operator takes effect on restart, not on process lifetime."""
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", "first@qnsc.vn")
    assert is_reset_operator("first@qnsc.vn") is True
    monkeypatch.setattr(settings, "FACTORY_RESET_ALLOWED_EMAILS", "second@qnsc.vn")
    assert is_reset_operator("first@qnsc.vn") is False
    assert is_reset_operator("second@qnsc.vn") is True


def test_several_operators_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        settings, "FACTORY_RESET_ALLOWED_EMAILS", "sinhhpt@qnsc.vn, ops@qnsc.vn"
    )
    assert is_reset_operator("sinhhpt@qnsc.vn") is True
    assert is_reset_operator("ops@qnsc.vn") is True


# --- reporting and storage ---------------------------------------------------


def test_the_report_never_leaks_object_keys():
    """`storage_keys` is a work list of private R2 paths, not a result to publish."""
    counts = ResetCounts(
        rows={"articles": 3, "gaps": 0},
        storage_keys=["s3://bucket/sources/secret.pdf"],
        storage_objects=1,
    )
    payload = counts.as_dict()
    assert "storage_keys" not in payload
    assert "secret.pdf" not in repr(payload)
    # Empty tables are omitted so the report shows what actually happened.
    assert payload["rows_by_table"] == {"articles": 3}
    assert payload["tables_cleared"] == 1
    assert payload["total_rows"] == 3


def test_a_storage_failure_does_not_fail_the_completed_reset(monkeypatch):
    """The rows are already durable when this runs.

    Raising here would return a 500 that implies nothing happened and invite the
    operator to run an irreversible operation a second time.
    """

    def explode(key):
        raise RuntimeError("R2 unavailable")

    monkeypatch.setattr("src.domain.factory_reset.delete_source", explode)
    counts = ResetCounts(storage_keys=["s3://bucket/sources/a.pdf"], storage_objects=1)

    asyncio.run(delete_reset_objects(counts))

    assert counts.storage_failures == ["s3://bucket/sources/a.pdf"]
    assert counts.storage_deleted == 0


def test_a_dry_run_carries_no_storage_work_list():
    """Nothing was deleted, so there is nothing for the caller to clean up after."""
    counts = ResetCounts(storage_keys=[], storage_objects=4)
    asyncio.run(delete_reset_objects(counts))
    assert counts.storage_deleted == 0
    assert counts.storage_failures == []
