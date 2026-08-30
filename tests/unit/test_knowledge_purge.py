"""The purge must delete all content, in the right order, and nothing else.

Every test here defends a property that is invisible in the code and expensive to discover
in production:

* Order. `document_sources.article_id` is ondelete=SET NULL, the only FK into articles that
  does not cascade. Deleting articles first strands those rows with a NULL article_id, and
  because their RLS policy is a join through articles, the orphans become permanently
  invisible to the tenant — along with the R2 objects they point at.
* Sync state. `external_documents` caches revision + content_hash per provider file and the
  sync path skips ingest when both match. Deleting articles while keeping that cache means
  the next reconcile walk finds every file unchanged and re-imports nothing, so the corpus
  ends up permanently empty rather than freshly empty.
* Scope. Users, roles, RBAC, departments, access groups and the audit log are not content.
  A purge that removes them is a factory reset, which is not what this is.
* Connectors survive. They hold the OAuth grant. Deleting them would force an operator to
  reconnect SharePoint by hand after every test reset.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from src.models.rbac import Role
from src.models.user import User


class _Result:
    """Enough of a SQLAlchemy result for both the SELECT and DELETE paths."""

    def __init__(self, rows=None, rowcount=0):
        self.rows = list(rows or [])
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _PurgeDB:
    """Records every statement in issue order and compiles it on demand."""

    def __init__(self, *, article_ids=None, user_ids=None, rowcount=1):
        self.statements: list[str] = []
        self.added: list[object] = []
        self.committed = False
        self._article_ids = article_ids if article_ids is not None else [uuid.uuid4()]
        self._user_ids = user_ids if user_ids is not None else [uuid.uuid4()]
        self._rowcount = rowcount

    async def execute(self, statement, params=None, *args, **kwargs):
        try:
            rendered = str(statement.compile(dialect=postgresql.dialect()))
        except Exception:
            rendered = str(statement)
        self.statements.append(rendered)
        lowered = rendered.lower()

        # The purge reads ids and storage keys before deleting; give it plausible rows.
        if lowered.startswith("select"):
            if "users.id" in lowered and "from users" in lowered:
                return _Result(self._user_ids)
            if "articles.id" in lowered and "from articles" in lowered:
                return _Result(self._article_ids)
            if "storage_key" in lowered:
                return _Result([])
            if "outbox_events" in lowered:
                return _Result([])
            return _Result([])
        return _Result(rowcount=self._rowcount)

    async def commit(self):
        self.committed = True

    async def refresh(self, _item):
        return None

    def add(self, item):
        self.added.append(item)


def _global_admin(*, company="acme.test") -> User:
    user = User(id=uuid.uuid4(), role="Admin", company_domain=company)
    user.roles = [Role(name="Admin", company_domain=None, active=True)]
    return user


def _index_of(statements: list[str], table: str, verb: str = "delete from") -> int:
    """Index of the statement whose TARGET is `table`.

    Anchored on the target rather than a substring match: the document_sources delete
    carries an `IN (SELECT articles.id ...)` subquery, so a plain `"articles" in sql`
    matches it too and an ordering assertion silently compares a statement with itself.
    """
    prefix = f"{verb} {table}"
    for index, sql in enumerate(statements):
        lowered = " ".join(sql.lower().split())
        if lowered.startswith(prefix):
            return index
    raise AssertionError(f"no '{verb} {table}' statement was issued")


def _run_purge(db, *, dry_run: bool):
    from src.domain.kb_purge import purge_knowledge_base

    return asyncio.run(purge_knowledge_base(db, "acme.test", dry_run=dry_run))


# ---------------------------------------------------------------- ordering and coverage


def test_document_sources_are_deleted_before_articles():
    """The SET NULL trap: reversing this orphans the rows and leaks their R2 objects."""
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    assert _index_of(db.statements, "document_sources") < _index_of(
        db.statements, "articles"
    )


def test_the_provider_skip_cache_is_deleted_so_the_next_sync_reimports():
    """Keeping external_documents makes a full reconcile walk find nothing changed."""
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    _index_of(db.statements, "external_documents")


def test_sync_cursors_are_reset_rather_than_left_pointing_at_deleted_state():
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    updates = [
        sql
        for sql in db.statements
        if sql.lower().startswith("update") and "sync_cursors" in sql.lower()
    ]
    assert updates, "sync cursors were never reset"
    assert "full_sync_required" in updates[0]
    assert "cursor_value" in updates[0]


def test_queued_connector_work_is_drained_so_it_cannot_reimport_after_the_purge():
    """A SyncRequest queued before the purge would otherwise execute right after it."""
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    _index_of(db.statements, "sync_requests")
    _index_of(db.statements, "connector_notifications")


def test_ingestion_fingerprints_are_purged_or_reupload_is_refused_forever():
    """An approved fingerprint outliving its article refuses the same file with a 409."""
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    _index_of(db.statements, "ingestion_fingerprints")


def test_every_content_table_the_cascades_do_not_reach_is_deleted_explicitly():
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    for table in (
        "articles",
        "document_sources",
        "pending_drafts",
        "external_documents",
        "ingestion_fingerprints",
        "gaps",
        "conflict_records",
        "index_reprocess_jobs",
        "ai_cache",
        "ai_conversations",
        "ai_usage_logs",
    ):
        _index_of(db.statements, table)


# ------------------------------------------------------------------------ what survives


def test_connectors_are_never_deleted_because_they_hold_the_oauth_grant():
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    deletes = [sql for sql in db.statements if sql.lower().startswith("delete from")]
    assert not [
        sql for sql in deletes if sql.lower().startswith("delete from connectors")
    ], "deleting connectors forces the operator to reconnect SharePoint by hand"


@pytest.mark.parametrize(
    "table",
    [
        "users",
        "roles",
        "permissions",
        "role_permissions",
        "departments",
        "access_groups",
        "audit_logs",
        "tag_catalog",
        "feature_flags",
    ],
)
def test_identity_and_config_tables_are_never_deleted(table):
    """A purge is content-only. audit_logs especially: it records that this happened."""
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    for sql in db.statements:
        lowered = sql.lower()
        if lowered.startswith("delete from"):
            assert not lowered.startswith(f"delete from {table} "), sql
            assert lowered.split("delete from ")[1].split()[0] != table, sql


def test_every_delete_is_scoped_to_one_tenant():
    """No statement may reach another tenant's rows, directly or through a join."""
    db = _PurgeDB()
    _run_purge(db, dry_run=False)

    for sql in db.statements:
        if not sql.lower().startswith(("delete from", "update")):
            continue
        lowered = sql.lower()
        assert (
            "company_domain" in lowered
            or "article_id" in lowered
            or "connector_id" in lowered
            or "user_id" in lowered
            or "owner_user_id" in lowered
            or "outbox_events" in lowered
        ), f"unscoped destructive statement: {sql}"


# ------------------------------------------------------------------------------ dry run


def test_a_dry_run_writes_nothing():
    db = _PurgeDB()
    counts = _run_purge(db, dry_run=True)

    assert not [
        sql
        for sql in db.statements
        if sql.lower().startswith(("delete", "update", "insert"))
    ]
    assert not db.committed
    assert counts.as_dict()["articles"] == len(db._article_ids)


def test_a_dry_run_still_reports_what_it_would_delete():
    db = _PurgeDB(article_ids=[uuid.uuid4(), uuid.uuid4()])
    counts = _run_purge(db, dry_run=True)

    assert counts.articles == 2


# ---------------------------------------------------------------------------- endpoint


def _call_endpoint(db, payload, user):
    from src.api.routers.governance import purge_knowledge

    return asyncio.run(purge_knowledge(payload=payload, current_user=user, db=db))


def _payload(**kwargs):
    from src.api.routers.governance import KnowledgePurgeRequest

    return KnowledgePurgeRequest(**kwargs)


def test_the_endpoint_defaults_to_a_dry_run():
    """A forgotten field must not erase a corpus."""
    db = _PurgeDB()
    result = _call_endpoint(db, _payload(), _global_admin())

    assert result["dry_run"] is True
    assert not db.committed


def test_a_real_run_without_the_matching_confirmation_is_refused():
    db = _PurgeDB()
    with pytest.raises(HTTPException) as error:
        _call_endpoint(db, _payload(dry_run=False), _global_admin())

    assert error.value.status_code == 409
    assert error.value.detail["code"] == "confirmation_mismatch"
    assert not db.committed
    assert not [sql for sql in db.statements if sql.lower().startswith("delete")]


def test_confirming_the_wrong_tenant_is_refused():
    """Typing the domain proves the operator knows WHICH corpus they are erasing."""
    db = _PurgeDB()
    with pytest.raises(HTTPException) as error:
        _call_endpoint(
            db, _payload(dry_run=False, confirm="other.test"), _global_admin()
        )

    assert error.value.status_code == 409
    assert error.value.detail["expected"] == "acme.test"


def test_a_confirmed_real_run_purges_and_audits_in_one_transaction():
    from src.models.governance import AuditLog

    db = _PurgeDB()
    result = _call_endpoint(
        db, _payload(dry_run=False, confirm="acme.test"), _global_admin()
    )

    assert result["dry_run"] is False
    assert db.committed
    audits = [item for item in db.added if isinstance(item, AuditLog)]
    assert len(audits) == 1
    assert audits[0].action == "knowledge_purge"
    assert audits[0].target_id == "acme.test"
    assert audits[0].detail_json["articles"] == 1


def test_the_audit_action_fits_the_column():
    """AuditLog.action is String(50); an overlong action fails at INSERT time."""
    from src.models.governance import AuditLog

    db = _PurgeDB()
    _call_endpoint(db, _payload(dry_run=False, confirm="acme.test"), _global_admin())
    action = [item for item in db.added if isinstance(item, AuditLog)][0].action

    assert len(action) <= 50


def test_an_account_without_a_company_domain_cannot_purge():
    db = _PurgeDB()
    user = _global_admin()
    user.company_domain = None

    with pytest.raises(HTTPException) as error:
        _call_endpoint(db, _payload(dry_run=False, confirm="acme.test"), user)

    assert error.value.status_code == 409
    assert error.value.detail["code"] == "tenant_unresolved"


def test_the_endpoint_requires_global_role_manage():
    """The gate is the dependency, so assert the dependency rather than a fake request."""
    import inspect

    from src.api.routers.governance import purge_knowledge

    dependency = inspect.signature(purge_knowledge).parameters["current_user"].default
    source = inspect.getsource(purge_knowledge)

    assert dependency is not inspect.Parameter.empty
    assert 'require_permission("role.manage", scope="global")' in source
