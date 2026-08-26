from src.domain.cloud_sync import (
    _acl_hash,
    _apply_mapped_groups,
    _needs_content_ingest,
    _record_permission_change_audits,
    _save_permissions,
    _sharepoint_acl_intersection,
    _upsert_document,
)
import asyncio
import uuid

import httpx
import pytest
from fastapi import HTTPException

from src.domain.cloud_sync import (
    _acl_hash,
    _apply_mapped_groups,
    _cleanup_unreferenced_source_keys,
    _handle_deleted_document,
    _needs_content_ingest,
    _save_permissions,
    _sharepoint_acl_intersection,
)
from src.domain.connector_adapters import (
    ConnectorAdapter,
    ConnectorProviderError,
    GoogleDriveAdapter,
    NormalizedChange,
)
from src.api.routers.connectors import _can_complete_oauth, _safe_connector_config
from src.models.ops import Connector
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User


def test_acl_hash_is_order_independent():
    left = [
        {"principal_type": "group", "principal_id": "g2", "role": "reader"},
        {"principal_type": "user", "principal_id": "u1", "role": "reader"},
    ]
    right = list(reversed(left))
    assert _acl_hash(left) == _acl_hash(right)


def test_repeated_provider_identity_reuses_external_document():
    """A second delta run must update one row, never create a duplicate."""
    from src.models.connectors import ExternalDocument, SourceScope

    connector_id = uuid.uuid4()
    scope_id = uuid.uuid4()
    connector = Connector(
        id=connector_id, company_domain="acme.test", system="sharepoint"
    )
    scope = SourceScope(
        id=scope_id,
        connector_id=connector_id,
        external_scope_id="site-1",
        scope_type="site",
        display_name="Site",
        selected=True,
    )
    existing = ExternalDocument(
        id=uuid.uuid4(),
        connector_id=connector_id,
        scope_id=scope_id,
        corpus_id="drive-1",
        external_id="file-1",
        name="old-name.md",
    )

    class Result:
        def scalar_one_or_none(self):
            return existing

    class Db:
        def __init__(self):
            self.statements = []
            self.added = []

        async def execute(self, statement):
            self.statements.append(str(statement))
            return Result()

        def add(self, item):
            self.added.append(item)

        async def flush(self):
            raise AssertionError(
                "an existing provider identity must not flush a new row"
            )

    change = NormalizedChange(
        external_id="file-1",
        corpus_id="drive-1",
        name="renamed.md",
        state="active",
        content_changed=False,
        permissions_changed=False,
        moved=False,
        revision="etag-2",
        mime_type="text/markdown",
        parent_external_id="folder-1",
        web_url="https://provider.example/file-1",
        metadata={"size": 10},
    )
    db = Db()

    first = asyncio.run(_upsert_document(db, connector, scope, change))
    second = asyncio.run(_upsert_document(db, connector, scope, change))

    assert first is existing
    assert second is existing
    assert existing.name == "renamed.md"
    assert existing.revision == "etag-2"
    assert existing.metadata_json == {"size": 10}
    assert db.added == []
    assert len(db.statements) == 2
    assert all(
        "external_documents.connector_id" in statement for statement in db.statements
    )


def test_permission_change_audits_target_each_reconciled_article():
    from src.models.governance import AuditLog

    article_id = uuid.uuid4()
    actor_id = uuid.uuid4()

    class Db:
        def __init__(self):
            self.added = []

        def add(self, item):
            self.added.append(item)

    db = Db()
    _record_permission_change_audits(db, [article_id, article_id], actor_id)

    assert len(db.added) == 1
    audit = db.added[0]
    assert isinstance(audit, AuditLog)
    assert audit.user_id == actor_id
    assert audit.action == "permission_change"
    assert audit.target_type == "article"
    assert audit.target_id == str(article_id)
    assert audit.outcome == "success"


def test_acl_mapping_change_reconciles_even_when_provider_acl_hash_is_unchanged():
    from src.models.connectors import ExternalDocument, ExternalGroupMapping

    connector_id = uuid.uuid4()
    access_group_id = uuid.uuid4()
    connector = Connector(id=connector_id, company_domain="acme.test")
    permissions = [
        {"principal_type": "group", "principal_id": "provider-group", "role": "read"},
        {"principal_type": "user", "principal_id": "provider-user", "role": "read"},
    ]
    document = ExternalDocument(
        id=uuid.uuid4(),
        connector_id=connector_id,
        corpus_id="drive",
        external_id="file-1",
        name="policy.md",
        acl_hash=_acl_hash(permissions),
        metadata_json={
            "sharepoint_acl_present": True,
            "mapped_access_group_ids": [],
            "unmapped_group_ids": ["provider-group"],
            "mapped_source_user_ids": [],
            "unmapped_source_user_ids": [],
        },
    )
    mapping = ExternalGroupMapping(
        connector_id=connector_id,
        external_group_id="provider-group",
        access_group_id=access_group_id,
        active=True,
    )

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def scalars(self):
            return self

        def all(self):
            return self.rows

    class Db:
        def __init__(self):
            self.statements = []

        async def execute(self, statement):
            self.statements.append(str(statement))
            if "external_group_mappings" in str(statement):
                assert "access_groups.company_domain" in str(statement)
                return Result([mapping])
            else:
                assert "external_identities" in str(statement)
                assert "users.company_domain" in str(statement)
                assert "users.active" in str(statement)
                return Result([])

    db = Db()
    changed = asyncio.run(_save_permissions(db, connector, document, permissions))

    assert changed is True
    assert document.metadata_json["mapped_access_group_ids"] == [str(access_group_id)]
    assert document.metadata_json["unmapped_group_ids"] == []


def test_sharepoint_acl_intersection_never_broadens_internal_policy():
    result = _sharepoint_acl_intersection(
        internal_visibility="department",
        internal_group_ids={"g-internal"},
        internal_user_ids={"u-internal", "u-other"},
        source_group_ids={"g-internal", "g-provider-only"},
        source_user_ids={"u-provider"},
        source_group_member_ids={"u-internal"},
        unmapped_principals=False,
        acl_present=True,
    )
    assert result["group_ids"] == {"g-internal"}
    assert result["direct_user_ids"] == set()
    assert result["internal_users_allowed_by_source"] == {"u-internal"}
    assert result["visibility"] == "department"


def test_empty_or_unmapped_sharepoint_acl_fails_closed():
    result = _sharepoint_acl_intersection(
        internal_visibility="public",
        internal_group_ids=set(),
        internal_user_ids={"u-internal"},
        source_group_ids=set(),
        source_user_ids=set(),
        source_group_member_ids=set(),
        unmapped_principals=True,
        acl_present=True,
    )
    assert result["group_ids"] == set()
    assert result["direct_user_ids"] == set()
    assert result["internal_users_allowed_by_source"] == set()
    assert result["visibility"] == "users"


def test_unsupported_provider_principal_is_retained_as_unmapped():
    from src.domain.connector_adapters import SharePointAdapter

    adapter = SharePointAdapter(Connector(id=uuid.uuid4(), company_domain="acme.test"))
    change = NormalizedChange(
        external_id="file-1",
        corpus_id="drive-1",
        name="policy.md",
        state="active",
        content_changed=True,
        permissions_changed=False,
        moved=False,
        revision="etag-1",
        mime_type="text/markdown",
        parent_external_id=None,
        web_url=None,
        metadata={},
    )

    async def fake_request(*_args, **_kwargs):
        return {
            "value": [
                {
                    "id": "permission-link",
                    "roles": ["read"],
                    "link": {"scope": "anonymous"},
                }
            ]
        }

    adapter._request = fake_request
    assert asyncio.run(adapter.permissions(change)) == [
        {
            "principal_type": "unknown",
            "principal_id": "permission-link",
            "role": "read",
        }
    ]


def test_unsupported_provider_principal_is_persisted_as_unmapped():
    from src.models.connectors import ExternalDocument

    connector = Connector(id=uuid.uuid4(), company_domain="acme.test")
    document = ExternalDocument(
        id=uuid.uuid4(),
        connector_id=connector.id,
        corpus_id="drive-1",
        external_id="file-1",
        name="policy.md",
    )
    permissions = [
        {"principal_type": "domain", "principal_id": "example.test", "role": "read"}
    ]

    class Result:
        def scalar_one_or_none(self):
            return None

    class Db:
        def __init__(self):
            self.added = []

        async def execute(self, _statement):
            return Result()

        def add(self, item):
            self.added.append(item)

        async def flush(self):
            return None

    asyncio.run(_save_permissions(Db(), connector, document, permissions))

    assert document.metadata_json["unmapped_principal_ids"] == ["domain:example.test"]


def test_sharepoint_permission_tightening_is_applied_on_resync():
    from src.models.article import Article, ArticleUserPermission
    from src.models.connectors import ExternalDocument
    from src.models.user import AccessGroup

    article_id = uuid.uuid4()
    internal_user_id = uuid.uuid4()
    group = AccessGroup(
        id=uuid.uuid4(), company_domain="acme.test", name="Security", bitmask_position=3
    )
    internal_override = ArticleUserPermission(
        article_id=article_id, user_id=internal_user_id, effect="allow"
    )
    article = Article(
        id=article_id,
        company_domain="acme.test",
        dept="Engineering",
        domain="Security",
        type="POLICY",
        sensitivity="internal",
        visibility="department",
        status="published",
        lifecycle_status="active",
        access_groups=[group],
        user_permissions=[internal_override],
    )
    document = ExternalDocument(
        id=uuid.uuid4(),
        connector_id=uuid.uuid4(),
        article_id=article_id,
        corpus_id="drive",
        external_id="file-1",
        name="policy.md",
        metadata_json={
            "sharepoint_acl_present": True,
            "mapped_access_group_ids": [str(group.id)],
            "mapped_source_user_ids": [],
            "unmapped_group_ids": [],
            "unmapped_source_user_ids": [],
        },
    )

    class FakeResult:
        def __init__(self, *, one=None, rows=None):
            self.one = one
            self.rows = rows or []

        def scalar_one_or_none(self):
            return self.one

        def scalars(self):
            return self

        def all(self):
            return self.rows

    class FakeDb:
        def __init__(self):
            self.calls = 0
            self.added = []

        async def execute(self, _statement):
            self.calls += 1
            if self.calls == 1:
                return FakeResult(one=article)
            if self.calls == 2 and document.metadata_json.get(
                "mapped_access_group_ids"
            ):
                sql = str(_statement)
                assert "users.company_domain" in sql
                assert "users.active" in sql
                return FakeResult(rows=[internal_user_id])
            if self.calls == 3 and document.metadata_json.get(
                "mapped_access_group_ids"
            ):
                return FakeResult(rows=[group])
            return FakeResult()

        def add(self, item):
            self.added.append(item)

        async def flush(self):
            return None

    db = FakeDb()
    asyncio.run(_apply_mapped_groups(db, object(), document))
    assert article.visibility == "department"
    assert article.access_groups == [group]

    document.metadata_json = {
        **document.metadata_json,
        "mapped_access_group_ids": [],
        "mapped_source_user_ids": [],
        "unmapped_group_ids": [],
        "unmapped_source_user_ids": ["provider-user-no-longer-mapped"],
    }
    db.calls = 0
    db.added.clear()
    asyncio.run(_apply_mapped_groups(db, object(), document))

    assert article.visibility == "users"
    assert any(
        isinstance(item, ArticleUserPermission)
        and item.user_id == internal_user_id
        and item.effect == "deny"
        and item.source == "sharepoint"
        for item in db.added
    )


def test_provider_deletion_rejects_pending_draft_inactivates_article_and_audits():
    from src.models.article import Article
    from src.models.connectors import ExternalDocument
    from src.models.governance import AuditLog, DraftTransition, PendingDraft

    actor_id = uuid.uuid4()
    article_id = uuid.uuid4()
    document_id = uuid.uuid4()
    storage_key = "s3://private-kb/sources/acme.test/transient.pdf"
    draft = PendingDraft(
        id=uuid.uuid4(),
        title="Provider policy",
        company_domain="acme.test",
        source_ref="sharepoint://drive/file-1",
        source_hash="a" * 64,
        storage_key=storage_key,
        status="pending",
        external_document_id=document_id,
    )
    article = Article(
        id=article_id,
        title="Provider policy",
        body_md="body",
        dept="Security",
        domain="Policy",
        company_domain="acme.test",
        type="POLICY",
        status="published",
        lifecycle_status="active",
    )
    document = ExternalDocument(
        id=document_id,
        connector_id=uuid.uuid4(),
        article_id=article_id,
        corpus_id="drive",
        external_id="file-1",
        name="policy.pdf",
        state="active",
    )

    class Result:
        def scalars(self):
            return self

        def all(self):
            return [draft]

    class Db:
        def __init__(self):
            self.added = []

        async def execute(self, _statement):
            return Result()

        async def get(self, model, _object_id):
            assert model is Article
            return article

        def add(self, item):
            self.added.append(item)

        async def flush(self):
            return None

    db = Db()
    cleanup_keys, deleted_article_id = asyncio.run(
        _handle_deleted_document(db, document, actor_id)
    )

    assert document.state == "deleted"
    assert draft.status == "rejected"
    assert draft.storage_key is None
    assert cleanup_keys == [storage_key]
    assert deleted_article_id == article_id
    assert article.lifecycle_status == "inactive"
    assert any(
        isinstance(item, DraftTransition)
        and item.from_status == "pending"
        and item.to_status == "rejected"
        and item.actor_id == actor_id
        for item in db.added
    )
    assert {
        (item.action, item.target_type, item.target_id, item.outcome)
        for item in db.added
        if isinstance(item, AuditLog)
    } == {
        ("delete", "article", str(article_id), "success"),
        ("delete", "external_document", str(document_id), "success"),
    }


def test_unreferenced_transient_source_is_deleted_but_referenced_key_is_retained(
    monkeypatch,
):
    deleted = []

    def fake_delete(storage_key):
        deleted.append(storage_key)

    monkeypatch.setattr("src.domain.cloud_sync.delete_source", fake_delete)

    class Db:
        def __init__(self, responses):
            self.responses = list(responses)

        async def scalar(self, _statement):
            return self.responses.pop(0)

    asyncio.run(
        _cleanup_unreferenced_source_keys(
            Db([None, None, None]),
            ["s3://private-kb/sources/acme.test/transient.pdf"] * 2,
        )
    )
    asyncio.run(
        _cleanup_unreferenced_source_keys(
            Db([uuid.uuid4()]),
            ["s3://private-kb/sources/acme.test/retained.pdf"],
        )
    )

    assert deleted == ["s3://private-kb/sources/acme.test/transient.pdf"]


def test_normalized_change_preserves_move_and_content_flags():
    change = NormalizedChange(
        external_id="file-1",
        corpus_id="drive-1",
        name="policy.pdf",
        state="active",
        content_changed=True,
        permissions_changed=False,
        moved=True,
        revision="etag-2",
        mime_type="application/pdf",
        parent_external_id="folder-2",
        web_url="https://provider.example/file-1",
        metadata={"size": 100},
    )
    assert change.external_id == "file-1"
    assert change.moved is True
    assert change.content_changed is True


def test_new_file_is_ingested_even_when_provider_content_flag_is_unset():
    assert _needs_content_ingest(
        is_file=True,
        previous_exists=False,
        previous_revision=None,
        current_revision="etag-1",
        has_content_hash=False,
        pending_draft_needs_candidates=False,
    )
    assert _needs_content_ingest(
        is_file=True,
        previous_exists=True,
        previous_revision="etag-1",
        current_revision="etag-1",
        has_content_hash=True,
        pending_draft_needs_candidates=True,
    )
    assert not _needs_content_ingest(
        is_file=True,
        previous_exists=True,
        previous_revision="etag-1",
        current_revision="etag-1",
        has_content_hash=True,
        pending_draft_needs_candidates=False,
    )


class _Adapter(ConnectorAdapter):
    allowed_api_hosts = frozenset({"api.example.com"})

    @property
    def access_token(self):
        return "provider-token"


class _MockAdapter(_Adapter):
    def __init__(self, handler):
        super().__init__(connector=None)  # type: ignore[arg-type]
        self.handler = handler

    def _http_client(self):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self.handler),
            follow_redirects=False,
            trust_env=False,
        )


def test_connector_rejects_provider_controlled_urls_outside_the_official_api():
    adapter = _Adapter(connector=None)  # type: ignore[arg-type]
    with pytest.raises(ConnectorProviderError, match="untrusted API URL"):
        adapter._validate_provider_url("https://metadata.internal/latest")


@pytest.mark.parametrize(
    "url",
    [
        "http://downloads.example.com/file",
        "https://127.0.0.1/file",
        "https://user:pass@downloads.example.com/file",
    ],
)
def test_connector_rejects_unsafe_download_redirects(url):
    with pytest.raises(ConnectorProviderError, match="unsafe download redirect"):
        _Adapter._validate_redirect_url(url)


def test_connector_config_rejects_nested_credentials_and_oversized_data():
    with pytest.raises(HTTPException, match="credentials"):
        _safe_connector_config(
            {"sync_mode": "daily", "nested": {"access_token": "do-not-store"}}
        )
    with pytest.raises(HTTPException, match="too large"):
        _safe_connector_config({"note": "x" * 20_000})
    assert _safe_connector_config({"sync_mode": "daily", "folders": ["abc"]}) == {
        "sync_mode": "daily",
        "folders": ["abc"],
    }


def test_oauth_callback_rechecks_the_authorizing_manager():
    connector = Connector(
        name="Drive", system="google_drive", company_domain="acme.test"
    )
    initiator = User(
        email="manager@acme.test",
        name="Manager",
        password_hash="hash",
        company_domain="acme.test",
        role="Staff",
        active=True,
    )
    permission = Permission(key="connector.manage", name="Connector management")
    role = Role(name="Connector manager", company_domain="acme.test", active=True)
    role.permissions.append(RolePermission(permission=permission, scope="company"))
    initiator.roles.append(role)
    assert _can_complete_oauth(initiator, connector)
    initiator.active = False
    assert not _can_complete_oauth(initiator, connector)


def test_connector_strips_bearer_token_before_provider_download_redirect():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "api.example.com":
            return httpx.Response(
                302,
                headers={"location": "https://downloads.example.com/file"},
                request=request,
            )
        return httpx.Response(
            200,
            content=b"document",
            headers={"content-type": "application/octet-stream"},
            request=request,
        )

    result = asyncio.run(
        _MockAdapter(handler)._request("GET", "https://api.example.com/download")
    )
    assert result == b"document"
    assert requests[0].headers["authorization"] == "Bearer provider-token"
    assert "authorization" not in requests[1].headers


def test_google_drive_scope_discovery_follows_drive_and_folder_pages():
    adapter = GoogleDriveAdapter(
        Connector(id=uuid.uuid4(), company_domain="acme.test", system="google_drive")
    )
    calls = []

    async def fake_request(method, url, **_kwargs):
        calls.append(url)
        if "/drives?" in url and "pageToken=" not in url:
            return {
                "drives": [{"id": "shared-1", "name": "Shared One"}],
                "nextPageToken": "drive-page-2",
            }
        if "/drives?" in url and "pageToken=drive-page-2" in url:
            return {"drives": [{"id": "shared-2", "name": "Shared Two"}]}
        if "/files?" in url and "pageToken=" not in url:
            return {
                "files": [{"id": "folder-1", "name": "Policies"}],
                "nextPageToken": "folder-page-2",
            }
        if "/files?" in url and "pageToken=folder-page-2" in url:
            return {"files": [{"id": "folder-2", "name": "Archive"}]}
        raise AssertionError(f"unexpected Google Drive request: {url}")

    adapter._request = fake_request
    scopes = asyncio.run(adapter.discover_scopes())
    scope_ids = {item["external_scope_id"] for item in scopes}
    assert {"shared-1", "shared-2", "user", "user:folder-1", "user:folder-2"}.issubset(scope_ids)
    assert any("pageToken=drive-page-2" in url for url in calls)
    assert any("pageToken=folder-page-2" in url for url in calls)


def test_google_drive_permissions_prefer_email_for_internal_identity_matching():
    adapter = GoogleDriveAdapter(
        Connector(id=uuid.uuid4(), company_domain="acme.test", system="google_drive")
    )
    change = NormalizedChange(
        external_id="file-1",
        corpus_id="shared-1",
        name="policy.md",
        state="active",
        content_changed=True,
        permissions_changed=False,
        moved=False,
        revision="1",
        mime_type="text/markdown",
        parent_external_id=None,
        web_url=None,
        metadata={},
    )

    async def fake_request(*_args, **_kwargs):
        return {
            "permissions": [
                {"id": "permission-id", "type": "user", "emailAddress": "owner@acme.test", "role": "reader"},
                {"id": "domain-id", "type": "domain", "domain": "acme.test", "role": "reader"},
            ]
        }

    adapter._request = fake_request
    assert asyncio.run(adapter.permissions(change)) == [
        {"principal_type": "user", "principal_id": "owner@acme.test", "role": "reader"},
        {"principal_type": "domain", "principal_id": "domain-id", "role": "reader"},
    ]
