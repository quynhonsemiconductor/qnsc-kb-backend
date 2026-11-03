"""Guards for the OneDrive connector and the provider-neutral ACL intersection.

OneDrive is Microsoft Graph, exactly like SharePoint, so the adapter was never the
risk. The risk was the dozen places that decided provider behaviour by comparing
``connector.system`` to the literal ``"sharepoint"``. Two of those places are the
source-ACL intersection in ``domain/permissions.py`` and ``repositories/article.py``:
a provider whose name did not match fell through to "no provider ACL applies" and a
company-wide reader saw a document the provider had never shared with them.

``asyncio.run`` rather than pytest-asyncio: the marker is not available in every
environment this suite runs in.
"""
from __future__ import annotations

import asyncio
import uuid

from src.domain.connector_adapters import (
    ConnectorProviderError,
    MicrosoftGraphAdapter,
    OneDriveAdapter,
    SharePointAdapter,
    adapter_for,
)
from src.domain.connector_providers import (
    REMOTE_PROVIDERS,
    SOURCE_ACL_PROVIDERS,
    cursor_type,
    identity_provider,
    is_microsoft_graph,
    supports_lifecycle_webhooks,
)
from src.domain.permissions import PermissionService
from src.models.ops import Connector


def _connector(system: str) -> Connector:
    return Connector(id=uuid.uuid4(), company_domain="acme.test", system=system)


class _Source:
    def __init__(self, source_system: str):
        self.source_system = source_system


class _Override:
    def __init__(self, user_id, effect: str, source: str | None):
        self.user_id = user_id
        self.effect = effect
        self.source = source


class _Department:
    def __init__(self):
        self.id = uuid.uuid4()


class _Article:
    def __init__(self, *, sources, user_permissions=(), departments=()):
        self.sources = list(sources)
        self.user_permissions = list(user_permissions)
        self.departments = list(departments)


class _User:
    def __init__(self, departments=()):
        self.id = uuid.uuid4()
        self.departments = list(departments)


# --- provider registry -------------------------------------------------------


def test_onedrive_is_a_graph_provider_and_shares_sharepoint_semantics():
    assert is_microsoft_graph("onedrive")
    # Graph exposes /delta with a deltaLink; a `changes` cursor would be read back
    # as a Google Drive page token and never resume the walk.
    assert cursor_type("onedrive") == cursor_type("sharepoint") == "delta"
    # SharePoint and OneDrive sit behind ONE Entra registration, so an ACL user
    # principal must resolve through the same ExternalIdentity rows.
    assert identity_provider("onedrive") == identity_provider("sharepoint") == "microsoft_entra"
    assert supports_lifecycle_webhooks("onedrive") is True


def test_google_drive_keeps_its_own_cursor_and_identity_provider():
    """The registry must not flatten every provider into Graph."""
    assert is_microsoft_graph("google_drive") is False
    assert cursor_type("google_drive") == "changes"
    assert identity_provider("google_drive") == "google_drive"
    assert supports_lifecycle_webhooks("google_drive") is False


def test_every_remote_provider_stamps_source_managed_permissions():
    """SOURCE_ACL_PROVIDERS is what both ACL predicates match on.

    A remote provider missing from this tuple writes provenance rows that neither
    predicate recognises, which is a silent fail-OPEN rather than fail-closed.
    """
    assert set(SOURCE_ACL_PROVIDERS) == set(REMOTE_PROVIDERS)
    assert "local_folder" not in SOURCE_ACL_PROVIDERS


# --- source-ACL intersection (the fail-open this change closes) --------------


def test_a_provider_backed_article_fails_closed_for_every_remote_provider():
    """The regression this whole change exists to prevent.

    The check matched the literal ``"sharepoint"``, so a OneDrive or Google Drive
    Article returned True immediately — "no provider ACL applies" — and the
    intersection never ran. The reader below shares no department with the Article
    and holds no source-qualified allow, so every remote provider must deny.
    """
    for provider in SOURCE_ACL_PROVIDERS:
        article = _Article(sources=[_Source(provider)], departments=[_Department()])
        assert (
            PermissionService._source_acl_allows(_User(), article) is False
        ), f"{provider} bypassed the provider-ACL intersection"


def test_a_manual_article_is_not_subject_to_a_provider_acl():
    """Nothing here is source-managed, so the internal policy is the whole policy."""
    article = _Article(sources=[_Source("manual")], departments=[_Department()])
    assert PermissionService._source_acl_allows(_User(), article) is True


def test_a_mapped_onedrive_group_admits_a_shared_department_member():
    """The positive path: mapped groups reach the reader through Article.departments."""
    shared = _Department()
    article = _Article(sources=[_Source("onedrive")], departments=[shared])
    assert PermissionService._source_acl_allows(_User(departments=[shared]), article) is True


def test_a_onedrive_source_allow_admits_a_mapped_direct_user():
    """A mapped direct user is stored as a source-qualified allow row.

    The row's ``source`` is ``connector.system``, so it must be matched against the
    Article's actual providers rather than a hardcoded name.
    """
    user = _User()
    article = _Article(
        sources=[_Source("onedrive")],
        user_permissions=[_Override(user.id, "allow", "onedrive")],
    )
    assert PermissionService._source_acl_allows(user, article) is True


def test_another_providers_allow_does_not_satisfy_this_providers_acl():
    """Provenance is per provider: a Google Drive grant is not OneDrive consent."""
    user = _User()
    article = _Article(
        sources=[_Source("onedrive")],
        user_permissions=[_Override(user.id, "allow", "google_drive")],
    )
    assert PermissionService._source_acl_allows(user, article) is False


# --- adapter -----------------------------------------------------------------


def test_onedrive_dispatches_to_its_own_adapter():
    adapter = adapter_for(_connector("onedrive"))
    assert isinstance(adapter, OneDriveAdapter)
    assert isinstance(adapter, MicrosoftGraphAdapter)
    assert adapter.provider == "onedrive"


def test_an_unregistered_provider_is_refused_rather_than_defaulted():
    try:
        adapter_for(_connector("dropbox"))
    except ConnectorProviderError as exc:
        assert exc.code == "unsupported_provider"
        assert exc.retryable is False
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("an unknown provider must not silently pick an adapter")


def test_onedrive_does_not_request_sharepoint_site_permission():
    """Least privilege: this provider never reads a site.

    Inheriting SharePoint's consent string would make a OneDrive-only integration a
    tenant-wide site reader, which is exactly the over-permission an allowlisted
    deployment is trying to avoid.
    """
    assert "Sites.Read.All" in SharePointAdapter.oauth_scope
    assert "Sites.Read.All" not in OneDriveAdapter.oauth_scope
    assert "Files.Read.All" in OneDriveAdapter.oauth_scope
    assert "offline_access" in OneDriveAdapter.oauth_scope


def test_onedrive_discovery_walks_the_allowlisted_user_drives(monkeypatch):
    monkeypatch.setattr(
        "src.domain.connector_adapters.settings.MICROSOFT_ONEDRIVE_USER_IDS",
        "alice@acme.test, bob@acme.test",
    )
    monkeypatch.setattr(
        "src.domain.connector_adapters.settings.MICROSOFT_CONNECTOR_AUTH_MODE",
        "application",
    )
    adapter = OneDriveAdapter(_connector("onedrive"))
    urls: list[str] = []

    async def fake_request(method, url, **_kwargs):
        urls.append(url)
        if "/drive?" in url:
            user = url.split("/users/")[1].split("/drive")[0]
            return {"id": f"drive-{user}", "name": "OneDrive", "webUrl": f"https://acme/{user}"}
        # Graph always sends a childCount on the folder facet; an empty dict is falsy
        # and would be treated as a file by the shared folder walk.
        return {"value": [{"id": "folder-1", "name": "Policies", "folder": {"childCount": 2}}]}

    adapter._request = fake_request  # type: ignore[method-assign]
    scopes = asyncio.run(adapter.discover_scopes())

    # /me/drive is never consulted when an allowlist exists: it resolves to whoever
    # clicked consent, which is not an administrable source.
    assert not any("/me/drive" in url for url in urls)
    assert [scope["scope_type"] for scope in scopes] == [
        "onedrive_drive",
        "onedrive_folder",
        "onedrive_drive",
        "onedrive_folder",
    ]
    folder = scopes[1]
    # drive:folder, because the delta call needs BOTH halves.
    assert folder["external_scope_id"] == "drive-alice%40acme.test:folder-1"
    assert folder["config"]["folder_id"] == "folder-1"
    assert folder["config"]["drive_id"] == "drive-alice%40acme.test"


def test_application_mode_refuses_to_discover_without_an_allowlist(monkeypatch):
    """App-only mode has no signed-in user, so there is nothing to fall back to."""
    monkeypatch.setattr(
        "src.domain.connector_adapters.settings.MICROSOFT_ONEDRIVE_USER_IDS", ""
    )
    monkeypatch.setattr(
        "src.domain.connector_adapters.settings.MICROSOFT_CONNECTOR_AUTH_MODE",
        "application",
    )
    adapter = OneDriveAdapter(_connector("onedrive"))
    try:
        asyncio.run(adapter.discover_scopes())
    except ConnectorProviderError as exc:
        assert exc.code == "scope_allowlist_required"
        assert exc.retryable is False
    else:  # pragma: no cover
        raise AssertionError("application mode must not enumerate the whole tenant")


def test_an_unprovisioned_drive_does_not_abort_the_remaining_allowlist(monkeypatch):
    """A licensed account with no OneDrive answers 404.

    One such account used to take the whole discovery call down, so a single stale
    entry in the allowlist hid every other user's drive.
    """
    monkeypatch.setattr(
        "src.domain.connector_adapters.settings.MICROSOFT_ONEDRIVE_USER_IDS",
        "ghost@acme.test,real@acme.test",
    )
    monkeypatch.setattr(
        "src.domain.connector_adapters.settings.MICROSOFT_CONNECTOR_AUTH_MODE",
        "delegated",
    )
    adapter = OneDriveAdapter(_connector("onedrive"))

    async def fake_request(method, url, **_kwargs):
        if "ghost" in url:
            raise ConnectorProviderError("Provider request failed (404)", code="404")
        if "/drive?" in url:
            return {"id": "drive-real", "name": "OneDrive"}
        return {"value": []}

    adapter._request = fake_request  # type: ignore[method-assign]
    scopes = asyncio.run(adapter.discover_scopes())

    assert [scope["external_scope_id"] for scope in scopes] == ["drive-real"]


def test_onedrive_reuses_the_graph_delta_walk():
    """Inherited, not reimplemented — including the deletion-sweep opt-in."""
    adapter = OneDriveAdapter(_connector("onedrive"))
    urls: list[str] = []

    async def fake_request(method, url, **_kwargs):
        urls.append(url)
        return {
            "value": [{"id": "file-1", "name": "one.docx", "file": {"mimeType": "application/vnd"}}],
            "@odata.deltaLink": "https://graph.microsoft.com/v1.0/next-delta",
        }

    adapter._request = fake_request  # type: ignore[method-assign]
    changes, cursor = asyncio.run(
        adapter.incremental_changes({"external_scope_id": "drive-1", "config": {}}, None)
    )

    assert "/delta" in urls[0]
    assert [change.external_id for change in changes] == ["file-1"]
    assert cursor == "https://graph.microsoft.com/v1.0/next-delta"
    assert OneDriveAdapter.full_walk_is_authoritative is True
