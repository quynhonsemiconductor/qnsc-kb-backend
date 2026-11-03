"""Guards for the parts of cloud sync that fail quietly rather than loudly.

Every case here shares a shape: the connector kept reporting healthy while the corpus
it was supposed to keep current drifted away from the provider. A broken file that
froze the whole walk, a Drive token nobody renewed, a folder whose subfolders were
never read, a deletion that was filtered out before anything could act on it — none of
them raised where an operator would see it, and the only symptom was a knowledge base
that quietly stopped matching the drive behind it.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest

from src.domain import connector_auth
from src.domain.cloud_sync import (
    _MAX_ITEM_ATTEMPTS,
    _SCOPE_FATAL_CODES,
    _is_item_level_failure,
    _is_quarantined,
    _unsupported_reason,
)
from src.domain.connector_adapters import (
    ConnectorProviderError,
    GoogleDriveAdapter,
    NormalizedChange,
)
from src.models.connectors import ExternalDocument
from src.models.ops import Connector


def _connector(system: str, **kwargs) -> Connector:
    return Connector(id=uuid.uuid4(), company_domain="acme.test", system=system, **kwargs)


def _change(external_id: str = "file-1", revision: str | None = "v1") -> NormalizedChange:
    return NormalizedChange(
        external_id=external_id,
        corpus_id="drive-1",
        name="handbook.pdf",
        state="active",
        content_changed=True,
        permissions_changed=False,
        moved=False,
        revision=revision,
        mime_type="application/pdf",
        parent_external_id=None,
        web_url=None,
        metadata={},
    )


# --------------------------------------------------------------- token renewal


def test_a_google_drive_connector_refreshes_its_own_token(monkeypatch):
    """Drive access tokens live one hour; only the refresh keeps a connector alive.

    ``ensure_connector_authorized`` used to return early for anything that was not
    SharePoint, so a Drive connector worked for exactly one token lifetime and then
    answered 401 — reported as non-retryable — forever after, with a perfectly good
    refresh token sitting unused in the same row.
    """

    connector = _connector(
        "google_drive",
        oauth_access_token="stale",
        oauth_refresh_token="refresh",
        oauth_expires_at=datetime.utcnow() - timedelta(minutes=1),
    )
    refreshed: list[str] = []

    class FakeAdapter:
        async def refresh_token(self):
            refreshed.append("called")
            return {"access_token": "fresh", "expires_in": 3600}

    monkeypatch.setattr(connector_auth, "adapter_for", lambda _connector: FakeAdapter())
    monkeypatch.setattr(connector_auth, "encrypt_secret", lambda value: value)

    class FakeSession:
        async def commit(self):
            return None

    asyncio.run(connector_auth.ensure_connector_authorized(FakeSession(), connector))

    assert refreshed == ["called"], "a non-SharePoint connector must renew its token too"
    assert connector.oauth_access_token == "fresh"
    assert connector.oauth_expires_at > datetime.utcnow()


def test_a_refresh_response_without_a_refresh_token_keeps_the_stored_one(monkeypatch):
    """Google returns a refresh token on the first consent only, never again."""

    connector = _connector(
        "google_drive",
        oauth_access_token="stale",
        oauth_refresh_token="the-only-one",
        oauth_expires_at=datetime.utcnow() - timedelta(minutes=1),
    )

    class FakeAdapter:
        async def refresh_token(self):
            return {"access_token": "fresh", "expires_in": 3600}

    monkeypatch.setattr(connector_auth, "adapter_for", lambda _connector: FakeAdapter())
    monkeypatch.setattr(connector_auth, "encrypt_secret", lambda value: value)

    class FakeSession:
        async def commit(self):
            return None

    asyncio.run(connector_auth.ensure_connector_authorized(FakeSession(), connector))

    assert connector.oauth_refresh_token == "the-only-one"


def test_a_token_expiring_mid_walk_is_renewed_before_the_walk_starts(monkeypatch):
    """A sync is a long walk. A token good for one more minute is not good enough."""

    connector = _connector(
        "google_drive",
        oauth_access_token="nearly-stale",
        oauth_refresh_token="refresh",
        oauth_expires_at=datetime.utcnow() + timedelta(minutes=1),
    )
    refreshed: list[str] = []

    class FakeAdapter:
        async def refresh_token(self):
            refreshed.append("called")
            return {"access_token": "fresh", "expires_in": 3600}

    monkeypatch.setattr(connector_auth, "adapter_for", lambda _connector: FakeAdapter())
    monkeypatch.setattr(connector_auth, "encrypt_secret", lambda value: value)

    class FakeSession:
        async def commit(self):
            return None

    asyncio.run(connector_auth.ensure_connector_authorized(FakeSession(), connector))
    assert refreshed == ["called"]


# ------------------------------------------------------- poison-pill isolation


@pytest.mark.parametrize(
    "code",
    ["401", "403", "429", "503", "not_authorized", "resync_required"],
)
def test_a_connector_wide_failure_is_never_blamed_on_one_document(code):
    """Otherwise an expired credential quarantines the drive one file at a time."""

    assert code in _SCOPE_FATAL_CODES
    assert not _is_item_level_failure(ConnectorProviderError("boom", code=code))


@pytest.mark.parametrize("code", ["404", "410", "response_too_large", "invalid_json"])
def test_a_failure_about_one_document_stays_about_that_document(code):
    assert _is_item_level_failure(ConnectorProviderError("boom", code=code))


def test_an_extraction_failure_is_isolated_rather_than_fatal():
    """A password-protected PDF is a bad file, not a broken connector."""

    assert _is_item_level_failure(ValueError("No readable text was found"))


def test_a_repeatedly_failing_revision_stops_being_retried():
    document = ExternalDocument(
        connector_id=uuid.uuid4(),
        corpus_id="drive-1",
        external_id="file-1",
        name="handbook.pdf",
        metadata_json={
            "ingest_failure": {"revision": "v1", "attempts": _MAX_ITEM_ATTEMPTS, "reason": "corrupt"}
        },
    )
    assert _is_quarantined(document, _change(revision="v1"))


def test_replacing_the_file_at_the_source_releases_the_quarantine():
    """The fix a person actually performs must be the fix that works.

    Nobody should have to find an admin screen to un-stick a document they have
    already repaired in SharePoint or Drive; a new revision is the signal.
    """

    document = ExternalDocument(
        connector_id=uuid.uuid4(),
        corpus_id="drive-1",
        external_id="file-1",
        name="handbook.pdf",
        metadata_json={
            "ingest_failure": {"revision": "v1", "attempts": 9, "reason": "corrupt"}
        },
    )
    assert not _is_quarantined(document, _change(revision="v2"))


def test_a_first_failure_is_retried_before_it_is_given_up_on():
    document = ExternalDocument(
        connector_id=uuid.uuid4(),
        corpus_id="drive-1",
        external_id="file-1",
        name="handbook.pdf",
        metadata_json={"ingest_failure": {"revision": "v1", "attempts": 1, "reason": "timeout"}},
    )
    assert not _is_quarantined(document, _change(revision="v1"))


# ------------------------------------------------------------- type pre-filter


@pytest.mark.parametrize("name", ["policy.pdf", "notes.MD", "sheet.xlsx", "scan.JPG"])
def test_an_indexable_file_is_not_pre_filtered(name):
    assert _unsupported_reason(name, "application/octet-stream") is None


@pytest.mark.parametrize("name", ["recording.mp4", "archive.zip", "notebook.one", "README"])
def test_a_file_that_could_never_be_indexed_is_never_downloaded(name):
    """The cheapest failure is the one the connector declines to attempt."""

    assert _unsupported_reason(name, None) is not None


def test_a_google_doc_is_exportable_and_a_google_form_is_not():
    assert _unsupported_reason("Handbook", "application/vnd.google-apps.document") is None
    assert _unsupported_reason("Sheet", "application/vnd.google-apps.spreadsheet") is None
    assert _unsupported_reason("Survey", "application/vnd.google-apps.form") is not None
    assert _unsupported_reason("Sketch", "application/vnd.google-apps.drawing") is not None


# ------------------------------------------------- Google Drive folder scoping


def _drive_adapter_returning(pages: list[dict], files: dict[str, list[str]]):
    adapter = GoogleDriveAdapter(_connector("google_drive"))
    remaining = list(pages)

    async def fake_request(method, url, **_kwargs):
        if "/files/" in url and "fields=id,parents" in url:
            file_id = url.split("/files/")[1].split("?")[0]
            return {"id": file_id, "parents": files.get(file_id, [])}
        if "startPageToken" in url:
            return {"startPageToken": "token-1"}
        return remaining.pop(0)

    adapter._request = fake_request  # type: ignore[method-assign]
    return adapter


def test_a_selected_drive_folder_includes_what_is_nested_inside_it():
    """Drive's change feed reports DIRECT parents only.

    Matching on those alone meant a selected folder contributed its immediate children
    and nothing else, so an admin who picked "Policies" got the loose files at its top
    level and none of the subfolders anyone actually files documents in — with no error
    anywhere to explain the gap.
    """

    adapter = _drive_adapter_returning(
        [
            {
                "changes": [
                    {
                        "fileId": "deep",
                        "file": {
                            "id": "deep",
                            "name": "2026-policy.pdf",
                            "mimeType": "application/pdf",
                            "parents": ["sub"],
                            "version": "3",
                        },
                    }
                ],
                "newStartPageToken": "token-2",
            }
        ],
        files={"sub": ["target-folder"], "target-folder": []},
    )

    changes, _ = asyncio.run(
        adapter.incremental_changes(
            {"external_scope_id": "drive-1", "config": {"drive_id": "drive-1", "folder_id": "target-folder"}},
            "cursor",
        )
    )
    assert [change.external_id for change in changes] == ["deep"]


def test_a_file_outside_the_selected_folder_is_still_excluded():
    adapter = _drive_adapter_returning(
        [
            {
                "changes": [
                    {
                        "fileId": "elsewhere",
                        "file": {
                            "id": "elsewhere",
                            "name": "payroll.xlsx",
                            "mimeType": "application/vnd.ms-excel",
                            "parents": ["other"],
                            "version": "1",
                        },
                    }
                ],
                "newStartPageToken": "token-2",
            }
        ],
        files={"other": ["some-root"], "some-root": []},
    )

    changes, _ = asyncio.run(
        adapter.incremental_changes(
            {"external_scope_id": "drive-1", "config": {"drive_id": "drive-1", "folder_id": "target-folder"}},
            "cursor",
        )
    )
    assert changes == []


def test_a_deletion_under_a_folder_scope_is_reported_not_swallowed():
    """A Drive removal carries no file resource, so it has no parents to test.

    Filtering on parents therefore discarded every deletion under a folder scope, and
    the knowledge base kept serving documents that no longer existed in Drive. The
    tombstone is emitted; the sync layer, which knows what it actually tracks, decides.
    """

    adapter = _drive_adapter_returning(
        [{"changes": [{"fileId": "gone", "removed": True}], "newStartPageToken": "token-2"}],
        files={},
    )

    changes, _ = asyncio.run(
        adapter.incremental_changes(
            {"external_scope_id": "drive-1", "config": {"drive_id": "drive-1", "folder_id": "target-folder"}},
            "cursor",
        )
    )
    assert [(change.external_id, change.state) for change in changes] == [("gone", "deleted")]


def test_ancestry_lookups_are_not_repeated_for_every_file_in_a_folder():
    """One walk of a folder must not re-resolve the same parent chain per file."""

    adapter = _drive_adapter_returning(
        [
            {
                "changes": [
                    {
                        "fileId": f"file-{index}",
                        "file": {
                            "id": f"file-{index}",
                            "name": f"doc-{index}.pdf",
                            "mimeType": "application/pdf",
                            "parents": ["sub"],
                            "version": "1",
                        },
                    }
                    for index in range(5)
                ],
                "newStartPageToken": "token-2",
            }
        ],
        files={"sub": ["target-folder"], "target-folder": []},
    )
    lookups: list[str] = []
    original = adapter._request

    async def counting_request(method, url, **kwargs):
        if "fields=id,parents" in url:
            lookups.append(url)
        return await original(method, url, **kwargs)

    adapter._request = counting_request  # type: ignore[method-assign]

    changes, _ = asyncio.run(
        adapter.incremental_changes(
            {"external_scope_id": "drive-1", "config": {"drive_id": "drive-1", "folder_id": "target-folder"}},
            "cursor",
        )
    )
    assert len(changes) == 5
    assert len(lookups) == 1, "the parent chain is resolved once per walk, not once per file"


# ------------------------------------------------------ one walk per connector


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeClaimSession:
    """Enough session to exercise the claim decision without a database.

    The advisory lock is issued as raw SQL and answered here the way Postgres would;
    the in-flight lookup is the ``scalar`` call.
    """

    def __init__(self, request, in_flight_id):
        self.request = request
        self.in_flight_id = in_flight_id
        self.commits = 0

    async def execute(self, statement, params=None):
        return _FakeResult(self.request)

    async def scalar(self, statement):
        return self.in_flight_id

    async def commit(self):
        self.commits += 1


def _queued_request(connector_id):
    from src.models.connectors import SyncRequest

    return SyncRequest(
        id=uuid.uuid4(),
        connector_id=connector_id,
        job_id=uuid.uuid4(),
        reason="polling",
        priority=50,
        status="queued",
        attempts=0,
        available_at=datetime.utcnow() - timedelta(seconds=1),
    )


def test_a_connector_already_walking_is_not_sent_a_second_walk():
    """Polling enqueues connector-wide work; reconciliation enqueues per scope.

    Those never coalesce with each other, so nothing stopped two or three walks of the
    SAME drive from running at once — each downloading the same files and each racing
    to write the same delta cursor, with the loser silently rewinding the winner.
    """

    from src.domain.sync_queue import claim_sync_request

    connector_id = uuid.uuid4()
    request = _queued_request(connector_id)
    session = _FakeClaimSession(request, in_flight_id=uuid.uuid4())

    claimed = asyncio.run(claim_sync_request(session, request.id))

    assert claimed is None
    assert request.status == "queued", "the request waits its turn rather than being lost"
    assert request.attempts == 0


def test_an_idle_connector_claims_normally():
    from src.domain.sync_queue import claim_sync_request

    connector_id = uuid.uuid4()
    request = _queued_request(connector_id)
    session = _FakeClaimSession(request, in_flight_id=None)

    claimed = asyncio.run(claim_sync_request(session, request.id))

    assert claimed is request
    assert request.status == "dispatched"
    assert request.attempts == 1
