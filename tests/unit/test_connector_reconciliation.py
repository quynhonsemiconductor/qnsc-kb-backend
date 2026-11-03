"""Guards for the connector reconciliation pass and the Graph lifecycle endpoint.

Both of these went wrong in a way no other test could see: the reconciliation sweep
deletes documents, and the lifecycle URL is only ever called by Microsoft.
"""
from __future__ import annotations

import asyncio
import uuid

from src.api.routers.connectors import sharepoint_lifecycle_webhook
from src.domain.connector_adapters import (
    ConnectorAdapter,
    GoogleDriveAdapter,
    SharePointAdapter,
)
from src.domain.sync_queue import COALESCIBLE_REQUEST_STATES
from src.models.ops import Connector


def _connector(system: str) -> Connector:
    return Connector(id=uuid.uuid4(), company_domain="acme.test", system=system)


def test_a_new_adapter_cannot_opt_into_the_deletion_sweep_by_accident():
    """The sweep reads "absent from the walk" as "deleted at the provider"."""
    assert ConnectorAdapter.full_walk_is_authoritative is False


def test_sharepoint_delta_without_a_token_replays_the_whole_scope():
    adapter = SharePointAdapter(_connector("sharepoint"))
    urls: list[str] = []

    async def fake_request(method, url, **_kwargs):
        urls.append(url)
        return {
            "value": [
                {"id": "file-1", "name": "one.docx", "file": {"mimeType": "application/vnd"}},
                {"id": "file-2", "name": "two.docx", "file": {"mimeType": "application/vnd"}},
            ],
            "@odata.deltaLink": "https://graph.microsoft.com/v1.0/next-delta",
        }

    adapter._request = fake_request  # type: ignore[method-assign]
    changes, cursor = asyncio.run(
        adapter.incremental_changes({"external_scope_id": "drive-1", "config": {}}, None)
    )

    assert "/delta" in urls[0], "a cursor-less call must start a fresh delta walk"
    assert {change.external_id for change in changes} == {"file-1", "file-2"}
    assert cursor == "https://graph.microsoft.com/v1.0/next-delta"
    assert SharePointAdapter.full_walk_is_authoritative is True


def test_google_drive_without_a_cursor_reports_nothing_rather_than_everything():
    """This is why Drive must never be swept.

    With no cursor the adapter asks for startPageToken — the marker for changes from NOW
    on — so a "full reconciliation" returns an EMPTY change list for a Drive holding
    thousands of indexed files. Reading that as the current state of the scope deleted
    the entire corpus on every reconciliation pass.
    """
    adapter = GoogleDriveAdapter(_connector("google_drive"))
    urls: list[str] = []

    async def fake_request(method, url, **_kwargs):
        urls.append(url)
        if "startPageToken" in url:
            return {"startPageToken": "token-1"}
        return {"changes": [], "newStartPageToken": "token-2"}

    adapter._request = fake_request  # type: ignore[method-assign]
    changes, cursor = asyncio.run(
        adapter.incremental_changes(
            {"external_scope_id": "drive-1", "config": {"drive_id": "drive-1"}}, None
        )
    )

    assert any("startPageToken" in url for url in urls)
    assert changes == [], "the changes feed is incremental-only, never an enumeration"
    assert cursor == "token-2"
    assert GoogleDriveAdapter.full_walk_is_authoritative is False


def test_a_notification_is_never_folded_into_a_run_already_in_flight():
    """A dispatched or running sync has already read its change list from the provider.

    Coalescing a fresh notification onto it drops the change until the next
    reconciliation, so only a request that has not started yet may absorb one.
    """
    assert COALESCIBLE_REQUEST_STATES == ("queued",)


def test_the_lifecycle_endpoint_echoes_the_graph_validation_token():
    """Graph validates lifecycleNotificationUrl while CREATING the subscription.

    It POSTs a validationToken and expects it back as text/plain. Without that the whole
    POST /subscriptions call fails and no SharePoint scope can enable notifications.
    """
    response = asyncio.run(
        sharepoint_lifecycle_webhook(None, validationToken="opaque-graph-token")
    )

    assert response.status_code == 200
    assert response.body == b"opaque-graph-token"
    assert response.media_type == "text/plain"
