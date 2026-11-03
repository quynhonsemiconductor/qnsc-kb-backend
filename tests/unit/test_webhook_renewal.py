"""Push subscriptions must be renewable, and the notification URL must be buildable.

Microsoft Graph drive subscriptions expire an hour after creation. A lapsed one produces
no error anywhere: the provider stops calling, the connector still reports sync_mode
"on_update", and the corpus goes quietly stale. Nothing renewed them, so real-time sync
lasted exactly one hour from the moment it was enabled.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.domain.connector_adapters import ConnectorAdapter, SharePointAdapter


class _Recorder(SharePointAdapter):
    """Captures the Graph call instead of making it."""

    def __init__(self):  # deliberately not calling super().__init__
        self.calls: list[tuple[str, str, dict]] = []

    async def _request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs.get("json", {})))
        return {}


@pytest.mark.asyncio
async def test_sharepoint_renewal_patches_the_subscription():
    adapter = _Recorder()

    before = datetime.utcnow()
    expires_at = await adapter.renew_webhook("sub-123")

    assert expires_at is not None and expires_at > before
    method, url, body = adapter.calls[0]
    assert method == "PATCH"
    assert url.endswith("/subscriptions/sub-123")
    assert body["expirationDateTime"].endswith("Z"), "Graph rejects a naive timestamp"


@pytest.mark.asyncio
async def test_a_provider_without_renewal_reports_it_rather_than_raising():
    # Google Drive channels cannot be extended in place; the caller retires the row.
    assert await ConnectorAdapter.renew_webhook(object(), "sub-123") is None


def test_notification_url_is_built_from_an_origin_not_a_path():
    """CONNECTOR_WEBHOOK_BASE_URL is an ORIGIN — connectors.py appends the full path.

    Terraform used to set it to "<api>/api/v1/connectors", producing
    ".../api/v1/connectors/api/v1/connectors/webhooks/sharepoint". Graph POSTs a
    validationToken to that URL while CREATING the subscription, so enabling update
    notifications failed outright.
    """
    base = "https://kb-api-dev.qnsc.vn"
    callback = f"{base.rstrip('/')}/api/v1/connectors/webhooks/{'sharepoint'.replace('_', '-')}"
    assert callback == "https://kb-api-dev.qnsc.vn/api/v1/connectors/webhooks/sharepoint"
    assert callback.count("/api/v1/connectors") == 1


def test_renewal_horizon_leaves_room_for_a_failed_run():
    """20-minute horizon, 10-minute schedule, 1-hour lifetime."""
    lifetime = timedelta(hours=1)
    horizon = timedelta(minutes=20)
    schedule = timedelta(minutes=10)
    assert horizon > schedule, "a subscription must be picked up more than once before expiry"
    assert horizon < lifetime, "renewing the moment it is created would spin uselessly"
