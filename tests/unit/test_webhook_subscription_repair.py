"""Guards for the two ways "24/7 sync" quietly stopped being 24/7.

Neither failure raises anything. The provider simply stops calling: the connector keeps
reporting `on_update`, the health endpoint keeps showing a subscription row, and the
corpus goes stale until a person happens to notice. So both are pinned here.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

from src.domain.connector_adapters import GoogleDriveAdapter, SharePointAdapter
from src.domain.webhook_subscriptions import (
    RENEWAL_HORIZON,
    ensure_webhook_subscriptions,
)
from src.models.connectors import SourceScope, WebhookSubscription
from src.models.ops import Connector


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Db:
    """Answers each select by model class, in the order the module queries them."""

    def __init__(self, scopes, subscriptions):
        self.scopes = scopes
        self.subscriptions = subscriptions
        self.added: list[object] = []

    async def execute(self, statement):
        text = str(statement)
        if "source_scopes" in text:
            return _Result(list(self.scopes))
        if "webhook_subscriptions" in text:
            return _Result(list(self.subscriptions))
        raise AssertionError(f"unexpected query: {text}")

    def add(self, item):
        self.added.append(item)

    async def commit(self):
        raise AssertionError("the caller owns the transaction, not the domain helper")


def _connector(system: str) -> Connector:
    return Connector(
        id=uuid.uuid4(),
        company_domain="acme.test",
        system=system,
        status="active",
        config_json={"webhook_enabled": True},
        oauth_access_token="token",
        oauth_expires_at=datetime.utcnow() + timedelta(hours=1),
    )


def _scope(connector: Connector) -> SourceScope:
    return SourceScope(
        id=uuid.uuid4(),
        connector_id=connector.id,
        external_scope_id="drive-1",
        scope_type="drive",
        display_name="Drive",
        selected=True,
        config_json={"drive_id": "drive-1"},
    )


def _subscribe(monkeypatch, connector, scopes, subscriptions, adapter, **kwargs):
    monkeypatch.setattr(
        "src.domain.webhook_subscriptions.adapter_for", lambda _connector: adapter
    )

    async def already_authorized(_db, _connector):
        return None

    monkeypatch.setattr(
        "src.domain.webhook_subscriptions.ensure_connector_authorized",
        already_authorized,
    )
    monkeypatch.setattr(
        "src.domain.webhook_subscriptions.settings.CONNECTOR_WEBHOOK_BASE_URL",
        "https://kb.example.com",
        raising=False,
    )
    db = _Db(scopes, subscriptions)
    created = asyncio.run(ensure_webhook_subscriptions(db, connector, **kwargs))
    return db, created


class _RecordingAdapter:
    def __init__(self):
        self.deleted: list[tuple[str, str | None]] = []
        self.created = 0

    async def delete_webhook(self, provider_subscription_id, resource=None):
        self.deleted.append((provider_subscription_id, resource))

    async def create_webhook(self, scope, callback_url, lifecycle_callback_url=None):
        self.created += 1
        return {
            "subscription_id": f"sub-{self.created}",
            "client_state": "secret",
            "expires_at": datetime.utcnow() + timedelta(days=1),
        }


def test_an_expired_subscription_is_replaced_without_an_admin(monkeypatch):
    """The Drive case: a channel cannot be renewed, only replaced.

    renew_webhook returns None for Drive, so the renewal pass retires the row. Before
    the repair pass existed, nothing created its successor and the connector fell back
    to polling permanently while still advertising on_update.
    """
    connector = _connector("google_drive")
    scope = _scope(connector)
    dead = WebhookSubscription(
        id=uuid.uuid4(),
        connector_id=connector.id,
        scope_id=scope.id,
        provider_subscription_id="channel-old",
        verification_token_hash="x" * 64,
        resource="resource-old",
        expires_at=datetime.utcnow() - timedelta(minutes=5),
        active=True,
    )
    adapter = _RecordingAdapter()

    db, created = _subscribe(monkeypatch, connector, [scope], [dead], adapter)

    assert len(created) == 1, "an expired subscription must be replaced automatically"
    assert dead.active is False
    assert len(db.added) == 1
    assert db.added[0].provider_subscription_id == "sub-1"


def test_a_healthy_subscription_is_left_alone(monkeypatch):
    """The repair pass runs every ten minutes over every connector.

    If it replaced a live subscription each time, it would burn provider quota and
    reset the notification stream on a schedule.
    """
    connector = _connector("sharepoint")
    scope = _scope(connector)
    live = WebhookSubscription(
        id=uuid.uuid4(),
        connector_id=connector.id,
        scope_id=scope.id,
        provider_subscription_id="sub-live",
        verification_token_hash="x" * 64,
        expires_at=datetime.utcnow() + RENEWAL_HORIZON + timedelta(hours=1),
        active=True,
    )
    adapter = _RecordingAdapter()

    db, created = _subscribe(monkeypatch, connector, [scope], [live], adapter)

    assert created == []
    assert adapter.created == 0
    assert live.active is True
    assert db.added == []


def test_the_old_subscription_is_retired_at_the_provider_first(monkeypatch):
    """Graph answers a duplicate changeType+resource with 409 Conflict.

    Creating the replacement without deleting the old one therefore fails precisely in
    the case repair exists for: a subscription that is half-alive at the provider.
    """
    connector = _connector("sharepoint")
    scope = _scope(connector)
    stale = WebhookSubscription(
        id=uuid.uuid4(),
        connector_id=connector.id,
        scope_id=scope.id,
        provider_subscription_id="sub-stale",
        verification_token_hash="x" * 64,
        expires_at=datetime.utcnow() + timedelta(minutes=1),
        active=True,
    )
    adapter = _RecordingAdapter()

    _db, created = _subscribe(monkeypatch, connector, [scope], [stale], adapter)

    assert adapter.deleted == [("sub-stale", None)]
    assert len(created) == 1


def test_a_provider_that_cannot_delete_still_gets_a_replacement(monkeypatch):
    """Cleanup is best effort. A dead channel must never block resubscription."""

    class _FailingDelete(_RecordingAdapter):
        async def delete_webhook(self, provider_subscription_id, resource=None):
            raise RuntimeError("channel already gone")

    connector = _connector("google_drive")
    scope = _scope(connector)
    dead = WebhookSubscription(
        id=uuid.uuid4(),
        connector_id=connector.id,
        scope_id=scope.id,
        provider_subscription_id="channel-old",
        verification_token_hash="x" * 64,
        expires_at=datetime.utcnow() - timedelta(days=1),
        active=True,
    )

    _db, created = _subscribe(monkeypatch, connector, [scope], [dead], _FailingDelete())

    assert len(created) == 1


def test_graph_is_asked_for_permission_change_notifications():
    """Without this header Graph notifies on CONTENT changes only.

    A file whose sharing was revoked would never wake the connector, so a user who lost
    access at SharePoint kept seeing the document in the KB until the next
    reconciliation pass — six hours by default.
    """
    adapter = SharePointAdapter(_connector("sharepoint"))
    seen: dict[str, object] = {}

    async def fake_request(method, url, **kwargs):
        seen.update(kwargs)
        seen["url"] = url
        return {"id": "sub-1"}

    adapter._request = fake_request  # type: ignore[method-assign]
    asyncio.run(
        adapter.create_webhook(
            {"external_scope_id": "drive-1", "config": {"drive_id": "drive-1"}},
            "https://kb.example.com/api/v1/connectors/webhooks/sharepoint",
        )
    )

    assert seen["headers"] == {"Prefer": "includesecuritywebhooks"}
    assert seen["json"]["resource"] == "/drives/drive-1/root"


def test_a_drive_channel_records_the_id_needed_to_stop_it():
    """channels.stop needs BOTH the channel id and the resourceId.

    Without the second one a superseded channel keeps delivering until it expires.
    """
    adapter = GoogleDriveAdapter(_connector("google_drive"))

    async def fake_request(method, url, **_kwargs):
        if "startPageToken" in url:
            return {"startPageToken": "token-1"}
        return {"resourceId": "resource-1", "expiration": "1700000000000"}

    adapter._request = fake_request  # type: ignore[method-assign]
    result = asyncio.run(
        adapter.create_webhook(
            {"external_scope_id": "drive-1", "config": {"drive_id": "drive-1"}},
            "https://kb.example.com/api/v1/connectors/webhooks/google-drive",
        )
    )

    assert result["resource"] == "resource-1"
    assert asyncio.run(adapter.renew_webhook(result["subscription_id"])) is None, (
        "a Drive channel cannot be extended; it must be reported as spent"
    )
