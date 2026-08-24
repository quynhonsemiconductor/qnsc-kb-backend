"""Provider push subscriptions — created, repaired and retired in one place.

A push subscription is the only thing separating "24/7" from "whatever the polling
interval happens to be", and it is the piece most likely to lapse quietly. Every
subscription this system holds is temporary by construction: Microsoft Graph caps a
drive subscription at under 30 days, and a Google Drive channel cannot be extended at
all — it can only be replaced.

Creation used to live inside the admin endpoint, so a lapsed subscription could only
come back when a human clicked the button again. A Drive connector therefore ran
real-time for a day and then silently degraded to polling forever. Everything that
creates a subscription now goes through :func:`ensure_webhook_subscriptions`, which
lets the renewal worker repair one through exactly the code path the admin uses.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.domain.connector_adapters import ConnectorProviderError, adapter_for
from src.domain.connector_auth import ensure_connector_authorized
from src.models.connectors import SourceScope, WebhookSubscription
from src.models.ops import Connector

logger = structlog.get_logger()

# Replace a subscription this long before it lapses. The repair pass runs every ten
# minutes, so a subscription gets at least two attempts inside the horizon and one
# failed provider call is never enough to drop it.
RENEWAL_HORIZON = timedelta(minutes=20)


class WebhookConfigurationError(RuntimeError):
    """The deployment or the connector cannot hold a subscription yet.

    Distinct from ConnectorProviderError: nothing was wrong at the provider, the
    request should not have been made. The API turns this into a 422 and the repair
    worker skips the connector instead of counting it as a provider failure.
    """


def callback_urls(connector: Connector) -> tuple[str, str | None]:
    """Return the (notification, lifecycle) URLs this connector's provider must call."""

    base = (settings.CONNECTOR_WEBHOOK_BASE_URL or "").rstrip("/")
    if not base:
        raise WebhookConfigurationError(
            "Configure CONNECTOR_WEBHOOK_BASE_URL before enabling update notifications"
        )
    callback = f"{base}/api/v1/connectors/webhooks/{connector.system.replace('_', '-')}"
    # Only Graph sends lifecycle events (subscriptionRemoved, reauthorizationRequired).
    lifecycle = f"{callback}/lifecycle" if connector.system == "sharepoint" else None
    return callback, lifecycle


async def ensure_webhook_subscriptions(
    db: AsyncSession,
    connector: Connector,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Give every selected scope a live push subscription.

    Idempotent by default: a scope whose subscription is active and expires beyond
    RENEWAL_HORIZON is left alone, so the repair worker can call this on every
    connector every ten minutes for the cost of one query. ``force`` replaces even a
    healthy subscription, which is what the admin endpoint means by "subscribe".

    Does not commit. The caller owns the transaction, because the admin endpoint also
    writes connector config in the same one.
    """

    if connector.system == "local_folder":
        raise WebhookConfigurationError("Local folders do not support webhooks")
    callback, lifecycle = callback_urls(connector)
    scopes = (
        (
            await db.execute(
                select(SourceScope).where(
                    SourceScope.connector_id == connector.id,
                    SourceScope.selected.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    if not scopes:
        raise WebhookConfigurationError(
            "Select at least one folder or drive before enabling update notifications"
        )

    adapter = adapter_for(connector)
    await ensure_connector_authorized(db, connector)
    deadline = datetime.utcnow() + RENEWAL_HORIZON
    created: list[dict[str, Any]] = []

    for scope in scopes:
        existing = (
            (
                await db.execute(
                    select(WebhookSubscription).where(
                        WebhookSubscription.connector_id == connector.id,
                        WebhookSubscription.scope_id == scope.id,
                        WebhookSubscription.active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        healthy = [
            subscription
            for subscription in existing
            if subscription.expires_at is not None and subscription.expires_at > deadline
        ]
        if healthy and not force:
            continue

        # Retire the old subscription at the PROVIDER before asking for a new one.
        # Graph rejects a second subscription with the same changeType and resource
        # with 409 Conflict, so skipping this makes every repair attempt fail exactly
        # when the subscription is still half-alive. Best effort: a subscription that
        # is already gone is the state we want, and a provider that cannot delete
        # (Drive channels stop by resourceId, which we may not hold) still lets the
        # replacement through — duplicate Drive notifications are deduplicated by the
        # notification inbox and the old channel expires on its own.
        for old in existing:
            try:
                await adapter.delete_webhook(old.provider_subscription_id, old.resource)
            except Exception:  # noqa: BLE001 - provider cleanup must never block repair
                pass
            old.active = False

        result = await adapter.create_webhook(
            {"external_scope_id": scope.external_scope_id, "config": scope.config_json or {}},
            callback,
            lifecycle,
        )
        db.add(
            WebhookSubscription(
                connector_id=connector.id,
                scope_id=scope.id,
                provider_subscription_id=result["subscription_id"],
                verification_token_hash=hashlib.sha256(
                    result["client_state"].encode("utf-8")
                ).hexdigest(),
                resource=result.get("resource"),
                lifecycle_notification_url=result.get("lifecycle_notification_url"),
                expires_at=result.get("expires_at"),
                active=True,
            )
        )
        created.append(
            {
                "scope_id": str(scope.id),
                "subscription_id": result["subscription_id"],
                "expires_at": result.get("expires_at"),
            }
        )

    return created


async def renew_due_subscriptions(db: AsyncSession) -> None:
    """Extend every subscription that lapses inside the horizon.

    A lapsed subscription is not an error anyone sees: the provider simply stops
    calling, the connector keeps reporting ``on_update``, and the corpus goes stale
    until a person notices. Renewal is therefore attempted well before expiry, and a
    subscription that cannot be renewed is deactivated rather than retried forever —
    :func:`repair_webhook_subscriptions` is what puts a live one back in its place.
    """

    horizon = datetime.utcnow() + RENEWAL_HORIZON
    due = (
        (
            await db.execute(
                select(WebhookSubscription).where(
                    WebhookSubscription.active.is_(True),
                    WebhookSubscription.expires_at.isnot(None),
                    WebhookSubscription.expires_at <= horizon,
                )
            )
        )
        .scalars()
        .all()
    )
    for subscription in due:
        connector = await db.get(Connector, subscription.connector_id)
        if connector is None or connector.status not in ("active", "error"):
            continue
        try:
            renewed = await adapter_for(connector).renew_webhook(
                subscription.provider_subscription_id
            )
        except ConnectorProviderError as exc:
            # Gone at the provider, or the token no longer grants it. Deactivate: the
            # polling loop still covers this connector, so the cost is latency, not
            # lost content, and the repair pass will try to subscribe again.
            logger.warning(
                "Webhook subscription renewal failed",
                connector_id=str(connector.id),
                subscription_id=subscription.provider_subscription_id,
                error=str(exc),
            )
            subscription.active = False
            continue
        if renewed is None:
            # Provider cannot extend in place (Google Drive channels). Retire the row so
            # it does not claim a liveness it no longer has.
            subscription.active = False
            continue
        subscription.expires_at = renewed
    await db.commit()


async def repair_webhook_subscriptions(db: AsyncSession, *, set_context=None) -> None:
    """Recreate subscriptions that renewal could not save.

    Renewal alone cannot keep a connector real-time. A Google Drive channel cannot be
    extended at all, and a Graph subscription deleted at the tenant or stripped of its
    authorization comes back as an error. In both cases the row is deactivated — and
    until this existed, nothing ever created a replacement. Real-time sync lasted one
    Drive channel lifetime and then decayed into polling with the UI still showing
    ``on_update``.

    Only connectors whose admin switched webhooks on are touched. One that still cannot
    subscribe is marked degraded rather than retried into a hot loop.

    ``set_context`` re-establishes the tenant/database context after a rollback, which
    discards it. The caller passes its own, because the API and the worker set it in
    different ways.
    """

    # Ids, not instances: a rollback below expires every loaded object, and refreshing an
    # expired attribute from inside an async loop is exactly the kind of implicit IO that
    # raises rather than reloads. Each connector is fetched fresh instead.
    connector_ids = (
        (
            await db.execute(
                select(Connector.id).where(
                    Connector.system.in_(["sharepoint", "google_drive"]),
                    Connector.status.in_(["active", "error"]),
                )
            )
        )
        .scalars()
        .all()
    )
    for connector_id in connector_ids:
        connector = await db.get(Connector, connector_id)
        if connector is None:
            continue
        config = connector.config_json or {}
        if not config.get("webhook_enabled"):
            continue
        try:
            created = await ensure_webhook_subscriptions(db, connector)
        except (ConnectorProviderError, WebhookConfigurationError) as exc:
            logger.warning(
                "Webhook subscription repair failed",
                connector_id=str(connector_id),
                error=str(exc),
            )
            await db.rollback()
            if set_context is not None:
                await set_context(db)
            connector = await db.get(Connector, connector_id)
            if connector is None:
                continue
            connector.config_json = {
                **(connector.config_json or {}),
                "webhook_degraded_at": config.get("webhook_degraded_at")
                or datetime.utcnow().isoformat(),
            }
            await db.commit()
            continue
        if created:
            logger.info(
                "Webhook subscriptions recreated",
                connector_id=str(connector_id),
                count=len(created),
            )
        if config.get("webhook_degraded_at"):
            connector.config_json = {
                key: value
                for key, value in (connector.config_json or {}).items()
                if key != "webhook_degraded_at"
            }
        await db.commit()
