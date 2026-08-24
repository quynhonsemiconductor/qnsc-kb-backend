"""Provider authentication for unattended connector workers."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.secrets import encrypt_secret
from src.domain.connector_adapters import ConnectorProviderError, adapter_for
from src.models.ops import Connector


async def ensure_connector_authorized(db: AsyncSession, connector: Connector) -> None:
    """Refresh delegated tokens or obtain an application token before Graph work."""

    if connector.system != "sharepoint":
        return
    mode = settings.microsoft_connector_auth_mode
    if mode not in {"delegated", "application"}:
        raise ConnectorProviderError("MICROSOFT_CONNECTOR_AUTH_MODE must be delegated or application", retryable=False, code="invalid_auth_mode")
    if connector.oauth_access_token and connector.oauth_expires_at and connector.oauth_expires_at > datetime.utcnow() + timedelta(minutes=2):
        return
    if mode == "delegated" and not connector.oauth_refresh_token:
        if connector.oauth_access_token:
            return
        raise ConnectorProviderError("Connector is not authorized", retryable=False, code="not_authorized")
    tokens = await adapter_for(connector).refresh_token()
    access_token = encrypt_secret(tokens.get("access_token"))
    if not access_token:
        # encrypt_secret returns None for an absent value, so assigning it straight
        # through would clear a working token and leave the connector unauthorized on
        # the strength of one malformed provider response.
        raise ConnectorProviderError(
            "Provider returned no access token", retryable=True, code="no_access_token"
        )
    connector.oauth_access_token = access_token
    connector.oauth_refresh_token = (
        encrypt_secret(tokens.get("refresh_token")) or connector.oauth_refresh_token
    )
    connector.oauth_expires_at = datetime.utcnow() + timedelta(seconds=int(tokens.get("expires_in", 3600)))
    if mode == "application":
        connector.oauth_subject = "application"
    await db.commit()
