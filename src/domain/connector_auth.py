"""Provider authentication for unattended connector workers."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.secrets import encrypt_secret
from src.domain.connector_adapters import ConnectorProviderError, adapter_for
from src.domain.connector_providers import is_microsoft_graph
from src.models.ops import Connector

# Refresh this far before the provider's stated expiry. A sync is a long walk; a token
# that is valid when the walk starts and expires halfway through fails every remaining
# item, and those failures look like provider errors rather than an expired credential.
_EXPIRY_SKEW = timedelta(minutes=5)


def _auth_mode(connector: Connector) -> str:
    """Return the OAuth grant this connector renews with.

    Only Microsoft offers an application (client-credentials) mode, and it covers
    every Graph provider — SharePoint and OneDrive share one Entra application
    registration. Every other provider is delegated: a refresh token obtained
    once by an admin.
    """

    if not is_microsoft_graph(connector.system):
        return "delegated"
    mode = settings.microsoft_connector_auth_mode
    if mode not in {"delegated", "application"}:
        raise ConnectorProviderError(
            "MICROSOFT_CONNECTOR_AUTH_MODE must be delegated or application",
            retryable=False,
            code="invalid_auth_mode",
        )
    return mode


async def ensure_connector_authorized(db: AsyncSession, connector: Connector) -> None:
    """Refresh delegated tokens or obtain an application token before provider work.

    Provider-independent by construction. This used to return early for anything that
    was not SharePoint, which meant a Google Drive connector was never refreshed at
    all: its access token is good for one hour, after which every Drive call came back
    401 — reported as non-retryable — and the connector sat in ``error`` until a human
    re-ran the OAuth flow by hand. The refresh token was there the whole time.
    """

    if connector.system == "local_folder":
        return
    mode = _auth_mode(connector)
    if (
        connector.oauth_access_token
        and connector.oauth_expires_at
        and connector.oauth_expires_at > datetime.utcnow() + _EXPIRY_SKEW
    ):
        return
    if mode == "delegated" and not connector.oauth_refresh_token:
        if connector.oauth_access_token:
            # No refresh token was ever issued (a consent that omitted offline access).
            # The stored token may still work; let the provider be the judge rather than
            # failing a sync that would have succeeded.
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
    # Google only returns a refresh token on the FIRST consent; every refresh response
    # after that omits it. Keeping the stored one is what makes the connector survive
    # past its first hour.
    connector.oauth_refresh_token = (
        encrypt_secret(tokens.get("refresh_token")) or connector.oauth_refresh_token
    )
    connector.oauth_expires_at = datetime.utcnow() + timedelta(seconds=int(tokens.get("expires_in", 3600)))
    if mode == "application":
        connector.oauth_subject = "application"
    await db.commit()
