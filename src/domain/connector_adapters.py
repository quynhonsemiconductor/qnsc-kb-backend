"""Provider adapters for the first-party SharePoint, OneDrive and Google Drive connectors.

Adapters return normalized changes; synchronization, persistence and retry
policy remain in the connector service so additional providers do not create
provider-specific ingestion paths.

Provider IDENTITY questions — is this Graph, which cursor, which identity
provider, which ACL key — belong in ``domain/connector_providers.py``, not in
string comparisons at call sites.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlsplit

import httpx

from src.core.config import settings
from src.core.secrets import decrypt_secret
from src.models.ops import Connector


@dataclass(frozen=True)
class NormalizedChange:
    external_id: str
    corpus_id: str
    name: str
    state: str
    content_changed: bool
    permissions_changed: bool
    moved: bool
    revision: str | None
    mime_type: str | None
    parent_external_id: str | None
    web_url: str | None
    metadata: dict[str, Any]


class ConnectorProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True, code: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.code = code


class ConnectorAdapter:
    provider = "base"
    allowed_api_hosts: frozenset[str] = frozenset()

    def __init__(self, connector: Connector):
        self.connector = connector

    @property
    def access_token(self) -> str:
        value = decrypt_secret(self.connector.oauth_access_token)
        if not value:
            raise ConnectorProviderError("Connector is not authorized", retryable=False, code="not_authorized")
        return value

    def _validate_provider_url(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.hostname.lower() not in self.allowed_api_hosts:
            raise ConnectorProviderError("Provider returned an untrusted API URL", retryable=False, code="untrusted_url")

    @staticmethod
    def _validate_redirect_url(url: str) -> None:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
            raise ConnectorProviderError("Provider returned an unsafe download redirect", retryable=False, code="unsafe_redirect")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if hostname == "localhost" or hostname.endswith(".localhost") or (address is not None and not address.is_global):
            raise ConnectorProviderError("Provider returned an unsafe download redirect", retryable=False, code="unsafe_redirect")

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=45.0, follow_redirects=False, trust_env=False)

    async def _read_response(self, client: httpx.AsyncClient, method: str, url: str, headers: dict[str, str], **kwargs: Any) -> tuple[int, dict[str, str], bytes]:
        async with client.stream(method, url, headers=headers, **kwargs) as response:
            response_headers = dict(response.headers)
            if response.is_redirect:
                return response.status_code, response_headers, b""
            is_json = "application/json" in response_headers.get("content-type", "").lower()
            max_bytes = settings.MAX_CONNECTOR_API_RESPONSE_BYTES if is_json else settings.MAX_SOURCE_UPLOAD_BYTES
            content_length = response_headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                raise ConnectorProviderError("Provider response exceeds the configured size limit", retryable=False, code="response_too_large")
            body = bytearray()
            async for chunk in response.aiter_bytes(1024 * 1024):
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ConnectorProviderError("Provider response exceeds the configured size limit", retryable=False, code="response_too_large")
            return response.status_code, response_headers, bytes(body)

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any] | list[Any] | bytes:
        self._validate_provider_url(url)
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.access_token}"
        async with self._http_client() as client:
            for attempt in range(4):
                target = url
                request_headers = headers
                status_code, response_headers, body = await self._read_response(client, method, target, request_headers, **kwargs)
                if 300 <= status_code < 400:
                    location = response_headers.get("location")
                    if method.upper() != "GET" or not location:
                        raise ConnectorProviderError("Provider returned an unsupported redirect", retryable=False, code="unsafe_redirect")
                    target = urljoin(target, location)
                    self._validate_redirect_url(target)
                    # Download CDNs do not need an OAuth bearer token. Never
                    # carry it to a provider-controlled redirect destination.
                    request_headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
                    status_code, response_headers, body = await self._read_response(client, method, target, request_headers, **kwargs)
                    if 300 <= status_code < 400:
                        raise ConnectorProviderError("Provider returned too many redirects", retryable=False, code="unsafe_redirect")
                if status_code in {429, 500, 502, 503, 504}:
                    retry_after = response_headers.get("retry-after")
                    # The provider chooses this number, so it is untrusted input: a
                    # misconfigured or hostile tenant answering `Retry-After: 86400`
                    # parks a sync worker for a day. Capped at the top of our own
                    # backoff curve; past the attempt ceiling the durable queue
                    # reschedules the whole request anyway.
                    delay = min(60.0, float(retry_after)) if retry_after and retry_after.isdigit() else min(16, 2 ** attempt) + secrets.randbelow(500) / 1000
                    if attempt == 3:
                        raise ConnectorProviderError(f"Provider retry limit reached ({status_code})", code=str(status_code))
                    await asyncio.sleep(delay)
                    continue
                if status_code in {401, 403}:
                    raise ConnectorProviderError(f"Provider authorization failed ({status_code})", retryable=False, code=str(status_code))
                if status_code >= 400:
                    detail = ""
                    if "application/json" in response_headers.get("content-type", "").lower():
                        try:
                            payload = json.loads(body)
                            error_payload = payload.get("error") if isinstance(payload, dict) else None
                            detail = str((error_payload or {}).get("message") or (error_payload or {}).get("code") or "")[:240]
                        except (TypeError, ValueError, json.JSONDecodeError):
                            detail = ""
                    suffix = f": {detail}" if detail else ""
                    raise ConnectorProviderError(f"Provider request failed ({status_code}){suffix}", code=str(status_code))
                if "application/json" in response_headers.get("content-type", "").lower():
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise ConnectorProviderError("Provider returned invalid JSON", retryable=False, code="invalid_json") from exc
                return body
        raise ConnectorProviderError("Provider request failed")

    async def exchange_code(self, code: str) -> dict[str, Any]:
        raise NotImplementedError

    async def refresh_token(self) -> dict[str, Any]:
        raise NotImplementedError

    async def discover_scopes(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    # Whether calling incremental_changes with NO cursor enumerates the entire scope.
    # Only then may "absent from the result" be read as "gone at the provider", which is
    # what the reconciliation sweep in cloud_sync does. Default False: a provider that
    # cannot enumerate must never have its corpus deleted for being unmentioned.
    full_walk_is_authoritative = False

    async def incremental_changes(self, scope: dict[str, Any], cursor: str | None) -> tuple[list[NormalizedChange], str | None]:
        raise NotImplementedError

    async def permissions(self, change: NormalizedChange) -> list[dict[str, str]]:
        raise NotImplementedError

    async def download(self, change: NormalizedChange) -> bytes:
        raise NotImplementedError

    async def create_webhook(
        self,
        scope: dict[str, Any],
        callback_url: str,
        lifecycle_callback_url: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def renew_webhook(self, provider_subscription_id: str) -> datetime | None:
        """Push the subscription's expiry out and return the new one.

        Returning None means this provider cannot extend a subscription in place and the
        caller should treat the subscription as expired. Raising is reserved for a
        provider that CAN renew and failed to.
        """
        return None

    async def delete_webhook(self, provider_subscription_id: str, resource: str | None = None) -> None:
        """Retire a subscription at the provider.

        Called before a replacement is created, because a provider may refuse a second
        subscription covering the same resource. Best effort by contract: the caller
        treats any failure as "already gone" rather than as a reason not to resubscribe.
        """
        return None


class MicrosoftGraphAdapter(ConnectorAdapter):
    """Shared Microsoft Graph transport for SharePoint and OneDrive.

    Both providers are drives behind the same Graph endpoints: identical delta
    semantics, identical permission entries, identical subscription lifecycle,
    identical Entra application registration. Only scope DISCOVERY differs —
    SharePoint enumerates sites and their libraries, OneDrive enumerates user
    drives — so that is the single method subclasses override.
    """

    graph = "https://graph.microsoft.com/v1.0"
    allowed_api_hosts = frozenset({"graph.microsoft.com"})
    # A /delta called without a token replays the drive (or folder subtree) from empty
    # and pages to the end, so the result is the full current state of the scope.
    full_walk_is_authoritative = True
    #: Delegated consent scopes. Declared per provider because OneDrive has no use
    #: for Sites.Read.All, and requesting permission a provider does not need is
    #: exactly how least privilege gets lost.
    oauth_scope = "offline_access openid profile User.Read Files.Read.All Sites.Read.All"
    #: The (drive-level, folder-level) ``scope_type`` values this provider emits.
    scope_types: tuple[str, str] = ("sharepoint_library", "sharepoint_folder")

    def oauth_url(self, state: str) -> str:
        params = {
            "client_id": settings.MICROSOFT_CLIENT_ID or "",
            "response_type": "code",
            "redirect_uri": settings.MICROSOFT_REDIRECT_URI or "",
            "response_mode": "query",
            "scope": self.oauth_scope,
            "state": state,
        }
        return f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/authorize?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/token",
                data={
                    "client_id": settings.MICROSOFT_CLIENT_ID,
                    "client_secret": settings.MICROSOFT_CLIENT_SECRET,
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": settings.MICROSOFT_REDIRECT_URI,
                    "scope": self.oauth_scope,
                },
            )
            if response.status_code >= 400:
                raise ConnectorProviderError("Microsoft OAuth exchange failed", retryable=False, code=str(response.status_code))
            return response.json()

    async def refresh_token(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if settings.microsoft_connector_auth_mode == "application":
                response = await client.post(
                    f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/token",
                    data={
                        "client_id": settings.MICROSOFT_CLIENT_ID,
                        "client_secret": settings.MICROSOFT_CLIENT_SECRET,
                        "grant_type": "client_credentials",
                        "scope": settings.MICROSOFT_GRAPH_SCOPE,
                    },
                )
                if response.status_code >= 400:
                    raise ConnectorProviderError("Microsoft application token request failed", retryable=False, code=str(response.status_code))
                return response.json()
            refresh = decrypt_secret(self.connector.oauth_refresh_token)
            if not refresh:
                raise ConnectorProviderError("Microsoft refresh token is missing", retryable=False, code="not_authorized")
            response = await client.post(
                f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/token",
                data={"client_id": settings.MICROSOFT_CLIENT_ID, "client_secret": settings.MICROSOFT_CLIENT_SECRET, "grant_type": "refresh_token", "refresh_token": refresh, "scope": self.oauth_scope},
            )
            if response.status_code >= 400:
                raise ConnectorProviderError("Microsoft token refresh failed", retryable=False, code=str(response.status_code))
            return response.json()

    @staticmethod
    def _configured_ids(raw: str) -> list[str]:
        """Split a comma-separated allowlist setting, dropping blanks."""
        return [value.strip() for value in (raw or "").split(",") if value.strip()]

    async def _folder_scopes(
        self, drive_id: str, *, base_config: dict[str, Any], location: str
    ) -> list[dict[str, Any]]:
        """The top-level folders of one drive, as individually selectable scopes.

        Folder scopes exist so an administrator can ingest one folder instead of a
        whole library. ``external_scope_id`` stays ``drive:folder`` because the sync
        path needs both halves to call ``/drives/{drive}/items/{folder}/delta``.
        """

        folders = await self._request(
            "GET",
            f"{self.graph}/drives/{quote(drive_id, safe='')}/root/children?$select=id,name,folder,webUrl",
        )
        result: list[dict[str, Any]] = []
        for folder in folders.get("value", []):  # type: ignore[union-attr]
            if not folder.get("folder"):
                continue
            folder_name = str(folder.get("name") or folder.get("id"))
            label = f"{location} / {folder_name}"
            result.append({
                "external_scope_id": f"{drive_id}:{folder['id']}",
                "scope_type": self.scope_types[1],
                "display_name": label,
                "config": {
                    **base_config,
                    "folder_id": folder["id"],
                    "web_url": folder.get("webUrl") or base_config.get("web_url"),
                    "location_label": label,
                },
            })
        return result

    async def create_webhook(
        self,
        scope: dict[str, Any],
        callback_url: str,
        lifecycle_callback_url: str | None = None,
    ) -> dict[str, Any]:
        client_state = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=settings.MICROSOFT_GRAPH_SUBSCRIPTION_MINUTES)
        payload: dict[str, Any] = {
            "changeType": "updated",
            "notificationUrl": callback_url,
            "resource": f"/drives/{scope['config'].get('drive_id', scope['external_scope_id'])}/root",
            "expirationDateTime": expires_at.isoformat(timespec="seconds") + "Z",
            "clientState": client_state,
        }
        if lifecycle_callback_url:
            payload["lifecycleNotificationUrl"] = lifecycle_callback_url
        # Without this header Graph notifies on CONTENT changes only. A file whose
        # sharing was revoked never wakes the connector, so a user who lost access at
        # SharePoint keeps seeing the document in the KB until the next reconciliation
        # pass — six hours by default. SharePoint and OneDrive for Business support the
        # header; consumer OneDrive ignores it, which is why it is safe to always send.
        result = await self._request(
            "POST",
            f"{self.graph}/subscriptions",
            json=payload,
            headers={"Prefer": "includesecuritywebhooks"},
        )
        return {
            "subscription_id": result["id"],
            "client_state": client_state,
            "expires_at": expires_at,
            "resource": payload["resource"],
            "lifecycle_notification_url": lifecycle_callback_url,
        }  # type: ignore[index]

    async def renew_webhook(self, provider_subscription_id: str) -> datetime | None:
        # Graph caps a drive subscription at roughly 30 days but rejects anything beyond
        # its own maximum, so this asks for the same hour create_webhook does and leans on
        # the renewal task running far more often than that.
        expires_at = datetime.utcnow() + timedelta(minutes=settings.MICROSOFT_GRAPH_SUBSCRIPTION_MINUTES)
        await self._request(
            "PATCH",
            f"{self.graph}/subscriptions/{provider_subscription_id}",
            json={"expirationDateTime": expires_at.isoformat(timespec="seconds") + "Z"},
        )
        return expires_at

    async def delete_webhook(self, provider_subscription_id: str, resource: str | None = None) -> None:
        # Graph refuses a second subscription with the same changeType and resource
        # (409 Conflict), so a replacement is only possible once this one is gone.
        await self._request("DELETE", f"{self.graph}/subscriptions/{provider_subscription_id}")

    async def incremental_changes(self, scope: dict[str, Any], cursor: str | None) -> tuple[list[NormalizedChange], str | None]:
        drive_id = scope["config"].get("drive_id", scope["external_scope_id"])
        root = f"items/{scope['config']['folder_id']}" if scope["config"].get("folder_id") else "root"
        url = cursor or f"{self.graph}/drives/{drive_id}/{root}/delta?$select=id,name,file,folder,parentReference,eTag,cTag,webUrl,deleted,lastModifiedDateTime"
        changes: list[NormalizedChange] = []
        next_cursor: str | None = None
        while url:
            page = await self._request("GET", url)
            for item in page.get("value", []):  # type: ignore[union-attr]
                deleted = bool(item.get("deleted"))
                file_info = item.get("file") or {}
                changes.append(NormalizedChange(
                    external_id=item["id"], corpus_id=drive_id, name=item.get("name", item["id"]),
                    state="deleted" if deleted else "active", content_changed=bool(file_info),
                    permissions_changed=False, moved=bool(item.get("parentReference")),
                    revision=item.get("eTag") or item.get("cTag"), mime_type=file_info.get("mimeType"),
                    parent_external_id=(item.get("parentReference") or {}).get("id"), web_url=item.get("webUrl"), metadata=item,
                ))
            url = page.get("@odata.nextLink")  # type: ignore[union-attr]
            next_cursor = page.get("@odata.deltaLink", next_cursor)  # type: ignore[union-attr]
        return changes, next_cursor

    async def permissions(self, change: NormalizedChange) -> list[dict[str, str]]:
        # A connector may select several drives; the normalized corpus is the
        # authoritative drive for this item, not connector-level config.
        drive_id = change.corpus_id
        data = await self._request("GET", f"{self.graph}/drives/{drive_id}/items/{change.external_id}/permissions")
        result: list[dict[str, str]] = []
        for item in data.get("value", []):  # type: ignore[union-attr]
            result.extend(sharepoint_permission_principals(item))
        return _dedupe_principals(result)

    async def download(self, change: NormalizedChange) -> bytes:
        drive_id = change.corpus_id
        return await self._request("GET", f"{self.graph}/drives/{drive_id}/items/{change.external_id}/content")  # type: ignore[return-value]


class SharePointAdapter(MicrosoftGraphAdapter):
    provider = "sharepoint"
    scope_types = ("sharepoint_library", "sharepoint_folder")

    async def discover_scopes(self) -> list[dict[str, Any]]:
        # ``/drives`` often returns only a generic library such as
        # "Documents". Resolve SharePoint sites first so reviewers can see the
        # real site/library/folder location instead of guessing where it lives.
        auth_mode = settings.microsoft_connector_auth_mode
        configured_site_ids = self._configured_ids(settings.MICROSOFT_SHAREPOINT_SITE_IDS)
        if auth_mode == "application" and not configured_site_ids:
            raise ConnectorProviderError(
                "Application mode requires MICROSOFT_SHAREPOINT_SITE_IDS",
                retryable=False,
                code="scope_allowlist_required",
            )
        if configured_site_ids:
            sites: list[Any] = [
                await self._request(
                    "GET",
                    f"{self.graph}/sites/{quote(site_id, safe='')}?$select=id,name,displayName,webUrl",
                )
                for site_id in configured_site_ids
            ]
        else:
            sites_data = await self._request("GET", f"{self.graph}/sites?search=*&$top=50&$select=id,name,displayName,webUrl")
            sites = sites_data.get("value", []) if isinstance(sites_data, dict) else []
        result: list[dict[str, Any]] = []

        for site in sites:
            site_id = str(site.get("id") or "")
            if not site_id:
                continue
            site_name = str(site.get("displayName") or site.get("name") or site_id)
            site_url = site.get("webUrl")
            drives_data = await self._request(
                "GET",
                f"{self.graph}/sites/{quote(site_id, safe='')}/drives?$select=id,name,driveType,webUrl",
            )
            for drive in drives_data.get("value", []):  # type: ignore[union-attr]
                drive_id = str(drive.get("id") or "")
                if not drive_id:
                    continue
                drive_name = str(drive.get("name") or drive_id)
                location = f"{site_name} / {drive_name}"
                drive_config = {
                    "site_id": site_id,
                    "site_name": site_name,
                    "site_url": site_url,
                    "drive_id": drive_id,
                    "drive_name": drive_name,
                    "web_url": drive.get("webUrl") or site_url,
                    "location_label": location,
                }
                result.append({
                    "external_scope_id": drive_id,
                    "scope_type": self.scope_types[0],
                    "display_name": location,
                    "config": drive_config,
                })
                result.extend(
                    await self._folder_scopes(drive_id, base_config=drive_config, location=location)
                )

        if result:
            return result

        # Keep a fallback for tenants where site search is disabled but the
        # delegated token can still enumerate drives.
        data = await self._request("GET", f"{self.graph}/drives?$select=id,name,driveType,webUrl")
        for item in data.get("value", []):  # type: ignore[union-attr]
            drive_id = str(item.get("id") or "")
            drive_name = str(item.get("name") or drive_id)
            location = f"Available SharePoint library / {drive_name}"
            drive_config = {"drive_id": drive_id, "drive_name": drive_name, "web_url": item.get("webUrl"), "location_label": location}
            result.append({"external_scope_id": drive_id, "scope_type": self.scope_types[0], "display_name": location, "config": drive_config})
            result.extend(
                await self._folder_scopes(drive_id, base_config=drive_config, location=location)
            )
        return result


class OneDriveAdapter(MicrosoftGraphAdapter):
    """Per-user OneDrive for Business drives.

    Sites.Read.All is deliberately absent from the consent request: this
    provider never reads a SharePoint site, and asking for the permission
    anyway would make a OneDrive-only integration a tenant-wide site reader.

    Discovery is allowlist-driven in BOTH auth modes. ``/me/drive`` is the only
    self-service alternative and it resolves to the single account that clicked
    consent, which is not a knowledge-base source anyone can administer. When
    the allowlist is empty in delegated mode the signed-in user's own drive is
    the honest interpretation, so it is offered explicitly rather than pretending
    a tenant-wide enumeration happened.
    """

    provider = "onedrive"
    oauth_scope = "offline_access openid profile User.Read Files.Read.All"
    scope_types = ("onedrive_drive", "onedrive_folder")

    async def _user_drive(self, user_id: str) -> dict[str, Any] | None:
        """One user's drive, or None when the account has none provisioned.

        A licensed account with OneDrive never provisioned answers 404, and one
        such account in the allowlist must not abort discovery for every other.
        """
        try:
            drive = await self._request(
                "GET",
                f"{self.graph}/users/{quote(user_id, safe='')}/drive?$select=id,name,driveType,webUrl",
            )
        except ConnectorProviderError as exc:
            if exc.code == "404":
                return None
            raise
        return drive if isinstance(drive, dict) else None

    async def discover_scopes(self) -> list[dict[str, Any]]:
        configured_user_ids = self._configured_ids(settings.MICROSOFT_ONEDRIVE_USER_IDS)
        if settings.microsoft_connector_auth_mode == "application" and not configured_user_ids:
            raise ConnectorProviderError(
                "Application mode requires MICROSOFT_ONEDRIVE_USER_IDS",
                retryable=False,
                code="scope_allowlist_required",
            )
        result: list[dict[str, Any]] = []
        for user_id in configured_user_ids:
            drive = await self._user_drive(user_id)
            drive_id = str((drive or {}).get("id") or "")
            if not drive_id:
                continue
            drive_name = str(drive.get("name") or "OneDrive")  # type: ignore[union-attr]
            location = f"OneDrive / {user_id} / {drive_name}"
            drive_config = {
                "drive_id": drive_id,
                "drive_name": drive_name,
                "user_id": user_id,
                "web_url": drive.get("webUrl"),  # type: ignore[union-attr]
                "location_label": location,
            }
            result.append({
                "external_scope_id": drive_id,
                "scope_type": self.scope_types[0],
                "display_name": location,
                "config": drive_config,
            })
            result.extend(
                await self._folder_scopes(drive_id, base_config=drive_config, location=location)
            )
        if result or configured_user_ids:
            return result

        drive = await self._request("GET", f"{self.graph}/me/drive?$select=id,name,driveType,webUrl,owner")
        drive_id = str(drive.get("id") or "") if isinstance(drive, dict) else ""
        if not drive_id:
            return result
        owner = ((drive.get("owner") or {}).get("user") or {}) if isinstance(drive, dict) else {}
        owner_label = str(owner.get("email") or owner.get("displayName") or "signed-in user")
        drive_name = str(drive.get("name") or "OneDrive")
        location = f"OneDrive / {owner_label} / {drive_name}"
        drive_config = {
            "drive_id": drive_id,
            "drive_name": drive_name,
            "owner_label": owner_label,
            "web_url": drive.get("webUrl"),
            "location_label": location,
        }
        result.append({
            "external_scope_id": drive_id,
            "scope_type": self.scope_types[0],
            "display_name": location,
            "config": drive_config,
        })
        result.extend(
            await self._folder_scopes(drive_id, base_config=drive_config, location=location)
        )
        return result


class GoogleDriveAdapter(ConnectorAdapter):
    provider = "google_drive"
    api = "https://www.googleapis.com/drive/v3"
    allowed_api_hosts = frozenset({"www.googleapis.com"})
    # DELIBERATELY False. With no cursor this adapter asks for startPageToken and pages
    # from there — changes from NOW ON — so a cursor-less call returns almost nothing.
    # It is not an enumeration, and treating it as one would delete the whole indexed
    # corpus of every Drive scope on each reconciliation pass. A real full walk here
    # means files.list, which this adapter does not implement.
    full_walk_is_authoritative = False

    def oauth_url(self, state: str) -> str:
        params = {"client_id": settings.GOOGLE_CLIENT_ID or "", "redirect_uri": settings.GOOGLE_REDIRECT_URI or "", "response_type": "code", "access_type": "offline", "prompt": "consent", "scope": "https://www.googleapis.com/auth/drive.readonly openid email profile", "state": state}
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://oauth2.googleapis.com/token", data={"client_id": settings.GOOGLE_CLIENT_ID, "client_secret": settings.GOOGLE_CLIENT_SECRET, "code": code, "grant_type": "authorization_code", "redirect_uri": settings.GOOGLE_REDIRECT_URI})
            if response.status_code >= 400:
                raise ConnectorProviderError("Google OAuth exchange failed", retryable=False, code=str(response.status_code))
            return response.json()

    async def refresh_token(self) -> dict[str, Any]:
        refresh = decrypt_secret(self.connector.oauth_refresh_token)
        if not refresh:
            raise ConnectorProviderError("Google refresh token is missing", retryable=False, code="not_authorized")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://oauth2.googleapis.com/token", data={"client_id": settings.GOOGLE_CLIENT_ID, "client_secret": settings.GOOGLE_CLIENT_SECRET, "grant_type": "refresh_token", "refresh_token": refresh})
            if response.status_code >= 400:
                raise ConnectorProviderError("Google token refresh failed", retryable=False, code=str(response.status_code))
            return response.json()

    async def discover_scopes(self) -> list[dict[str, Any]]:
        drives: list[dict[str, Any]] = []
        drive_page_token: str | None = None
        while True:
            query = {"pageSize": "100", "fields": "nextPageToken,drives(id,name,webViewLink)"}
            if drive_page_token:
                query["pageToken"] = drive_page_token
            page = await self._request("GET", f"{self.api}/drives?{urlencode(query)}")
            drives.extend(page.get("drives", []))  # type: ignore[union-attr]
            drive_page_token = page.get("nextPageToken")  # type: ignore[union-attr]
            if not drive_page_token:
                break

        result = [{"external_scope_id": item["id"], "scope_type": "shared_drive", "display_name": item.get("name", item["id"]), "config": {"drive_id": item["id"], "web_url": item.get("webViewLink")}} for item in drives]
        result.append({"external_scope_id": "user", "scope_type": "drive", "display_name": "My Drive", "config": {"corpus": "user"}})
        for drive in [*drives, {"id": None, "name": "My Drive"}]:
            params: dict[str, str] = {"q": "mimeType = 'application/vnd.google-apps.folder' and trashed = false", "pageSize": "100", "fields": "nextPageToken,files(id,name,parents,webViewLink,driveId)", "includeItemsFromAllDrives": "true", "supportsAllDrives": "true"}
            if drive.get("id"):
                params.update({"corpora": "drive", "driveId": drive["id"]})
            folder_page_token: str | None = None
            while True:
                page_params = {**params, **({"pageToken": folder_page_token} if folder_page_token else {})}
                folders = await self._request("GET", f"{self.api}/files?{urlencode(page_params)}")
                for folder in folders.get("files", []):  # type: ignore[union-attr]
                    drive_id = drive.get("id") or "user"
                    result.append({"external_scope_id": f"{drive_id}:{folder['id']}", "scope_type": "folder", "display_name": f"{drive.get('name', 'My Drive')} / {folder.get('name', folder['id'])}", "config": {"drive_id": drive.get("id"), "folder_id": folder["id"], "corpus": drive_id, "web_url": folder.get("webViewLink")}})
                folder_page_token = folders.get("nextPageToken")  # type: ignore[union-attr]
                if not folder_page_token:
                    break
        return result

    async def create_webhook(
        self,
        scope: dict[str, Any],
        callback_url: str,
        lifecycle_callback_url: str | None = None,
    ) -> dict[str, Any]:
        config = scope["config"]
        start = await self._request("GET", f"{self.api}/changes/startPageToken" + (f"?driveId={config['drive_id']}&supportsAllDrives=true" if config.get("drive_id") else ""))
        channel_id = str(uuid.uuid4())
        client_state = secrets.token_urlsafe(32)
        result = await self._request("POST", f"{self.api}/changes/watch", params={"pageToken": start["startPageToken"], "supportsAllDrives": "true"}, json={"id": channel_id, "type": "web_hook", "address": callback_url, "token": client_state})
        expiration = result.get("expiration")
        expires_at = datetime.utcfromtimestamp(int(expiration) / 1000) if expiration else datetime.utcnow() + timedelta(days=1)
        # Google sends the channel id in X-Goog-Channel-ID; keep that id as
        # our subscription key so the webhook can resolve the connector. resourceId is
        # stored because channels.stop needs BOTH ids — without it a superseded channel
        # keeps delivering until it expires on its own.
        return {"subscription_id": channel_id, "client_state": client_state, "expires_at": expires_at, "resource": result.get("resourceId")}  # type: ignore[union-attr]

    async def renew_webhook(self, provider_subscription_id: str) -> datetime | None:
        # A Drive channel cannot be extended: changes.watch mints a new one and stops
        # the old. Returning None tells the worker this subscription is spent, and the
        # repair pass in webhook_subscriptions creates its replacement.
        return None

    async def delete_webhook(self, provider_subscription_id: str, resource: str | None = None) -> None:
        if not resource:
            # Pre-existing rows predate storing resourceId. Nothing can stop the channel;
            # it lapses within a day and the notification inbox deduplicates until then.
            return None
        await self._request("POST", f"{self.api}/channels/stop", json={"id": provider_subscription_id, "resourceId": resource})

    async def _parents_of(self, file_id: str, cache: dict[str, list[str]]) -> list[str]:
        """Return a file's parents, memoized for the lifetime of one walk."""

        if file_id in cache:
            return cache[file_id]
        try:
            data = await self._request(
                "GET",
                f"{self.api}/files/{quote(file_id, safe='')}?supportsAllDrives=true&fields=id,parents",
            )
        except ConnectorProviderError:
            # An ancestor we cannot read is an ancestor we cannot claim the file sits
            # under. Fail closed: the item is left out of a folder scope rather than
            # pulled into one it may not belong to.
            data = {}
        parents = [str(item) for item in (data.get("parents") or [])] if isinstance(data, dict) else []
        cache[file_id] = parents
        return parents

    async def _within_folder(
        self,
        parents: list[str],
        folder_id: str,
        cache: dict[str, list[str]],
        depth: int = 0,
    ) -> bool:
        """Whether any ancestor chain from ``parents`` reaches ``folder_id``.

        Drive's changes feed reports only DIRECT parents, so testing membership with
        ``folder_id in parents`` matched a selected folder's immediate children and
        nothing else. Everything one level deeper — the usual shape of a real shared
        folder — was silently dropped, and the admin who selected the folder saw a
        fraction of it appear in the KB with no error anywhere to explain the rest.
        """

        # Drive nesting is shallow in practice; the bound is a cycle guard, not a policy.
        if depth > 20 or not parents:
            return False
        if folder_id in parents:
            return True
        for parent in parents:
            if await self._within_folder(
                await self._parents_of(parent, cache), folder_id, cache, depth + 1
            ):
                return True
        return False

    async def incremental_changes(self, scope: dict[str, Any], cursor: str | None) -> tuple[list[NormalizedChange], str | None]:
        config = scope["config"]
        params = {"pageToken": cursor or "", "pageSize": "100", "includeRemoved": "true", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true", "fields": "nextPageToken,newStartPageToken,changes(fileId,removed,file(id,name,mimeType,parents,webViewLink,version,md5Checksum,trashed,modifiedTime))"}
        if not cursor:
            start = await self._request("GET", f"{self.api}/changes/startPageToken" + (f"?driveId={config['drive_id']}&supportsAllDrives=true" if config.get("drive_id") else ""))
            params["pageToken"] = start["startPageToken"]  # type: ignore[index]
        changes: list[NormalizedChange] = []
        next_cursor = None
        folder_id = config.get("folder_id")
        ancestry: dict[str, list[str]] = {}
        while params.get("pageToken"):
            query = urlencode(params)
            page = await self._request("GET", f"{self.api}/changes?{query}")
            for entry in page.get("changes", []):  # type: ignore[union-attr]
                file = entry.get("file") or {}
                removed = bool(entry.get("removed")) or bool(file.get("trashed"))
                if folder_id and not removed:
                    if not await self._within_folder(
                        [str(item) for item in (file.get("parents") or [])], folder_id, ancestry
                    ):
                        continue
                # A removal carries no file resource at all, so it has no parents to
                # test — the old filter therefore discarded EVERY deletion under a
                # folder scope and the KB kept serving documents that were gone from
                # Drive. Emit the tombstone and let the sync layer, which knows what it
                # actually tracks for this scope, decide whether it is relevant.
                changes.append(NormalizedChange(
                    external_id=entry["fileId"], corpus_id=config.get("drive_id", "user"), name=file.get("name", entry["fileId"]), state="deleted" if removed else "active", content_changed=bool(file.get("md5Checksum") or file.get("version")), permissions_changed=False, moved=bool(file.get("parents")), revision=str(file.get("version") or file.get("modifiedTime") or "unknown"), mime_type=file.get("mimeType"), parent_external_id=(file.get("parents") or [None])[0], web_url=file.get("webViewLink"), metadata=file,
                ))
            if page.get("nextPageToken"):  # type: ignore[union-attr]
                params["pageToken"] = page["nextPageToken"]  # type: ignore[index]
            else:
                next_cursor = page.get("newStartPageToken")  # type: ignore[union-attr]
                params["pageToken"] = ""
        return changes, next_cursor

    async def permissions(self, change: NormalizedChange) -> list[dict[str, str]]:
        data = await self._request("GET", f"{self.api}/files/{change.external_id}/permissions?supportsAllDrives=true&fields=permissions(id,type,emailAddress,domain,role,displayName)")
        result = []
        for item in data.get("permissions", []):  # type: ignore[union-attr]
            principal_type = item.get("type", "user")
            # Google exposes a stable permission id, but the email is the
            # useful tenant-local identity for reconciling access to an
            # internal account. Keep the id as a fallback for principals that
            # do not expose an address (for example legacy group entries).
            principal_id = item.get("emailAddress") or item.get("id") or item.get("domain")
            if principal_id:
                result.append({"principal_type": principal_type, "principal_id": str(principal_id), "role": str(item.get("role", "reader"))})
        return result

    async def download(self, change: NormalizedChange) -> bytes:
        if change.mime_type and change.mime_type.startswith("application/vnd.google-apps"):
            export_mime = "application/pdf" if change.mime_type.endswith("document") or change.mime_type.endswith("presentation") else "text/csv"
            return await self._request("GET", f"{self.api}/files/{change.external_id}/export?mimeType={export_mime}")  # type: ignore[return-value]
        return await self._request("GET", f"{self.api}/files/{change.external_id}?alt=media&supportsAllDrives=true")  # type: ignore[return-value]


#: Provider name -> adapter. A table rather than an if-chain so registering a
#: provider is one line and cannot half-happen.
_ADAPTERS: dict[str, type[ConnectorAdapter]] = {
    SharePointAdapter.provider: SharePointAdapter,
    OneDriveAdapter.provider: OneDriveAdapter,
    GoogleDriveAdapter.provider: GoogleDriveAdapter,
}


def adapter_for(connector: Connector) -> ConnectorAdapter:
    adapter = _ADAPTERS.get(connector.system)
    if adapter is None:
        raise ConnectorProviderError(f"Unsupported connector provider: {connector.system}", retryable=False, code="unsupported_provider")
    return adapter(connector)


#: The identity kinds a SharePoint identity set can carry, and what each is to us.
_SHAREPOINT_IDENTITY_KINDS = {
    "group": "group",
    "siteGroup": "group",
    "user": "user",
    "siteUser": "user",
    "application": "application",
    "device": "device",
}


def _identity_sets(item: dict) -> list[dict]:
    """Every identity set on one Graph permission entry.

    Graph reports a grant in FOUR shapes, and only the two singular ones were read:

        grantedToV2            one identity set
        grantedTo              one identity set  (legacy)
        grantedToIdentitiesV2  a LIST of identity sets
        grantedToIdentities    a LIST of identity sets  (legacy)

    The plural forms are what SharePoint uses for most library permissions, and they
    were being missed entirely. A permission that only had them fell through to the
    fallback below and was recorded as principal_type "unknown" with the PERMISSION
    entry's own id standing in for a principal id -- an identifier that names nobody, is
    different on every file, and can never be mapped to anything meaningful. That is why
    a single library produced dozens of unmappable GUIDs that blocked every approval.
    """
    sets: list[dict] = []
    for key in ("grantedToV2", "grantedTo"):
        value = item.get(key)
        if isinstance(value, dict):
            sets.append(value)
    for key in ("grantedToIdentitiesV2", "grantedToIdentities"):
        for value in item.get(key) or []:
            if isinstance(value, dict):
                sets.append(value)
    return sets


def sharepoint_permission_principals(item: dict) -> list[dict[str, str]]:
    """Turn one Graph permission entry into the principals it actually grants to.

    A link or an invitation is a real grant and is kept, but named for what it is
    ("link:anonymous") rather than by the permission's own id. That identifier is stable
    across files, so one mapping decision covers a whole library instead of one per
    document -- and it says what is being decided, which a bare GUID never did.
    """
    role = ",".join(item.get("roles") or [])
    principals: list[dict[str, str]] = []

    for identity_set in _identity_sets(item):
        for key, kind in _SHAREPOINT_IDENTITY_KINDS.items():
            identity = identity_set.get(key)
            if not isinstance(identity, dict) or not identity.get("id"):
                continue
            principals.append(
                {
                    "principal_type": kind,
                    "principal_id": str(identity["id"]),
                    "principal_name": str(
                        identity.get("displayName") or identity.get("email") or ""
                    ),
                    "role": role,
                }
            )
    if principals:
        return principals

    # No identity at all: a sharing link, an outstanding invitation, or something Graph
    # has not told us about. These are NOT dropped -- an ACL that looks narrower than it
    # is would let an unsafe approval through -- but they are named usefully.
    link = item.get("link") or {}
    scope = link.get("scope")
    if scope:
        return [
            {
                "principal_type": "link",
                "principal_id": f"link:{scope}",
                "principal_name": f"Sharing link ({scope})",
                "role": role,
            }
        ]
    email = (item.get("invitation") or {}).get("email")
    if email:
        return [
            {
                "principal_type": "user",
                "principal_id": str(email).lower(),
                "principal_name": str(email),
                "role": role,
            }
        ]
    if item.get("id"):
        return [
            {
                "principal_type": "unknown",
                "principal_id": str(item["id"]),
                "principal_name": "",
                "role": role,
            }
        ]
    return []


def _dedupe_principals(principals: list[dict[str, str]]) -> list[dict[str, str]]:
    """One row per principal. The same identity appears in several sets on one item, and
    the storage layer has a uniqueness constraint on (snapshot, type, id)."""
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for principal in principals:
        key = (principal["principal_type"], principal["principal_id"])
        existing = seen.get(key)
        if existing is None:
            seen[key] = dict(principal)
            continue
        # Keep the widest role and any name we managed to learn.
        roles = {part for part in (existing["role"], principal["role"]) if part}
        existing["role"] = ",".join(sorted(roles))
        existing["principal_name"] = existing.get("principal_name") or principal.get(
            "principal_name", ""
        )
    return list(seen.values())
