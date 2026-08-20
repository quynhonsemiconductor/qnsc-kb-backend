"""Provider adapters for the first-party SharePoint and Google Drive MVP.

Adapters return normalized changes; synchronization, persistence and retry
policy remain in the connector service so additional providers do not create
provider-specific ingestion paths.
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
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(16, 2 ** attempt) + secrets.randbelow(500) / 1000
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

    async def incremental_changes(self, scope: dict[str, Any], cursor: str | None) -> tuple[list[NormalizedChange], str | None]:
        raise NotImplementedError

    async def permissions(self, change: NormalizedChange) -> list[dict[str, str]]:
        raise NotImplementedError

    async def download(self, change: NormalizedChange) -> bytes:
        raise NotImplementedError

    async def create_webhook(self, scope: dict[str, Any], callback_url: str) -> dict[str, Any]:
        raise NotImplementedError

    async def renew_webhook(self, provider_subscription_id: str) -> datetime | None:
        """Push the subscription's expiry out and return the new one.

        Returning None means this provider cannot extend a subscription in place and the
        caller should treat the subscription as expired. Raising is reserved for a
        provider that CAN renew and failed to.
        """
        return None


class SharePointAdapter(ConnectorAdapter):
    provider = "sharepoint"
    graph = "https://graph.microsoft.com/v1.0"
    allowed_api_hosts = frozenset({"graph.microsoft.com"})

    def oauth_url(self, state: str) -> str:
        params = {
            "client_id": settings.MICROSOFT_CLIENT_ID or "",
            "response_type": "code",
            "redirect_uri": settings.MICROSOFT_REDIRECT_URI or "",
            "response_mode": "query",
            "scope": "offline_access openid profile User.Read Files.Read.All Sites.Read.All",
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
                    "scope": "offline_access openid profile User.Read Files.Read.All Sites.Read.All",
                },
            )
            if response.status_code >= 400:
                raise ConnectorProviderError("Microsoft OAuth exchange failed", retryable=False, code=str(response.status_code))
            return response.json()

    async def refresh_token(self) -> dict[str, Any]:
        refresh = decrypt_secret(self.connector.oauth_refresh_token)
        if not refresh:
            raise ConnectorProviderError("Microsoft refresh token is missing", retryable=False, code="not_authorized")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/token",
                data={"client_id": settings.MICROSOFT_CLIENT_ID, "client_secret": settings.MICROSOFT_CLIENT_SECRET, "grant_type": "refresh_token", "refresh_token": refresh, "scope": "offline_access openid profile User.Read Files.Read.All Sites.Read.All"},
            )
            if response.status_code >= 400:
                raise ConnectorProviderError("Microsoft token refresh failed", retryable=False, code=str(response.status_code))
            return response.json()

    async def discover_scopes(self) -> list[dict[str, Any]]:
        # ``/drives`` often returns only a generic library such as
        # "Documents". Resolve SharePoint sites first so reviewers can see the
        # real site/library/folder location instead of guessing where it lives.
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
                    "scope_type": "sharepoint_library",
                    "display_name": location,
                    "config": drive_config,
                })
                folders = await self._request(
                    "GET",
                    f"{self.graph}/drives/{quote(drive_id, safe='')}/root/children?$select=id,name,folder,webUrl",
                )
                for folder in folders.get("value", []):  # type: ignore[union-attr]
                    if folder.get("folder"):
                        folder_name = str(folder.get("name") or folder.get("id"))
                        result.append({
                            "external_scope_id": f"{drive_id}:{folder['id']}",
                            "scope_type": "sharepoint_folder",
                            "display_name": f"{location} / {folder_name}",
                            "config": {
                                **drive_config,
                                "folder_id": folder["id"],
                                "web_url": folder.get("webUrl") or drive.get("webUrl") or site_url,
                                "location_label": f"{location} / {folder_name}",
                            },
                        })

        if result:
            return result

        # Keep a fallback for tenants where site search is disabled but the
        # delegated token can still enumerate drives.
        data = await self._request("GET", f"{self.graph}/drives?$select=id,name,driveType,webUrl")
        for item in data.get("value", []):  # type: ignore[union-attr]
            drive_id = str(item.get("id") or "")
            drive_name = str(item.get("name") or drive_id)
            location = f"Available SharePoint library / {drive_name}"
            result.append({"external_scope_id": drive_id, "scope_type": "sharepoint_library", "display_name": location, "config": {"drive_id": drive_id, "drive_name": drive_name, "web_url": item.get("webUrl"), "location_label": location}})
            folders = await self._request("GET", f"{self.graph}/drives/{quote(drive_id, safe='')}/root/children?$select=id,name,folder,webUrl")
            for folder in folders.get("value", []):  # type: ignore[union-attr]
                if folder.get("folder"):
                    folder_name = str(folder.get("name") or folder.get("id"))
                    result.append({"external_scope_id": f"{drive_id}:{folder['id']}", "scope_type": "sharepoint_folder", "display_name": f"{location} / {folder_name}", "config": {"drive_id": drive_id, "folder_id": folder["id"], "web_url": folder.get("webUrl") or item.get("webUrl"), "location_label": f"{location} / {folder_name}"}})
        return result

    async def create_webhook(self, scope: dict[str, Any], callback_url: str) -> dict[str, Any]:
        client_state = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=1)
        result = await self._request("POST", f"{self.graph}/subscriptions", json={
            "changeType": "updated",
            "notificationUrl": callback_url,
            "resource": f"/drives/{scope['config'].get('drive_id', scope['external_scope_id'])}/root",
            "expirationDateTime": expires_at.isoformat(timespec="seconds") + "Z",
            "clientState": client_state,
        })
        return {"subscription_id": result["id"], "client_state": client_state, "expires_at": expires_at}  # type: ignore[index]

    async def renew_webhook(self, provider_subscription_id: str) -> datetime | None:
        # Graph caps a drive subscription at roughly 30 days but rejects anything beyond
        # its own maximum, so this asks for the same hour create_webhook does and leans on
        # the renewal task running far more often than that.
        expires_at = datetime.utcnow() + timedelta(hours=1)
        await self._request(
            "PATCH",
            f"{self.graph}/subscriptions/{provider_subscription_id}",
            json={"expirationDateTime": expires_at.isoformat(timespec="seconds") + "Z"},
        )
        return expires_at

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
        result = []
        for item in data.get("value", []):  # type: ignore[union-attr]
            identities = item.get("grantedToV2") or item.get("grantedTo") or {}
            principal = identities.get("user") or identities.get("group") or identities.get("siteUser") or identities.get("siteGroup")
            if principal and principal.get("id"):
                result.append({"principal_type": "group" if "group" in identities or "siteGroup" in identities else "user", "principal_id": str(principal["id"]), "role": ",".join(item.get("roles") or [])})
            elif item.get("id"):
                # Preserve link/domain/other permission entries as explicit
                # unresolved principals. Dropping them would make a provider
                # ACL look narrower than it is and could allow an unsafe
                # approval or mapping decision.
                result.append({"principal_type": "unknown", "principal_id": str(item["id"]), "role": ",".join(item.get("roles") or [])})
        return result

    async def download(self, change: NormalizedChange) -> bytes:
        drive_id = change.corpus_id
        return await self._request("GET", f"{self.graph}/drives/{drive_id}/items/{change.external_id}/content")  # type: ignore[return-value]


class GoogleDriveAdapter(ConnectorAdapter):
    provider = "google_drive"
    api = "https://www.googleapis.com/drive/v3"
    allowed_api_hosts = frozenset({"www.googleapis.com"})

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

    async def create_webhook(self, scope: dict[str, Any], callback_url: str) -> dict[str, Any]:
        config = scope["config"]
        start = await self._request("GET", f"{self.api}/changes/startPageToken" + (f"?driveId={config['drive_id']}&supportsAllDrives=true" if config.get("drive_id") else ""))
        channel_id = str(uuid.uuid4())
        client_state = secrets.token_urlsafe(32)
        result = await self._request("POST", f"{self.api}/changes/watch", params={"pageToken": start["startPageToken"], "supportsAllDrives": "true"}, json={"id": channel_id, "type": "web_hook", "address": callback_url, "token": client_state})
        expiration = result.get("expiration")
        expires_at = datetime.utcfromtimestamp(int(expiration) / 1000) if expiration else datetime.utcnow() + timedelta(days=1)
        # Google sends the channel id in X-Goog-Channel-ID; keep that id as
        # our subscription key so the webhook can resolve the connector.
        return {"subscription_id": channel_id, "client_state": client_state, "expires_at": expires_at}  # type: ignore[union-attr]

    async def incremental_changes(self, scope: dict[str, Any], cursor: str | None) -> tuple[list[NormalizedChange], str | None]:
        config = scope["config"]
        params = {"pageToken": cursor or "", "pageSize": "100", "includeRemoved": "true", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true", "fields": "nextPageToken,newStartPageToken,changes(fileId,removed,file(id,name,mimeType,parents,webViewLink,version,md5Checksum,trashed,modifiedTime))"}
        if not cursor:
            start = await self._request("GET", f"{self.api}/changes/startPageToken" + (f"?driveId={config['drive_id']}&supportsAllDrives=true" if config.get("drive_id") else ""))
            params["pageToken"] = start["startPageToken"]  # type: ignore[index]
        changes: list[NormalizedChange] = []
        next_cursor = None
        while params.get("pageToken"):
            query = urlencode(params)
            page = await self._request("GET", f"{self.api}/changes?{query}")
            for entry in page.get("changes", []):  # type: ignore[union-attr]
                file = entry.get("file") or {}
                if config.get("folder_id") and config["folder_id"] not in (file.get("parents") or []):
                    continue
                removed = bool(entry.get("removed")) or bool(file.get("trashed"))
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


def adapter_for(connector: Connector) -> ConnectorAdapter:
    if connector.system == "sharepoint":
        return SharePointAdapter(connector)
    if connector.system == "google_drive":
        return GoogleDriveAdapter(connector)
    raise ConnectorProviderError(f"Unsupported connector provider: {connector.system}", retryable=False, code="unsupported_provider")
