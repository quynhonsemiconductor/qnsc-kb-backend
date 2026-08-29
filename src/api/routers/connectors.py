import uuid
import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timedelta
import jwt
from typing import Any, Literal
from urllib.parse import urlencode
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.api.deps import SessionLocal, get_db, get_current_user, require_permission, set_database_context
from src.models import User
from src.models.governance import AuditLog, PendingDraft
from src.models.user import AccessGroup, ExternalIdentity
from src.models.ops import Connector, ConnectorJob
from src.models.connectors import ConnectorNotification, ExternalAclPrincipal, ExternalDocument, ExternalGroupMapping, PermissionSnapshot, SourceScope, SyncCursor, SyncError, SyncRequest, WebhookSubscription
from src.repositories.user import UserRepository
from src.domain.connectors import sync_local_folder
from src.core.config import settings
from src.domain.rbac import AuthorizationService
from src.domain.connector_adapters import adapter_for, ConnectorProviderError, SharePointAdapter, GoogleDriveAdapter
from src.domain.connector_auth import ensure_connector_authorized
from src.domain.webhook_subscriptions import WebhookConfigurationError, ensure_webhook_subscriptions
from src.domain.sync_queue import claim_sync_request, enqueue_connector_sync
from src.core.secrets import encrypt_secret
from src.domain.departments import resolve_active_departments

router = APIRouter()

class ConnectorCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    system: Literal["local_folder", "google_drive", "sharepoint"] = "local_folder"
    path: str = Field(default="", max_length=1_000)
    config: dict[str, Any] = Field(default_factory=dict)


class ConnectorUpdate(BaseModel):
    sync_mode: Literal["manual", "daily", "on_update"] | None = None
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)


SYNC_MODES = {"manual", "daily", "on_update"}


class ScopeSelection(BaseModel):
    scope_ids: list[str] = Field(default_factory=list, max_length=500)


class GroupMappingRequest(BaseModel):
    access_group_id: uuid.UUID
    external_group_name: str | None = Field(default=None, max_length=255)


_SENSITIVE_CONFIG_KEYS = {"clientsecret", "accesstoken", "refreshtoken", "token", "apikey", "password", "secret"}
_MAX_CONNECTOR_CONFIG_BYTES = 16_384


def _safe_connector_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject secrets and oversized data before persisting connector config."""
    try:
        encoded = json.dumps(config, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Connector configuration must contain JSON values only") from exc
    if len(encoded.encode("utf-8")) > _MAX_CONNECTOR_CONFIG_BYTES:
        raise HTTPException(status_code=422, detail="Connector configuration is too large")

    def inspect(value: Any, depth: int = 0) -> None:
        if depth > 8:
            raise HTTPException(status_code=422, detail="Connector configuration is nested too deeply")
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = "".join(character for character in str(key).lower() if character.isalnum())
                if normalized_key in _SENSITIVE_CONFIG_KEYS:
                    raise HTTPException(status_code=422, detail="Connector credentials must be authorized through OAuth, not saved in configuration")
                inspect(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                inspect(item, depth + 1)

    inspect(config)
    return json.loads(encoded)


async def _apply_connector_departments(db: AsyncSession, connector: Connector, config: dict[str, Any], department_ids: list[uuid.UUID] | None) -> dict[str, Any]:
    """Store canonical default routing for drafts created by this connector."""
    if department_ids is None:
        return config
    departments = await resolve_active_departments(db, connector.company_domain, department_ids, required=False)
    next_config = {**config}
    next_config["department_ids"] = [str(department.id) for department in departments]
    next_config["department_names"] = [department.name for department in departments]
    return next_config

def _response(connector: Connector) -> dict[str, Any]:
    config = connector.config_json or {}
    application_authorized = bool(
        connector.system == "sharepoint"
        and settings.microsoft_connector_auth_mode == "application"
        and settings.MICROSOFT_CLIENT_ID
        and settings.MICROSOFT_CLIENT_SECRET
        and settings.MICROSOFT_TENANT_ID
    )
    return {
        "id": str(connector.id), "name": connector.name, "system": connector.system,
        "status": connector.status, "company_domain": connector.company_domain,
        "last_sync": connector.last_sync, "last_error": connector.last_error,
        "authorized": bool(connector.oauth_refresh_token or connector.oauth_access_token) or application_authorized,
        "path": config.get("path"),
        "sync_mode": config.get("sync_mode", "manual" if connector.system == "local_folder" else "daily"),
        "webhook_enabled": bool(config.get("webhook_enabled")),
        # webhook_enabled is the admin's INTENT and stays true once switched on. This is
        # what is actually happening: set by the repair worker when it could not keep a
        # subscription alive, so "on_update" stops claiming a liveness it does not have.
        "webhook_degraded": bool(config.get("webhook_degraded_at")),
        "webhook_degraded_since": config.get("webhook_degraded_at"),
        "department_ids": [str(item) for item in config.get("department_ids", [])],
        "department_names": [str(item) for item in config.get("department_names", [])],
    }


def _job_response(job: ConnectorJob, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": str(job.id),
        "status": job.status,
        "attempts": job.attempts,
        "last_error": job.last_error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "summary": summary if summary is not None else (job.summary_json or {}),
    }


async def _legacy_job_summary(db: AsyncSession, job: ConnectorJob) -> dict[str, Any] | None:
    """Give pre-summary jobs a useful best-effort file list after migration."""
    if job.summary_json or job.status != "completed":
        return job.summary_json
    stmt = select(ExternalDocument).where(
        ExternalDocument.connector_id == job.connector_id,
        ExternalDocument.mime_type.is_not(None),
        ExternalDocument.created_at >= job.created_at,
    )
    if job.completed_at:
        stmt = stmt.where(ExternalDocument.created_at <= job.completed_at)
    documents = (await db.execute(stmt.order_by(ExternalDocument.created_at))).scalars().all()
    if not documents:
        return None
    return {
        "changes_seen": len(documents),
        "files_seen": len(documents),
        "imported": len(documents),
        "updated": 0,
        "deleted": 0,
        "unchanged": 0,
        "permissions_updated": 0,
        "items": [{"name": document.name, "action": "processed", "web_url": document.web_url} for document in documents[:200]],
        "legacy_backfill": True,
    }


def _can_complete_oauth(initiator: User | None, connector: Connector) -> bool:
    return bool(
        initiator
        and initiator.active
        and initiator.company_domain == connector.company_domain
        and AuthorizationService.has_permission(initiator, "connector.manage", requested_scope="company")
    )


async def _connector_for_user(db: AsyncSession, connector_id: uuid.UUID, current_user: User) -> Connector | None:
    """Load a connector with the caller's tenant scope in the SQL query."""
    stmt = select(Connector).where(Connector.id == connector_id)
    if not AuthorizationService.has_permission(current_user, "connector.manage", requested_scope="global"):
        stmt = stmt.where(Connector.company_domain == current_user.company_domain)
    return (await db.execute(stmt)).scalar_one_or_none()

@router.get("")
async def list_connectors(
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    stmt = select(Connector).order_by(Connector.company_domain, Connector.name)
    if not AuthorizationService.has_permission(current_user, "connector.manage", requested_scope="global"):
        stmt = stmt.where(Connector.company_domain == current_user.company_domain)
    return [_response(item) for item in (await db.execute(stmt)).scalars().all()]

@router.post("", status_code=201)
async def create_connector(
    request: ConnectorCreate,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if request.system not in {"local_folder", "google_drive", "sharepoint"}:
        raise HTTPException(status_code=422, detail="Unsupported connector provider")
    folder = None
    if request.system == "local_folder":
        from src.domain.connectors import _safe_folder
        folder = _safe_folder(request.path)
    safe_config = _safe_connector_config(request.config)
    sync_mode = safe_config.get("sync_mode", "manual" if request.system == "local_folder" else "daily")
    if sync_mode not in SYNC_MODES:
        raise HTTPException(status_code=422, detail="Sync mode must be manual, daily, or on_update")
    safe_config["sync_mode"] = sync_mode
    raw_department_ids = safe_config.pop("department_ids", None)
    if raw_department_ids is not None:
        try:
            department_ids = [uuid.UUID(str(item)) for item in raw_department_ids]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Connector departments must be valid department IDs") from exc
        temporary_connector = Connector(company_domain=current_user.company_domain)
        safe_config = await _apply_connector_departments(db, temporary_connector, safe_config, department_ids)
    if folder:
        safe_config["path"] = str(folder)
    connector = Connector(name=request.name, system=request.system, company_domain=current_user.company_domain, created_by=current_user.id, config_json=safe_config)
    db.add(connector)
    await db.commit()
    await db.refresh(connector)
    return _response(connector)


@router.patch("/{connector_id}")
async def update_connector(
    connector_id: uuid.UUID,
    request: ConnectorUpdate,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    if request.sync_mode is not None:
        if request.sync_mode not in SYNC_MODES:
            raise HTTPException(status_code=422, detail="Sync mode must be manual, daily, or on_update")
        connector.config_json = {**(connector.config_json or {}), "sync_mode": request.sync_mode}
    if request.department_ids is not None:
        connector.config_json = await _apply_connector_departments(db, connector, connector.config_json or {}, request.department_ids)
        department_ids = set(connector.config_json.get("department_ids", []))
        department_names = connector.config_json.get("department_names", [])
        primary_department = department_names[0] if department_names else None
        pending_drafts = (await db.execute(
            select(PendingDraft)
            .join(ExternalDocument, PendingDraft.external_document_id == ExternalDocument.id)
            .where(
                ExternalDocument.connector_id == connector.id,
                PendingDraft.status == "pending",
                PendingDraft.dept.is_(None),
            )
        )).scalars().all()
        for draft in pending_drafts:
            metadata = {**(draft.content_metadata or {}), "department_ids": list(department_ids), "department_names": department_names, "submission_kind": "connector_import"}
            draft.dept = primary_department
            draft.content_metadata = metadata
    await db.commit()
    await db.refresh(connector)
    return _response(connector)


def _oauth_frontend_redirect(*, connector_id: str | None = None, success: bool = False) -> RedirectResponse:
    params = {"oauth": "success" if success else "error"}
    if connector_id:
        params["connector_id"] = connector_id
    return RedirectResponse(
        url=f"{settings.FRONTEND_URL.rstrip('/')}/admin/connectors?{urlencode(params)}",
        status_code=303,
    )


@router.get("/oauth/callback")
async def oauth_callback_entry(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> Response:
    # Keep the provider redirect URI static (required by Google/Microsoft),
    # while the signed state identifies the connector being authorized.
    if error or not code or not state:
        return _oauth_frontend_redirect()
    try:
        result = await oauth_callback(code=code, state=state, db=db)
    except Exception as exc:
        # OAuth must always return the user to the application. Provider,
        # database, or callback-state failures should never leave the browser
        # on a generic 500 page.
        import structlog
        structlog.get_logger().exception("Connector OAuth callback failed", error=str(exc))
        return _oauth_frontend_redirect()
    return _oauth_frontend_redirect(connector_id=result["connector_id"], success=True)


@router.get("/{connector_id}/oauth/start")
async def start_oauth(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    if connector.system == "local_folder":
        raise HTTPException(status_code=422, detail="Local folders do not require OAuth")
    if connector.system == "sharepoint" and settings.microsoft_connector_auth_mode == "application":
        raise HTTPException(status_code=422, detail="Microsoft connector is using app-only mode; configure Entra application permissions and use Discover scopes directly")
    if connector.system == "sharepoint" and not all((settings.MICROSOFT_CLIENT_ID, settings.MICROSOFT_CLIENT_SECRET, settings.MICROSOFT_REDIRECT_URI)):
        raise HTTPException(status_code=422, detail="Microsoft connector is not configured. Set MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, and MICROSOFT_REDIRECT_URI in the API environment.")
    if connector.system == "google_drive" and not all((settings.GOOGLE_CLIENT_ID, settings.GOOGLE_CLIENT_SECRET, settings.GOOGLE_REDIRECT_URI)):
        raise HTTPException(status_code=422, detail="Google Drive connector is not configured. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in the API environment.")
    state = jwt.encode({"type": "connector_oauth", "connector_id": str(connector.id), "user_id": str(current_user.id), "exp": datetime.utcnow() + timedelta(minutes=10)}, settings.SECRET_KEY, algorithm="HS256")
    connector.oauth_state_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
    connector.oauth_state_expires_at = datetime.utcnow() + timedelta(minutes=10)
    await db.commit()
    adapter = adapter_for(connector)
    if isinstance(adapter, SharePointAdapter):
        url = adapter.oauth_url(state)
    elif isinstance(adapter, GoogleDriveAdapter):
        url = adapter.oauth_url(state)
    else:
        raise HTTPException(status_code=422, detail="OAuth is not configured for this provider")
    return {"authorization_url": url}


@router.get("/{connector_id}/oauth/callback", include_in_schema=False)
async def oauth_callback(
    code: str,
    state: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        claims = jwt.decode(state, settings.SECRET_KEY, algorithms=["HS256"])
        if claims.get("type") != "connector_oauth":
            raise jwt.InvalidTokenError("invalid connector state")
        connector_id = uuid.UUID(str(claims["connector_id"]))
        initiator_id = uuid.UUID(str(claims["user_id"]))
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="OAuth state is invalid or expired") from exc
    # Provider callbacks are authenticated by the short-lived signed state,
    # not a browser session.  Use explicit internal context so RLS can safely
    # protect the connector root table.
    await set_database_context(db, None, True)
    connector = (await db.execute(select(Connector).where(Connector.id == connector_id))).scalar_one_or_none()
    if not connector or not connector.oauth_state_hash or connector.oauth_state_hash != hashlib.sha256(state.encode("utf-8")).hexdigest() or not connector.oauth_state_expires_at or connector.oauth_state_expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="OAuth state is invalid or expired")
    initiator = await UserRepository(db).get_by_id(initiator_id)
    if not _can_complete_oauth(initiator, connector):
        connector.oauth_state_hash = None
        connector.oauth_state_expires_at = None
        await db.commit()
        raise HTTPException(status_code=403, detail="The user who started this authorization no longer has connector access")
    try:
        if claims.get("connector_id") != str(connector.id):
            raise jwt.InvalidTokenError("invalid connector state")
        tokens = await adapter_for(connector).exchange_code(code)
    except (jwt.PyJWTError, ConnectorProviderError) as exc:
        connector.status = "auth_failed"
        connector.last_error = str(exc)
        await db.commit()
        raise HTTPException(status_code=400, detail="Provider authorization failed") from exc
    connector.oauth_access_token = encrypt_secret(tokens.get("access_token"))
    connector.oauth_refresh_token = encrypt_secret(tokens.get("refresh_token")) or connector.oauth_refresh_token
    # Deliberately NOT read out of tokens["id_token"]. This column records HOW a
    # connector was authorized -- connector_auth.py writes "application" for app-only
    # mode -- and nothing anywhere reads it for an account name: it is serialized into
    # no response, queried by no code, and absent from the frontend.
    #
    # It used to hold the id_token's `sub`, decoded without signature verification. That
    # decode is what broke this endpoint (see the test), and keeping it would mean
    # either shipping an unverified decode past Semgrep or fetching provider JWKS on the
    # callback path -- another step that could fail an authorization -- to fill a field
    # with no consumer. If an account label is ever actually wanted, add it deliberately,
    # with a verification decision and somewhere to display it.
    connector.oauth_subject = "delegated"
    if tokens.get("expires_in"):
        connector.oauth_expires_at = datetime.utcnow() + timedelta(seconds=int(tokens["expires_in"]))
    connector.oauth_state_hash = None
    connector.oauth_state_expires_at = None
    connector.status = "active"
    connector.last_error = None
    await db.commit()
    return {"connector_id": str(connector.id), "status": connector.status, "message": "Connector authorized; select scopes before syncing."}


@router.get("/{connector_id}/scopes")
async def discover_scopes(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    try:
        await ensure_connector_authorized(db, connector)
        scopes = await adapter_for(connector).discover_scopes()
    except ConnectorProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return scopes


@router.get("/{connector_id}/preview")
async def preview_connector(
    connector_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Preview the provider corpus without creating documents or drafts."""
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    if connector.system == "local_folder":
        raise HTTPException(status_code=422, detail="Preview is not available for local folders")
    scopes = (await db.execute(select(SourceScope).where(SourceScope.connector_id == connector.id, SourceScope.selected.is_(True)))).scalars().all()
    if not scopes:
        raise HTTPException(status_code=409, detail="Select at least one SharePoint library or folder before previewing")
    adapter = adapter_for(connector)
    await ensure_connector_authorized(db, connector)
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for scope in scopes:
        try:
            changes, _ = await adapter.incremental_changes({"external_scope_id": scope.external_scope_id, "config": scope.config_json or {}}, None)
            for item in changes:
                if item.state != "deleted":
                    items.append({"external_id": item.external_id, "name": item.name, "mime_type": item.mime_type, "web_url": item.web_url, "revision": item.revision, "scope": scope.display_name})
                if len(items) >= limit:
                    break
        except ConnectorProviderError as exc:
            errors.append(f"{scope.display_name}: {exc}")
        if len(items) >= limit:
            break
    return {"connector_id": str(connector.id), "provider": connector.system, "scopes": len(scopes), "files": items[:limit], "files_returned": min(len(items), limit), "truncated": len(items) > limit, "errors": errors, "writes_performed": False}


@router.put("/{connector_id}/scopes")
async def select_scopes(
    connector_id: uuid.UUID,
    request: ScopeSelection,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    await ensure_connector_authorized(db, connector)
    selected = set(request.scope_ids)
    existing = (await db.execute(select(SourceScope).where(SourceScope.connector_id == connector.id))).scalars().all()
    discovered = {str(item["external_scope_id"]): item for item in await adapter_for(connector).discover_scopes()}
    for scope in existing:
        scope.selected = scope.external_scope_id in selected
        item = discovered.get(scope.external_scope_id)
        if item:
            scope.scope_type = item["scope_type"]
            scope.display_name = item["display_name"]
            scope.config_json = item.get("config")
    known = {scope.external_scope_id for scope in existing}
    for item in discovered.values():
        if item["external_scope_id"] in selected and item["external_scope_id"] not in known:
            db.add(SourceScope(connector_id=connector.id, external_scope_id=item["external_scope_id"], scope_type=item["scope_type"], display_name=item["display_name"], selected=True, config_json=item.get("config")))
    await db.commit()
    return [{"external_scope_id": scope.external_scope_id, "display_name": scope.display_name, "scope_type": scope.scope_type, "selected": scope.selected, "config": scope.config_json or {}} for scope in (await db.execute(select(SourceScope).where(SourceScope.connector_id == connector.id))).scalars().all()]


@router.post("/{connector_id}/webhooks/subscribe")
async def subscribe_webhooks(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    try:
        # force: the admin pressing this button means "give me a fresh subscription",
        # not "leave the one that is about to expire alone".
        created = await ensure_webhook_subscriptions(db, connector, force=True)
    except WebhookConfigurationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    config = {**(connector.config_json or {}), "webhook_enabled": True, "sync_mode": "on_update"}
    # The repair worker sets this when it cannot keep a subscription alive; a successful
    # manual subscribe is exactly the event that clears it.
    config.pop("webhook_degraded_at", None)
    connector.config_json = config
    await db.commit()
    return {"connector_id": str(connector.id), "subscriptions": created, "sync_mode": "on_update"}


@router.get("/{connector_id}/group-mappings")
async def list_group_mappings(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    mappings = (await db.execute(select(ExternalGroupMapping, AccessGroup.name).join(AccessGroup, AccessGroup.id == ExternalGroupMapping.access_group_id).where(
        ExternalGroupMapping.connector_id == connector.id,
        AccessGroup.company_domain == connector.company_domain,
    ))).all()
    return [{"external_group_id": mapping.external_group_id, "external_group_name": mapping.external_group_name, "access_group_id": str(mapping.access_group_id), "access_group_name": name, "active": mapping.active} for mapping, name in mappings]


@router.get("/{connector_id}/acl-principals")
async def list_acl_principals(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List observed provider principals and their current mapping state.

    The connector lookup supplies the tenant boundary before any provider ACL
    row is returned. Unmapped principals are deliberately visible only to
    connector managers so they can make an explicit, audited mapping choice.
    """
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    mapping_rows = (await db.execute(select(ExternalGroupMapping, AccessGroup.name).join(
        AccessGroup, AccessGroup.id == ExternalGroupMapping.access_group_id,
    ).where(
        ExternalGroupMapping.connector_id == connector.id,
        AccessGroup.company_domain == connector.company_domain,
    ))).all()
    group_mappings = {
        mapping.external_group_id: {
            "external_group_name": mapping.external_group_name,
            "access_group_id": str(mapping.access_group_id),
            "access_group_name": name,
            "active": mapping.active,
        }
        for mapping, name in mapping_rows
    }

    principals = (await db.execute(
        select(ExternalAclPrincipal)
        .join(PermissionSnapshot, PermissionSnapshot.id == ExternalAclPrincipal.permission_snapshot_id)
        .join(ExternalDocument, ExternalDocument.id == PermissionSnapshot.external_document_id)
        .where(
            ExternalDocument.connector_id == connector.id,
            ExternalDocument.state != "deleted",
            PermissionSnapshot.active.is_(True),
        )
    )).scalars().all()
    by_principal: dict[tuple[str, str], dict[str, Any]] = {}
    for principal in principals:
        key = (principal.principal_type, principal.principal_id)
        entry = by_principal.setdefault(key, {"principal_type": principal.principal_type, "principal_id": principal.principal_id, "roles": set()})
        if principal.role:
            entry["roles"].update(item.strip() for item in principal.role.split(",") if item.strip())

    user_subjects = [principal_id for principal_type, principal_id in by_principal if principal_type in {"user", "siteUser"}]
    identities = (await db.execute(
        select(ExternalIdentity).join(User, User.id == ExternalIdentity.user_id).where(
            ExternalIdentity.provider == "microsoft_entra",
            ExternalIdentity.subject.in_(user_subjects),
            User.company_domain == connector.company_domain,
        )
    )).scalars().all() if user_subjects else []
    mapped_users = {str(identity.subject): str(identity.user_id) for identity in identities}

    response: list[dict[str, Any]] = []
    for (principal_type, principal_id), entry in sorted(by_principal.items()):
        mapping = group_mappings.get(principal_id) if principal_type in {"group", "siteGroup"} else None
        mapped_user_id = mapped_users.get(principal_id) if principal_type in {"user", "siteUser"} else None
        active_mapping = mapping and mapping["active"]
        response.append({
            "principal_type": principal_type,
            "principal_id": principal_id,
            "roles": sorted(entry["roles"]),
            "mapping_status": "mapped" if active_mapping or mapped_user_id else "unmapped",
            "external_group_name": mapping["external_group_name"] if mapping else None,
            "access_group_id": mapping["access_group_id"] if active_mapping else None,
            "access_group_name": mapping["access_group_name"] if active_mapping else None,
            "internal_user_id": mapped_user_id,
        })
    return response


@router.put("/{connector_id}/group-mappings/{external_group_id}")
async def set_group_mapping(
    connector_id: uuid.UUID,
    request: GroupMappingRequest,
    external_group_id: str = Path(..., min_length=1, max_length=512),
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    group = (await db.execute(select(AccessGroup).where(
        AccessGroup.id == request.access_group_id,
        AccessGroup.company_domain == connector.company_domain,
    ))).scalar_one_or_none()
    if not group:
        raise HTTPException(status_code=404, detail="Connector or access group not found")
    mapping = (await db.execute(select(ExternalGroupMapping).where(ExternalGroupMapping.connector_id == connector.id, ExternalGroupMapping.external_group_id == external_group_id))).scalar_one_or_none()
    if mapping is None:
        mapping = ExternalGroupMapping(connector_id=connector.id, external_group_id=external_group_id, external_group_name=request.external_group_name, access_group_id=group.id, active=True)
        db.add(mapping)
    else:
        mapping.external_group_name = request.external_group_name or mapping.external_group_name
        mapping.access_group_id = group.id
        mapping.active = True
    try:
        await db.flush()
        from src.domain.cloud_sync import reconcile_connector_acl_mappings, _record_permission_change_audits
        changed_article_ids = await reconcile_connector_acl_mappings(db, connector)
        _record_permission_change_audits(db, changed_article_ids, current_user.id)
        db.add(AuditLog(user_id=current_user.id, action="connector_permission_mapping", target_type="connector", target_id=str(connector.id), outcome="success"))
        await db.commit()
    except Exception:
        await db.rollback()
        db.add(AuditLog(user_id=current_user.id, action="connector_permission_mapping", target_type="connector", target_id=str(connector.id), outcome="failure"))
        await db.commit()
        raise
    from src.domain.events import event_bus
    for article_id in changed_article_ids:
        await event_bus.publish("PermissionChanged", {"article_id": str(article_id)})
    return {"external_group_id": mapping.external_group_id, "access_group_id": str(mapping.access_group_id), "active": mapping.active, "articles_reconciled": len(changed_article_ids)}


@router.delete("/{connector_id}/group-mappings/{external_group_id}", status_code=204)
async def delete_group_mapping(
    connector_id: uuid.UUID,
    external_group_id: str = Path(..., min_length=1, max_length=512),
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> None:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    mapping = (await db.execute(select(ExternalGroupMapping).where(ExternalGroupMapping.connector_id == connector.id, ExternalGroupMapping.external_group_id == external_group_id))).scalar_one_or_none()
    if mapping:
        mapping.active = False
        try:
            await db.flush()
            from src.domain.cloud_sync import reconcile_connector_acl_mappings, _record_permission_change_audits
            changed_article_ids = await reconcile_connector_acl_mappings(db, connector)
            _record_permission_change_audits(db, changed_article_ids, current_user.id)
            db.add(AuditLog(user_id=current_user.id, action="connector_permission_mapping", target_type="connector", target_id=str(connector.id), outcome="success"))
            await db.commit()
        except Exception:
            await db.rollback()
            db.add(AuditLog(user_id=current_user.id, action="connector_permission_mapping", target_type="connector", target_id=str(connector.id), outcome="failure"))
            await db.commit()
            raise
        from src.domain.events import event_bus
        for article_id in changed_article_ids:
            await event_bus.publish("PermissionChanged", {"article_id": str(article_id)})

async def _run_cloud_sync_inline(
    connector_id: uuid.UUID,
    job_id: uuid.UUID,
    sync_request_id: uuid.UUID | None = None,
) -> None:
    """Inline-mode cloud sync on a dedicated session, mirroring the Celery task.

    Runs detached on the API event loop (same tradeoff as other inline-mode
    lifecycle work): the HTTP response returns immediately and a process
    restart abandons the job, which the connector's queued-job expiry and the
    next manual/scheduled sync recover from.
    """
    from src.domain.cloud_sync import sync_cloud_connector
    from src.models.ops import Connector as ConnectorModel
    from src.models.ops import ConnectorJob as ConnectorJobModel

    try:
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            if sync_request_id:
                from src.domain.sync_queue import mark_sync_request_running
                await mark_sync_request_running(db, sync_request_id)
            sync_request = await db.get(SyncRequest, sync_request_id) if sync_request_id else None
            connector = await db.get(ConnectorModel, connector_id)
            job = await db.get(ConnectorJobModel, job_id)
            if not connector or not job:
                if sync_request_id:
                    from src.domain.sync_queue import finish_sync_request
                    await finish_sync_request(db, sync_request_id, success=False, error="Connector or job no longer exists", retryable=False)
                return
            await sync_cloud_connector(
                db,
                connector,
                job,
                scope_id=sync_request.scope_id if sync_request else None,
            )
            if sync_request_id:
                from src.domain.sync_queue import finish_sync_request
                await finish_sync_request(db, sync_request_id, success=True)
    except Exception:
        if sync_request_id:
            from src.domain.sync_queue import finish_sync_request
            async with SessionLocal() as error_db:
                await set_database_context(error_db, None, True)
                await finish_sync_request(error_db, sync_request_id, success=False, error="Inline connector sync failed")
        import structlog

        structlog.get_logger().exception(
            "Inline cloud connector sync failed", connector_id=str(connector_id)
        )


def _dispatch_cloud_sync(
    connector_id: uuid.UUID,
    job_id: uuid.UUID,
    sync_request_id: uuid.UUID | None = None,
) -> None:
    """Dispatch a queued cloud-sync job according to the deployment job mode.

    Calling Celery's .delay() without a broker raises and leaves the job row
    stuck "queued", so inline deployments must run the same work in-process.
    """
    if settings.JOB_MODE == "celery":
        from src.workers.tasks import sync_cloud_connector_task

        sync_cloud_connector_task.delay(str(connector_id), str(job_id), str(sync_request_id) if sync_request_id else None)
    else:
        asyncio.create_task(_run_cloud_sync_inline(connector_id, job_id, sync_request_id))


@router.post("/{connector_id}/sync")
async def sync_connector(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    if connector.system != "local_folder":
        if settings.microsoft_connector_auth_mode != "application" and not connector.oauth_access_token and not connector.oauth_refresh_token:
            raise HTTPException(status_code=409, detail="Authorize the connector before syncing")
        active_job = (await db.execute(
            select(ConnectorJob)
            .where(ConnectorJob.connector_id == connector.id, ConnectorJob.status.in_(["queued", "running"]))
            .order_by(ConnectorJob.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if active_job and active_job.status == "queued" and active_job.created_at and active_job.created_at < datetime.utcnow() - timedelta(minutes=10):
            active_job.status = "failed"
            active_job.last_error = "Sync job expired before the worker started"
            active_job.completed_at = datetime.utcnow()
            await db.commit()
            active_job = None
        if active_job:
            return {"connector_id": str(connector.id), "job_id": str(active_job.id), "status": active_job.status, "last_sync": connector.last_sync, "already_running": True}
        request = await enqueue_connector_sync(db, connector.id, reason="manual", requested_by=current_user.id, priority=10)
        request_id = request.id
        job = await db.get(ConnectorJob, request.job_id) if request.job_id else None
        await db.commit()
        # Claim through the queue rather than flipping the row here. The claim is what
        # enforces one walk per connector, and hand-marking it "dispatched" bypassed
        # that check entirely — a manual sync pressed during a reconciliation ran a
        # second concurrent walk of the same drive.
        claimed = await claim_sync_request(db, request_id)
        if claimed and job:
            _dispatch_cloud_sync(connector.id, job.id, claimed.id)
        status = claimed.status if claimed else "queued"
        return {"connector_id": str(connector.id), "job_id": str(job.id) if job else None, "status": status, "last_sync": connector.last_sync}
    job = await sync_local_folder(db, connector, current_user.id)
    return {"connector_id": str(connector.id), "job_id": str(job.id), "status": job.status, "last_sync": connector.last_sync}


@router.get("/{connector_id}/jobs")
async def list_connector_jobs(
    connector_id: uuid.UUID,
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    jobs = (await db.execute(
        select(ConnectorJob)
        .where(ConnectorJob.connector_id == connector.id)
        .order_by(ConnectorJob.created_at.desc())
        .limit(limit)
    )).scalars().all()
    result = []
    for job in jobs:
        result.append(_job_response(job, await _legacy_job_summary(db, job)))
    return result


@router.get("/{connector_id}/health")
async def connector_health(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Expose freshness, cursor, subscription and queue health for operations."""

    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    scopes = (await db.execute(select(SourceScope).where(SourceScope.connector_id == connector.id))).scalars().all()
    cursors = (await db.execute(select(SyncCursor).where(SyncCursor.connector_id == connector.id))).scalars().all()
    subscriptions = (await db.execute(select(WebhookSubscription).where(WebhookSubscription.connector_id == connector.id))).scalars().all()
    queued = int(await db.scalar(select(func.count(SyncRequest.id)).where(SyncRequest.connector_id == connector.id, SyncRequest.status.in_(("queued", "dispatched", "running")))) or 0)
    notifications_24h = int(await db.scalar(select(func.count(ConnectorNotification.id)).where(ConnectorNotification.connector_id == connector.id, ConnectorNotification.received_at >= datetime.utcnow() - timedelta(hours=24))) or 0)
    # Per-document failures no longer stop the sync, which is what makes them easy to
    # miss: the job reports "completed" and a handful of files simply never arrive.
    # Surfacing them here is what turns a silent gap into something an admin can act on.
    item_errors = (await db.execute(
        select(SyncError, ExternalDocument.name, ExternalDocument.web_url)
        .outerjoin(ExternalDocument, ExternalDocument.id == SyncError.external_document_id)
        .where(SyncError.connector_id == connector.id, SyncError.stage == "ingest")
        .order_by(SyncError.created_at.desc())
        .limit(25)
    )).all()
    quarantined = int(await db.scalar(
        select(func.count(ExternalDocument.id)).where(
            ExternalDocument.connector_id == connector.id,
            ExternalDocument.state != "deleted",
            ExternalDocument.metadata_json["ingest_failure"].is_not(None),
        )
    ) or 0)
    now = datetime.utcnow()
    return {
        "connector_id": str(connector.id),
        "status": connector.status,
        "last_sync": connector.last_sync,
        "last_error": connector.last_error,
        "queue_depth": queued,
        "notifications_last_24h": notifications_24h,
        "documents_needing_attention": quarantined,
        "recent_document_errors": [
            {
                "document_id": str(error.external_document_id) if error.external_document_id else None,
                "name": name,
                "web_url": web_url,
                "code": error.error_code,
                "message": error.message,
                "attempts": error.attempts,
                "retryable": error.retryable,
                "at": error.created_at,
            }
            for error, name, web_url in item_errors
        ],
        "scopes": [
            {
                "id": str(scope.id),
                "name": scope.display_name,
                "cursor_status": next((cursor.status for cursor in cursors if cursor.scope_id == scope.id), "missing"),
                "full_sync_required": next((cursor.full_sync_required for cursor in cursors if cursor.scope_id == scope.id), True),
                "last_delta_success": next((cursor.last_success_at for cursor in cursors if cursor.scope_id == scope.id), None),
                "last_error": next((cursor.last_error for cursor in cursors if cursor.scope_id == scope.id), None),
            }
            for scope in scopes
        ],
        "subscriptions": [
            {
                "id": str(subscription.id),
                "provider_subscription_id": subscription.provider_subscription_id,
                "active": subscription.active,
                "expires_at": subscription.expires_at,
                "seconds_to_expiry": int((subscription.expires_at - now).total_seconds()) if subscription.expires_at else None,
                "reauthorization_required": subscription.reauthorization_required,
                "last_notification_at": subscription.last_notification_at,
                "last_error": subscription.last_error,
            }
            for subscription in subscriptions
        ],
    }


@router.post("/{connector_id}/retry-failed")
async def retry_failed_documents(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Let a sync try quarantined documents again.

    A document that fails to ingest three times on one revision stops being retried, and
    only a new revision at the provider clears that. The reasoning holds when the file is
    the problem: replacing it at the source is how a person fixes it, and needing to find
    an admin screen as well would be worse.

    It does not hold when the failure was ours. A server-side defect quarantines every
    document it touches, and once it is fixed there is nothing at the provider for anyone
    to repair -- the files were always fine. Without this the corpus stays empty and the
    sync reports "skipped" forever, which reads as a problem with the documents.

    Releasing does not re-ingest. It clears the mark so the next sync treats these as
    unseen; run a sync afterwards.
    """
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    from src.domain.cloud_sync import _clear_ingest_failure

    documents = (
        await db.execute(
            select(ExternalDocument).where(
                ExternalDocument.connector_id == connector.id,
                ExternalDocument.state != "deleted",
                ExternalDocument.metadata_json["ingest_failure"].is_not(None),
            )
        )
    ).scalars().all()
    for document in documents:
        _clear_ingest_failure(document)
    db.add(
        AuditLog(
            user_id=current_user.id,
            action="connector_retry_failed_documents",
            target_type="connector",
            target_id=str(connector.id),
            outcome="success",
            detail_json={"released": len(documents)},
        )
    )
    await db.commit()
    return {"released": len(documents)}


@router.get("/{connector_id}/source-tree")
async def connector_source_tree(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Expose the provider folder/file hierarchy already observed by sync."""
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    scopes = (await db.execute(select(SourceScope).where(SourceScope.connector_id == connector.id, SourceScope.selected.is_(True)).order_by(SourceScope.display_name))).scalars().all()
    documents = (await db.execute(select(ExternalDocument).where(ExternalDocument.connector_id == connector.id, ExternalDocument.state != "deleted").order_by(ExternalDocument.name))).scalars().all()
    by_scope: dict[uuid.UUID, list[ExternalDocument]] = {}
    for document in documents:
        if document.scope_id:
            by_scope.setdefault(document.scope_id, []).append(document)
    result_scopes = []
    for scope in scopes:
        rows = by_scope.get(scope.id, [])
        nodes = [{"id": document.external_id, "name": document.name, "parent_external_id": document.parent_external_id, "is_folder": bool(document.mime_type and document.mime_type.endswith(".folder")), "state": document.state, "article_id": str(document.article_id) if document.article_id else None, "web_url": document.web_url} for document in rows]
        result_scopes.append({"id": str(scope.id), "external_scope_id": scope.external_scope_id, "display_name": scope.display_name, "scope_type": scope.scope_type, "nodes": nodes})
    return {"connector_id": str(connector.id), "connector_name": connector.name, "system": connector.system, "scopes": result_scopes, "files_indexed": len(documents)}


@router.get("/{connector_id}/readme")
async def connector_readme(
    connector_id: uuid.UUID,
    current_user: User = Depends(require_permission("connector.manage")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Generate a reviewable source README from current connector metadata."""
    connector = await _connector_for_user(db, connector_id, current_user)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    scopes = (await db.execute(select(SourceScope).where(SourceScope.connector_id == connector.id, SourceScope.selected.is_(True)).order_by(SourceScope.display_name))).scalars().all()
    jobs = (await db.execute(select(ConnectorJob).where(ConnectorJob.connector_id == connector.id).order_by(ConnectorJob.created_at.desc()).limit(5))).scalars().all()
    lines = [f"# {connector.name}", "", f"- Provider: {connector.system}", f"- Sync mode: {(connector.config_json or {}).get('sync_mode', 'manual')}", f"- Last sync: {connector.last_sync.isoformat() if connector.last_sync else 'Not synced'}", "", "## Selected locations", ""]
    if scopes:
        lines.extend(f"- {scope.display_name} ({scope.scope_type})" for scope in scopes)
    else:
        lines.append("- No locations selected")
    lines.extend(["", "## Recent sync activity", ""])
    if jobs:
        lines.extend(f"- {job.created_at.isoformat() if job.created_at else 'Unknown'} — {job.status} — {job.summary_json.get('changes_seen', 0) if isinstance(job.summary_json, dict) else 0} changes" for job in jobs)
    else:
        lines.append("- No sync jobs recorded")
    lines.extend(["", "## Access boundary", "", "Provider ACLs are intersected with internal article policy. Unmapped or ambiguous principals fail closed."])
    return {"connector_id": str(connector.id), "generated_at": datetime.utcnow(), "markdown": "\n".join(lines)}


async def _enqueue_webhook(request: Request, provider: str, lifecycle_only: bool = False) -> Response:
    """Persist Graph/Drive notifications and enqueue one coalesced delta sync."""

    headers = request.headers
    body: dict[str, Any] = {}
    subscription_id = headers.get("x-goog-channel-id") if provider == "google_drive" else None
    client_state = headers.get("x-goog-channel-token") if provider == "google_drive" else None
    content_length = headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > 1_048_576:
        return Response(status_code=202)
    try:
        raw = (await request.body())[:1_048_576]
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                body = parsed
    except (ValueError, UnicodeDecodeError):
        return Response(status_code=202)

    values = body.get("value") if isinstance(body.get("value"), list) else []
    if not values:
        values = [body] if body else [{
            "subscriptionId": subscription_id,
            "clientState": client_state,
            "changeType": headers.get("x-goog-resource-state"),
        }]

    dispatches: dict[tuple[uuid.UUID, uuid.UUID, uuid.UUID], None] = {}
    claimable: dict[uuid.UUID, tuple[uuid.UUID, uuid.UUID]] = {}
    async with SessionLocal() as db_session:
        await set_database_context(db_session, None, True)
        for raw_item in values:
            item = raw_item if isinstance(raw_item, dict) else {}
            item_subscription_id = str(item.get("subscriptionId") or body.get("subscriptionId") or subscription_id or "")
            item_client_state = str(item.get("clientState") or body.get("clientState") or client_state or "")
            if not item_subscription_id:
                continue
            subscription = (
                await db_session.execute(
                    select(WebhookSubscription).where(
                        WebhookSubscription.provider_subscription_id == item_subscription_id
                    )
                )
            ).scalar_one_or_none()
            if not subscription or not subscription.active:
                continue
            expected_token = subscription.verification_token_hash
            received_token = hashlib.sha256(item_client_state.encode("utf-8")).hexdigest() if item_client_state else None
            if expected_token and (not received_token or not hmac.compare_digest(expected_token, received_token)):
                continue

            lifecycle_event = str(item.get("lifecycleEvent") or body.get("lifecycleEvent") or "") or None
            if lifecycle_only and not lifecycle_event:
                # The lifecycle URL only ever carries lifecycle events. Anything else
                # arriving here is unsolicited and must not queue provider work.
                lifecycle_event = "unknown"
            change_type = str(item.get("changeType") or body.get("changeType") or "") or None
            payload_hash = hashlib.sha256(
                json.dumps({"provider": provider, "subscription": item_subscription_id, "item": item}, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            existing = (
                await db_session.execute(
                    select(ConnectorNotification.id).where(
                        ConnectorNotification.provider == provider,
                        ConnectorNotification.subscription_id == item_subscription_id,
                        ConnectorNotification.payload_hash == payload_hash,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                continue
            notification = ConnectorNotification(
                connector_id=subscription.connector_id,
                scope_id=subscription.scope_id,
                provider=provider,
                subscription_id=item_subscription_id,
                payload_hash=payload_hash,
                resource=item.get("resource"),
                change_type=change_type,
                lifecycle_event=lifecycle_event,
                payload=item,
                status="received",
            )
            db_session.add(notification)
            now = datetime.utcnow()
            if lifecycle_event:
                notification.status = "processed"
                notification.processed_at = now
                subscription.last_lifecycle_at = now
                if lifecycle_event == "reauthorizationRequired":
                    subscription.reauthorization_required = True
                    subscription.last_error = "Microsoft Graph requires connector reauthorization"
                continue
            subscription.last_notification_at = now
            notification.status = "queued"
            sync_request = await enqueue_connector_sync(
                db_session,
                subscription.connector_id,
                scope_id=subscription.scope_id,
                reason="webhook",
                priority=10,
            )
            if sync_request.job_id and sync_request.status == "queued":
                claimable[sync_request.id] = (subscription.connector_id, sync_request.job_id)
        await db_session.commit()

    # Claimed after the notification inbox is durable, and through the queue's own
    # claim so a connector already walking is not woken a second time in parallel.
    # Anything the claim declines stays queued for dispatch_pending_sync_requests, so
    # the notification is never lost — only deferred.
    for request_id, (connector_id, job_id) in claimable.items():
        async with SessionLocal() as claim_db:
            await set_database_context(claim_db, None, True)
            claimed = await claim_sync_request(claim_db, request_id)
        if claimed:
            dispatches[(connector_id, job_id, request_id)] = None

    for connector_id, job_id, sync_request_id in dispatches:
        _dispatch_cloud_sync(connector_id, job_id, sync_request_id)
    return Response(status_code=202)


@router.post("/webhooks/sharepoint")
async def sharepoint_webhook(request: Request, validationToken: str | None = Query(default=None)) -> Response:
    if validationToken:
        return Response(content=validationToken, media_type="text/plain")
    return await _enqueue_webhook(request, "sharepoint")


@router.post("/webhooks/sharepoint/lifecycle")
async def sharepoint_lifecycle_webhook(
    request: Request, validationToken: str | None = Query(default=None)
) -> Response:
    # Graph validates the lifecycleNotificationUrl exactly as it validates the
    # notificationUrl: it POSTs a validationToken while CREATING the subscription and
    # expects it echoed back as text/plain. Without this the whole POST /subscriptions
    # call fails, so enabling update notifications fails for every SharePoint scope.
    if validationToken:
        return Response(content=validationToken, media_type="text/plain")
    return await _enqueue_webhook(request, "sharepoint", lifecycle_only=True)


@router.post("/webhooks/google-drive")
async def google_drive_webhook(request: Request) -> Response:
    return await _enqueue_webhook(request, "google_drive")


@router.post("/webhooks/google-drive/lifecycle")
async def google_drive_lifecycle_webhook(
    request: Request, validationToken: str | None = Query(default=None)
) -> Response:
    if validationToken:
        return Response(content=validationToken, media_type="text/plain")
    return await _enqueue_webhook(request, "google_drive", lifecycle_only=True)
