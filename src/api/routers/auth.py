import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any
from fastapi import APIRouter, Body, Depends, status, HTTPException, Query, Request, Response, Cookie
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
import jwt
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_db, get_current_user, require_permission, set_database_context
from src.repositories.user import UserRepository
from src.domain.auth import AuthService
from src.core.config import settings
from src.core.security import get_password_hash
from src.repositories.audit import AuditRepository
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import AccessGroup, Department, DepartmentManager, User, user_departments
from src.models.user import ExternalIdentity
from src.models.article import Article, article_departments
from src.models.chunk import ArticleChunk
from src.models.governance import Gap, PendingDraft
from src.models.ops import FeatureFlag
from src.models.sessions import RefreshSession
from src.domain.rbac import AuthorizationService, SCOPES, bootstrap_rbac
from src.domain.departments import resolve_active_department, normalize_department_name, lock_company_access_groups
from src.core.rate_limit import auth_rate_limiter
from src.domain import entra_auth
from sqlalchemy import delete, select, func, update
from sqlalchemy.orm import selectinload

router = APIRouter()

MANAGED_PRIMARY_ROLES = {"Admin", "CEO", "Reviewer", "Staff"}
LEGACY_DEPARTMENT_OWNER_ROLE = "Department Owner"


def _reject_cross_site_auth_request(request: Request) -> None:
    """Protect cookie-mutating auth endpoints from browser CSRF.

    Non-browser API clients do not normally send ``Origin`` and remain
    supported. Browsers do, so reject origins outside the explicit CORS
    allow-list before issuing, rotating, or clearing an authentication
    cookie.
    """
    origin = request.headers.get("origin")
    if origin and origin not in settings.cors_origin_list:
        raise HTTPException(status_code=403, detail="Cross-site authentication requests are not allowed")

class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    dept: str | None
    departments: list[dict[str, Any]] = []
    owned_departments: list[dict[str, Any]] = []
    role: str
    company_domain: str
    active: bool
    roles: list["RoleSummary"] = []
    permissions: list[str] = []
    permission_scopes: dict[str, str] = {}
    
    model_config = ConfigDict(from_attributes=True)


class RoleSummary(BaseModel):
    id: uuid.UUID
    name: str
    company_domain: str | None

    model_config = ConfigDict(from_attributes=True)

class TokenResponse(BaseModel):
    access_token: str
    # Refresh tokens are delivered only through the HttpOnly cookie.  The
    # optional field remains for compatibility with older API clients.
    refresh_token: str | None = None
    token_type: str
    user: UserResponse

class ManagedUserCreate(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=12, max_length=72)
    dept: str | None = Field(default=None, max_length=100)
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    role: str = Field(default="Staff", min_length=1, max_length=100)
    role_ids: list[uuid.UUID] | None = Field(default=None, min_length=1, max_length=20)
    owned_department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)


class ManagedUserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    dept: str | None = Field(default=None, max_length=100)
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    role: str | None = Field(default=None, min_length=1, max_length=100)
    password: str | None = Field(default=None, min_length=12, max_length=72)
    email: EmailStr | None = None
    active: bool | None = None
    role_ids: list[uuid.UUID] | None = Field(default=None, min_length=1, max_length=20)
    owned_department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)


class RoleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1_000)
    company_domain: str | None = Field(default=None, max_length=255)


class RoleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1_000)
    active: bool | None = None


class RolePermissionInput(BaseModel):
    permission_key: str = Field(min_length=1, max_length=100)
    scope: str = Field(default="company", min_length=1, max_length=20)


class AccessGroupInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AccessGroupMembersInput(BaseModel):
    user_ids: list[uuid.UUID] = Field(max_length=1_000)


class DepartmentInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    company_domain: str | None = Field(default=None, max_length=255)


class DepartmentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    active: bool | None = None


def _user_response(user: Any) -> dict[str, Any]:
    effective = AuthorizationService.get_effective_permissions(user)
    return {
        "id": user.id, "email": user.email, "name": user.name, "dept": user.dept,
        "departments": [
            {"id": department.id, "name": department.name}
            for department in getattr(user, "departments", [])
            if department.active and department.company_domain == user.company_domain
        ],
        "owned_departments": [
            {"id": assignment.department.id, "name": assignment.department.name}
            for assignment in getattr(user, "department_ownerships", [])
            if assignment.active and assignment.department.active
        ],
        "role": user.role, "company_domain": user.company_domain, "active": user.active,
        "roles": [{"id": role.id, "name": role.name, "company_domain": role.company_domain} for role in user.roles],
        "permissions": [item["key"] for item in effective],
        "permission_scopes": {item["key"]: item["scope"] for item in effective},
    }


async def _resolve_owned_departments(db: AsyncSession, company_domain: str, department_ids: list[uuid.UUID] | None) -> list[Department]:
    if not department_ids:
        return []
    unique_ids = list(dict.fromkeys(department_ids))
    departments = list((await db.execute(select(Department).where(
        Department.id.in_(unique_ids),
        Department.company_domain == company_domain,
        Department.active.is_(True),
    ))).scalars().all())
    if len(departments) != len(unique_ids):
        raise HTTPException(status_code=422, detail="Every owned department must be active and belong to the user's company")
    return departments


async def _resolve_membership_departments(
    db: AsyncSession,
    company_domain: str,
    department_ids: list[uuid.UUID] | None,
    primary_name: str | None,
) -> list[Department]:
    """Resolve a user's multi-department membership and canonicalize it."""
    if department_ids is None:
        department = await resolve_active_department(db, company_domain, primary_name, required=False)
        return [department] if department else []
    unique_ids = list(dict.fromkeys(department_ids))
    if not unique_ids:
        return []
    departments = list((await db.execute(select(Department).where(
        Department.id.in_(unique_ids),
        Department.company_domain == company_domain,
        Department.active.is_(True),
    ))).scalars().all())
    if len(departments) != len(unique_ids):
        raise HTTPException(status_code=422, detail="Every department membership must be active and belong to the user's company")
    by_id = {department.id: department for department in departments}
    return [by_id[department_id] for department_id in unique_ids]


async def _ensure_departments_have_single_owner(
    db: AsyncSession,
    user_id: uuid.UUID,
    departments: list[Department],
) -> None:
    if not departments:
        return
    conflicts = (await db.execute(
        select(Department.name)
        .join(DepartmentManager, DepartmentManager.department_id == Department.id)
        .join(User, User.id == DepartmentManager.user_id)
        .where(
            Department.id.in_([department.id for department in departments]),
            DepartmentManager.active.is_(True),
            DepartmentManager.user_id != user_id,
            User.active.is_(True),
        )
    )).scalars().all()
    if conflicts:
        raise HTTPException(status_code=409, detail=f"Department already has an owner: {', '.join(sorted(set(conflicts)))}")


def _ensure_owned_departments_are_memberships(memberships: list[Department], owned: list[Department]) -> None:
    membership_ids = {department.id for department in memberships}
    if any(department.id not in membership_ids for department in owned):
        raise HTTPException(status_code=422, detail="A user must be a member of every department they own")


def _apply_department_ownership(user: Any, departments: list[Department]) -> None:
    """Synchronize ownership rows without reinserting existing assignments."""
    desired_by_id = {department.id: department for department in departments}
    existing_ids: set[uuid.UUID] = set()
    for assignment in list(user.department_ownerships):
        department_id = assignment.department_id or getattr(assignment.department, "id", None)
        if department_id in desired_by_id:
            assignment.department = desired_by_id[department_id]
            assignment.active = True
            existing_ids.add(department_id)
        else:
            # Keep the historical row for auditability and to avoid a
            # delete/insert race; inactive assignments do not grant access.
            assignment.active = False
    for department_id, department in desired_by_id.items():
        if department_id not in existing_ids:
            user.department_ownerships.append(DepartmentManager(department=department, user=user, active=True))


def _is_global_user_manager(user: Any) -> bool:
    return AuthorizationService.has_permission(user, "user.manage", requested_scope="global")


def _is_global_role_manager(user: Any) -> bool:
    return AuthorizationService.has_permission(user, "role.manage", requested_scope="global")


async def _system_role(db: AsyncSession, name: str, company_domain: str | None) -> Role:
    role = (await db.execute(select(Role).where(Role.name == name, Role.company_domain == company_domain))).scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=422, detail=f"The requested system role is unavailable: {name}")
    return role


async def _set_primary_role(db: AsyncSession, user: Any, name: str) -> None:
    """Keep the legacy display role and authoritative RBAC relationship aligned."""
    role_company = None if name == "Admin" else user.company_domain
    role = await _system_role(db, name, role_company)
    user.roles = [role]
    user.role = name


def _validate_role_assignment_authority(current_user: Any, roles: list[Role]) -> None:
    """Company managers must never grant a global role or broader permissions."""
    if any(role.name.casefold() == LEGACY_DEPARTMENT_OWNER_ROLE.casefold() for role in roles):
        raise HTTPException(status_code=422, detail="Department ownership is managed separately from roles")
    if _is_global_user_manager(current_user):
        return
    if any(role.company_domain is None or role.name in {"Admin", "CEO"} for role in roles):
        raise HTTPException(status_code=403, detail="Only global user managers can assign global or executive roles")
    rank = {"own": 1, "department": 2, "company": 3, "global": 4}
    effective = {item["key"]: item["scope"] for item in AuthorizationService.get_effective_permissions(current_user)}
    for role in roles:
        for assignment in role.permissions:
            if assignment.permission and rank.get(assignment.scope, 0) > rank.get(effective.get(assignment.permission.key, ""), 0):
                raise HTTPException(
                    status_code=403,
                    detail="You cannot assign a role with permissions broader than your own",
                )


async def _store_refresh_session(db: AsyncSession, user: Any, token: str) -> None:
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    expires_at = datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc).replace(tzinfo=None)
    db.add(RefreshSession(
        user_id=user.id,
        token_hash=hashlib.sha256(token.encode("utf-8")).hexdigest(),
        expires_at=expires_at,
    ))


def _refresh_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _refresh_claims_from_token(refresh_value: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(refresh_value, settings.SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != "refresh" or not isinstance(payload.get("sub"), str):
            raise jwt.InvalidTokenError("Not a refresh token")
        subject = payload["sub"].strip().lower()
        if "@" not in subject:
            raise jwt.InvalidTokenError("Invalid refresh subject")
        payload["sub"] = subject
        return payload
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Refresh token is invalid or expired") from exc


def _refresh_subject_from_token(refresh_value: str) -> str:
    return str(_refresh_claims_from_token(refresh_value)["sub"])


async def _revoke_refresh_sessions(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Make every refresh token unusable after a sensitive account change."""
    await db.execute(
        RefreshSession.__table__.update()
        .where(RefreshSession.user_id == user_id, RefreshSession.revoked_at.is_(None))
        .values(revoked_at=datetime.utcnow())
    )


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: Any = Depends(require_permission("user.read")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    users = await UserRepository(db).list_users(offset=offset, limit=limit, viewer=current_user)
    return [_user_response(user) for user in users]


@router.get("/groups")
async def list_access_groups(current_user: Any = Depends(require_permission("user.read")), db: AsyncSession = Depends(get_db)) -> Any:
    global_admin = AuthorizationService.can_view_all_access_groups(current_user)
    groups = await UserRepository(db).get_all_groups(None if global_admin else current_user.company_domain)
    return [{"id": group.id, "name": group.name, "company_domain": group.company_domain, "bitmask_position": group.bitmask_position} for group in groups]


@router.get("/departments")
async def list_departments(current_user: Any = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> Any:
    global_admin = AuthorizationService.can_view_all_access_groups(current_user)
    can_manage = AuthorizationService.has_permission(current_user, "user.manage", requested_scope="company")
    stmt = select(Department).options(
        selectinload(Department.managers).selectinload(DepartmentManager.user)
    ).order_by(Department.company_domain, Department.name)
    if not global_admin:
        stmt = stmt.where(Department.company_domain == current_user.company_domain)
        if not can_manage:
            stmt = stmt.where(Department.active.is_(True))
    rows = (await db.execute(stmt)).scalars().all()
    return [{
        "id": item.id,
        "name": item.name,
        "company_domain": item.company_domain,
        "active": item.active,
        "owner": next((
            {"id": assignment.user.id, "name": assignment.user.name, "email": assignment.user.email}
            for assignment in item.managers
            if assignment.active and assignment.user and assignment.user.active
        ), None),
    } for item in rows]


@router.post("/departments", status_code=status.HTTP_201_CREATED)
async def create_department(payload: DepartmentInput, current_user: Any = Depends(require_permission("user.manage")), db: AsyncSession = Depends(get_db)) -> Any:
    name = normalize_department_name(payload.name)
    if name is None:
        raise HTTPException(status_code=422, detail="Department name cannot be blank")
    can_manage_globally = _is_global_user_manager(current_user)
    company_domain = (payload.company_domain or current_user.company_domain).strip().lower()
    if not can_manage_globally and company_domain != current_user.company_domain:
        raise HTTPException(status_code=403, detail="You can only create departments inside your company")
    existing = (await db.execute(select(Department).where(Department.company_domain == company_domain, func.lower(Department.name) == name.lower()))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Department already exists")
    await lock_company_access_groups(db, f"department:{company_domain}")
    department_group = (await db.execute(select(AccessGroup).where(
        AccessGroup.company_domain == company_domain,
        func.lower(AccessGroup.name) == f"dept_{name.lower()}".lower(),
    ))).scalar_one_or_none()
    if department_group is None:
        next_position = int((await db.execute(select(func.coalesce(func.max(AccessGroup.bitmask_position), -1)).where(
            AccessGroup.company_domain == company_domain,
        ))).scalar_one()) + 1
        if next_position >= 62:
            raise HTTPException(status_code=422, detail="The maximum number of access groups has been reached")
        db.add(AccessGroup(name=f"dept_{name.lower()}", company_domain=company_domain, bitmask_position=next_position))
    item = Department(company_domain=company_domain, name=name, active=True)
    db.add(item)
    await db.commit()
    await db.refresh(item)
    await AuditRepository(db).record(current_user.id, "department_create", "department", str(item.id))
    return {"id": item.id, "name": item.name, "company_domain": item.company_domain, "active": item.active}


async def _department_for_management(db: AsyncSession, department_id: uuid.UUID, current_user: Any) -> Department:
    stmt = select(Department).where(Department.id == department_id)
    if not _is_global_user_manager(current_user):
        stmt = stmt.where(Department.company_domain == current_user.company_domain)
    item = (await db.execute(stmt)).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Department not found")
    return item


@router.patch("/departments/{department_id}")
async def update_department(
    department_id: uuid.UUID,
    payload: DepartmentUpdate,
    current_user: Any = Depends(require_permission("user.manage")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    item = await _department_for_management(db, department_id, current_user)
    if payload.name is None and payload.active is None:
        raise HTTPException(status_code=422, detail="Provide a new name or active status")

    if payload.name is not None:
        name = normalize_department_name(payload.name)
        if name is None:
            raise HTTPException(status_code=422, detail="Department name cannot be blank")
        duplicate = (await db.execute(select(Department).where(
            Department.company_domain == item.company_domain,
            Department.id != item.id,
            func.lower(Department.name) == name.lower(),
        ))).scalar_one_or_none()
        if duplicate:
            raise HTTPException(status_code=409, detail="Department already exists")
        old_name = item.name
        old_group_name = f"dept_{old_name.lower()}"
        new_group_name = f"dept_{name.lower()}"
        old_group = (await db.execute(select(AccessGroup).where(
            AccessGroup.company_domain == item.company_domain,
            func.lower(AccessGroup.name) == old_group_name.lower(),
        ))).scalar_one_or_none()
        new_group = (await db.execute(select(AccessGroup).where(
            AccessGroup.company_domain == item.company_domain,
            func.lower(AccessGroup.name) == new_group_name.lower(),
        ))).scalar_one_or_none()
        if old_group and new_group and old_group.id != new_group.id:
            raise HTTPException(status_code=409, detail="The target department name already has an access group; consolidate that group before renaming")
        item.name = name
        # Department names are currently denormalized in content and user
        # records. Keep the rename atomic so access rules do not point at a
        # stale department name after the master record changes.
        await db.execute(update(User).where(User.company_domain == item.company_domain, User.dept == old_name).values(dept=name))
        await db.execute(update(Article).where(Article.company_domain == item.company_domain, Article.dept == old_name).values(dept=name))
        await db.execute(update(ArticleChunk).where(
            ArticleChunk.department_id == old_name,
            ArticleChunk.article_id.in_(select(Article.id).where(
                Article.company_domain == item.company_domain,
                Article.dept == name,
            )),
        ).values(department_id=name))
        await db.execute(update(PendingDraft).where(PendingDraft.company_domain == item.company_domain, PendingDraft.dept == old_name).values(dept=name))
        await db.execute(update(Gap).where(Gap.company_domain == item.company_domain, Gap.dept == old_name).values(dept=name))
        await db.execute(update(FeatureFlag).where(FeatureFlag.department == old_name).values(department=name))
        if old_group:
            old_group.name = new_group_name
    if payload.active is not None:
        item.active = payload.active
    await db.commit()
    await db.refresh(item)
    await AuditRepository(db).record(current_user.id, "department_update", "department", str(item.id))
    return {"id": item.id, "name": item.name, "company_domain": item.company_domain, "active": item.active}


@router.delete("/departments/{department_id}")
async def delete_department(
    department_id: uuid.UUID,
    current_user: Any = Depends(require_permission("user.manage")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    item = await _department_for_management(db, department_id, current_user)
    references = {
        "users": int((await db.execute(select(func.count()).select_from(User).where(User.company_domain == item.company_domain, User.dept == item.name))).scalar_one()),
        "user memberships": int((await db.execute(select(func.count()).select_from(user_departments).where(user_departments.c.department_id == item.id))).scalar_one()),
        "articles": int((await db.execute(select(func.count()).select_from(Article).where(Article.company_domain == item.company_domain, Article.dept == item.name))).scalar_one()),
        "article associations": int((await db.execute(select(func.count()).select_from(article_departments).where(article_departments.c.department_id == item.id))).scalar_one()),
        "pending drafts": int((await db.execute(select(func.count()).select_from(PendingDraft).where(PendingDraft.company_domain == item.company_domain, PendingDraft.dept == item.name))).scalar_one()),
        "gaps": int((await db.execute(select(func.count()).select_from(Gap).where(Gap.company_domain == item.company_domain, Gap.dept == item.name))).scalar_one()),
        "owners": int((await db.execute(select(func.count()).select_from(DepartmentManager).where(DepartmentManager.department_id == item.id))).scalar_one()),
        "feature flags": int((await db.execute(select(func.count()).select_from(FeatureFlag).where(FeatureFlag.department == item.name))).scalar_one()),
    }
    used_by = [label for label, count in references.items() if count]
    if used_by:
        raise HTTPException(
            status_code=409,
            detail=f"Department is still in use by {', '.join(used_by)}. Deactivate it instead of deleting it.",
        )
    await db.delete(item)
    await db.commit()
    await AuditRepository(db).record(current_user.id, "department_delete", "department", str(department_id))
    return {"id": department_id, "deleted": True}


@router.post("/groups", status_code=status.HTTP_201_CREATED)
async def create_access_group(payload: AccessGroupInput, current_user: Any = Depends(require_permission("user.manage")), db: AsyncSession = Depends(get_db)) -> Any:
    company_domain = current_user.company_domain
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Access group name cannot be blank")
    await lock_company_access_groups(db, company_domain)
    exists = (await db.execute(select(AccessGroup).where(AccessGroup.company_domain == company_domain, func.lower(AccessGroup.name) == name.lower()))).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=409, detail="An access group with this name already exists")
    next_position = int((await db.execute(select(func.coalesce(func.max(AccessGroup.bitmask_position), 0)).where(AccessGroup.company_domain == company_domain))).scalar_one()) + 1
    if next_position >= 62:
        raise HTTPException(status_code=422, detail="The maximum number of access groups has been reached")
    group = AccessGroup(name=name, company_domain=company_domain, bitmask_position=next_position)
    created = await UserRepository(db).create_group(group)
    await AuditRepository(db).record(current_user.id, "group_create", "access_group", str(created.id))
    return {"id": created.id, "name": created.name, "company_domain": created.company_domain, "bitmask_position": created.bitmask_position}


@router.put("/groups/{group_id}/members")
async def replace_access_group_members(group_id: uuid.UUID, payload: AccessGroupMembersInput, current_user: Any = Depends(require_permission("user.manage")), db: AsyncSession = Depends(get_db)) -> Any:
    can_manage_globally = AuthorizationService.has_permission(current_user, "user.manage", requested_scope="global")
    group = await UserRepository(db).get_group_by_id(
        group_id,
        company_domain=None if can_manage_globally else current_user.company_domain,
    )
    if not group:
        raise HTTPException(status_code=404, detail="Access group not found")
    users = await UserRepository(db).get_by_ids(
        list(set(payload.user_ids)),
        company_domain=group.company_domain,
    )
    if len(users) != len(set(payload.user_ids)):
        raise HTTPException(status_code=422, detail="Every group member must be in the same company")
    group.users = list(users)
    await db.commit()
    await AuditRepository(db).record(current_user.id, "group_members_update", "access_group", str(group.id))
    return {"id": group.id, "member_ids": [str(user.id) for user in users]}


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_managed_user(
    user_in: ManagedUserCreate,
    current_user: Any = Depends(require_permission("user.manage")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    domain = str(user_in.email).lower().rsplit("@", 1)[-1]
    can_manage_globally = _is_global_user_manager(current_user)
    if not can_manage_globally and domain != current_user.company_domain:
        raise HTTPException(status_code=403, detail="Users must be created inside your company")
    if not can_manage_globally and user_in.role in {"Admin", "CEO"}:
        raise HTTPException(status_code=403, detail="Only global user managers can create executive or Admin accounts")
    memberships = await _resolve_membership_departments(db, domain, user_in.department_ids, user_in.dept)
    department = await resolve_active_department(db, domain, user_in.dept, required=False) if user_in.dept else (memberships[0] if memberships else None)
    if department and memberships and department.id not in {item.id for item in memberships}:
        raise HTTPException(status_code=422, detail="The primary department must be one of the selected memberships")
    owned_departments = await _resolve_owned_departments(db, domain, user_in.owned_department_ids)
    _ensure_owned_departments_are_memberships(memberships, owned_departments)
    requested_roles: list[Role] | None = None
    if user_in.role_ids is not None:
        requested_roles = list((await db.execute(
            select(Role).where(
                Role.id.in_(user_in.role_ids),
                (Role.company_domain == domain) | Role.company_domain.is_(None),
            )
            .options(selectinload(Role.permissions).selectinload(RolePermission.permission))
        )).scalars().all())
        if len(requested_roles) != len(set(user_in.role_ids)) or any(role.company_domain not in (None, domain) for role in requested_roles):
            raise HTTPException(status_code=403, detail="Invalid role assignment")
        if any(not role.active for role in requested_roles):
            raise HTTPException(status_code=422, detail="Inactive roles cannot be assigned")
        if not requested_roles:
            raise HTTPException(status_code=422, detail="At least one role is required")
        _validate_role_assignment_authority(current_user, requested_roles)
    if user_in.role.casefold() == LEGACY_DEPARTMENT_OWNER_ROLE.casefold():
        raise HTTPException(status_code=422, detail="Department ownership is managed separately from roles")
    await _ensure_departments_have_single_owner(db, uuid.UUID(int=0), owned_departments)
    password = user_in.password
    user = await AuthService(UserRepository(db)).register_user(
        email=str(user_in.email), name=user_in.name, password=password,
        dept=department.name if department else None, role=user_in.role,
        # Executive/Admin creation was checked above. Preserve valid
        # company-scoped Reviewer assignments.
        allow_privileged_role=True,
    )
    await bootstrap_rbac(db)
    user = await UserRepository(db).get_by_id(user.id)
    user.departments = memberships
    user.dept = department.name if department else None
    if requested_roles is not None:
        user.roles = list(requested_roles)
        user.role = next((role.name for role in requested_roles if role.name in MANAGED_PRIMARY_ROLES), requested_roles[0].name)
    else:
        user.role = user.role if user.role in MANAGED_PRIMARY_ROLES else "Staff"
    owned_departments = await _resolve_owned_departments(db, user.company_domain, user_in.owned_department_ids)
    await _ensure_departments_have_single_owner(db, user.id, owned_departments)
    if user_in.owned_department_ids is not None:
        _apply_department_ownership(user, owned_departments)
    await db.commit()
    # Re-FETCH rather than refresh(). db.refresh() expires the instance and reloads only
    # its column attributes, discarding the eager loads this function set up above — and
    # _user_response walks user.roles -> role.permissions, which then lazy-loads inside an
    # async request and raises MissingGreenlet. The 500 is reported after the user has
    # already been committed, so it reads as "creation failed" when the row exists.
    user = await UserRepository(db).get_by_id(user.id)
    await AuditRepository(db).record(current_user.id, "user_create", "user", str(user.id))
    return _user_response(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_managed_user(
    user_id: uuid.UUID,
    user_in: ManagedUserUpdate,
    current_user: Any = Depends(require_permission("user.manage")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id, viewer=current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not AuthorizationService.has_permission(current_user, "user.manage", requested_scope="global") and user.company_domain != current_user.company_domain:
        raise HTTPException(status_code=403, detail="Users must belong to your company")
    can_manage_globally = _is_global_user_manager(current_user)
    if user.id == current_user.id and user_in.role and user_in.role != "Admin":
        raise HTTPException(status_code=400, detail="You cannot remove your own Admin role")
    if user_in.name is not None:
        user.name = user_in.name
    memberships_changed = False
    if "dept" in user_in.model_fields_set:
        memberships_changed = True
        memberships = await _resolve_membership_departments(db, user.company_domain, user_in.department_ids, user_in.dept)
        primary = await resolve_active_department(db, user.company_domain, user_in.dept, required=False) if user_in.dept else None
        if primary and memberships and primary.id not in {item.id for item in memberships}:
            raise HTTPException(status_code=422, detail="The primary department must be one of the selected memberships")
        user.departments = memberships
        user.dept = primary.name if primary else (memberships[0].name if memberships else None)
    elif user_in.department_ids is not None:
        memberships_changed = True
        memberships = await _resolve_membership_departments(db, user.company_domain, user_in.department_ids, user.dept)
        user.departments = memberships
        membership_names = {item.name for item in memberships}
        user.dept = user.dept if user.dept in membership_names else (memberships[0].name if memberships else None)
    if user_in.role is not None:
        if user_in.role not in MANAGED_PRIMARY_ROLES:
            raise HTTPException(status_code=422, detail="Invalid role")
        if not can_manage_globally and user_in.role in {"Admin", "CEO"}:
            raise HTTPException(status_code=403, detail="Company managers cannot assign global roles")
        if user_in.role_ids is None:
            await _set_primary_role(db, user, user_in.role)
    if user_in.role_ids is not None:
        roles = (await db.execute(
            select(Role).where(
                Role.id.in_(user_in.role_ids),
                (Role.company_domain == user.company_domain) | Role.company_domain.is_(None),
            )
            .options(selectinload(Role.permissions).selectinload(RolePermission.permission))
        )).scalars().all()
        if any(role.company_domain not in (None, user.company_domain) for role in roles):
            raise HTTPException(status_code=403, detail="A user can only receive roles from their company")
        if any(not role.active for role in roles):
            raise HTTPException(status_code=422, detail="Inactive roles cannot be assigned")
        if not roles:
            raise HTTPException(status_code=422, detail="At least one role is required")
        _validate_role_assignment_authority(current_user, roles)
        if user.id == current_user.id and can_manage_globally and not any(role.name == "Admin" and role.company_domain is None for role in roles):
            raise HTTPException(status_code=400, detail="You cannot remove your own global Admin role")
        user.roles = list(roles)
        user.role = next((role.name for role in roles if role.name in MANAGED_PRIMARY_ROLES), roles[0].name)
    if user_in.owned_department_ids is not None:
        owned_departments = await _resolve_owned_departments(db, user.company_domain, user_in.owned_department_ids)
    else:
        owned_departments = [assignment.department for assignment in user.department_ownerships if assignment.active and assignment.department.active]
    if memberships_changed:
        if user_in.owned_department_ids is not None:
            _ensure_owned_departments_are_memberships(memberships, owned_departments)
        else:
            membership_ids = {department.id for department in memberships}
            retained_owned = [department for department in owned_departments if department.id in membership_ids]
            if len(retained_owned) != len(owned_departments):
                _apply_department_ownership(user, retained_owned)
            owned_departments = retained_owned
    await _ensure_departments_have_single_owner(db, user.id, owned_departments)
    if user_in.owned_department_ids is not None:
        _apply_department_ownership(user, owned_departments)
    if user_in.email is not None:
        new_email = str(user_in.email).lower()
        if not can_manage_globally and new_email.rsplit("@", 1)[-1] != current_user.company_domain:
            raise HTTPException(status_code=403, detail="Employee email must remain in the company domain")
        if new_email.rsplit("@", 1)[-1] != user.company_domain:
            raise HTTPException(status_code=422, detail="Changing a user's company domain is not supported; create a new account instead")
        user.email = new_email
        user.company_domain = new_email.rsplit("@", 1)[-1]
    if user_in.active is not None:
        if user.id == current_user.id and not user_in.active:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
        user.active = user_in.active
    if user_in.password:
        from src.core.security import get_password_hash
        user.password_hash = get_password_hash(user_in.password)
        user.auth_version += 1
    if user_in.password or user_in.active is False:
        await _revoke_refresh_sessions(db, user.id)
    updated = await repo.update(user)
    await AuditRepository(db).record(current_user.id, "user_update", "user", str(user.id))
    return _user_response(updated)


@router.delete("/users/{user_id}", response_model=UserResponse)
async def deactivate_user(
    user_id: uuid.UUID,
    current_user: Any = Depends(require_permission("user.manage")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id, viewer=current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not AuthorizationService.has_permission(current_user, "user.manage", requested_scope="global") and user.company_domain != current_user.company_domain:
        raise HTTPException(status_code=403, detail="Users must belong to your company")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
    user.active = False
    for assignment in user.department_ownerships:
        assignment.active = False
    await _revoke_refresh_sessions(db, user.id)
    if user.role == "CEO" or any(role.name == "CEO" for role in user.roles):
        await _set_primary_role(db, user, "Staff")
    updated = await repo.update(user)
    await AuditRepository(db).record(current_user.id, "user_deactivate", "user", str(user.id))
    return _user_response(updated)


@router.post("/companies/{company_domain}/ceo", response_model=UserResponse)
async def assign_company_ceo(
    company_domain: str,
    user_id: uuid.UUID,
    current_user: Any = Depends(require_permission("user.manage")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    if not _is_global_user_manager(current_user):
        raise HTTPException(status_code=403, detail="Only global user managers can assign a company CEO")
    repo = UserRepository(db)
    users = await repo.list_users(viewer=current_user)
    company_users = [user for user in users if user.company_domain == company_domain.lower()]
    target = next((user for user in company_users if user.id == user_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Employee not found in this company")
    if not target.active:
        raise HTTPException(status_code=422, detail="A deactivated employee cannot be assigned as CEO")
    ceo_role = await _system_role(db, "CEO", target.company_domain)
    staff_role = await _system_role(db, "Staff", target.company_domain)
    for user in company_users:
        if (user.role == "CEO" or any(role.name == "CEO" for role in user.roles)) and user.id != target.id:
            user.role = "Staff"
            user.roles = [role for role in user.roles if role.name != "CEO"] or [staff_role]
            await repo.update(user)
            await AuditRepository(db).record(current_user.id, "ceo_change", "user", str(user.id))
    target.role = "CEO"
    target.roles = [role for role in target.roles if role.name != "CEO"] + [ceo_role]
    target = await repo.update(target)
    await AuditRepository(db).record(current_user.id, "ceo_change", "user", str(target.id))
    return _user_response(target)


@router.get("/permissions")
async def list_permissions(current_user: Any = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> Any:
    if not (
        AuthorizationService.has_permission(current_user, "permission.manage", requested_scope="company")
        or AuthorizationService.has_permission(current_user, "role.manage", requested_scope="company")
    ):
        raise HTTPException(status_code=403, detail="Missing permission to view the permission catalog")
    permissions = (await db.execute(select(Permission).order_by(Permission.key))).scalars().all()
    return [{"id": item.id, "key": item.key, "name": item.name, "description": item.description, "system": item.system} for item in permissions]


@router.get("/roles")
async def list_roles(current_user: Any = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> Any:
    if not (
        AuthorizationService.has_permission(current_user, "role.manage", requested_scope="company")
        or AuthorizationService.has_permission(current_user, "user.manage", requested_scope="company")
    ):
        raise HTTPException(status_code=403, detail="Missing permission to view assignable roles")
    stmt = select(Role).where(func.lower(Role.name) != LEGACY_DEPARTMENT_OWNER_ROLE.casefold()).options(selectinload(Role.permissions).selectinload(RolePermission.permission)).order_by(Role.company_domain, Role.name)
    if not (
        AuthorizationService.has_permission(current_user, "role.manage", requested_scope="global")
        or AuthorizationService.has_permission(current_user, "user.manage", requested_scope="global")
    ):
        stmt = stmt.where((Role.company_domain == current_user.company_domain) | (Role.company_domain.is_(None)))
    roles = (await db.execute(stmt)).scalars().all()
    return [{"id": role.id, "name": role.name, "description": role.description, "company_domain": role.company_domain, "active": role.active, "system": role.system,
             "permissions": [{"permission_key": item.permission.key, "scope": item.scope} for item in role.permissions if item.permission]} for role in roles]


@router.post("/roles", status_code=status.HTTP_201_CREATED)
async def create_role(payload: RoleCreate, current_user: Any = Depends(require_permission("role.manage")), db: AsyncSession = Depends(get_db)) -> Any:
    can_manage_globally = AuthorizationService.has_permission(current_user, "role.manage", requested_scope="global")
    company_domain = payload.company_domain.lower() if payload.company_domain else (None if can_manage_globally else current_user.company_domain)
    if not can_manage_globally and company_domain != current_user.company_domain:
        raise HTTPException(status_code=403, detail="Roles must belong to your company")
    role_name = payload.name.strip()
    if not role_name:
        raise HTTPException(status_code=422, detail="Role name cannot be blank")
    if role_name.casefold() == LEGACY_DEPARTMENT_OWNER_ROLE.casefold():
        raise HTTPException(status_code=422, detail="Department ownership is managed separately from roles")
    exists = (await db.execute(select(Role).where(func.lower(Role.name) == role_name.lower(), Role.company_domain == company_domain))).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=409, detail="A role with this name already exists")
    role = Role(name=role_name, description=payload.description, company_domain=company_domain, system=False, active=True)
    db.add(role)
    await db.commit()
    await db.refresh(role)
    await AuditRepository(db).record(current_user.id, "role_create", "role", str(role.id))
    return {"id": role.id, "name": role.name, "description": role.description, "company_domain": role.company_domain, "active": role.active, "system": role.system, "permissions": []}


@router.patch("/roles/{role_id}")
async def update_role(role_id: uuid.UUID, payload: RoleUpdate, current_user: Any = Depends(require_permission("role.manage")), db: AsyncSession = Depends(get_db)) -> Any:
    can_manage_globally = AuthorizationService.has_permission(current_user, "role.manage", requested_scope="global")
    role_stmt = select(Role).where(Role.id == role_id)
    if not can_manage_globally:
        role_stmt = role_stmt.where(Role.company_domain == current_user.company_domain)
    role = (await db.execute(role_stmt.options(selectinload(Role.permissions).selectinload(RolePermission.permission)))).scalar_one_or_none()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    # Global roles are shared by every company.  Letting a company-scoped
    # manager rename, deactivate, or otherwise edit one would change another
    # tenant's authorization model even if that manager cannot assign it.
    if (role.company_domain is None and not can_manage_globally) or (
        role.company_domain is not None and role.company_domain != current_user.company_domain and not can_manage_globally
    ):
        raise HTTPException(status_code=403, detail="Role is outside your management scope")
    if role.system and payload.name and payload.name != role.name:
        raise HTTPException(status_code=400, detail="System role names cannot be changed")
    if payload.name is not None:
        role_name = payload.name.strip()
        if not role_name:
            raise HTTPException(status_code=422, detail="Role name cannot be blank")
        if role_name.casefold() == LEGACY_DEPARTMENT_OWNER_ROLE.casefold():
            raise HTTPException(status_code=422, detail="Department ownership is managed separately from roles")
        duplicate = (await db.execute(select(Role).where(
            Role.id != role.id,
            Role.company_domain == role.company_domain,
            func.lower(Role.name) == role_name.lower(),
        ))).scalar_one_or_none()
        if duplicate:
            raise HTTPException(status_code=409, detail="A role with this name already exists")
        role.name = role_name
    if payload.description is not None: role.description = payload.description
    if payload.active is not None and not role.system: role.active = payload.active
    await db.commit()
    await AuditRepository(db).record(current_user.id, "role_update", "role", str(role.id))
    return {"id": role.id, "name": role.name, "description": role.description, "company_domain": role.company_domain, "active": role.active, "system": role.system,
            "permissions": [{"key": item.permission.key, "scope": item.scope} for item in role.permissions if item.permission]}


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(role_id: uuid.UUID, current_user: Any = Depends(require_permission("role.manage")), db: AsyncSession = Depends(get_db)) -> None:
    can_manage_globally = AuthorizationService.has_permission(current_user, "role.manage", requested_scope="global")
    role_stmt = select(Role).where(Role.id == role_id)
    if not can_manage_globally:
        role_stmt = role_stmt.where(Role.company_domain == current_user.company_domain)
    role = (await db.execute(role_stmt.options(selectinload(Role.users)))).scalar_one_or_none()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    if role.system:
        raise HTTPException(status_code=400, detail="System roles cannot be deleted")
    if (role.company_domain is None and not can_manage_globally) or (
        role.company_domain is not None and role.company_domain != current_user.company_domain and not can_manage_globally
    ):
        raise HTTPException(status_code=403, detail="Role is outside your management scope")
    if role.users:
        raise HTTPException(status_code=409, detail="Remove users from this role before deleting it")
    await db.delete(role)
    await db.commit()
    await AuditRepository(db).record(current_user.id, "role_delete", "role", str(role.id))


@router.put("/roles/{role_id}/permissions")
async def replace_role_permissions(role_id: uuid.UUID, payload: list[RolePermissionInput] = Body(..., max_length=100), current_user: Any = Depends(require_permission("permission.manage")), db: AsyncSession = Depends(get_db)) -> Any:
    can_manage_globally = AuthorizationService.has_permission(current_user, "permission.manage", requested_scope="global")
    role_stmt = select(Role).where(Role.id == role_id)
    if not can_manage_globally:
        role_stmt = role_stmt.where(Role.company_domain == current_user.company_domain)
    role = (await db.execute(role_stmt)).scalar_one_or_none()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    if role.system:
        raise HTTPException(status_code=400, detail="System role permissions cannot be changed")
    if (role.company_domain is None and not can_manage_globally) or (
        role.company_domain is not None and role.company_domain != current_user.company_domain and not can_manage_globally
    ):
        raise HTTPException(status_code=403, detail="Role is outside your management scope")
    if any(item.scope not in SCOPES for item in payload):
        raise HTTPException(status_code=422, detail="Invalid permission scope")
    permission_keys = [item.permission_key for item in payload]
    if len(permission_keys) != len(set(permission_keys)):
        raise HTTPException(status_code=422, detail="A permission can only be assigned once per role")
    if (role.company_domain is not None or not can_manage_globally) and any(item.scope == "global" for item in payload):
        raise HTTPException(status_code=403, detail="Only global permission managers can grant global scopes to global roles")
    keys = {item.permission_key for item in payload}
    permissions = (await db.execute(select(Permission).where(Permission.key.in_(keys)))).scalars().all()
    by_key = {item.key: item for item in permissions}
    if len(by_key) != len(keys):
        raise HTTPException(status_code=422, detail="Unknown permission key")
    rank = {"own": 1, "department": 2, "company": 3, "global": 4}
    effective_scopes = {item["key"]: item["scope"] for item in AuthorizationService.get_effective_permissions(current_user)}
    if not can_manage_globally and any(rank[item.scope] > rank.get(effective_scopes.get(item.permission_key, ""), 0) for item in payload):
        raise HTTPException(status_code=403, detail="You cannot delegate permissions broader than your own")
    await db.execute(delete(RolePermission).where(RolePermission.role_id == role.id))
    for item in payload:
        db.add(RolePermission(role_id=role.id, permission_id=by_key[item.permission_key].id, scope=item.scope))
    await db.commit()
    await AuditRepository(db).record(current_user.id, "role_permissions_update", "role", str(role.id))
    return {"role_id": role.id, "permissions": [item.model_dump() for item in payload]}


@router.get("/users/{user_id}/roles")
async def get_user_roles(user_id: uuid.UUID, current_user: Any = Depends(require_permission("user.read")), db: AsyncSession = Depends(get_db)) -> Any:
    user = await UserRepository(db).get_by_id(user_id, viewer=current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return [{"id": role.id, "name": role.name, "company_domain": role.company_domain} for role in user.roles]


@router.put("/users/{user_id}/roles")
async def replace_user_roles(user_id: uuid.UUID, role_ids: list[uuid.UUID], current_user: Any = Depends(require_permission("user.manage")), db: AsyncSession = Depends(get_db)) -> Any:
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id, viewer=current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not role_ids:
        raise HTTPException(status_code=422, detail="At least one role is required")
    roles = (await db.execute(
        select(Role).where(
            Role.id.in_(role_ids),
            (Role.company_domain == user.company_domain) | Role.company_domain.is_(None),
        )
        .options(selectinload(Role.permissions).selectinload(RolePermission.permission))
    )).scalars().all()
    if len(roles) != len(set(role_ids)) or any(role.company_domain not in (None, user.company_domain) for role in roles):
        raise HTTPException(status_code=403, detail="Invalid role assignment")
    if any(not role.active for role in roles):
        raise HTTPException(status_code=422, detail="Inactive roles cannot be assigned")
    _validate_role_assignment_authority(current_user, roles)
    if user.id == current_user.id and not any(role.name == "Admin" and role.company_domain is None for role in roles) and AuthorizationService.has_permission(current_user, "user.manage", requested_scope="global"):
        raise HTTPException(status_code=400, detail="You cannot remove your own global Admin role")
    user.roles = list(roles)
    user.role = next((role.name for role in roles if role.name in MANAGED_PRIMARY_ROLES), roles[0].name)
    await db.commit()
    # Re-FETCH, not refresh() — same reason as create_managed_user above: refresh discards
    # the eager loads, and _user_response then lazy-loads user.roles inside an async
    # request and raises MissingGreenlet.
    user = await UserRepository(db).get_by_id(user.id)
    await AuditRepository(db).record(current_user.id, "user_roles_update", "user", str(user.id))
    return _user_response(user)

@router.post("/login", response_model=TokenResponse, response_model_exclude_none=True)
async def login(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> Any:
    _reject_cross_site_auth_request(request)
    # OAuth2 request form uses username as email field
    client_ip = request.client.host if request and request.client else "unknown"
    username = form_data.username.strip().lower()
    if "@" not in username:
        raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    await set_database_context(db, username.rsplit("@", 1)[1])
    # Limit each account across every source IP as well as each source IP
    # across accounts.  The former prevents a distributed credential-stuffing
    # attack from bypassing an IP+username composite key.
    account_allowed, account_retry = await auth_rate_limiter.allow(f"account:{username}")
    ip_allowed, ip_retry = await auth_rate_limiter.allow(f"ip:{client_ip}")
    if not account_allowed or not ip_allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts",
            headers={"Retry-After": str(max(account_retry, ip_retry))},
        )
    user_repo = UserRepository(db)
    auth_service = AuthService(user_repo)
    user = await auth_service.authenticate_user(email=form_data.username, password=form_data.password)
    if not user.roles:
        await bootstrap_rbac(db)
        user = await user_repo.get_by_id(user.id)
    token = auth_service.create_token(user)
    refresh = auth_service.create_refresh_token(user)
    await _store_refresh_session(db, user, refresh)
    await db.commit()
    _set_auth_cookies(response, token, refresh)
    return {"access_token": token, "token_type": "bearer", "user": _user_response(user)}

class RefreshRequest(BaseModel):
    refresh_token: str


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    secure = settings.ENVIRONMENT.lower() in {"production", "prod"}
    response.set_cookie("access_token", access_token, httponly=True, secure=secure, samesite="lax", max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, path="/")
    response.set_cookie("refresh_token", refresh_token, httponly=True, secure=secure, samesite="lax", max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400, path=f"{settings.API_V1_STR}/auth")


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path=f"{settings.API_V1_STR}/auth")

@router.post("/refresh", response_model=TokenResponse, response_model_exclude_none=True)
async def refresh_token(
    request: Request,
    response: Response,
    req: RefreshRequest | None = None,
    refresh_token_cookie: str | None = Cookie(default=None, alias="refresh_token"),
    db: AsyncSession = Depends(get_db),
) -> Any:
    _reject_cross_site_auth_request(request)
    refresh_value = req.refresh_token if req and req.refresh_token else refresh_token_cookie
    if not refresh_value:
        raise HTTPException(status_code=401, detail="Refresh token is required")
    payload = _refresh_claims_from_token(refresh_value)
    refresh_subject = str(payload["sub"])
    await set_database_context(db, refresh_subject.rsplit("@", 1)[1])
    session = (await db.execute(
        select(RefreshSession).where(
            RefreshSession.token_hash == _refresh_hash(refresh_value),
            RefreshSession.revoked_at.is_(None),
            RefreshSession.expires_at > datetime.utcnow(),
        )
    )).scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=401, detail="Refresh token has been revoked or expired")
    user = await UserRepository(db).get_by_email(refresh_subject)
    if not user or not user.active or user.id != session.user_id or int(payload.get("av", 0)) != user.auth_version:
        raise HTTPException(status_code=401, detail="Account is unavailable")
    if not user.roles:
        await bootstrap_rbac(db)
        user = await UserRepository(db).get_by_id(user.id)
    await set_database_context(
        db,
        user.company_domain,
        AuthorizationService.is_global_administrator(user),
        str(user.id),
        AuthorizationService.has_global_article_access(user),
        AuthorizationService.has_global_identity_management(user),
        AuthorizationService.has_global_connector_management(user),
        AuthorizationService.has_global_governance_access(user),
    )
    service = AuthService(UserRepository(db))
    session.revoked_at = datetime.utcnow()
    replacement = service.create_refresh_token(user)
    await _store_refresh_session(db, user, replacement)
    await db.commit()
    access = service.create_token(user)
    _set_auth_cookies(response, access, replacement)
    return {"access_token": access, "token_type": "bearer", "user": _user_response(user)}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    req: RefreshRequest | None = None,
    refresh_token_cookie: str | None = Cookie(default=None, alias="refresh_token"),
    db: AsyncSession = Depends(get_db),
) -> None:
    _reject_cross_site_auth_request(request)
    refresh_value = req.refresh_token if req and req.refresh_token else refresh_token_cookie
    session = None
    if refresh_value:
        # RLS on refresh_sessions needs the signed token's tenant context.
        # Logout remains idempotent for malformed/expired tokens, but a valid
        # current token must be revoked before its cookie is cleared.
        try:
            refresh_subject = _refresh_subject_from_token(refresh_value)
            await set_database_context(db, refresh_subject.rsplit("@", 1)[1])
            session = (await db.execute(
                select(RefreshSession).where(RefreshSession.token_hash == _refresh_hash(refresh_value))
            )).scalar_one_or_none()
        except HTTPException:
            session = None
    if session and session.revoked_at is None:
        session.revoked_at = datetime.utcnow()
        await db.commit()
    _clear_auth_cookies(response)

@router.get("/me", response_model=UserResponse)
async def read_current_user(current_user: Any = Depends(get_current_user)) -> Any:
    return _user_response(current_user)


@router.get("/oidc/config")
async def oidc_config() -> dict[str, object]:
    """Expose non-secret OIDC configuration for a future Entra/Workspace login UI."""
    return {
        "enabled": entra_auth.configured() or bool(settings.OIDC_ISSUER_URL and settings.OIDC_CLIENT_ID and settings.OIDC_REDIRECT_URI),
        "issuer_url": settings.OIDC_ISSUER_URL,
        "client_id": settings.OIDC_CLIENT_ID,
        "redirect_uri": settings.OIDC_REDIRECT_URI,
        "scopes": settings.OIDC_SCOPES.split(),
        "entra_enabled": entra_auth.configured(),
    }


@router.get("/entra/login")
async def entra_login() -> dict[str, str]:
    """Return the Microsoft Entra authorization URL for the login screen."""
    if not entra_auth.configured():
        raise HTTPException(status_code=503, detail="Microsoft Entra login is not configured")
    nonce = entra_auth.new_nonce()
    state = jwt.encode({"type": "entra_login", "nonce": nonce, "exp": datetime.utcnow() + timedelta(minutes=10)}, settings.SECRET_KEY, algorithm="HS256")
    return {"authorization_url": entra_auth.authorization_url(state, nonce), "state_expires_in": "600"}


@router.get("/entra/callback", response_model=TokenResponse, response_model_exclude_none=True)
async def entra_callback(
    code: str,
    state: str,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Exchange and verify Entra claims, then link to an existing account."""
    if not entra_auth.configured():
        raise HTTPException(status_code=503, detail="Microsoft Entra login is not configured")
    try:
        state_claims = jwt.decode(state, settings.SECRET_KEY, algorithms=["HS256"])
        if state_claims.get("type") != "entra_login" or not state_claims.get("nonce"):
            raise JWTError("Invalid Entra login state")
        tokens = await entra_auth.exchange_code(code)
        claims = await entra_auth.verify_id_token(str(tokens.get("id_token") or ""), str(state_claims["nonce"]))
    except (JWTError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Microsoft sign-in could not be verified") from exc
    email = str(claims["email"]).lower()
    await set_database_context(db, email.rsplit("@", 1)[1])
    identity = (await db.execute(select(ExternalIdentity).where(
        ExternalIdentity.provider == "microsoft_entra",
        ExternalIdentity.subject == str(claims["subject"]),
    ))).scalar_one_or_none()
    user = await UserRepository(db).get_by_id(identity.user_id) if identity else await UserRepository(db).get_by_email(email)
    if not user or not user.active:
        raise HTTPException(status_code=403, detail="Your Microsoft account is not linked to an active internal KB account")
    if user.email.lower() != email:
        raise HTTPException(status_code=403, detail="The Microsoft account email does not match the linked internal account")
    if identity is None:
        db.add(ExternalIdentity(
            user_id=user.id,
            provider="microsoft_entra",
            subject=str(claims["subject"]),
            email=email,
            tenant_id=str(claims.get("tid") or "")[:255] or None,
        ))
        await AuditRepository(db).record(user.id, "entra_identity_link", "user", str(user.id), outcome="success")
    if not user.roles:
        await bootstrap_rbac(db)
        user = await UserRepository(db).get_by_id(user.id)
    service = AuthService(UserRepository(db))
    access = service.create_token(user)
    refresh = service.create_refresh_token(user)
    await _store_refresh_session(db, user, refresh)
    await db.commit()
    _set_auth_cookies(response, access, refresh)
    return {"access_token": access, "token_type": "bearer", "user": _user_response(user)}
