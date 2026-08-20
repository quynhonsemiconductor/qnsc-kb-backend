from __future__ import annotations

from typing import Iterable
import hashlib
import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import set_committed_value
from src.models import Permission, Role, RolePermission, User

SCOPES = {"own", "department", "company", "global"}

PERMISSION_CATALOG = {
    "article.read": ("Read articles", "View authorized knowledge-base articles."),
    "article.create": ("Create articles", "Create new articles or drafts."),
    "article.edit": ("Edit articles", "Edit authorized articles."),
    "article.publish": ("Publish articles", "Publish or approve articles."),
    "article.delete": ("Delete articles", "Archive or delete articles."),
    "article.review": ("Review articles", "Review pending articles and drafts."),
    "user.read": ("View users", "View users in the permitted scope."),
    "user.manage": ("Manage users", "Create, update, and deactivate users."),
    "role.manage": ("Manage roles", "Create and manage company roles."),
    "permission.manage": ("Manage permissions", "Assign catalog permissions to roles."),
    "connector.manage": ("Manage connectors", "Create and synchronize source connectors."),
    "ai.ask": ("Ask AI", "Use knowledge-base AI retrieval."),
    "governance.read": ("View governance", "View governance and review workflows."),
}

DEFAULT_ROLE_PERMISSIONS = {
    "Admin": {key: "global" for key in PERMISSION_CATALOG},
    "CEO": {
        "article.read": "company", "article.create": "company", "article.edit": "company",
        "article.publish": "company", "article.review": "company", "article.delete": "company", "user.read": "company",
        "user.manage": "company", "connector.manage": "company", "governance.read": "company",
        "ai.ask": "company",
    },
    "Reviewer": {
        "article.read": "company", "article.review": "company", "governance.read": "company", "ai.ask": "company",
    },
    "Staff": {
        "article.read": "company", "article.create": "own", "article.edit": "own", "ai.ask": "company",
    },
}


async def bootstrap_rbac(db: AsyncSession) -> None:
    """Create the catalog and compatibility roles, then backfill existing users."""
    permissions: dict[str, Permission] = {}
    for key, (name, description) in PERMISSION_CATALOG.items():
        permission = (await db.execute(select(Permission).where(Permission.key == key))).scalar_one_or_none()
        if permission is None:
            permission = Permission(key=key, name=name, description=description, system=True)
            db.add(permission)
            await db.flush()
        permissions[key] = permission

    users = (await db.execute(select(User).options(selectinload(User.roles)))).scalars().all()
    domains = {user.company_domain for user in users if user.company_domain}
    role_by_name: dict[tuple[str, str | None], Role] = {}

    async def ensure_role(name: str, company_domain: str | None, system: bool = True) -> Role:
        cache_key = (name, company_domain)
        if cache_key in role_by_name:
            return role_by_name[cache_key]
        role = (await db.execute(select(Role).where(Role.name == name, Role.company_domain == company_domain).options(selectinload(Role.permissions)))).scalar_one_or_none()
        if role is None:
            role = Role(name=name, company_domain=company_domain, system=system, active=True)
            db.add(role)
            await db.flush()
        assigned_permission_ids = set(
            (await db.execute(
                select(RolePermission.permission_id).where(RolePermission.role_id == role.id)
            )).scalars().all()
        )
        for key, scope in DEFAULT_ROLE_PERMISSIONS[name].items():
            if permissions[key].id not in assigned_permission_ids:
                db.add(RolePermission(role_id=role.id, permission_id=permissions[key].id, scope=scope))
                assigned_permission_ids.add(permissions[key].id)
        role_by_name[cache_key] = role
        return role

    admin_role = await ensure_role("Admin", None)
    for domain in domains:
        for name in DEFAULT_ROLE_PERMISSIONS:
            if name != "Admin":
                await ensure_role(name, domain)

    for user in users:
        if user.roles:
            continue
        role = admin_role if user.role == "Admin" else await ensure_role(user.role if user.role in DEFAULT_ROLE_PERMISSIONS else "Staff", user.company_domain)
        user.roles.append(role)
    await db.commit()


class AuthorizationService:
    """Permission and scope evaluation over eagerly loaded user roles."""

    @staticmethod
    def owned_department_names(user: User) -> set[str]:
        """Return active departments explicitly owned by this user."""
        return {
            assignment.department.name
            for assignment in getattr(user, "department_ownerships", [])
            if assignment.active and assignment.department.active and assignment.department.company_domain == user.company_domain
        }

    @staticmethod
    def member_department_names(user: User) -> set[str]:
        """Return active department memberships used as the article boundary."""
        return {
            department.name
            for department in getattr(user, "departments", [])
            if department.active and department.company_domain == user.company_domain
        }

    @classmethod
    def article_department_names(cls, article: object) -> set[str]:
        names = {
            getattr(department, "name", None)
            for department in getattr(article, "departments", [])
            if getattr(department, "active", True)
        }
        primary = getattr(article, "dept", None)
        if primary:
            names.add(primary)
        return {name for name in names if name}

    @classmethod
    def can_access_article_departments(cls, user: User, article: object) -> bool:
        """Evaluate the shared audience boundary for an Article.

        Global article readers and full-company Admin/CEO identities are the
        deliberate exceptions. Ownership is not used as a read boundary;
        ownership must already be a membership by data validation.
        """
        if cls.has_permission(user, "article.read", requested_scope="global") or cls.has_full_company_article_access(user):
            return True
        member_departments = cls.member_department_names(user)
        article_departments = cls.article_department_names(article)
        if member_departments & article_departments:
            return True
        user_group_ids = {group.id for group in getattr(user, "groups", []) or []}
        article_group_ids = {group.id for group in getattr(article, "access_groups", []) or []}
        # Access groups are read audiences just like departments.  They may
        # cross the organisation tree, but are not used for approval routing.
        if user_group_ids & article_group_ids:
            return True
        user_access_ids = {department.id for department in getattr(user, "departments", []) if getattr(department, "kind", "org") == "access"}
        article_access_ids = {department.id for department in getattr(article, "departments", []) if getattr(department, "kind", "org") == "access"}
        return bool(user_access_ids & article_access_ids)

    @classmethod
    def restrict_article_metadata(cls, user: User, article: object) -> None:
        """Hide non-member department labels from an otherwise visible article."""
        if cls.has_permission(user, "article.read", requested_scope="global") or cls.has_full_company_article_access(user):
            return
        member_departments = cls.member_department_names(user)
        visible = [department for department in getattr(article, "departments", []) if department.name in member_departments]
        # These are persistent, session-attached ORM entities.  Assigning the
        # relationship normally marks the association table dirty; a search
        # request commits its SearchLog in the same session, which would then
        # flush and delete departments the viewer was never allowed to see.
        # Set the in-memory display value as already committed so serializers
        # see the redacted view without creating a persistence mutation.
        set_committed_value(article, "departments", visible)
        if getattr(article, "dept", None) not in member_departments and visible:
            set_committed_value(article, "dept", visible[0].name)

    @classmethod
    def has_department_ownership(cls, user: User) -> bool:
        return bool(cls.owned_department_names(user))

    @classmethod
    def has_full_company_article_access(cls, user: User) -> bool:
        """Whether the identity is an Admin/CEO with company-wide article access."""
        if cls.has_permission(user, "article.read", requested_scope="global"):
            return True
        privileged_role = user.role in {"Admin", "CEO"} if not user.roles else any(
            role.active is not False
            and role.name in {"Admin", "CEO"}
            and role.company_domain in {None, user.company_domain}
            for role in user.roles
        )
        return privileged_role and cls.has_permission(user, "article.read", requested_scope="company")

    @classmethod
    def has_narrow_article_access(cls, user: User, resource: object | None = None) -> bool:
        """Whether own/department scope grants access without group membership."""
        return (
            not cls.has_permission(user, "article.read", requested_scope="company")
            and any(
                cls.has_permission(user, "article.read", resource, scope)
                for scope in ("own", "department")
            )
        )

    @staticmethod
    def _scope_allows(granted: str, requested: str, user: User, resource: object | None) -> bool:
        if granted not in SCOPES or requested not in SCOPES:
            return False
        if granted == "global":
            return True
        if resource is None:
            return requested in {"own", "department", "company"} and granted in {requested, "company", "global"}
        if granted == "company":
            return requested in {"own", "department", "company"} and getattr(resource, "company_domain", user.company_domain) == user.company_domain
        if granted == "department":
            resource_departments = {getattr(department, "name", None) for department in getattr(resource, "departments", [])}
            resource_departments.discard(None)
            resource_departments.add(getattr(resource, "dept", None))
            return (
                requested in {"own", "department"}
                and getattr(resource, "company_domain", user.company_domain) == user.company_domain
                and bool(resource_departments & AuthorizationService.owned_department_names(user))
            )
        return requested == "own" and getattr(resource, "owner_id", None) == user.id

    @classmethod
    def has_permission(cls, user: User, key: str, resource: object | None = None, requested_scope: str = "company") -> bool:
        # Compatibility fallback for users created before RBAC bootstrap and
        # for isolated domain tests; normal production users are backfilled.
        if not user.roles and user.role in DEFAULT_ROLE_PERMISSIONS:
            scope = DEFAULT_ROLE_PERMISSIONS[user.role].get(key)
            if scope:
                return cls._scope_allows(scope, requested_scope, user, resource)
        for role in user.roles:
            if role.active is False or (role.company_domain is not None and role.company_domain != user.company_domain):
                continue
            for assignment in role.permissions:
                if assignment.permission and assignment.permission.key == key and cls._scope_allows(assignment.scope, requested_scope, user, resource):
                    return True
        return False

    @staticmethod
    def is_global_administrator(user: User) -> bool:
        """Whether this identity may bypass tenant RLS.

        A global *article* permission is deliberately not enough: it grants
        cross-company document access, not unrestricted access to every
        user's sessions, conversations, audit trail, or group membership.
        The database-wide bypass is reserved for the seeded global Admin
        role. Other global permissions still go through their route-specific
        checks and tenant-aware queries.
        """
        return any(
            role.active is not False and role.name == "Admin" and role.company_domain is None
            for role in user.roles
        )

    @classmethod
    def can_view_all_access_groups(cls, user: User) -> bool:
        """Access-group membership metadata belongs to identity management.

        A global article reader may retrieve documents across companies, but
        must not automatically enumerate every company's groups and names.
        """
        return cls.has_permission(user, "user.read", requested_scope="global")

    @classmethod
    def has_global_identity_management(cls, user: User) -> bool:
        """Whether tenant RLS may expose cross-company identity records."""
        return any(
            cls.has_permission(user, permission, requested_scope="global")
            for permission in ("user.read", "user.manage", "role.manage", "permission.manage")
        )

    @classmethod
    def has_global_connector_management(cls, user: User) -> bool:
        """Whether connector RLS may expose cross-company connector state."""
        return cls.has_permission(user, "connector.manage", requested_scope="global")

    @classmethod
    def has_global_governance_access(cls, user: User) -> bool:
        """Whether governance RLS may expose cross-company workflow records."""
        return cls.has_permission(user, "governance.read", requested_scope="global")

    @classmethod
    def has_global_article_access(cls, user: User) -> bool:
        """Whether article RLS may expose cross-company article records."""
        return any(cls.has_permission(user, key, requested_scope="global") for key in (
            "article.read", "article.create", "article.edit", "article.publish", "article.delete", "article.review",
        ))

    @classmethod
    def get_effective_permissions(cls, user: User) -> list[dict[str, str]]:
        result: dict[str, str] = {}
        rank = {"own": 1, "department": 2, "company": 3, "global": 4}
        for role in user.roles:
            if role.active is False or (role.company_domain is not None and role.company_domain != user.company_domain):
                continue
            for assignment in role.permissions:
                if assignment.permission and rank.get(assignment.scope, 0) > rank.get(result.get(assignment.permission.key, ""), 0):
                    result[assignment.permission.key] = assignment.scope
        if not user.roles:
            for key, scope in DEFAULT_ROLE_PERMISSIONS.get(user.role, {}).items():
                result[key] = scope
        return [{"key": key, "scope": scope} for key, scope in sorted(result.items())]

    @classmethod
    def authorization_fingerprint(cls, user: User) -> str:
        """Return a stable, user-specific authorization snapshot.

        Cached answers contain document text, so a coarse access bitmap is not
        sufficient.  Include the user identity because ``own`` scope can vary
        even when two users have identical roles, and include groups/roles so
        role or group changes naturally select a new cache namespace.
        """
        roles = sorted(
            (str(role.id), role.name, role.company_domain, bool(role.active))
            for role in user.roles
        )
        groups = sorted(
            (str(group.id), group.name, group.bitmask_position)
            for group in user.groups
        )
        payload = {
            "user_id": str(user.id),
            "company_domain": user.company_domain,
            "dept": user.dept,
            "owned_departments": sorted(cls.owned_department_names(user)),
            "roles": roles,
            "permissions": cls.get_effective_permissions(user),
            "groups": groups,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()


def legacy_roles_for_permission(permission: str) -> list[str]:
    return [name for name, values in DEFAULT_ROLE_PERMISSIONS.items() if permission in values]
