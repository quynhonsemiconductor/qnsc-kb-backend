import uuid
from typing import Sequence
from sqlalchemy import select, and_, false, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models.user import User, Department, DepartmentManager
from src.models.rbac import Role, RolePermission
from src.domain.rbac import AuthorizationService


class AmbiguousEmailError(ValueError):
    """More than one account matches the same address case-insensitively.

    Raised instead of letting SQLAlchemy's ``MultipleResultsFound`` escape: an Access
    token can only be mapped to one person, and a data collision must be an explicit,
    logged refusal rather than an opaque 500 from the ORM.
    """


class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _identity_scope(viewer: User | None):
        if viewer is None:
            return None
        can_read_global = AuthorizationService.has_permission(
            viewer, "user.read", requested_scope="global"
        )
        can_manage_global = AuthorizationService.has_permission(
            viewer, "user.manage", requested_scope="global"
        )
        if can_read_global or can_manage_global:
            return None
        if not (
            AuthorizationService.has_permission(
                viewer, "user.read", requested_scope="company"
            )
            or AuthorizationService.has_permission(
                viewer, "user.manage", requested_scope="company"
            )
        ):
            return false()
        return User.company_domain == viewer.company_domain

    async def get_by_id(
        self, user_id: uuid.UUID, viewer: User | None = None
    ) -> User | None:
        filters = [User.id == user_id]
        scope = self._identity_scope(viewer)
        if scope is not None:
            filters.append(scope)
        result = await self.db.execute(
            select(User)
            .where(and_(*filters))
            .options(selectinload(User.departments))
            .options(
                selectinload(User.department_ownerships).selectinload(
                    DepartmentManager.department
                )
            )
            .options(
                selectinload(User.roles)
                .selectinload(Role.permissions)
                .selectinload(RolePermission.permission)
            )
        )
        return result.scalar_one_or_none()

    async def get_by_ids(
        self, user_ids: list[uuid.UUID], company_domain: str | None = None
    ) -> Sequence[User]:
        if not user_ids:
            return []
        stmt = select(User).where(User.id.in_(set(user_ids)))
        if company_domain is not None:
            stmt = stmt.where(User.company_domain == company_domain)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    def _user_select(self):
        return (
            select(User)
            .options(selectinload(User.departments))
            .options(
                selectinload(User.department_ownerships).selectinload(
                    DepartmentManager.department
                )
            )
            .options(
                selectinload(User.roles)
                .selectinload(Role.permissions)
                .selectinload(RolePermission.permission)
            )
        )

    async def get_by_email(self, email: str) -> User | None:
        """Find one account by address, deterministically.

        The exact comparison runs first, so a normal lower-case row uses the unique index
        and wins even if a differently-cased duplicate exists. Only a miss falls back to
        ``lower(email)``; if that finds more than one row the lookup fails explicitly.
        """
        raw = email.strip()
        normalized = raw.lower()
        # Prefer the address exactly as given (so an exact mixed-case row wins), then the
        # normalised form, before the case-insensitive fallback.
        for candidate in dict.fromkeys((raw, normalized)):
            if not candidate:
                continue
            exact = (
                await self.db.execute(
                    self._user_select().where(User.email == candidate)
                )
            ).scalar_one_or_none()
            if exact is not None:
                return exact

        matches = (
            (
                await self.db.execute(
                    self._user_select()
                    .where(func.lower(User.email) == normalized)
                    .limit(2)
                )
            )
            .scalars()
            .all()
        )
        if len(matches) > 1:
            raise AmbiguousEmailError(
                "more than one account matches this email case-insensitively"
            )
        return matches[0] if matches else None

    async def create(self, user: User) -> User:
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_department_by_name(
        self, name: str, company_domain: str | None = None
    ) -> Department | None:
        stmt = select(Department).where(Department.name == name)
        if company_domain is not None:
            stmt = stmt.where(Department.company_domain == company_domain)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_departments_by_ids(
        self, department_ids: list[uuid.UUID], company_domain: str | None = None
    ) -> Sequence[Department]:
        stmt = select(Department).where(Department.id.in_(department_ids))
        if company_domain:
            stmt = stmt.where(Department.company_domain == company_domain)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def list_users(
        self,
        offset: int = 0,
        limit: int = 100,
        viewer: User | None = None,
        *,
        company_domain: str | None = None,
        active: bool | None = None,
        exclude_id: uuid.UUID | None = None,
    ) -> Sequence[User]:
        filters = []
        scope = self._identity_scope(viewer)
        if scope is not None:
            filters.append(scope)
        if company_domain is not None:
            filters.append(User.company_domain == company_domain)
        if active is not None:
            filters.append(User.active.is_(active))
        if exclude_id is not None:
            filters.append(User.id != exclude_id)
        stmt = (
            select(User)
            .options(
                selectinload(User.departments),
                selectinload(User.department_ownerships).selectinload(
                    DepartmentManager.department
                ),
                selectinload(User.roles)
                .selectinload(Role.permissions)
                .selectinload(RolePermission.permission),
            )
            .order_by(User.company_domain, User.name)
            .offset(offset)
            .limit(limit)
        )
        if filters:
            stmt = stmt.where(and_(*filters))
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def update(self, user: User) -> User:
        self.db.add(user)
        await self.db.commit()
        # Re-FETCH rather than refresh(). refresh() expires the instance and reloads its
        # COLUMNS only, so every relationship loaded by get_by_id is dropped — and each
        # caller of this method hands the result to a response builder that walks
        # user.roles -> role.permissions. Those then lazy-load inside an async request,
        # which asyncpg cannot do, and the endpoint 500s with MissingGreenlet AFTER the
        # write has already committed.
        return await self.get_by_id(user.id) or user

    async def update_user_departments(
        self, user: User, departments: list[Department]
    ) -> User:
        user.departments = departments
        self.db.add(user)
        await self.db.commit()
        # Same reason as update() above.
        return await self.get_by_id(user.id) or user
