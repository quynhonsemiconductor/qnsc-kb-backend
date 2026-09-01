import uuid
from typing import Sequence
from sqlalchemy import select, and_, false
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models.user import User, Department, DepartmentManager
from src.models.rbac import Role, RolePermission
from src.domain.rbac import AuthorizationService

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _identity_scope(viewer: User | None):
        if viewer is None:
            return None
        can_read_global = AuthorizationService.has_permission(viewer, "user.read", requested_scope="global")
        can_manage_global = AuthorizationService.has_permission(viewer, "user.manage", requested_scope="global")
        if can_read_global or can_manage_global:
            return None
        if not (
            AuthorizationService.has_permission(viewer, "user.read", requested_scope="company")
            or AuthorizationService.has_permission(viewer, "user.manage", requested_scope="company")
        ):
            return false()
        return User.company_domain == viewer.company_domain

    async def get_by_id(self, user_id: uuid.UUID, viewer: User | None = None) -> User | None:
        filters = [User.id == user_id]
        scope = self._identity_scope(viewer)
        if scope is not None:
            filters.append(scope)
        result = await self.db.execute(
            select(User)
            .where(and_(*filters))
            .options(selectinload(User.departments))
            .options(selectinload(User.department_ownerships).selectinload(DepartmentManager.department))
            .options(selectinload(User.roles).selectinload(Role.permissions).selectinload(RolePermission.permission))
        )
        return result.scalar_one_or_none()

    async def get_by_ids(self, user_ids: list[uuid.UUID], company_domain: str | None = None) -> Sequence[User]:
        if not user_ids:
            return []
        stmt = select(User).where(User.id.in_(set(user_ids)))
        if company_domain is not None:
            stmt = stmt.where(User.company_domain == company_domain)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(
            select(User)
            .where(User.email == email)
            .options(selectinload(User.departments))
            .options(selectinload(User.department_ownerships).selectinload(DepartmentManager.department))
            .options(selectinload(User.roles).selectinload(Role.permissions).selectinload(RolePermission.permission))
        )
        return result.scalar_one_or_none()

    async def create(self, user: User) -> User:
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_department_by_name(self, name: str, company_domain: str | None = None) -> Department | None:
        stmt = select(Department).where(Department.name == name)
        if company_domain is not None:
            stmt = stmt.where(Department.company_domain == company_domain)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_departments_by_ids(self, department_ids: list[uuid.UUID], company_domain: str | None = None) -> Sequence[Department]:
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
        stmt = select(User).options(
                selectinload(User.departments),
                selectinload(User.department_ownerships).selectinload(DepartmentManager.department),
                selectinload(User.roles).selectinload(Role.permissions).selectinload(RolePermission.permission),
            ).order_by(User.company_domain, User.name).offset(offset).limit(limit)
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

    async def update_user_departments(self, user: User, departments: list[Department]) -> User:
        user.departments = departments
        self.db.add(user)
        await self.db.commit()
        # Same reason as update() above.
        return await self.get_by_id(user.id) or user
