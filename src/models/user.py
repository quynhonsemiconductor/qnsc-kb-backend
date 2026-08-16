import uuid
from datetime import datetime
from sqlalchemy import (
    Table,
    Column,
    ForeignKey,
    String,
    Integer,
    Boolean,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin

# Association table for User <-> AccessGroup (Many-to-Many)
user_groups = Table(
    "user_groups",
    Base.metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "group_id", ForeignKey("access_groups.id", ondelete="CASCADE"), primary_key=True
    ),
)

# User <-> Department membership. ``users.dept`` remains the primary/legacy
# department for compatibility with existing integrations; this relation is
# the authoritative multi-department membership model.
user_departments = Table(
    "user_departments",
    Base.metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "department_id",
        ForeignKey("departments.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class AccessGroup(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "access_groups"
    __table_args__ = (
        Index("uq_access_groups_company_name", "company_domain", "name", unique=True),
        Index(
            "uq_access_groups_company_bit_position",
            "company_domain",
            "bitmask_position",
            unique=True,
        ),
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    company_domain: Mapped[str] = mapped_column(
        String(255), nullable=False, default="local", index=True
    )
    # position of the bit in the bitmask (0, 1, 2, ... up to 63 for 64-bit integer, or higher if using numeric)
    bitmask_position: Mapped[int] = mapped_column(Integer, nullable=False)

    users: Mapped[list["User"]] = relationship(
        "User", secondary=user_groups, back_populates="groups"
    )


class Department(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "departments"
    __table_args__ = (
        UniqueConstraint("company_domain", "name", name="uq_departments_company_name"),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    managers: Mapped[list["DepartmentManager"]] = relationship(
        "DepartmentManager",
        back_populates="department",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    members: Mapped[list["User"]] = relationship(
        "User", secondary=user_departments, back_populates="departments"
    )


class DepartmentManager(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Explicit ownership of one or more departments by a user."""

    __tablename__ = "department_managers"
    __table_args__ = (
        UniqueConstraint(
            "department_id", "user_id", name="uq_department_manager_assignment"
        ),
    )

    department_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("departments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    department: Mapped[Department] = relationship(
        "Department", back_populates="managers"
    )
    user: Mapped["User"] = relationship("User", back_populates="department_ownerships")


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    company_domain: Mapped[str] = mapped_column(
        String(255), index=True, nullable=False, default="local"
    )
    dept: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Admin, CEO, Reviewer, Staff; department ownership is managed separately.
    role: Mapped[str] = mapped_column(String(50), default="Staff", nullable=False)
    active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, index=True
    )
    # Incrementing this invalidates every signed access/refresh token issued
    # before a password or account-security change.
    auth_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    groups: Mapped[list[AccessGroup]] = relationship(
        "AccessGroup", secondary=user_groups, back_populates="users"
    )
    departments: Mapped[list[Department]] = relationship(
        "Department",
        secondary=user_departments,
        back_populates="members",
        lazy="selectin",
    )
    department_ownerships: Mapped[list[DepartmentManager]] = relationship(
        "DepartmentManager",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    roles: Mapped[list["Role"]] = relationship(
        "Role", secondary="user_roles", back_populates="users"
    )
    identities: Mapped[list["ExternalIdentity"]] = relationship(
        "ExternalIdentity",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ExternalIdentity(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Provider subject linked to one internal user account."""

    __tablename__ = "external_identities"
    __table_args__ = (
        UniqueConstraint(
            "provider", "subject", name="uq_external_identity_provider_subject"
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    user: Mapped[User] = relationship("User", back_populates="identities")
