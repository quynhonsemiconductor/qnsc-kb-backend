from typing import TYPE_CHECKING
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
    DateTime,
    Text,
    JSON,
)
from sqlalchemy import func, true
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin

# User <-> Department membership. ``users.dept`` remains the primary
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


if TYPE_CHECKING:
    from src.models.rbac import Role


class Department(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "departments"
    __table_args__ = (
        UniqueConstraint("company_domain", "name", name="uq_departments_company_name"),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    #: Whether a draft for this department waits for a human decision. False publishes it
    #: without one (src/domain/department_auto_publish.py). True is the default and the only
    #: behaviour before this column existed, so nothing changes until an admin opts a
    #: department out.
    review_required: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=true(), nullable=False
    )
    #: The Admin/CEO who turned review off. Auto-published drafts are approved AS this person,
    #: through the same checks a manual approval goes through, so the decision has a name
    #: behind it; if they lose the role or are deactivated, publishing stops and drafts wait.
    auto_publish_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
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

    @validates("email")
    def _normalise_email(self, key: str, value: str) -> str:
        # Every create and every assignment: an IdP may hand back a different case for the
        # same mailbox, and a case variant must never become a second account. Return the
        # original object when it is already canonical so an unchanged assign is a no-op.
        if not isinstance(value, str):
            return value
        normalised = value.strip().lower()
        return normalised if normalised != value else value

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
    #: The first Cloudflare Access `sub` seen for this account. It binds the account to one
    #: Access identity, so a reassigned mailbox cannot inherit it through the email
    #: fallback. NULL until the first Access login; an administrator clears it to rebind.
    access_subject: Mapped[str | None] = mapped_column(
        String(255), unique=True, index=True, nullable=True
    )

    # The Access lookup falls back to lower(email); without this expression index that
    # fallback is a sequential scan on a table read on every authenticated request.
    __table_args__ = (Index("ix_users_email_lower", func.lower(email)),)

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


class Invitation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Single-use Entra invitation, scoped to a tenant and email address."""

    __tablename__ = "invitations"
    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_invitations_token_hash"),
        Index("ix_invitations_email_domain", "email", "company_domain"),
    )

    @validates("email")
    def _normalise_email(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            return value
        normalised = value.strip().lower()
        return normalised if normalised != value else value

    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="Staff")
    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    audience_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    invited_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    accepted_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
