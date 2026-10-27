from typing import TYPE_CHECKING
import uuid
from datetime import datetime
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin

if TYPE_CHECKING:
    from src.models.user import User


class PersonalAccessToken(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A long-lived credential a user hands to an MCP client (Claude, Cursor, ...).

    Only the SHA-256 of the token is stored, like `RefreshSession` and `Invitation`: a
    database dump must not hand out working credentials. The raw value is shown once, at
    creation. `token_prefix` keeps the first characters so the owner can tell their tokens
    apart in the list.

    The token acts as its owner and nothing more: every call it makes is authorised by the
    owner's roles, department and ACL at the time of the call. `scope` can only narrow
    that (`read` never reaches a write tool).
    """

    __tablename__ = "personal_access_tokens"
    __table_args__ = (
        CheckConstraint(
            "scope IN ('read', 'write')", name="ck_personal_access_tokens_scope"
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    token_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    token_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    scope: Mapped[str] = mapped_column(String(10), nullable=False, default="read")
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    user: Mapped["User"] = relationship("User")
