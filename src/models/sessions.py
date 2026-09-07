import uuid
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin


class RefreshSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "refresh_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    user: Mapped["User"] = relationship("User")


class PasswordResetToken(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single-use, short-lived password reset grant.

    Only the SHA-256 of the token is stored, exactly as `Invitation` and `RefreshSession`
    do it: a database dump must not hand out working reset links. The raw token exists
    only in the email that carried it.

    `used_at` rather than a delete, so a replayed link can be told apart from an unknown
    one in the audit trail, and so a stolen-but-spent link cannot be reused.
    """

    __tablename__ = "password_reset_tokens"
    #: Declared here, not only in the migration: `alembic check` compares the live schema
    #: against this metadata, so an index created by raw SQL alone reads as drift and CI
    #: asks to DROP it. Same mistake caught by the same gate in 20260830_67.
    #:
    #: Partial because the only query is "does this user already have a live grant", used
    #: to rate-limit repeat requests. Spent and expired rows are the majority and are
    #: never looked up.
    __table_args__ = (
        Index(
            "ix_password_reset_tokens_active",
            "user_id",
            text("expires_at DESC"),
            postgresql_where=text("used_at IS NULL"),
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    #: Which address the link was sent to, for the audit trail. The user's current email
    #: can change between request and use; this records what was actually mailed.
    requested_for_email: Mapped[str] = mapped_column(String(255), nullable=False)
    user: Mapped["User"] = relationship("User")
