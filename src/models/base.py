from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func
import uuid


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    # Both a Python-side default AND a server default, deliberately.
    #
    # With only a server default, SQLAlchemy has to read the generated values back, so
    # every ORM insert becomes `INSERT ... RETURNING created_at, updated_at`. Under
    # row-level security a RETURNING clause additionally applies the table's SELECT
    # policy to the new row: a row you are allowed to WRITE but not to READ makes the
    # insert fail with "new row violates row-level security policy" — naming the table
    # but not the RETURNING, which is what made it hard to place.
    #
    # That is not hypothetical. notification_queue may be written for any user in the
    # same company (tenant_notification_delivery) but read only by its recipient
    # (recipient_notifications_read), so approving a draft — a reviewer notifying the
    # author — failed on the insert while assigning an approver, done by a global admin
    # who can read everything, succeeded.
    #
    # A Python-side default puts the value in the INSERT itself, so no RETURNING is
    # emitted and write permission alone is enough. The server default stays for rows
    # written outside the ORM (migrations, psql, the bootstrap scripts).
    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, server_default=func.now(), onupdate=datetime.utcnow
    )


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
