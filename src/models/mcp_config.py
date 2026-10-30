import uuid

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class McpConfig(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """The team's standing instructions for every AI that uses the knowledge base.

    One current text per company (`name` leaves room for more later), edited by an admin
    and versioned. An AI client reads it through the `get_project_config` MCP tool and keeps
    a copy in its own CLAUDE.md; the version is what lets it notice that its copy is stale
    without anyone telling it. Lookups always filter on `company_domain`.
    """

    __tablename__ = "mcp_configs"
    __table_args__ = (
        UniqueConstraint("company_domain", "name", name="uq_mcp_configs_company_name"),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, default="default")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content_md: Mapped[str] = mapped_column(Text, nullable=False)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class McpConfigVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Every text the config has had, so an edit can be compared with or restored from."""

    __tablename__ = "mcp_config_versions"
    __table_args__ = (
        UniqueConstraint("config_id", "version", name="uq_mcp_config_versions_version"),
    )

    config_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("mcp_configs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_md: Mapped[str] = mapped_column(Text, nullable=False)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
