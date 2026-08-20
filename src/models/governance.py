import uuid
from datetime import datetime
from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    Integer,
    JSON,
    UniqueConstraint,
    Boolean,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin


class PendingDraft(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "pending_drafts"

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # Keep the queue tenant-scoped.  Source text is sensitive before it is
    # approved, so deriving tenancy only from an optional creator is unsafe.
    company_domain: Mapped[str] = mapped_column(
        String(255), nullable=False, default="local", index=True
    )
    dept: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    source_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    restructured_body_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Keep an AI candidate separately when preservation checks reject it. The
    # reviewer can inspect it and explicitly accept or discard it; the source
    # extraction in ``summary`` is never overwritten.
    restructure_candidate_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    restructure_decision: Mapped[str] = mapped_column(
        String(30), default="not_reviewed", nullable=False
    )
    restructure_status: Mapped[str] = mapped_column(
        String(40), default="pending", nullable=False
    )
    restructure_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    restructure_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(150), nullable=True)
    page_texts: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    similarity_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    similarity_matches: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    requires_update_confirmation: Mapped[bool] = mapped_column(
        default=False, nullable=False
    )
    update_target_article_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("articles.id", ondelete="SET NULL"), nullable=True
    )
    related_article_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    # Metadata submitted with a manually authored document.  Uploads retain
    # their source-derived fields above; this lets both paths share one
    # independent-approval workflow without creating a publishable Article
    # before review.
    content_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # pending, approved, rejected
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    assigned_approver_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    assigned_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("external_documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    candidates: Mapped[list["DraftCandidate"]] = relationship(
        "DraftCandidate",
        back_populates="draft",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    creator: Mapped["User | None"] = relationship("User", foreign_keys=[created_by])
    assigned_approver: Mapped["User | None"] = relationship(
        "User", foreign_keys=[assigned_approver_id]
    )
    assigner: Mapped["User | None"] = relationship("User", foreign_keys=[assigned_by])
    reviewer: Mapped["User | None"] = relationship("User", foreign_keys=[reviewed_by])


class DraftTransition(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Immutable state-machine history for a pending draft."""

    __tablename__ = "draft_transitions"

    draft_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pending_drafts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    from_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    to_status: Mapped[str] = mapped_column(String(30), nullable=False)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[str] = mapped_column(String(30), nullable=False, default="applied")

    draft: Mapped[PendingDraft] = relationship("PendingDraft")
    actor: Mapped["User | None"] = relationship("User", foreign_keys=[actor_id])


class DraftCandidate(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A structure-aware article candidate awaiting batch review."""

    __tablename__ = "draft_candidates"
    __table_args__ = (
        UniqueConstraint("draft_id", "position", name="uq_draft_candidate_position"),
    )

    draft_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pending_drafts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body_md: Mapped[str] = mapped_column(Text, nullable=False)
    source_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_position: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Reviewer-editable routing chosen from the department suggestions generated
    # after the document has been formatted and split.
    department_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    department_suggestions: Mapped[list[dict] | None] = mapped_column(
        JSON, nullable=True
    )
    proposed_department: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="candidate")
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    draft: Mapped[PendingDraft] = relationship(
        "PendingDraft", back_populates="candidates"
    )


class ApproverRule(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Company department rule for automatically selecting an approver."""

    __tablename__ = "approver_rules"
    __table_args__ = (
        UniqueConstraint(
            "company_domain", "dept", name="uq_approver_rules_company_dept"
        ),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dept: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    approver_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    approver: Mapped["User"] = relationship("User", foreign_keys=[approver_id])
    creator: Mapped["User | None"] = relationship("User", foreign_keys=[created_by])


class IngestionFingerprint(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Tenant-scoped reservation for a source hash.

    Keeping pending reservations in the database closes the duplicate-upload
    race between two concurrent requests.
    """

    __tablename__ = "ingestion_fingerprints"
    __table_args__ = (
        UniqueConstraint(
            "company_domain", "source_hash", name="uq_ingestion_fingerprint_tenant_hash"
        ),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    draft_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pending_drafts.id", ondelete="SET NULL"), nullable=True
    )
    article_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("articles.id", ondelete="SET NULL"), nullable=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class Gap(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "gaps"

    __table_args__ = (
        Index("uq_gaps_company_query", "company_domain", "query", unique=True),
    )

    company_domain: Mapped[str] = mapped_column(
        String(255), nullable=False, default="local", index=True
    )
    query: Mapped[str] = mapped_column(String(255), nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    dept: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # open, assigned, dismissed
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)


class ArticleEditRequest(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A correction request from a reader who cannot edit an article directly."""

    __tablename__ = "article_edit_requests"
    __table_args__ = (
        Index("ix_article_edit_requests_company_status", "company_domain", "status"),
        Index("ix_article_edit_requests_article", "article_id"),
    )

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False
    )
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    request_text: Mapped[str] = mapped_column(Text, nullable=False)
    # open, accepted, rejected, completed
    status: Mapped[str] = mapped_column(String(30), default="open", nullable=False)
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConflictRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Persisted, reviewable contradiction detected across published articles."""

    __tablename__ = "conflict_records"
    __table_args__ = (Index("ix_conflicts_company_status", "company_domain", "status"),)

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    fact: Mapped[str] = mapped_column(String(255), nullable=False)
    article_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    evidence: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="open", server_default="open", nullable=False)
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "audit_logs"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # create, update, delete, permission_change, approve
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    target_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # article, user, group, draft
    target_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    outcome: Mapped[str] = mapped_column(
        String(30), nullable=False, default="success", server_default="success"
    )
    # Structured before/after/context data.  Keep the compact legacy columns
    # for filtering and make details optional so historical rows remain valid.
    detail_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    user: Mapped["User | None"] = relationship("User")
