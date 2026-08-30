import uuid
from datetime import datetime
from sqlalchemy import Boolean, ForeignKey, String, Text, Integer, JSON, Float, DateTime, UniqueConstraint, Index, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin

class Connector(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "connectors"
    __table_args__ = (Index("uq_connectors_company_name", "company_domain", "name", unique=True),)

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    system: Mapped[str] = mapped_column(String(50), nullable=False)  # google_drive, sharepoint
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)  # active, paused
    last_sync: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    company_domain: Mapped[str] = mapped_column(String(255), index=True, nullable=False, default="local")
    # NO sync_interval_minutes. It was kept here "for compatibility with pre-Alembic
    # connector rows" while being read by nothing — and because SQLAlchemy SELECTs every
    # mapped column, that compatibility gesture broke every database Alembic actually
    # built. No migration ever creates the column (20260802_08 patches on the oauth
    # columns beside it and skips this one), so schedule_cloud_connector_syncs died on
    # every beat tick with
    #
    #   UndefinedColumnError: column connectors.sync_interval_minutes does not exist
    #
    # taking cloud connector polling down entirely. A legacy database keeps its column;
    # unmapped, it is simply never referenced again.
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    oauth_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    oauth_access_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    oauth_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    oauth_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    oauth_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    oauth_state_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    jobs: Mapped[list["ConnectorJob"]] = relationship("ConnectorJob", back_populates="connector", cascade="all, delete-orphan")

class ConnectorJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "connector_jobs"

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    requested_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)  # pending, running, completed, failed
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    connector: Mapped[Connector] = relationship("Connector", back_populates="jobs")


class IndexReprocessJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Durable bulk indexing progress and retry state."""
    __tablename__ = "index_reprocess_jobs"

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    requested_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="queued", index=True)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    target_article_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

class NotificationQueue(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "notification_queue"

    recipient_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # email, slack
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)  # pending, sent, failed
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

class DeadLetterJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "dead_letter_jobs"

    source_queue: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    failed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OutboxEvent(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "outbox_events"

    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class SearchLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "search_logs"

    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    # Retained only for schema compatibility; new rows use a redacted marker.
    query: Mapped[str] = mapped_column(Text, nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    user: Mapped["User | None"] = relationship("User")


class ApiRequestMetric(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "api_request_metrics"
    #: Declared here, not only in the migration: `alembic check` compares the live schema
    #: against this metadata, so an index created by raw SQL alone reads as drift and the
    #: CI gate asks to DROP it. Measured -- that is exactly how this failed first.
    #:
    #: Partial and DESC to match the only query /governance/request-failures issues:
    #: newest failures first. Indexing the 200s would cover most of the table for none of
    #: the queries actually run.
    __table_args__ = (
        Index(
            "ix_api_request_metrics_failures",
            text("created_at DESC"),
            "path",
            postgresql_where=text("status_code >= 500"),
        ),
    )

    request_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(12), nullable=False)
    path: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    #: WHY THESE EXIST. The middleware already catches every unhandled exception and
    #: logs it, but only the status code was persisted — so diagnosing a 500 meant
    #: reading CloudWatch, which needs an AWS role switch nobody has to hand. The
    #: exception was there and thrown away. These two columns keep it.
    #:
    #: Populated ONLY for failures; a successful request leaves both NULL, so the
    #: table stays the same size for the traffic that does not need explaining.
    error_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    #: The message plus the innermost frames. Truncated when stored, because a deep
    #: traceback is unbounded and this row is written on the request path.
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class FeatureFlag(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "feature_flags"

    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    rollout_percent: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    department: Mapped[str | None] = mapped_column(String(100), nullable=True)


class LLMProviderConfig(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "llm_provider_configs"

    # There is one workspace-wide provider configuration. Keeping this as a
    # row gives administrators a durable setting without putting secrets in
    # frontend code or requiring a container restart after every change.
    config_key: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, default="workspace")
    enabled: Mapped[bool] = mapped_column(default=False, nullable=False)
    provider: Mapped[str] = mapped_column(String(30), nullable=False, default="openai")
    model: Mapped[str] = mapped_column(String(150), nullable=False)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    encrypted_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)

class EvalQuestion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "eval_questions"

    question: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    expected_chunk_ids: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list of parent chunk IDs
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    eval_set_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("eval_sets.id", ondelete="SET NULL"), nullable=True, index=True)
    eval_set: Mapped["EvalSet | None"] = relationship("EvalSet", back_populates="questions")
    runs: Mapped[list["EvalRun"]] = relationship("EvalRun", back_populates="question")


class EvalSet(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Versioned, approver-owned acceptance corpus."""

    __tablename__ = "eval_sets"
    __table_args__ = (UniqueConstraint("company_domain", "name", "version", name="uq_eval_set_company_name_version"),)

    company_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    environment: Mapped[str] = mapped_column(String(50), nullable=False, default="uat")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="draft")
    approved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    questions: Mapped[list[EvalQuestion]] = relationship("EvalQuestion", back_populates="eval_set")

class EvalRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "eval_runs"

    eval_question_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("eval_questions.id", ondelete="CASCADE"), nullable=False)
    retrieval_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    context_recall: Mapped[float] = mapped_column(Float, nullable=False)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=False)
    answer_correctness: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # True only when the answer exposed a citation outside the authorized
    # retrieval set used for this evaluation run.
    permission_leakage: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")

    question: Mapped[EvalQuestion] = relationship("EvalQuestion", back_populates="runs")
