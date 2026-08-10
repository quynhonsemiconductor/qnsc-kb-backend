import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, String, Text, Integer, JSON, Float, DateTime, UniqueConstraint, Index
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
    # Retained for compatibility with pre-Alembic connector rows. Current
    # scheduling uses connector config and job mode; no API exposes this field.
    sync_interval_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60, server_default="60")
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

    request_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(12), nullable=False)
    path: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)


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

    runs: Mapped[list["EvalRun"]] = relationship("EvalRun", back_populates="question", cascade="all, delete-orphan")

class EvalRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "eval_runs"

    eval_question_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("eval_questions.id", ondelete="CASCADE"), nullable=False)
    retrieval_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    context_recall: Mapped[float] = mapped_column(Float, nullable=False)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=False)
    answer_correctness: Mapped[float] = mapped_column(Float, nullable=False)

    question: Mapped[EvalQuestion] = relationship("EvalQuestion", back_populates="runs")
