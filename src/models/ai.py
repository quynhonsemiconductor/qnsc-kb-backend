import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, String, Text, Integer, BigInteger, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin

class AiUsageLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "ai_usage_logs"

    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    # Telemetry only: raw conversation content lives in user-owned messages.
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    retrieval_version: Mapped[str] = mapped_column(String(50), nullable=False)
    reranker_version: Mapped[str] = mapped_column(String(50), nullable=False)
    retrieved_chunk_ids: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User | None"] = relationship("User")
    feedback: Mapped[list["AiFeedback"]] = relationship("AiFeedback", back_populates="usage_log", cascade="all, delete-orphan")

class AiConversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "ai_conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New conversation")

    user: Mapped["User"] = relationship("User")
    messages: Mapped[list["AiMessage"]] = relationship(
        "AiMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="AiMessage.created_at"
    )

class AiMessage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "ai_messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ai_conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    grounded_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    extended_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    usage_log_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("ai_usage_logs.id", ondelete="SET NULL"), nullable=True)

    conversation: Mapped[AiConversation] = relationship("AiConversation", back_populates="messages")

class AiCache(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "ai_cache"

    cache_key: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    # Cache entries contain generated private-answer text. Cache keys are
    # user-specific, and this ownership column lets database RLS protect the
    # payload even if another repository query is accidentally broadened.
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    question_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # Exact authorization context used to create the answer.  The legacy
    # bitmap is retained for compatibility, but must never be used as the
    # authority for serving cached answer text.
    authorization_fingerprint: Mapped[str] = mapped_column(String(64), index=True, nullable=False, default="legacy")
    access_group_bitmap: Mapped[int] = mapped_column(BigInteger, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[str] = mapped_column(Text, nullable=False)  # JSON serialized array of citations
    article_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

class AiFeedback(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "ai_feedback"

    ai_usage_log_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ai_usage_logs.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # e.g., 1 for thumbs up, -1 for thumbs down
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    usage_log: Mapped[AiUsageLog] = relationship("AiUsageLog", back_populates="feedback")
    user: Mapped["User | None"] = relationship("User")

class PromptVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "prompt_versions"

    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(default=True, nullable=False)
