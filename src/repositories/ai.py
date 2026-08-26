import uuid
from datetime import datetime
from typing import Sequence
from sqlalchemy import select, and_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.ai import (
    AiUsageLog,
    AiCache,
    AiFeedback,
    PromptVersion,
    AiConversation,
    AiMessage,
)
import json


class AIRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # AI Logs
    async def log_usage(self, log: AiUsageLog) -> AiUsageLog:
        self.db.add(log)
        await self.db.commit()
        await self.db.refresh(log)
        return log

    async def get_usage_log(
        self, log_id: uuid.UUID, user_id: uuid.UUID
    ) -> AiUsageLog | None:
        result = await self.db.execute(
            select(AiUsageLog).where(
                AiUsageLog.id == log_id,
                AiUsageLog.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def create_conversation(
        self, user_id: uuid.UUID, title: str = "New conversation"
    ) -> AiConversation:
        conversation = AiConversation(
            user_id=user_id, title=title[:255] or "New conversation"
        )
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    async def list_conversations(self, user_id: uuid.UUID) -> list[AiConversation]:
        result = await self.db.execute(
            select(AiConversation)
            .where(AiConversation.user_id == user_id)
            .order_by(AiConversation.updated_at.desc())
            .limit(200)
        )
        return list(result.scalars().all())

    async def get_conversation(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> AiConversation | None:
        result = await self.db.execute(
            select(AiConversation).where(
                AiConversation.id == conversation_id,
                AiConversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_messages(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[AiMessage]:
        result = await self.db.execute(
            select(AiMessage)
            .join(AiConversation, AiConversation.id == AiMessage.conversation_id)
            .where(
                AiMessage.conversation_id == conversation_id,
                AiConversation.user_id == user_id,
            )
            .order_by(AiMessage.created_at.asc())
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        usage_log_id: uuid.UUID | None = None,
        grounded_content: str | None = None,
        extended_content: str | None = None,
        action_data: dict | None = None,
    ) -> AiMessage:
        message = AiMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            grounded_content=grounded_content,
            extended_content=extended_content,
            citations=json.dumps(citations or []),
            action_data=action_data,
            usage_log_id=usage_log_id,
        )
        self.db.add(message)
        await self.db.execute(
            update(AiConversation)
            .where(AiConversation.id == conversation_id)
            .values(updated_at=datetime.utcnow())
        )
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def delete_conversation(self, conversation: AiConversation) -> None:
        await self.db.delete(conversation)
        await self.db.commit()

    async def rename_conversation(
        self, conversation: AiConversation, title: str
    ) -> AiConversation:
        conversation.title = title[:255].strip() or "New conversation"
        conversation.updated_at = datetime.utcnow()
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    # AI Cache
    async def get_cached(
        self,
        question_hash: str,
        authorization_fingerprint: str,
        owner_user_id: uuid.UUID,
    ) -> AiCache | None:
        # Cached answer text is served only to the exact authorization context
        # that produced it.  Do not use bitmap overlap here: all users share
        # the public bit and overlap is not proof that restricted citations
        # remain authorized.
        now = datetime.utcnow()
        result = await self.db.execute(
            select(AiCache).where(
                and_(
                    AiCache.question_hash == question_hash,
                    AiCache.expires_at > now,
                    AiCache.authorization_fingerprint == authorization_fingerprint,
                    AiCache.owner_user_id == owner_user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def cache_answer(self, cache: AiCache) -> AiCache:
        # Multiple tabs/retries can finish the same question at nearly the
        # same time. The cache key is intentionally unique, so use a
        # PostgreSQL upsert instead of allowing a duplicate-key error to turn
        # a successfully generated answer into a failed chat request.
        cache_id = cache.id or uuid.uuid4()
        statement = (
            pg_insert(AiCache)
            .values(
                id=cache_id,
                cache_key=cache.cache_key,
                owner_user_id=cache.owner_user_id,
                question_hash=cache.question_hash,
                authorization_fingerprint=cache.authorization_fingerprint,
                access_group_bitmap=cache.access_group_bitmap,
                answer=cache.answer,
                citations=cache.citations,
                article_ids=cache.article_ids,
                expires_at=cache.expires_at,
            )
            .on_conflict_do_update(
                index_elements=[AiCache.cache_key],
                set_={
                    "question_hash": cache.question_hash,
                    "owner_user_id": cache.owner_user_id,
                    "authorization_fingerprint": cache.authorization_fingerprint,
                    "access_group_bitmap": cache.access_group_bitmap,
                    "answer": cache.answer,
                    "citations": cache.citations,
                    "article_ids": cache.article_ids,
                    "expires_at": cache.expires_at,
                },
            )
            .returning(AiCache.id)
        )
        result = await self.db.execute(statement)
        await self.db.commit()
        cache.id = result.scalar_one()
        return cache

    # Feedback
    async def log_feedback(self, feedback: AiFeedback) -> AiFeedback:
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    # Prompt Versions
    async def get_active_prompt(self) -> PromptVersion | None:
        result = await self.db.execute(
            select(PromptVersion)
            .where(PromptVersion.active == True)
            .order_by(PromptVersion.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create_prompt(self, prompt: PromptVersion) -> PromptVersion:
        # Set all other active prompt templates to inactive
        from sqlalchemy import update

        await self.db.execute(
            update(PromptVersion)
            .where(PromptVersion.active == True)
            .values(active=False)
        )
        self.db.add(prompt)
        await self.db.commit()
        await self.db.refresh(prompt)
        return prompt
