import asyncio
import json
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from src.api.deps import SessionLocal, get_current_user, get_db, set_database_context
from src.domain.ai_service import AIService, normalize_answer_markdown
from src.domain.search_service import SearchService
from src.domain.rbac import AuthorizationService
from src.models import User
from src.models.article import Article
from src.models.chunk import ArticleChunk, ParentChunk
from src.repositories.ai import AIRepository
from src.repositories.chunk import ChunkRepository
from src.repositories.governance import GovernanceRepository
from src.repositories.article import ArticleRepository
from src.repositories.user import UserRepository
from src.core.rate_limit import ai_rate_limiter
from src.repositories.feature_flags import FeatureFlagRepository
from src.rag.citations import extract_citation_ids

router = APIRouter()
logger = structlog.get_logger()


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: uuid.UUID | None = None
    language: Literal["en", "vi"] = "en"



class ConversationCreate(BaseModel):
    title: str | None = None


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class FeedbackRequest(BaseModel):
    ai_usage_log_id: uuid.UUID
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2_000)


def _historical_source_id(citation: dict) -> str | None:
    """Normalize the stored marker without trusting any other citation field."""
    value = citation.get("source_id")
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized.startswith("C"):
            normalized = normalized[1:]
        if normalized.isdigit() and int(normalized) > 0:
            return f"C{int(normalized)}"
    try:
        source_index = int(citation.get("source_index"))
    except (TypeError, ValueError):
        source_index = 0
    return f"C{source_index}" if source_index > 0 else None


def get_ai_service(db: AsyncSession) -> AIService:
    chunk_repo = ChunkRepository(db)
    gov_repo = GovernanceRepository(db)
    return AIService(
        AIRepository(db),
        SearchService(chunk_repo, gov_repo, FeatureFlagRepository(db)),
        gov_repo,
    )


async def _hydrate_citations(
    db: AsyncSession, user: User, citations: list[dict]
) -> list[dict]:
    """Refresh historical citations with their authorized parent passage."""
    chunk_ids = []
    for citation in citations:
        try:
            if isinstance(citation, dict) and citation.get("chunk_id"):
                chunk_ids.append(uuid.UUID(str(citation["chunk_id"])))
        except (ValueError, TypeError):
            continue
    if not chunk_ids:
        return citations

    # Citation hydration is a data-access path, not a presentation filter.
    # Reuse the complete SQL Article predicate so explicit denies, explicit
    # user visibility, groups, tenant, department, lifecycle, and workflow
    # rules are enforced before source text enters the application process.
    conditions = [
        ArticleChunk.id.in_(chunk_ids),
        Article.status == "published",
        Article.index_status == "ready",
        *ArticleRepository._authorized_article_filters(user),
    ]
    result = await db.execute(
        select(ArticleChunk)
        .join(Article, Article.id == ArticleChunk.article_id)
        .options(
            selectinload(ArticleChunk.parent_chunk).selectinload(
                ParentChunk.child_chunks
            ),
            selectinload(ArticleChunk.article),
        )
        .where(*conditions)
    )
    chunks_by_id = {str(chunk.id): chunk for chunk in result.scalars().all()}
    hydrated = []
    for citation in citations:
        chunk = chunks_by_id.get(str(citation.get("chunk_id")))
        if (
            chunk
            and chunk.parent_chunk
            and chunk.article
            and _historical_source_id(citation)
        ):
            parent = chunk.parent_chunk
            article = chunk.article
            source_id = _historical_source_id(citation)
            child_chunks = sorted(
                parent.child_chunks, key=lambda item: item.chunk_index
            )
            page_number = chunk.page_number or parent.page_number
            hydrated.append(
                {
                    "source_id": source_id,
                    "source_index": int(source_id[1:]),
                    "chunk_id": str(chunk.id),
                    "child_chunk_ids": [str(item.id) for item in child_chunks]
                    or [str(chunk.id)],
                    "parent_chunk_id": str(parent.id),
                    "article_id": str(chunk.article_id),
                    "title": article.title,
                    "section_ref": parent.section_ref,
                    "heading": chunk.heading or parent.heading,
                    "source_ref": f"{article.title} - {chunk.heading or parent.heading or parent.section_ref or 'General'}",
                    "excerpt": parent.text[:1800],
                    "highlight_text": chunk.chunk_text[:500],
                    "highlight_texts": [child.chunk_text for child in child_chunks]
                    or [chunk.chunk_text],
                    "page_number": page_number,
                    "source_url": f"/api/v1/articles/{chunk.article_id}/source"
                    + (f"?page={page_number}" if page_number else ""),
                }
            )
        # Every persisted citation must identify a concrete retrieved chunk.
        # Do not preserve metadata-only or malformed entries: they can carry
        # forged excerpts, titles, or source URLs and cannot be re-authorized.
    return hydrated


def _parse_historical_citations(value: str | None) -> tuple[list[dict], bool]:
    """Parse stored citations and flag any shape that cannot be re-authorized."""
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], bool(value)
    if not isinstance(parsed, list):
        return [], bool(parsed)

    citations: list[dict] = []
    malformed = False
    for item in parsed:
        if (
            not isinstance(item, dict)
            or not item.get("chunk_id")
            or not _historical_source_id(item)
        ):
            malformed = True
            continue
        try:
            uuid.UUID(str(item["chunk_id"]))
        except (ValueError, TypeError, AttributeError):
            malformed = True
            continue
        citations.append(item)
    return citations, malformed


@router.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Any:
    conversations = await AIRepository(db).list_conversations(current_user.id)
    return [
        {"id": str(item.id), "title": item.title, "updated_at": item.updated_at}
        for item in conversations
    ]


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    req: ConversationCreate | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    conversation = await AIRepository(db).create_conversation(
        current_user.id, (req.title if req else None) or "New conversation"
    )
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "updated_at": conversation.updated_at,
    }


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    repo = AIRepository(db)
    if not await repo.get_conversation(conversation_id, current_user.id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await repo.list_messages(conversation_id, current_user.id)
    response = []
    for message in messages:
        raw_citations, malformed_citations = _parse_historical_citations(
            message.citations
        )
        citations = await _hydrate_citations(db, current_user, raw_citations)
        stored_markers = {
            marker
            for marker in (_historical_source_id(item) for item in raw_citations)
            if marker
        }
        answer_markers = (
            set(extract_citation_ids(message.grounded_content or message.content))
            if message.role == "assistant"
            else set()
        )
        inaccessible_source = message.role == "assistant" and (
            not raw_citations
            or malformed_citations
            or len(citations) < len(raw_citations)
            or answer_markers != stored_markers
        )
        response.append(
            {
                "id": str(message.id),
                "role": message.role,
                # Cached conversation text can contain verbatim source passages.
                # Do not preserve it after any cited source is no longer allowed.
                "content": (
                    "This historical answer is no longer available because your access to one or more source documents changed."
                    if message.role == "assistant" and inaccessible_source
                    else normalize_answer_markdown(message.content)
                ),
                # Do not leave the original answer in a secondary field after a
                # cited source has become unauthorized.
                "answer_grounded": (
                    (message.grounded_content or message.content)
                    if message.role == "assistant" and not inaccessible_source
                    else ""
                ),
                "answer_extended": (
                    (message.extended_content or "")
                    if message.role == "assistant" and not inaccessible_source
                    else ""
                ),
                "has_extended": (
                    bool(message.extended_content)
                    if message.role == "assistant" and not inaccessible_source
                    else False
                ),
                "citations": [] if inaccessible_source else citations,
                "usage_log_id": (
                    str(message.usage_log_id) if message.usage_log_id else None
                ),
                "created_at": message.created_at,
            }
        )
    return response


@router.delete(
    "/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    repo = AIRepository(db)
    conversation = await repo.get_conversation(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await repo.delete_conversation(conversation)


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: uuid.UUID,
    req: ConversationRename,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    repo = AIRepository(db)
    conversation = await repo.get_conversation(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    renamed = await repo.rename_conversation(conversation, req.title)
    return {
        "id": str(renamed.id),
        "title": renamed.title,
        "updated_at": renamed.updated_at,
    }


@router.post("/ask")
async def ask_question(
    req: AskRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    allowed, retry_after = await ai_rate_limiter.allow(str(current_user.id))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="AI request limit exceeded. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )
    repo = AIRepository(db)
    conversation = (
        await repo.get_conversation(req.conversation_id, current_user.id)
        if req.conversation_id
        else None
    )
    if req.conversation_id and not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not conversation:
        conversation = await repo.create_conversation(
            current_user.id, req.question[:80]
        )
    elif conversation.title == "New conversation":
        conversation.title = req.question[:80]
        await db.commit()

    await repo.add_message(conversation.id, "user", req.question)
    data = await get_ai_service(db).ask(
        current_user, req.question, conversation_id=conversation.id
    )
    await repo.add_message(
        conversation.id,
        "assistant",
        data["answer"],
        data.get("citations"),
        uuid.UUID(data["log_id"]) if data.get("log_id") else None,
        data.get("answer_grounded"),
        data.get("answer_extended"),
    )
    data["conversation_id"] = str(conversation.id)
    return data


@router.post("/ask/stream")
async def ask_question_stream(
    req: AskRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream a grounded answer and its citations using the DocNexus SSE contract."""
    # The SSE generator starts only after FastAPI has returned the response,
    # at which point request-scoped dependencies (including ``db``) may have
    # been closed.  Keep only scalar authorization context here; never close
    # over the request-bound SQLAlchemy ``User`` instance in ``event_stream``.
    user_id = current_user.id
    company_domain = current_user.company_domain
    is_global_admin = AuthorizationService.is_global_administrator(current_user)
    has_global_article_access = AuthorizationService.has_global_article_access(
        current_user
    )
    has_global_identity_access = AuthorizationService.has_global_identity_management(
        current_user
    )
    has_global_connector_access = AuthorizationService.has_global_connector_management(
        current_user
    )
    has_global_governance_access = AuthorizationService.has_global_governance_access(
        current_user
    )

    allowed, retry_after = await ai_rate_limiter.allow(str(user_id))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="AI request limit exceeded. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )

    repo = AIRepository(db)
    conversation = (
        await repo.get_conversation(req.conversation_id, current_user.id)
        if req.conversation_id
        else None
    )
    if req.conversation_id and not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not conversation:
        conversation = await repo.create_conversation(
            current_user.id, req.question[:80]
        )
    elif conversation.title == "New conversation":
        conversation.title = req.question[:80]
        await db.commit()

    await repo.add_message(conversation.id, "user", req.question)
    conversation_id = str(conversation.id)

    async def event_stream():
        queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()

        # Streaming responses outlive the endpoint dependency context. Use a
        # dedicated session and reload the user so authorization and database
        # work never touch detached ORM instances.
        async with SessionLocal() as stream_db:
            await set_database_context(
                stream_db,
                company_domain,
                is_global_admin,
                str(user_id),
                has_global_article_access,
                has_global_identity_access,
                has_global_connector_access,
                has_global_governance_access,
            )
            stream_user = await UserRepository(stream_db).get_by_id(user_id)
            if stream_user is None:
                yield f"data: {json.dumps({'type': 'error', 'detail': 'User account is no longer available.'})}\n\n"
                return
            stream_repo = AIRepository(stream_db)

            async def on_token(content: str) -> None:
                await queue.put({"type": "token", "content": content})

            async def on_replace(content: str) -> None:
                await queue.put({"type": "replace", "content": content})

            task = asyncio.create_task(
                get_ai_service(stream_db).ask(
                    stream_user,
                    req.question,
                    conversation_id=uuid.UUID(conversation_id),
                    language=req.language,
                    on_token=on_token,
                    on_replace=on_replace,
                )
            )
            streamed_content = False
            while not task.done() or not queue.empty():
                if queue.empty():
                    await asyncio.sleep(0.01)
                    continue
                event = await queue.get()
                if event.get("type") in {"token", "replace"}:
                    streamed_content = True
                yield f"data: {json.dumps(event)}\n\n"

            try:
                data = await task
            except HTTPException as exc:
                logger.error(
                    "AI stream task failed",
                    status_code=exc.status_code,
                    detail=str(exc.detail),
                )
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc.detail)})}\n\n"
                return
            except Exception as exc:
                logger.exception("AI stream task failed unexpectedly", error=str(exc))
                yield f"data: {json.dumps({'type': 'error', 'detail': 'AI generation failed. Please try again.'})}\n\n"
                return
            if not streamed_content:
                answer = data.get("answer", "")
                for start in range(0, len(answer), 48):
                    yield f"data: {json.dumps({'type': 'token', 'content': answer[start:start + 48]})}\n\n"
                    await asyncio.sleep(0)
            await stream_repo.add_message(
                uuid.UUID(conversation_id),
                "assistant",
                data["answer"],
                data.get("citations"),
                uuid.UUID(data["log_id"]) if data.get("log_id") else None,
                data.get("answer_grounded"),
                data.get("answer_extended"),
            )
            yield f"data: {json.dumps({'type': 'sources', 'sources': data.get('citations', [])})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id, 'log_id': data.get('log_id'), 'prompt_version': data.get('prompt_version'), 'retrieval_version': data.get('retrieval_version'), 'answer_grounded': data.get('answer_grounded', ''), 'answer_extended': data.get('answer_extended', ''), 'has_extended': data.get('has_extended', False)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    success = await get_ai_service(db).submit_feedback(
        user=current_user,
        log_id=req.ai_usage_log_id,
        rating=req.rating,
        comment=req.comment,
    )
    return {"success": success}
