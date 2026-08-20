import asyncio
import hashlib
import json
import re
import uuid
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Annotated, Any, Sequence
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.api.deps import get_db, get_current_user
from src.models import User
from src.repositories.article import ArticleRepository
from src.repositories.user import UserRepository
from src.domain.articles import ArticleService
from src.domain.source_extraction import (
    SUPPORTED_EXTENSIONS,
    SourceExtractionError,
    extract_source_markdown,
    extract_source_pages,
)
from src.domain.source_storage import (
    save_source,
    delete_source,
    load_source,
    head_source,
    source_storage_key,
    create_presigned_source_url,
    safe_source_media_type,
    source_should_display_inline,
)
from src.models.article import Article, DocumentSource, TagCatalog
from src.models.interaction import ArticleFollower
from src.models.governance import (
    ArticleEditRequest,
    DraftTransition,
    DraftCandidate,
    PendingDraft,
    IngestionFingerprint,
)
from src.models.ops import NotificationQueue
from src.repositories.governance import GovernanceRepository
from src.repositories.feature_flags import FeatureFlagRepository
from src.core.config import settings
from src.repositories.audit import AuditRepository
from src.domain.similarity import find_similar_documents, classify_similarity
from src.domain.content_restructure import restructure_document
from src.domain.document_splitter import split_document_candidates
from src.domain.governance import GovernanceService
from src.domain.llm_client import complete, resolve_provider
from src.domain.events import event_bus
from src.domain.permissions import PermissionService
from src.domain.article_edit_requests import create_article_edit_request as create_edit_request
from src.domain.rbac import AuthorizationService
from src.domain.departments import resolve_active_department, resolve_active_departments
from src.domain.departments import lock_company_access_groups
from src.core.rate_limit import source_upload_rate_limiter
import structlog

router = APIRouter()
logger = structlog.get_logger()
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


def _add_split_candidates(db: AsyncSession, draft: PendingDraft, text: str) -> None:
    """Persist ordered candidates from an already formatted reading view.

    Use the ORM relationship rather than ``draft.id`` directly: a newly
    created draft receives its UUID when SQLAlchemy flushes it.  The
    relationship makes the unit of work insert the draft first and propagate
    its ID to each candidate.
    """
    for item in split_document_candidates(
        draft.title, text, prefer_markdown_sections=True, page_texts=draft.page_texts
    ):
        db.add(DraftCandidate(draft=draft, **item))


TagInput = Annotated[str, Field(min_length=1, max_length=50)]


async def _read_upload_limited(file: UploadFile) -> bytes:
    """Read an upload without allowing a forged multipart part to exhaust RAM.

    ``UploadFile.size`` is populated by Starlette for normal multipart
    requests, but it is not a security boundary.  Enforce the limit while
    consuming the stream as well, because a caller can omit or falsify the
    part's content-length header.
    """
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > settings.MAX_SOURCE_UPLOAD_BYTES:
        await file.close()
        raise HTTPException(status_code=413, detail="Files must be 25 MB or smaller")

    data = bytearray()
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > settings.MAX_SOURCE_UPLOAD_BYTES:
            await file.close()
            raise HTTPException(
                status_code=413, detail="Files must be 25 MB or smaller"
            )
    return bytes(data)


async def _resolve_upload_departments(
    db: AsyncSession,
    current_user: User,
    dept: str | None,
    department_ids: str | None,
) -> list[Any]:
    if department_ids:
        try:
            raw_ids = json.loads(department_ids)
            if not isinstance(raw_ids, list):
                raise ValueError
            selected_ids = [uuid.UUID(str(item)) for item in raw_ids]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422, detail="The department selection is invalid"
            ) from exc
        return await resolve_active_departments(
            db, current_user.company_domain, selected_ids
        )
    return [
        await resolve_active_department(
            db, current_user.company_domain, dept or current_user.dept, required=True
        )
    ]


# Schema definitions
class ArticleCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    external_id: str | None = Field(default=None, max_length=120)
    body_md: str = Field(min_length=1, max_length=2_000_000)
    dept: str | None = Field(default=None, min_length=1, max_length=100)
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    language: str = Field(default="vi", min_length=2, max_length=20)
    tags: list[TagInput] = Field(default_factory=list, max_length=20)
    next_review: datetime | None = None
    visibility: str = Field(default="public", pattern="^(public|department)$")
    explicit_user_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    denied_user_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)


class ArticleUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    body_md: str | None = Field(default=None, min_length=1, max_length=2_000_000)
    dept: str | None = Field(default=None, min_length=1, max_length=100)
    department_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    language: str | None = Field(default=None, min_length=2, max_length=20)
    status: str | None = Field(default=None, min_length=1, max_length=30)
    tags: list[TagInput] | None = Field(default=None, max_length=20)
    next_review: datetime | None = None
    visibility: str | None = Field(default=None, pattern="^(public|department)$")
    explicit_user_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)
    denied_user_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)


class ArticleEditRequestCreate(BaseModel):
    request_text: str = Field(min_length=5, max_length=5_000)


class AutoTagRequest(BaseModel):
    article_ids: list[uuid.UUID] = Field(min_length=1, max_length=20)


class AutoTagResponse(BaseModel):
    results: list[dict]
    updated_count: int = 0
    requires_confirmation: bool = True


class ConfirmTagItem(BaseModel):
    article_id: uuid.UUID
    tags: list[str] = Field(max_length=20)


class ConfirmTagsRequest(BaseModel):
    items: list[ConfirmTagItem] = Field(min_length=1, max_length=20)


class GroupResponse(BaseModel):
    id: uuid.UUID
    name: str
    bitmask_position: int
    model_config = ConfigDict(from_attributes=True)


class TagResponse(BaseModel):
    tag: str
    model_config = ConfigDict(from_attributes=True)


class OwnerResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: str

    model_config = ConfigDict(from_attributes=True)


class DepartmentResponse(BaseModel):
    id: uuid.UUID
    name: str
    model_config = ConfigDict(from_attributes=True)


class ArticleResponse(BaseModel):
    id: uuid.UUID
    external_id: str | None = None
    title: str
    body_md: str
    dept: str
    departments: list[DepartmentResponse] = []
    domain: str
    company_domain: str
    type: str
    sensitivity: str
    visibility: str = "department"
    explicit_user_ids: list[uuid.UUID] = []
    explicit_denied_user_ids: list[uuid.UUID] = []
    language: str
    self_approved: bool = False
    source_changed: bool = False
    source_changed_at: datetime | None = None
    owner_id: uuid.UUID | None = None
    owner: OwnerResponse | None = None
    status: str
    lifecycle_status: str = "active"
    related_article_ids: list[str] | None = None
    source_position: dict[str, Any] | None = None
    version: int
    created_at: datetime
    next_review: datetime | None = None
    last_reviewed: datetime | None = None
    needs_update: bool = False
    index_status: str = "pending"
    index_error: str | None = None
    source_available: bool = False
    access_groups: list[GroupResponse] = []
    tags: list[TagResponse] = []

    model_config = ConfigDict(from_attributes=True)


class VersionResponse(BaseModel):
    id: uuid.UUID
    article_id: uuid.UUID
    version: int
    snapshot: dict
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DraftSubmissionResponse(BaseModel):
    id: uuid.UUID
    title: str
    status: str
    workflow: str
    message: str


class SourceUploadIntent(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    source_hash: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    content_length: int = Field(gt=0)
    dept: str | None = Field(default=None, min_length=1, max_length=100)
    department_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)
    tags: list[TagInput] = Field(default_factory=list, max_length=20)


class SourceUploadComplete(BaseModel):
    content_length: int | None = Field(default=None, gt=0)


@router.post("/auto-tags", response_model=AutoTagResponse)
async def auto_tag_articles(
    request: AutoTagRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AutoTagResponse:
    """Generate tag suggestions for a user-selected article batch.

    Suggestions are deliberately never persisted here.  The caller must make
    an explicit confirmation request after a human has reviewed them.
    """
    unique_ids = list(dict.fromkeys(request.article_ids))
    if not unique_ids:
        raise HTTPException(status_code=400, detail="Select at least one article")
    if len(unique_ids) > 20:
        raise HTTPException(
            status_code=400, detail="Select no more than 20 articles at a time"
        )

    provider = resolve_provider()
    if not provider:
        raise HTTPException(
            status_code=503,
            detail="AI tagging is unavailable because no LLM provider is configured",
        )

    article_repo = ArticleRepository(db)
    articles = []
    for article_id in unique_ids:
        article = await article_repo.get_by_id(article_id, user=current_user)
        if not article or article.status == "deleted":
            raise HTTPException(
                status_code=404, detail=f"Article {article_id} not found"
            )
        if not PermissionService.can_edit_article(current_user, article):
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to tag one or more selected articles",
            )
        articles.append(article)

    documents = []
    for index, article in enumerate(articles, start=1):
        existing = [tag.tag for tag in article.tags]
        documents.append(
            f"ARTICLE {index}\nID: {article.id}\nTITLE: {article.title}\n"
            f"TYPE: {article.type}\nEXISTING TAGS: {', '.join(existing) or '(none)'}\n"
            f"CONTENT:\n{article.body_md[:5000]}"
        )
    prompt = (
        "Generate concise search tags for each article below. Return ONLY valid JSON in this exact shape: "
        '{"articles":[{"id":"article UUID","tags":["tag1","tag2"]}]}.'
        "Use 3 to 8 specific tags per article, in lowercase, with letters, numbers, spaces, or hyphens. "
        "Do not invent tags unrelated to the article. Keep existing tags when relevant and add useful missing tags.\n\n"
        + "\n\n".join(documents)
    )
    try:
        answer, _, _, _ = await complete(
            [
                {
                    "role": "system",
                    "content": "You are a precise knowledge-base taxonomy assistant. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        cleaned = answer.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        payload = json.loads(cleaned)
        raw_results = payload.get("articles", []) if isinstance(payload, dict) else []
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error(
            "AI tag response was not valid JSON", error=str(exc), model=provider.model
        )
        raise HTTPException(
            status_code=502, detail="AI returned an invalid tag response"
        )
    except Exception as exc:
        logger.error("AI tag generation failed", error=str(exc), model=provider.model)
        raise HTTPException(
            status_code=502, detail="AI tagging failed. Please try again."
        )

    suggestions_by_id: dict[str, list[str]] = {}
    try:
        catalogue = set((await db.execute(select(TagCatalog.normalized_tag).where(TagCatalog.company_domain == current_user.company_domain, TagCatalog.active.is_(True)))).scalars().all())
    except AttributeError:
        # Lightweight unit doubles may not expose a database; production
        # requests always use the tenant catalogue query above.
        catalogue = None
    normalize_tag = lambda value: re.sub(r"\s+", " ", "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)).strip().casefold())
    for item in raw_results:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            continue
        cleaned_tags = []
        for tag in tags:
            value = re.sub(r"\s+", " ", str(tag).strip().lower())
            if (
                value
                and len(value) <= 50
                and re.fullmatch(r"[\w -]+", value, flags=re.UNICODE)
                and value not in cleaned_tags
            ):
                cleaned_tags.append(value)
        # AI may propose only customer-approved vocabulary. An empty catalogue
        # deliberately produces no automatic suggestions until the customer
        # has loaded its taxonomy.
        if catalogue == set():
            cleaned_tags = []
        elif catalogue is not None:
            cleaned_tags = [tag for tag in cleaned_tags if normalize_tag(tag) in catalogue]
        suggestions_by_id[str(item["id"])] = cleaned_tags[:8]

    results = []
    for article in articles:
        current_tags = [tag.tag for tag in article.tags]
        suggestions = suggestions_by_id.get(str(article.id), [])
        proposed_tags = list(dict.fromkeys([*current_tags, *suggestions]))[:20]
        results.append(
            {
                "article_id": str(article.id),
                "title": article.title,
                "current_tags": current_tags,
                "suggested_tags": suggestions,
                "proposed_tags": proposed_tags,
                "added_tags": [tag for tag in proposed_tags if tag not in current_tags],
            }
        )
    logger.info(
        "AI tag suggestions generated",
        article_count=len(articles),
        model=provider.model,
        user_id=str(current_user.id),
    )
    return AutoTagResponse(results=results, updated_count=0, requires_confirmation=True)


@router.post("/tags/confirm")
async def confirm_article_tags(
    request: ConfirmTagsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Persist only the exact tag sets explicitly confirmed by a human."""
    article_repo = ArticleRepository(db)
    seen: set[uuid.UUID] = set()
    pending: list[tuple[Article, list[str]]] = []
    for item in request.items:
        if item.article_id in seen:
            raise HTTPException(
                status_code=422, detail="Each article may appear only once"
            )
        seen.add(item.article_id)
        article = await article_repo.get_by_id(item.article_id, user=current_user)
        if not article or article.status == "deleted":
            raise HTTPException(
                status_code=404, detail=f"Article {item.article_id} not found"
            )
        if not PermissionService.can_edit_article(current_user, article):
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to confirm tags for one or more selected articles",
            )
        cleaned_tags: list[str] = []
        for raw_tag in item.tags:
            value = re.sub(r"\s+", " ", str(raw_tag).strip().lower())
            if (
                value
                and len(value) <= 50
                and re.fullmatch(r"[\w -]+", value, flags=re.UNICODE)
                and value not in cleaned_tags
            ):
                cleaned_tags.append(value)
        pending.append((article, cleaned_tags[:20]))

    # Validate the complete batch before making the first write. This avoids a
    # partial confirmation when a later item is unauthorized or missing.
    confirmed: list[dict[str, Any]] = []
    published_ids: list[uuid.UUID] = []
    for article, cleaned_tags in pending:
        article.index_status = "pending"
        article.index_error = None
        article.updated_at = datetime.utcnow()
        await article_repo.sync_tags(article.id, cleaned_tags, commit=False)
        confirmed.append({"article_id": str(article.id), "tags": cleaned_tags})
        if article.status == "published":
            published_ids.append(article.id)
    await db.commit()
    for article, _ in pending:
        await AuditRepository(db).record(
            current_user.id,
            "tags_confirm",
            "article",
            str(article.id),
            outcome="success",
        )
    for article_id in published_ids:
        await event_bus.publish("ArticleUpdated", {"article_id": str(article_id)})
    return {"confirmed": confirmed, "confirmed_count": len(confirmed)}


@router.post("/upload-source", status_code=status.HTTP_201_CREATED)
async def upload_source(
    file: UploadFile = File(...),
    tags: str | None = Form(None),
    dept: str | None = Form(None),
    department_ids: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    # A batch shares one DB session and should consume one request quota, not
    # one quota unit per file. Individual requests still receive their own
    # marker because FastAPI creates a fresh session for each request.
    rate_marker = f"source_upload_rate_checked:{current_user.id}"
    if not db.info.get(rate_marker):
        allowed, retry_after = await source_upload_rate_limiter.allow(
            str(current_user.id)
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Upload rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        db.info[rate_marker] = True
    selected_departments = await _resolve_upload_departments(
        db, current_user, dept, department_ids
    )
    upload_dept = selected_departments[0].name
    upload_resource = Article(
        company_domain=current_user.company_domain,
        dept=upload_dept,
        owner_id=current_user.id,
        departments=selected_departments,
    )
    if not any(
        AuthorizationService.has_permission(
            current_user, "article.create", upload_resource, scope
        )
        for scope in ("own", "department", "company", "global")
    ):
        raise HTTPException(status_code=403, detail="Not authorized to upload sources")
    data = await _read_upload_limited(file)
    try:
        raw_tags = json.loads(tags) if tags else []
    except json.JSONDecodeError:
        raw_tags = tags.split(",") if tags else []
    requested_tags = [
        tag.strip() for tag in raw_tags if isinstance(tag, str) and tag.strip()
    ][:20]
    logger.info(
        "Source upload received",
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(data),
        user_id=str(current_user.id),
    )
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded file is empty")
    filename = (
        Path(file.filename or "uploaded-source").name.strip()[:255] or "uploaded-source"
    )
    try:
        extracted_pages = await asyncio.to_thread(extract_source_pages, filename, data)
    except SourceExtractionError as exc:
        logger.warning("Source extraction rejected", filename=filename, error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Source extraction failed", filename=filename, error=str(exc))
        raise HTTPException(
            status_code=422, detail="Could not process uploaded source"
        ) from exc

    source_hash = hashlib.sha256(data).hexdigest()
    # Article deletion is soft-delete. Ignore source rows belonging to deleted
    # or inactive articles so a document can be uploaded again after removal.
    # Reserve the tenant/hash pair while the expensive extraction and storage
    # work is still in progress. This closes the concurrent-upload race.
    await lock_company_access_groups(
        db, f"upload:{current_user.company_domain}:{source_hash}"
    )
    fingerprint = await db.scalar(
        select(IngestionFingerprint).where(
            IngestionFingerprint.company_domain == current_user.company_domain,
            IngestionFingerprint.source_hash == source_hash,
        )
    )
    if fingerprint and fingerprint.status == "uploading":
        # Older browser-to-R2 attempts can be abandoned before completion.
        # They have no extracted content and are intentionally hidden from the
        # review queue, so release the reservation before this authenticated
        # API upload creates the real pending draft.
        abandoned_draft = (
            await db.get(PendingDraft, fingerprint.draft_id)
            if fingerprint.draft_id
            else None
        )
        if abandoned_draft and abandoned_draft.status == "draft":
            if abandoned_draft.storage_key:
                try:
                    await asyncio.to_thread(delete_source, abandoned_draft.storage_key)
                except Exception:
                    logger.warning(
                        "Could not remove abandoned presigned upload object",
                        draft_id=str(abandoned_draft.id),
                    )
            await db.delete(abandoned_draft)
        await db.delete(fingerprint)
        await db.flush()
    if fingerprint and fingerprint.status in {"pending", "approved"}:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_document",
                "message": "This document already exists.",
            },
        )
    exact_stmt = (
        select(DocumentSource)
        .join(DocumentSource.article)
        .where(
            DocumentSource.source_hash == source_hash,
            Article.status != "deleted",
            Article.lifecycle_status == "active",
        )
    )
    # Duplicate ownership is tenant-scoped even for a global administrator.
    # A global manager uploading into tenant A must not be blocked by tenant B.
    exact_stmt = exact_stmt.where(Article.company_domain == current_user.company_domain)
    exact_source = (await db.execute(exact_stmt)).scalars().first()
    if exact_source:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_document",
                "message": "This document already exists.",
            },
        )
    extracted = await asyncio.to_thread(
        extract_source_markdown, filename, data, extracted_pages
    )
    restructuring_enabled = (
        settings.RESTRUCTURE_ENABLED
        and await FeatureFlagRepository(db).is_enabled(
            "ai.document_restructure", current_user
        )
    )
    matches = await find_similar_documents(db, current_user, extracted)
    similarity_level = classify_similarity(matches)
    if similarity_level == "exact":
        # Extracted PDF text can collapse to boilerplate or omit image-only
        # pages, so a text-perfect match is evidence for review—not proof
        # that the uploaded binary is a duplicate. The hash check above is
        # the only hard duplicate gate.
        similarity_level = "very_high"
    storage_key = await asyncio.to_thread(
        save_source, source_hash, filename, data, current_user.company_domain
    )
    draft = PendingDraft(
        title=filename.rsplit(".", 1)[0][:255],
        company_domain=current_user.company_domain,
        dept=upload_dept,
        source_ref=f"upload://{filename}",
        source_hash=source_hash,
        summary=extracted,
        # Upload persistence must not wait for the remote LLM. The worker will
        # fill the reading view after the draft is visible in the review queue.
        restructured_body_md=None,
        restructure_status="queued" if restructuring_enabled else "disabled",
        restructure_model=(
            settings.RESTRUCTURE_MODEL if restructuring_enabled else "none"
        ),
        restructure_error=None,
        storage_key=storage_key,
        original_filename=filename,
        mime_type=safe_source_media_type(filename),
        page_texts=extracted_pages,
        created_by=current_user.id,
        status="draft",
        similarity_level=similarity_level,
        similarity_matches=matches,
        requires_update_confirmation=similarity_level == "very_high",
        related_article_ids=(
            [item["article_id"] for item in matches]
            if similarity_level == "partial"
            else None
        ),
        tags=requested_tags,
        content_metadata={
            "department_ids": [
                str(department.id) for department in selected_departments
            ]
        },
    )
    db.add(
        IngestionFingerprint(
            company_domain=current_user.company_domain,
            source_hash=source_hash,
            status="pending",
            draft_id=draft.id,
            created_by=current_user.id,
        )
    )
    try:
        await GovernanceRepository(db).create_draft(draft)
        await GovernanceService(
            GovernanceRepository(db), ArticleRepository(db)
        ).submit_draft(current_user, draft.id)
    except Exception:
        try:
            await asyncio.to_thread(delete_source, storage_key)
        except Exception:
            logger.exception(
                "Failed to clean up source after draft persistence failure",
                storage_key=storage_key,
            )
        raise
    if restructuring_enabled:
        # Dispatch only after the draft transaction is committed so the
        # formatter can immediately load the stored source text and update
        # the same pending draft safely.
        from src.workers.tasks import dispatch_restructure_pending_draft

        if not dispatch_restructure_pending_draft(
            str(draft.id), current_user.company_domain, str(current_user.id)
        ):
            # A missing broker must not turn a successful upload into a
            # failure. Reviewers can still retry formatting from the queue.
            draft.restructure_status = "fallback_formatting"
            draft.restructure_model = "lossless-markdown"
            draft.restructure_error = (
                "AI formatting could not be queued; retry from Pending Drafts."
            )
            await db.commit()
    logger.info(
        "Source upload queued as pending draft",
        draft_id=str(draft.id),
        filename=filename,
        page_count=len(extracted_pages),
        extracted_characters=len(extracted),
        page_text_characters=sum(len(str(page["text"])) for page in extracted_pages),
        restructure_status=draft.restructure_status,
        restructure_model=draft.restructure_model,
    )
    return {
        "id": str(draft.id),
        "title": draft.title,
        "source_ref": draft.source_ref,
        "source_hash": draft.source_hash,
        "status": draft.status,
        "company_domain": draft.company_domain,
        "dept": draft.dept,
        "departments": [
            {"id": str(department.id), "name": department.name}
            for department in selected_departments
        ],
        "assigned_approver_id": None,
        "extracted_characters": len(extracted),
        "page_count": len(extracted_pages),
        "restructure_status": draft.restructure_status,
        "restructure_model": draft.restructure_model,
        "restructure_decision": draft.restructure_decision,
        "similarity_level": similarity_level,
        "similarity_matches": matches,
        "requires_update_confirmation": similarity_level == "very_high",
        "related_article_ids": (
            [item["article_id"] for item in matches]
            if similarity_level == "partial"
            else []
        ),
        "tags": requested_tags,
        "message": (
            "Source stored and queued for reviewer approval. AI reading view is formatting in the background."
            if restructuring_enabled
            else "Source extracted and queued for reviewer approval."
        ),
    }


@router.post("/upload-sources", status_code=status.HTTP_201_CREATED)
async def upload_sources(
    files: list[UploadFile] = File(...),
    # Browsers send one JSON-encoded tag array per file (for example
    # `["body"]`). Accept that shape as well as repeated multipart fields.
    tags: str | list[str] | None = Form(None),
    dept: str | None = Form(None),
    department_ids: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Process a batch without letting one bad or duplicate file stop the batch."""
    if not files:
        raise HTTPException(status_code=422, detail="Select at least one file")
    if len(files) > 20:
        raise HTTPException(status_code=413, detail="Upload at most 20 files per batch")
    tag_values = tags if isinstance(tags, list) else ([tags] if tags else [])
    results: list[dict[str, Any]] = []
    for index, file in enumerate(files):
        try:
            result = await upload_source(
                file=file,
                tags=tag_values[index] if index < len(tag_values) else None,
                dept=dept,
                department_ids=department_ids,
                current_user=current_user,
                db=db,
            )
            results.append({"filename": file.filename, "status": "queued", **result})
        except HTTPException as exc:
            detail = (
                exc.detail
                if isinstance(exc.detail, dict)
                else {"message": str(exc.detail)}
            )
            results.append(
                {
                    "filename": file.filename,
                    "status": (
                        "duplicate"
                        if exc.status_code == 409
                        and detail.get("code") == "duplicate_document"
                        else "failed"
                    ),
                    "status_code": exc.status_code,
                    "detail": detail,
                }
            )
    return {
        "results": results,
        "queued_count": sum(item["status"] == "queued" for item in results),
        "duplicate_count": sum(item["status"] == "duplicate" for item in results),
        "failed_count": sum(item["status"] == "failed" for item in results),
    }


@router.post("/source-uploads", status_code=status.HTTP_201_CREATED)
async def create_source_upload_intent(
    request: SourceUploadIntent,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Reserve a draft and return a short-lived private R2 PUT URL.

    The API never accepts the source bytes on this path. The client uploads
    directly to the private bucket, then calls the completion endpoint so the
    server can verify the object, extract it, and submit it for review.
    """
    if request.content_length > settings.MAX_SOURCE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Files must be 25 MB or smaller")
    filename = Path(request.filename).name.strip()[:255] or "uploaded-source"
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported file type")
    selected_departments = (
        await resolve_active_departments(
            db, current_user.company_domain, request.department_ids
        )
        if request.department_ids
        else [
            await resolve_active_department(
                db,
                current_user.company_domain,
                request.dept or current_user.dept,
                required=True,
            )
        ]
    )
    upload_resource = Article(
        company_domain=current_user.company_domain,
        dept=selected_departments[0].name,
        owner_id=current_user.id,
        departments=selected_departments,
    )
    if not any(
        AuthorizationService.has_permission(
            current_user, "article.create", upload_resource, scope
        )
        for scope in ("own", "department", "company", "global")
    ):
        raise HTTPException(status_code=403, detail="Not authorized to upload sources")

    await lock_company_access_groups(
        db,
        f"presigned-upload:{current_user.company_domain}:{request.source_hash.lower()}",
    )
    fingerprint = await db.scalar(
        select(IngestionFingerprint).where(
            IngestionFingerprint.company_domain == current_user.company_domain,
            IngestionFingerprint.source_hash == request.source_hash.lower(),
        )
    )
    if fingerprint:
        # A browser can fail after the intent has been saved but before its PUT
        # reaches R2.  The draft is deliberately not shown in Pending Drafts
        # until completion, so let its uploader retry that same reservation
        # instead of reporting an invisible upload as a duplicate.
        existing_draft = (
            await db.get(PendingDraft, fingerprint.draft_id)
            if fingerprint.draft_id
            else None
        )
        if (
            fingerprint.status == "uploading"
            and existing_draft
            and existing_draft.status == "draft"
            and existing_draft.created_by == current_user.id
            and existing_draft.storage_key
        ):
            try:
                upload_url = await asyncio.to_thread(
                    create_presigned_source_url,
                    existing_draft.storage_key,
                    operation="put_object",
                    expires_in=900,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="Private R2 upload storage is unavailable"
                ) from exc
            await AuditRepository(db).record(
                current_user.id,
                "source_upload_intent_reissued",
                "draft",
                str(existing_draft.id),
            )
            return {
                "draft_id": str(existing_draft.id),
                "upload_url": upload_url,
                "expires_in": 900,
                "required_headers": {},
                "status": existing_draft.status,
                "filename": existing_draft.original_filename or filename,
            }
        if fingerprint.status in {"pending", "approved"} or (
            fingerprint.status == "uploading"
            and existing_draft
            and existing_draft.status == "draft"
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "duplicate_document",
                    "message": "This document already exists or is being uploaded.",
                },
            )
        # Recover reservations orphaned by an older failed intent (or a draft
        # that was cancelled before completion).  Without this, the unique
        # fingerprint constraint makes every later retry look duplicate.
        await db.delete(fingerprint)
        await db.flush()

    storage_key = source_storage_key(
        request.source_hash.lower(), filename, current_user.company_domain
    )
    draft = PendingDraft(
        title=filename.rsplit(".", 1)[0][:255],
        company_domain=current_user.company_domain,
        dept=selected_departments[0].name,
        source_ref=f"upload://{filename}",
        source_hash=request.source_hash.lower(),
        storage_key=storage_key,
        original_filename=filename,
        mime_type=safe_source_media_type(filename),
        created_by=current_user.id,
        status="draft",
        tags=list(
            dict.fromkeys(str(tag).strip() for tag in request.tags if str(tag).strip())
        )[:20],
        content_metadata={
            "upload_mode": "presigned",
            "expected_content_length": request.content_length,
            "department_ids": [
                str(department.id) for department in selected_departments
            ],
        },
    )
    db.add(draft)
    await db.flush()
    db.add(
        DraftTransition(
            draft_id=draft.id,
            from_status=None,
            to_status="draft",
            actor_id=current_user.id,
            reason="Draft created for direct R2 upload",
            outcome="applied",
        )
    )
    fingerprint = IngestionFingerprint(
        company_domain=current_user.company_domain,
        source_hash=request.source_hash.lower(),
        status="uploading",
        draft_id=draft.id,
        created_by=current_user.id,
    )
    db.add(fingerprint)
    try:
        await db.commit()
        await db.refresh(draft)
        upload_url = await asyncio.to_thread(
            create_presigned_source_url,
            storage_key,
            operation="put_object",
            expires_in=900,
        )
    except Exception as exc:
        await db.rollback()
        try:
            stale_fingerprint = await db.scalar(
                select(IngestionFingerprint).where(
                    IngestionFingerprint.company_domain == current_user.company_domain,
                    IngestionFingerprint.source_hash == request.source_hash.lower(),
                )
            )
            if stale_fingerprint:
                await db.delete(stale_fingerprint)
            stale = await db.get(PendingDraft, draft.id)
            if stale:
                await db.delete(stale)
            await db.commit()
        except Exception:
            await db.rollback()
        raise HTTPException(
            status_code=503, detail="Private R2 upload storage is unavailable"
        ) from exc

    await AuditRepository(db).record(
        current_user.id, "source_upload_intent", "draft", str(draft.id)
    )
    return {
        "draft_id": str(draft.id),
        "upload_url": upload_url,
        "expires_in": 900,
        "required_headers": {},
        "status": draft.status,
        "filename": filename,
    }


@router.post("/source-uploads/{draft_id}/complete", status_code=status.HTTP_201_CREATED)
async def complete_source_upload(
    draft_id: uuid.UUID,
    request: SourceUploadComplete | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Verify a direct R2 upload, extract it, and submit the draft."""
    draft = await GovernanceRepository(db).get_draft_for_user(
        draft_id, current_user, for_update=True
    )
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    if draft.created_by != current_user.id and not AuthorizationService.has_permission(
        current_user, "article.publish", requested_scope="global"
    ):
        raise HTTPException(
            status_code=403,
            detail="Only the uploader or a global publisher can complete this upload",
        )
    if (
        draft.status != "draft"
        or not draft.storage_key
        or (draft.content_metadata or {}).get("upload_mode") != "presigned"
    ):
        raise HTTPException(
            status_code=409, detail="This draft is not awaiting a presigned upload"
        )

    expected_length = int(
        (draft.content_metadata or {}).get("expected_content_length") or 0
    )

    async def discard_failed_upload(reason: str) -> None:
        try:
            await asyncio.to_thread(delete_source, draft.storage_key)
        except Exception:
            logger.warning(
                "Could not remove failed presigned upload object",
                draft_id=str(draft.id),
            )
        draft.status = "rejected"
        draft.review_note = reason[:2000]
        fingerprint = await db.scalar(
            select(IngestionFingerprint).where(
                IngestionFingerprint.company_domain == draft.company_domain,
                IngestionFingerprint.source_hash == draft.source_hash,
            )
        )
        if fingerprint:
            await db.delete(fingerprint)
        await db.commit()

    try:
        metadata = await asyncio.to_thread(head_source, draft.storage_key)
        object_length = int(metadata.get("content_length") or 0)
        if (
            not object_length
            or object_length > settings.MAX_SOURCE_UPLOAD_BYTES
            or (expected_length and object_length != expected_length)
            or (
                request
                and request.content_length
                and object_length != request.content_length
            )
        ):
            raise HTTPException(
                status_code=422,
                detail="The uploaded object size does not match the upload intent",
            )
        data = await asyncio.to_thread(load_source, draft.storage_key)
        if (
            len(data) != object_length
            or hashlib.sha256(data).hexdigest() != draft.source_hash
        ):
            raise HTTPException(
                status_code=422,
                detail="The uploaded object failed integrity verification",
            )
        extracted_pages = await asyncio.to_thread(
            extract_source_pages, draft.original_filename or draft.title, data
        )
        extracted = await asyncio.to_thread(
            extract_source_markdown,
            draft.original_filename or draft.title,
            data,
            extracted_pages,
        )
    except HTTPException as exc:
        await discard_failed_upload(str(exc.detail))
        raise
    except (FileNotFoundError, SourceExtractionError) as exc:
        await discard_failed_upload(str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Presigned source completion failed", draft_id=str(draft.id), error=str(exc)
        )
        await discard_failed_upload(
            "The uploaded object could not be verified or processed"
        )
        raise HTTPException(
            status_code=422,
            detail="The uploaded object could not be verified or processed",
        ) from exc

    matches = await find_similar_documents(db, current_user, extracted)
    similarity_level = classify_similarity(matches)
    if similarity_level == "exact":
        similarity_level = "very_high"
    restructuring_enabled = (
        settings.RESTRUCTURE_ENABLED
        and await FeatureFlagRepository(db).is_enabled(
            "ai.document_restructure", current_user
        )
    )
    draft.summary = extracted
    draft.page_texts = extracted_pages
    draft.restructure_status = "queued" if restructuring_enabled else "disabled"
    draft.restructure_model = (
        settings.RESTRUCTURE_MODEL if restructuring_enabled else "none"
    )
    draft.similarity_level = similarity_level
    draft.similarity_matches = matches
    draft.requires_update_confirmation = similarity_level == "very_high"
    draft.related_article_ids = (
        [item["article_id"] for item in matches]
        if similarity_level == "partial"
        else None
    )
    draft.content_metadata = {
        **(draft.content_metadata or {}),
        "upload_mode": "presigned_complete",
    }
    fingerprint = await db.scalar(
        select(IngestionFingerprint).where(
            IngestionFingerprint.company_domain == draft.company_domain,
            IngestionFingerprint.source_hash == draft.source_hash,
        )
    )
    if fingerprint:
        fingerprint.status = "pending"
        fingerprint.draft_id = draft.id
    updated = await GovernanceRepository(db).update_draft(draft)
    updated = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).submit_draft(current_user, updated.id)
    await AuditRepository(db).record(
        current_user.id, "source_upload_complete", "draft", str(updated.id)
    )
    if restructuring_enabled:
        from src.workers.tasks import dispatch_restructure_pending_draft

        if not dispatch_restructure_pending_draft(
            str(updated.id), current_user.company_domain, str(current_user.id)
        ):
            updated.restructure_status = "fallback_formatting"
            updated.restructure_model = "lossless-markdown"
            updated.restructure_error = (
                "AI formatting could not be queued; retry from Pending Drafts."
            )
            await db.commit()
    return {
        "id": str(updated.id),
        "title": updated.title,
        "source_ref": updated.source_ref,
        "source_hash": updated.source_hash,
        "status": updated.status,
        "company_domain": updated.company_domain,
        "dept": updated.dept,
        "assigned_approver_id": None,
        "extracted_characters": len(extracted),
        "page_count": len(extracted_pages),
        "restructure_status": updated.restructure_status,
        "similarity_level": similarity_level,
        "similarity_matches": matches,
        "requires_update_confirmation": updated.requires_update_confirmation,
        "tags": updated.tags or [],
    }


@router.get("/{id}/source")
async def view_article_source(
    id: uuid.UUID,
    page: int | None = Query(default=None, ge=1),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Stream the authorized original source for inline PDF/page review."""
    article = await ArticleService(
        ArticleRepository(db), UserRepository(db)
    ).get_article(current_user, id)
    result = await db.execute(
        select(DocumentSource)
        .where(DocumentSource.article_id == article.id)
        .order_by(DocumentSource.ingested_at.desc())
    )
    source = result.scalars().first()
    if not source or not source.storage_key:
        raise HTTPException(
            status_code=404, detail="Original source is not available for this article"
        )
    download_name = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", source.original_filename or article.title)
        or "source.bin"
    )
    media_type = safe_source_media_type(download_name)
    disposition = (
        "inline" if source_should_display_inline(download_name) else "attachment"
    )
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "sandbox; default-src 'none'; base-uri 'none'; form-action 'none'",
    }
    try:
        data = await asyncio.to_thread(load_source, source.storage_key)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="Original source file is missing"
        ) from exc
    if page is not None and source.page_texts:
        selected_page = next(
            (
                item
                for item in source.page_texts
                if int(item.get("page_number") or 0) == page
            ),
            None,
        )
        if selected_page is None:
            raise HTTPException(
                status_code=404, detail="The cited source page is not available"
            )
        return Response(
            content=str(selected_page.get("text") or "").encode("utf-8"),
            media_type="text/plain",
            headers={
                **security_headers,
                "X-Source-Page": str(page),
                "Content-Disposition": f'inline; filename="{download_name}.page-{page}.txt"',
            },
        )
    return Response(
        content=data,
        media_type=media_type,
        headers={
            **security_headers,
            "Content-Disposition": f'{disposition}; filename="{download_name}"',
        },
    )


@router.get("/{id}/source-url")
async def get_article_source_url(
    id: uuid.UUID,
    expires_in: int = Query(300, ge=1, le=900),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return a short-lived R2 URL only after Article authorization."""
    article = await ArticleService(
        ArticleRepository(db), UserRepository(db)
    ).get_article(current_user, id)
    result = await db.execute(
        select(DocumentSource)
        .where(DocumentSource.article_id == article.id)
        .order_by(DocumentSource.ingested_at.desc())
    )
    source = result.scalars().first()
    if not source or not source.storage_key:
        raise HTTPException(
            status_code=404, detail="Original source is not available for this article"
        )
    try:
        url = await asyncio.to_thread(
            create_presigned_source_url, source.storage_key, expires_in=expires_in
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503, detail="Private R2 source storage is unavailable"
        ) from exc
    return {"url": url, "expires_in": expires_in, "article_id": str(article.id)}


@router.post(
    "/", response_model=DraftSubmissionResponse, status_code=status.HTTP_201_CREATED
)
async def create_article(
    article_in: ArticleCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Submit authored content for independent approval.

    A manually written article is just as sensitive as an uploaded source.
    Persist it only as a ``PendingDraft`` so the author cannot self-publish by
    holding a publisher role; an eligible, separately assigned reviewer must
    make the final publication decision.
    """
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article_service = ArticleService(article_repo, user_repo, AuditRepository(db))
    if article_in.department_ids is not None:
        selected_departments = await resolve_active_departments(
            db, current_user.company_domain, article_in.department_ids
        )
        if article_in.dept:
            primary_department = await resolve_active_department(
                db, current_user.company_domain, article_in.dept
            )
            if primary_department.id not in {
                department.id for department in selected_departments
            }:
                raise HTTPException(
                    status_code=422,
                    detail="The primary department must be one of the selected departments",
                )
        else:
            primary_department = selected_departments[0]
    else:
        primary_department = await resolve_active_department(
            db, current_user.company_domain, article_in.dept, required=True
        )
        selected_departments = [primary_department]
    if article_in.visibility == "users" and not article_in.explicit_user_ids:
        raise HTTPException(
            status_code=422,
            detail="Explicit-user visibility requires at least one user",
        )
    if (
        article_in.explicit_user_ids or article_in.denied_user_ids
    ) and not AuthorizationService.has_permission(
        current_user, "permission.manage", requested_scope="company"
    ):
        raise HTTPException(
            status_code=403,
            detail="Only permission managers can set explicit Article users",
        )
    if set(article_in.explicit_user_ids) & set(article_in.denied_user_ids):
        raise HTTPException(
            status_code=422,
            detail="A user cannot be both explicitly allowed and denied",
        )
    selected_explicit_users = await UserRepository(db).get_by_ids(
        article_in.explicit_user_ids + article_in.denied_user_ids,
        current_user.company_domain,
    )
    if len(selected_explicit_users) != len(
        set(article_in.explicit_user_ids + article_in.denied_user_ids)
    ):
        raise HTTPException(
            status_code=422,
            detail="Every explicit Article user must belong to your company",
        )
    for department in selected_departments:
        article_service.ensure_can_create(current_user, department.name)
    restructuring_enabled = (
        settings.RESTRUCTURE_ENABLED
        and await FeatureFlagRepository(db).is_enabled(
            "ai.document_restructure", current_user
        )
    )
    restructure_result = await restructure_document(
        article_in.title, article_in.body_md, enabled=restructuring_enabled
    )
    draft_body = restructure_result.body_md or article_in.body_md
    matches = await find_similar_documents(db, current_user, draft_body)
    similarity_level = classify_similarity(matches)
    if similarity_level == "exact":
        similarity_level = "very_high"
    submitted_tags = list(
        dict.fromkeys(tag.strip() for tag in article_in.tags if tag.strip())
    )[:20]
    normalized_body = re.sub(r"\s+", " ", article_in.body_md).strip().lower()
    source_hash = hashlib.sha256(
        f"{current_user.company_domain}\0{normalized_body}".encode("utf-8")
    ).hexdigest()
    await lock_company_access_groups(
        db, f"upload:{current_user.company_domain}:{source_hash}"
    )
    existing_fingerprint = await db.scalar(
        select(IngestionFingerprint).where(
            IngestionFingerprint.company_domain == current_user.company_domain,
            IngestionFingerprint.source_hash == source_hash,
            IngestionFingerprint.status.in_({"pending", "approved"}),
        )
    )
    if existing_fingerprint:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_document",
                "message": "This article content already exists.",
            },
        )
    draft = PendingDraft(
        title=article_in.title,
        company_domain=current_user.company_domain,
        dept=primary_department.name,
        source_ref=f"manual://{uuid.uuid4()}",
        source_hash=source_hash,
        summary=article_in.body_md,
        restructured_body_md=draft_body,
        restructure_candidate_md=restructure_result.candidate_body_md,
        restructure_decision=(
            "pending_review"
            if restructure_result.candidate_body_md
            else (
                "ai_ready" if restructure_result.status == "llm" else "lossless_ready"
            )
        ),
        restructure_status=restructure_result.status,
        restructure_model=restructure_result.model,
        restructure_error=restructure_result.error,
        created_by=current_user.id,
        status="draft",
        similarity_level=similarity_level,
        similarity_matches=matches,
        requires_update_confirmation=similarity_level == "very_high",
        related_article_ids=(
            [item["article_id"] for item in matches]
            if similarity_level == "partial"
            else None
        ),
        tags=submitted_tags,
        content_metadata={
            "external_id": article_in.external_id,
            "domain": "General",
            "type": "REFERENCE",
            "sensitivity": "public",
            "visibility": article_in.visibility,
            "explicit_user_ids": [
                str(user_id) for user_id in article_in.explicit_user_ids
            ],
            "denied_user_ids": [str(user_id) for user_id in article_in.denied_user_ids],
            "language": article_in.language,
            "department_ids": [
                str(department.id) for department in selected_departments
            ],
            "next_review": (
                article_in.next_review.isoformat() if article_in.next_review else None
            ),
            "submission_kind": "manual",
        },
    )
    _add_split_candidates(db, draft, draft_body)
    db.add(
        IngestionFingerprint(
            company_domain=current_user.company_domain,
            source_hash=source_hash,
            status="pending",
            draft_id=draft.id,
            created_by=current_user.id,
        )
    )
    created = await GovernanceRepository(db).create_draft(draft)
    created = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).submit_draft(current_user, created.id)
    await AuditRepository(db).record(
        current_user.id, "manual_draft_submit", "draft", str(created.id)
    )
    return DraftSubmissionResponse(
        id=created.id,
        title=created.title,
        status=created.status,
        workflow="pending_approval",
        message="Draft submitted. Assign an independent approver before it can be published.",
    )


@router.get("/", response_model=list[ArticleResponse])
async def list_articles(
    dept: str | None = Query(None),
    topic: str | None = Query(None, max_length=80),
    status: str | None = Query(None),
    q: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    article_repo = ArticleRepository(db)
    return await article_repo.list_articles(
        user=current_user, dept=dept, topic=topic, status=status, search_query=q,
        limit=limit, offset=offset,
    )


@router.post("/{id}/follow", status_code=status.HTTP_201_CREATED)
async def follow_article(id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    article = await ArticleRepository(db).get_by_id(id, user=current_user)
    if not article or article.status == "deleted":
        raise HTTPException(status_code=404, detail="Article not found")
    existing = await db.scalar(select(ArticleFollower).where(ArticleFollower.user_id == current_user.id, ArticleFollower.article_id == id))
    if not existing:
        # Make the idempotent operation race-safe: two tabs may both observe
        # no row before either commits.
        await db.execute(pg_insert(ArticleFollower).values(user_id=current_user.id, article_id=id).on_conflict_do_nothing())
        await db.commit()
    return {"article_id": str(id), "following": True}


@router.delete("/{id}/follow", status_code=status.HTTP_204_NO_CONTENT)
async def unfollow_article(id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> None:
    article = await ArticleRepository(db).get_by_id(id, user=current_user)
    if not article or article.status == "deleted":
        raise HTTPException(status_code=404, detail="Article not found")
    await db.execute(delete(ArticleFollower).where(ArticleFollower.user_id == current_user.id, ArticleFollower.article_id == id))
    await db.commit()


@router.get("/{id}/follow")
async def article_follow_status(id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    article = await ArticleRepository(db).get_by_id(id, user=current_user)
    if not article or article.status == "deleted":
        raise HTTPException(status_code=404, detail="Article not found")
    followed = await db.scalar(select(ArticleFollower).where(ArticleFollower.user_id == current_user.id, ArticleFollower.article_id == id))
    return {"article_id": str(id), "following": bool(followed)}


@router.get("/{id}/related", response_model=list[ArticleResponse])
async def get_related_articles(
    id: uuid.UUID,
    limit: int = Query(6, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article = await ArticleService(article_repo, user_repo).get_article(
        current_user, id
    )
    return await article_repo.list_related(current_user, article, limit)


@router.get("/{id}", response_model=ArticleResponse)
async def get_article(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article_service = ArticleService(article_repo, user_repo, AuditRepository(db))
    article = await article_service.get_article(current_user, id)
    AuthorizationService.restrict_article_metadata(current_user, article)
    return article


@router.post("/{id}/edit-requests", status_code=status.HTTP_201_CREATED)
async def create_article_edit_request(
    id: uuid.UUID,
    request_in: ArticleEditRequestCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Record a reader correction and notify users who can edit this article."""
    return await create_edit_request(db, current_user, id, request_in.request_text, source="manual")


@router.put("/{id}", response_model=DraftSubmissionResponse)
async def update_article(
    id: uuid.UUID,
    article_in: ArticleUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article_service = ArticleService(article_repo, user_repo, AuditRepository(db))
    current = await article_service.get_article(current_user, id)
    if not PermissionService.can_edit_article(current_user, current):
        raise HTTPException(
            status_code=403, detail="Not authorized to edit this article"
        )
    if article_in.status is not None and article_in.status != current.status:
        raise HTTPException(
            status_code=422,
            detail="Article status changes must be approved through the review workflow",
        )

    title = article_in.title if article_in.title is not None else current.title
    body_md = article_in.body_md if article_in.body_md is not None else current.body_md
    dept = article_in.dept if article_in.dept is not None else current.dept
    language = (
        article_in.language if article_in.language is not None else current.language
    )
    destination_department = await resolve_active_department(
        db, current.company_domain, dept
    )
    dept = destination_department.name
    if article_in.department_ids is not None:
        selected_departments = await resolve_active_departments(
            db, current.company_domain, article_in.department_ids
        )
        if destination_department.id not in {
            department.id for department in selected_departments
        }:
            raise HTTPException(
                status_code=422,
                detail="The primary department must be one of the selected departments",
            )
    elif current.departments:
        selected_departments = list(current.departments)
    else:
        selected_departments = [destination_department]
    if dept != current.dept or {
        department.id for department in selected_departments
    } != {department.id for department in current.departments}:
        destination = Article(
            company_domain=current.company_domain,
            dept=dept,
            owner_id=current.owner_id,
            departments=selected_departments,
        )
        if not any(
            AuthorizationService.has_permission(
                current_user, "article.edit", destination, scope
            )
            for scope in ("department", "company", "global")
        ):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to move an article to this department",
            )

    visibility = (
        article_in.visibility
        if article_in.visibility is not None
        else (current.visibility or "department")
    )
    if (
        article_in.explicit_user_ids is not None
        or article_in.denied_user_ids is not None
        or article_in.visibility is not None
    ):
        if not AuthorizationService.has_permission(
            current_user, "permission.manage", requested_scope="company"
        ):
            raise HTTPException(
                status_code=403,
                detail="Only permission managers can change Article visibility",
            )
    explicit_user_ids = (
        article_in.explicit_user_ids
        if article_in.explicit_user_ids is not None
        else list(current.explicit_user_ids)
    )
    denied_user_ids = (
        article_in.denied_user_ids
        if article_in.denied_user_ids is not None
        else list(current.explicit_denied_user_ids)
    )
    if set(explicit_user_ids) & set(denied_user_ids):
        raise HTTPException(
            status_code=422,
            detail="A user cannot be both explicitly allowed and denied",
        )
    if visibility == "users" and not explicit_user_ids:
        raise HTTPException(
            status_code=422,
            detail="Explicit-user visibility requires at least one user",
        )
    selected_explicit_users = await UserRepository(db).get_by_ids(
        explicit_user_ids + denied_user_ids, current.company_domain
    )
    if len(selected_explicit_users) != len(set(explicit_user_ids + denied_user_ids)):
        raise HTTPException(
            status_code=422,
            detail="Every explicit Article user must belong to the Article company",
        )

    # Legacy domain/type/sensitivity and ACL values are preserved from the
    # synchronized article. They are no longer editable through this API.
    groups = list(current.access_groups)
    tags = (
        article_in.tags
        if article_in.tags is not None
        else [tag.tag for tag in current.tags]
    )
    submitted_tags = list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))[
        :20
    ]
    next_review = (
        article_in.next_review
        if "next_review" in article_in.model_fields_set
        else current.next_review
    )
    restructuring_enabled = (
        settings.RESTRUCTURE_ENABLED
        and await FeatureFlagRepository(db).is_enabled(
            "ai.document_restructure", current_user
        )
    )
    restructure_result = await restructure_document(
        title, body_md, enabled=restructuring_enabled
    )
    draft_body = restructure_result.body_md or body_md
    matches = await find_similar_documents(db, current_user, draft_body)
    current_match = {
        "article_id": str(current.id),
        "title": current.title,
        "score": 1.0,
        "lifecycle_status": current.lifecycle_status,
    }
    matches = [
        current_match,
        *[match for match in matches if match.get("article_id") != str(current.id)],
    ]
    normalized_body = re.sub(r"\s+", " ", body_md).strip().lower()
    source_hash = hashlib.sha256(
        f"{current.company_domain}\0{normalized_body}".encode("utf-8")
    ).hexdigest()
    await lock_company_access_groups(
        db, f"upload:{current.company_domain}:{source_hash}"
    )
    duplicate_fingerprint = await db.scalar(
        select(IngestionFingerprint).where(
            IngestionFingerprint.company_domain == current.company_domain,
            IngestionFingerprint.source_hash == source_hash,
            IngestionFingerprint.status.in_({"pending", "approved"}),
            IngestionFingerprint.article_id != current.id,
        )
    )
    if duplicate_fingerprint:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_document",
                "message": "This article content already exists.",
            },
        )
    draft = PendingDraft(
        title=title,
        company_domain=current.company_domain,
        dept=dept,
        source_ref=f"manual-update://{current.id}/{uuid.uuid4()}",
        source_hash=source_hash,
        summary=body_md,
        restructured_body_md=draft_body,
        restructure_candidate_md=restructure_result.candidate_body_md,
        restructure_decision=(
            "pending_review"
            if restructure_result.candidate_body_md
            else (
                "ai_ready" if restructure_result.status == "llm" else "lossless_ready"
            )
        ),
        restructure_status=restructure_result.status,
        restructure_model=restructure_result.model,
        restructure_error=restructure_result.error,
        created_by=current_user.id,
        status="draft",
        similarity_level="very_high",
        similarity_matches=matches,
        requires_update_confirmation=True,
        update_target_article_id=current.id,
        tags=submitted_tags,
        content_metadata={
            "external_id": current.external_id,
            "domain": current.domain,
            "type": current.type,
            "sensitivity": current.sensitivity,
            "visibility": visibility,
            "explicit_user_ids": [str(user_id) for user_id in explicit_user_ids],
            "denied_user_ids": [str(user_id) for user_id in denied_user_ids],
            "language": language,
            "department_ids": [
                str(department.id) for department in selected_departments
            ],
            "access_group_ids": [str(group.id) for group in groups],
            "next_review": next_review.isoformat() if next_review else None,
            "submission_kind": "manual_update",
            "suggested_update_article_id": str(current.id),
        },
    )
    _add_split_candidates(db, draft, draft_body)
    db.add(
        IngestionFingerprint(
            company_domain=current.company_domain,
            source_hash=source_hash,
            status="pending",
            draft_id=draft.id,
            created_by=current_user.id,
        )
    )
    created = await GovernanceRepository(db).create_draft(draft)
    created = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).submit_draft(current_user, created.id)
    await AuditRepository(db).record(
        current_user.id, "article_change_submit", "draft", str(created.id)
    )
    return DraftSubmissionResponse(
        id=created.id,
        title=created.title,
        status=created.status,
        workflow="pending_approval",
        message="Changes submitted. An independent approver must review them before the article is updated.",
    )


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_article(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article_service = ArticleService(article_repo, user_repo, AuditRepository(db))
    await article_service.soft_delete_article(current_user, id)


@router.get("/{id}/versions", response_model=list[VersionResponse])
async def list_versions(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article_service = ArticleService(article_repo, user_repo)
    return await article_service.get_history(current_user, id)


@router.get("/{id}/versions/{version_num}", response_model=VersionResponse)
async def get_version(
    id: uuid.UUID,
    version_num: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article_service = ArticleService(article_repo, user_repo)
    return await article_service.get_version(current_user, id, version_num)


@router.post(
    "/{id}/versions/{version_num}/restore", response_model=DraftSubmissionResponse
)
async def restore_version(
    id: uuid.UUID,
    version_num: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    article_service = ArticleService(article_repo, user_repo, AuditRepository(db))
    current = await article_service.get_article(current_user, id)
    if not PermissionService.can_edit_article(current_user, current):
        raise HTTPException(
            status_code=403, detail="Not authorized to restore this article version"
        )
    historical = await article_service.get_version(current_user, id, version_num)
    if historical.version == current.version:
        raise HTTPException(status_code=409, detail="This version is already active")
    snapshot = historical.snapshot or {}
    title = str(snapshot.get("title") or current.title)
    body_md = str(snapshot.get("body_md") or current.body_md)
    tags = [str(tag).strip() for tag in snapshot.get("tags", []) if str(tag).strip()]
    draft = PendingDraft(
        title=title,
        company_domain=current.company_domain,
        dept=str(snapshot.get("dept") or current.dept),
        source_ref=f"restore://{current.id}/version/{version_num}",
        source_hash=hashlib.sha256(
            f"restore\0{current.id}\0{version_num}\0{body_md}".encode("utf-8")
        ).hexdigest(),
        summary=body_md,
        restructured_body_md=body_md,
        restructure_status="historical",
        created_by=current_user.id,
        status="draft",
        similarity_level="very_high",
        similarity_matches=[
            {
                "article_id": str(current.id),
                "title": current.title,
                "score": 1.0,
                "lifecycle_status": current.lifecycle_status,
            }
        ],
        requires_update_confirmation=True,
        update_target_article_id=current.id,
        tags=list(dict.fromkeys(tags))[:20],
        content_metadata={
            "external_id": current.external_id,
            "domain": str(snapshot.get("domain") or current.domain),
            "type": str(snapshot.get("type") or current.type),
            "sensitivity": str(snapshot.get("sensitivity") or current.sensitivity),
            "language": str(snapshot.get("language") or current.language),
            "access_group_ids": [str(group.id) for group in current.access_groups],
            "next_review": (
                current.next_review.isoformat() if current.next_review else None
            ),
            "submission_kind": "manual_update",
            "suggested_update_article_id": str(current.id),
            "restored_from_version": version_num,
        },
    )
    _add_split_candidates(db, draft, body_md)
    created = await GovernanceRepository(db).create_draft(draft)
    created = await GovernanceService(
        GovernanceRepository(db), ArticleRepository(db)
    ).submit_draft(current_user, created.id)
    await AuditRepository(db).record(
        current_user.id, "article_restore_submit", "draft", str(created.id)
    )
    return DraftSubmissionResponse(
        id=created.id,
        title=created.title,
        status=created.status,
        workflow="pending_approval",
        message=f"Version {version_num} submitted for independent approval.",
    )
