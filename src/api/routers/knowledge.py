"""Knowledge surfaces backed by the live article and governance models."""
import json
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db, get_current_user, require_permission
from src.models import User
from src.models.article import Article, ArticleTag, DocumentSource
from src.models.governance import Gap, PendingDraft, ConflictRecord
from src.models.ai import AiUsageLog
from src.models.chunk import ArticleChunk
from src.models.interaction import ArticleFollower
from src.models.user import Department
from src.repositories.article import ArticleRepository
from src.repositories.governance import GovernanceRepository
from src.domain.rbac import AuthorizationService
from src.core.config import settings

router = APIRouter()

class ContentRequest(BaseModel):
    query: str = Field(min_length=2, max_length=255)
    dept: str | None = Field(default=None, max_length=100)


class RolePreviewRequest(BaseModel):
    role: str = Field(min_length=2, max_length=50)
    dept: str | None = Field(default=None, max_length=100)


class ConflictResolutionRequest(BaseModel):
    note: str = Field(min_length=1, max_length=2000)


def _browse_filters(
    repo: ArticleRepository,
    current_user: User,
    dept: str | None = None,
    topic: str | None = None,
    query: str | None = None,
) -> list[Any]:
    filters = [*repo._authorized_article_filters(current_user), Article.status == "published"]
    if dept:
        filters.append((Article.dept == dept) | Article.departments.any(name=dept))
    if topic:
        filters.append(
            ~Article.tags.any() if topic == "General knowledge"
            else Article.tags.any(ArticleTag.tag == topic)
        )
    if query:
        filters.append(Article.title.ilike(f"%{query.strip()}%"))
    return filters


def _article_card(article: Article) -> dict[str, Any]:
    return {
        "id": str(article.id), "title": article.title, "dept": article.dept,
        "company_domain": article.company_domain,
        "departments": [{"id": str(department.id), "name": department.name} for department in getattr(article, "departments", [])],
        "external_id": article.external_id, "status": article.status,
        "language": article.language,
        "version": article.version, "owner": article.owner.name if article.owner else None,
        "owner_id": str(article.owner_id) if article.owner_id else None,
        "tags": [tag.tag for tag in article.tags], "related_article_ids": article.related_article_ids or [],
        "next_review": article.next_review, "last_reviewed": article.last_reviewed,
        "needs_update": article.needs_update or bool(article.next_review and article.next_review < datetime.utcnow()),
        "source_changed": bool(getattr(article, "source_changed", False)),
        "source_count": len(article.sources),
    }


@router.get("/home")
async def home_summary(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    article_filters = ArticleRepository._authorized_article_filters(current_user)
    published_filters = [*article_filters, Article.status == "published"]
    home_has_full_company_access = (
        AuthorizationService.has_permission(current_user, "governance.read", requested_scope="global")
        or AuthorizationService.has_full_company_article_access(current_user)
    )
    home_departments = AuthorizationService.member_department_names(current_user)
    gaps_stmt = select(func.count()).select_from(Gap).where(
        Gap.status.in_(["open", "assigned"]),
        Gap.company_domain == current_user.company_domain,
    )
    if not home_has_full_company_access:
        gaps_stmt = gaps_stmt.where(Gap.dept.in_(home_departments))
    pending = await GovernanceRepository(db).count_pending_for_user(current_user)
    overdue_pending = int(await db.scalar(select(func.count(PendingDraft.id)).where(
        PendingDraft.company_domain == current_user.company_domain,
        PendingDraft.status == "pending",
        PendingDraft.assigned_at.is_not(None),
        PendingDraft.assigned_at < datetime.utcnow() - timedelta(days=settings.REVIEW_SLA_DAYS),
    )) or 0)
    gaps = await db.scalar(gaps_stmt) or 0
    total_articles = int(await db.scalar(select(func.count(Article.id)).where(and_(*published_filters))) or 0)
    owned_articles = int(await db.scalar(select(func.count(Article.id)).where(and_(*published_filters, Article.owner_id.is_not(None)))) or 0)
    needs_review = int(await db.scalar(select(func.count(Article.id)).where(and_(*published_filters, (Article.needs_update.is_(True) | (Article.next_review.is_not(None) & (Article.next_review < datetime.utcnow())))))) or 0)
    department_count = int(await db.scalar(select(func.count(func.distinct(Article.dept))).where(and_(*published_filters))) or 0)
    articles = list(await ArticleRepository(db).list_articles(current_user, status="published", limit=6))
    return {
        "total_articles": total_articles, "departments": department_count,
        "with_owner_percent": round(owned_articles / total_articles * 100) if total_articles else 0,
        "needs_review": needs_review,
        "pending_drafts": pending, "overdue_drafts": overdue_pending, "open_gaps": gaps,
        "recent": [_article_card(article) for article in articles[:6]],
    }


@router.get("/browse")
async def browse_knowledge(
    dept: str | None = Query(None),
    topic: str | None = Query(None, max_length=80),
    q: str | None = Query(None, max_length=255),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    repo = ArticleRepository(db)
    articles = list(await repo.list_articles(
        current_user, dept=dept, topic=topic, search_query=q, status="published", limit=limit, offset=offset,
    ))
    filters = _browse_filters(repo, current_user, dept=dept, topic=topic, query=q)
    total = int(await db.scalar(select(func.count(Article.id)).where(and_(*filters))) or 0)
    return {"articles": [_article_card(article) for article in articles], "total": total, "limit": limit, "offset": offset}


@router.get("/catalog")
async def knowledge_catalog(
    dept: str | None = Query(None),
    q: str | None = Query(None, max_length=255),
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return database-backed hierarchy counts without loading the document corpus."""
    repo = ArticleRepository(db)
    filters = _browse_filters(repo, current_user, dept=dept, query=q)
    total = int(await db.scalar(select(func.count(Article.id)).where(and_(*filters))) or 0)

    department_rows = await db.execute(
        select(Article.dept, func.count(Article.id))
        .where(and_(*filters))
        .group_by(Article.dept)
        .order_by(func.count(Article.id).desc(), Article.dept.asc())
    )
    department_items = [{"name": name or "Unassigned", "count": int(count)} for name, count in department_rows.all()]

    topic_filters = list(filters)
    topic_rows = await db.execute(
        select(ArticleTag.tag, func.count(func.distinct(Article.id)))
        .join(Article, Article.id == ArticleTag.article_id)
        .where(and_(*topic_filters))
        .group_by(ArticleTag.tag)
        .order_by(func.count(func.distinct(Article.id)).desc(), ArticleTag.tag.asc())
    )
    topic_items = [{"name": name, "count": int(count)} for name, count in topic_rows.all() if name]

    tagged_total = int(await db.scalar(
        select(func.count(func.distinct(Article.id)))
        .join(ArticleTag, ArticleTag.article_id == Article.id)
        .where(and_(*filters))
    ) or 0)
    if total > tagged_total:
        topic_items.append({"name": "General knowledge", "count": total - tagged_total})

    return {"total": total, "departments": department_items, "topics": topic_items}


@router.get("/sources")
async def sources(
    q: str | None = Query(None, max_length=255),
    source_system: str | None = Query(None, max_length=100),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    filters = [*ArticleRepository._authorized_article_filters(current_user)]
    if source_system:
        filters.append(DocumentSource.source_system == source_system)
    if q:
        search = f"%{q.strip()}%"
        filters.append(
            DocumentSource.original_filename.ilike(search)
            | DocumentSource.source_ref.ilike(search)
            | Article.title.ilike(search)
        )
    stmt = (select(DocumentSource, Article).join(Article, DocumentSource.article_id == Article.id)
            .where(and_(*filters)).order_by(DocumentSource.ingested_at.desc()).offset(offset).limit(limit))
    rows = (await db.execute(stmt)).all()
    total = int(await db.scalar(select(func.count(DocumentSource.id)).join(Article, DocumentSource.article_id == Article.id).where(and_(*filters))) or 0)
    result = [{"id": str(source.id), "article_id": str(article.id), "article_title": article.title,
               "source_system": source.source_system, "source_ref": source.source_ref,
               "filename": source.original_filename, "mime_type": source.mime_type,
               "ingested_at": source.ingested_at, "has_file": bool(source.storage_key),
               "source_changed": bool(article.source_changed)} for source, article in rows]
    return {"sources": result, "total": total, "limit": limit, "offset": offset}


@router.post("/content-requests", status_code=201)
async def content_request(request: ContentRequest, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    gap = await GovernanceRepository(db).log_gap(request.query.strip(), current_user.company_domain, request.dept)
    return {"id": str(gap.id), "query": gap.query, "dept": gap.dept, "count": gap.count, "status": gap.status}


@router.post("/role-preview")
async def role_preview(request: RolePreviewRequest, current_user: User = Depends(require_permission("permission.manage")), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Preview a real read surface using an ephemeral user policy."""
    if request.role not in {"Admin", "CEO", "Reviewer", "Staff"}:
        raise HTTPException(status_code=422, detail="Unsupported role preview")
    preview_user = User(id=current_user.id, email=current_user.email, name=current_user.name,
                        password_hash="preview", company_domain=current_user.company_domain,
                        dept=request.dept, role=request.role, active=True)
    if request.dept:
        preview_department = await db.scalar(select(Department).where(
            Department.company_domain == current_user.company_domain,
            Department.name == request.dept,
            Department.active.is_(True),
        ))
        if preview_department:
            preview_user.departments = [preview_department]
    articles = await ArticleRepository(db).list_articles(preview_user, status="published", limit=20)
    from src.repositories.audit import AuditRepository
    await AuditRepository(db).record(current_user.id, "role_preview", "user", str(current_user.id),
                                     detail={"preview_role": request.role, "preview_dept": request.dept, "article_count": len(articles)})
    return {"role": request.role, "dept": request.dept, "read_only": True,
            "articles": [_article_card(item) for item in articles], "article_count": len(articles)}


@router.get("/coverage")
async def knowledge_coverage(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Combine open gaps, AI demand and article citation usage into one map."""
    gap_stmt = select(Gap.dept, func.sum(Gap.count)).where(Gap.company_domain == current_user.company_domain, Gap.status.in_(["open", "assigned"]))
    if not AuthorizationService.has_full_company_article_access(current_user):
        gap_stmt = gap_stmt.where(Gap.dept.in_(AuthorizationService.member_department_names(current_user)))
    gaps = [{"dept": dept or "Unassigned", "gap_count": int(count or 0)} for dept, count in (await db.execute(gap_stmt.group_by(Gap.dept))).all()]
    # Do not use the repository's bounded list API here: the dashboard's
    # headline counts must describe the complete authorized corpus, not just
    # the first 200 articles.
    visible_filters = [Article.status == "published", *ArticleRepository._authorized_article_filters(current_user)]
    visible_rows = (await db.execute(
        select(Article.id, Article.title, Article.dept).where(and_(*visible_filters))
    )).all()
    article_ids = {str(article_id) for article_id, _, _ in visible_rows}
    chunk_rows = (await db.execute(select(ArticleChunk.article_id, ArticleChunk.id).where(ArticleChunk.article_id.in_([article_id for article_id, _, _ in visible_rows])))).all() if visible_rows else []
    chunk_to_article = {str(chunk_id): str(article_id) for article_id, chunk_id in chunk_rows}
    usage_rows = (await db.execute(select(AiUsageLog.retrieved_chunk_ids).where(AiUsageLog.user_id == current_user.id))).scalars().all()
    cited: set[str] = set()
    for payload in usage_rows:
        try:
            chunk_ids = json.loads(payload or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(chunk_ids, list):
            cited.update(chunk_to_article[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_to_article)
    never_cited = [{"id": str(article_id), "title": title, "dept": dept} for article_id, title, dept in visible_rows if str(article_id) not in cited]
    return {"gaps_by_department": gaps, "never_cited": never_cited, "cited_article_count": len(cited & article_ids), "visible_article_count": len(article_ids)}


@router.get("/owner-dashboard")
async def owner_dashboard(current_user: User = Depends(require_permission("governance.read")), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return await knowledge_coverage(current_user, db)


@router.get("/conflicts")
async def list_conflicts(current_user: User = Depends(require_permission("governance.read")), db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (await db.execute(select(ConflictRecord).where(ConflictRecord.company_domain == current_user.company_domain, ConflictRecord.status == "open").order_by(ConflictRecord.created_at.desc()).limit(200))).scalars().all()
    return [{"id": str(item.id), "fact": item.fact, "article_ids": item.article_ids, "evidence": item.evidence or [], "status": item.status, "created_at": item.created_at} for item in rows]


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: str, request: ConflictResolutionRequest, current_user: User = Depends(require_permission("governance.read")), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    import uuid
    item = await db.scalar(select(ConflictRecord).where(ConflictRecord.id == uuid.UUID(conflict_id), ConflictRecord.company_domain == current_user.company_domain, ConflictRecord.status == "open"))
    if not item:
        raise HTTPException(status_code=404, detail="Conflict not found")
    item.status = "resolved"
    item.resolved_by = current_user.id
    item.resolved_at = datetime.utcnow()
    item.resolution_note = request.note.strip()
    await db.commit()
    return {"id": str(item.id), "status": item.status, "resolution_note": item.resolution_note}
