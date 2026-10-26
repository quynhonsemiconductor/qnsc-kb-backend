"""Counting and listing the knowledge base itself, for questions about the collection.

"How many documents do we have?" is a question about the library, not about what any
document says, so it is answered by counting rows. Retrieval cannot answer it: the top-k
passages nearest to the question are a sample of the text, never a census of the documents.

Everything here reads only what `ArticleRepository._authorized_article_filters` lets the
user read, in SQL, so a total never includes a document the asker could not open. The
wording is produced by code, not by a model: a count that a language model has re-typed is
a count that can be wrong, and this path is meant to cost nothing and be exact.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.models.article import Article, DocumentSource
from src.models.evidence import DocumentEvidence
from src.models.user import User
from src.rag.reranker import fold_diacritics
from src.repositories.article import ArticleRepository

#: A catalog counts the documents that are live in the knowledge base. Drafts and archived
#: rows are visible to some users but are not "the documents we have".
PUBLISHED_STATUS = "published"

#: The source rows read to tally formats and pages. A bound, not a sample: past it the
#: answer says its format and page figures are partial.
MAX_SOURCE_ROWS = 50_000

#: Stands for "everything not listed" and for a format that could not be told. Wording is
#: chosen at display time, because the builder does not know the answer language.
OTHER_KEY = "__other__"

_MIME_FORMATS = {
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/csv": "csv",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/html": "html",
}
_FORMAT_ALIASES = {"jpeg": "jpg", "tiff": "tif", "htm": "html", "markdown": "md"}
_EXTENSION_RE = re.compile(r"\.([A-Za-z0-9]{1,5})$")


@dataclass
class CatalogFilter:
    """What the question narrowed the catalog to. Empty means the whole collection."""

    doc_types: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    since: datetime | None = None
    until: datetime | None = None
    #: "today" | "week" | "month" | "year" | "days:<n>" | "" -- for the sentence only.
    period: str = ""

    @property
    def is_narrowed(self) -> bool:
        return bool(
            self.doc_types or self.departments or self.since is not None or self.until
        )


@dataclass
class CatalogDocument:
    article_id: str
    title: str
    doc_type: str
    department: str
    updated_at: datetime | None
    pages: int


@dataclass
class CatalogResult:
    total: int = 0
    by_type: list[tuple[str, int]] = field(default_factory=list)
    by_department: list[tuple[str, int]] = field(default_factory=list)
    by_language: list[tuple[str, int]] = field(default_factory=list)
    by_format: list[tuple[str, int]] = field(default_factory=list)
    by_month: list[tuple[str, int]] = field(default_factory=list)
    pages: int = 0
    evidence: dict[str, int] = field(default_factory=dict)
    recent: list[CatalogDocument] = field(default_factory=list)
    #: True when more groups existed than were returned, or the source scan hit its bound.
    partial: bool = False
    generated_at: datetime = field(default_factory=datetime.utcnow)
    filter: CatalogFilter = field(default_factory=CatalogFilter)


# ── pure helpers ──────────────────────────────────────────────────────────────────────────


def file_format(filename: str | None, mime_type: str | None) -> str:
    """The short format name of one source file ("pdf", "docx"), or "" when unknown."""
    match = _EXTENSION_RE.search((filename or "").strip())
    if match:
        extension = match.group(1).lower()
        return _FORMAT_ALIASES.get(extension, extension)
    return _MIME_FORMATS.get((mime_type or "").split(";")[0].strip().lower(), "")


def summarise_sources(
    rows: Iterable[tuple[str | None, str | None, int | None]],
) -> tuple[Counter[str], int]:
    """Tally formats and pages over `(filename, mime_type, pages)` rows."""
    formats: Counter[str] = Counter()
    pages = 0
    for filename, mime_type, page_count in rows:
        formats[file_format(filename, mime_type)] += 1
        pages += max(0, int(page_count or 0))
    return formats, pages


def top_groups(
    counts: Iterable[tuple[str, int]], limit: int
) -> tuple[list[tuple[str, int]], bool]:
    """The `limit` biggest groups, the rest folded into one `OTHER_KEY` row.

    A group that is already `OTHER_KEY` (an unknown format) is ranked like any other and
    merged with the fold when there is one, so there is never more than one such row.
    """
    ordered = sorted(
        ((str(key), int(count)) for key, count in counts if int(count) > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if len(ordered) <= limit:
        return ordered, False
    keep = max(1, limit - 1)
    head = [item for item in ordered[:keep] if item[0] != OTHER_KEY]
    folded = sum(count for _key, count in ordered[keep:])
    folded += sum(count for key, count in ordered[:keep] if key == OTHER_KEY)
    return [*head, (OTHER_KEY, folded)], True


_PERIOD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("today", r"\b(hom nay|today)\b"),
    ("week", r"\b(tuan nay|this week)\b"),
    ("month", r"\b(thang nay|this month)\b"),
    ("year", r"\b(nam nay|this year)\b"),
)
_LAST_DAYS_RE = re.compile(
    r"\b(\d{1,3}) ngay (qua|gan day|truoc)\b|\b(?:last|past) (\d{1,3}) days\b"
)


def parse_period(
    folded_question: str, now: datetime
) -> tuple[datetime | None, datetime | None, str]:
    """`(since, until, period)` for "this month", "last 7 days" and the like.

    Takes the diacritic-folded question (`agent.plan._folded`). A period nobody asked for is
    `(None, None, "")`, never a default window: "how many documents" means all of them.
    """
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    last_days = _LAST_DAYS_RE.search(folded_question)
    if last_days:
        days = int(last_days.group(1) or last_days.group(3))
        if 1 <= days <= 3650:
            return today - timedelta(days=days - 1), None, f"days:{days}"
    for name, pattern in _PERIOD_PATTERNS:
        if not re.search(pattern, folded_question):
            continue
        if name == "today":
            return today, None, name
        if name == "week":
            return today - timedelta(days=today.weekday()), None, name
        if name == "month":
            return today.replace(day=1), None, name
        return today.replace(month=1, day=1), None, name
    return None, None, ""


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+!|~])")


def escape_markdown(text: str) -> str:
    """Titles are typed by uploaders. They are shown as text, never as markup or links."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


# ── queries ───────────────────────────────────────────────────────────────────────────────


def _pages_expression() -> Any:
    """Pages held in `document_sources.page_texts`.

    The column is JSON, and SQLAlchemy stores a Python None in it as the JSON value `null`
    rather than SQL NULL; `json_array_length` raises on a scalar, so the type is checked.
    """
    return case(
        (
            func.json_typeof(DocumentSource.page_texts) == "array",
            func.json_array_length(DocumentSource.page_texts),
        ),
        else_=0,
    )


def catalog_conditions(user: User, flt: CatalogFilter) -> list[Any]:
    """Every row-level condition a catalog query carries.

    The authorization predicate comes first and is the same one every article read uses.
    """
    conditions: list[Any] = [
        *ArticleRepository._authorized_article_filters(user),
        Article.status == PUBLISHED_STATUS,
    ]
    if flt.doc_types:
        conditions.append(Article.type.in_(flt.doc_types))
    if flt.departments:
        conditions.append(Article.dept.in_(flt.departments))
    if flt.since is not None:
        conditions.append(Article.created_at >= flt.since)
    if flt.until is not None:
        conditions.append(Article.created_at < flt.until)
    return conditions


async def _group(
    db: AsyncSession, column: Any, conditions: Sequence[Any]
) -> list[tuple[str, int]]:
    # Type, department and language are low-cardinality, so every group is read and the
    # fold into "other" in `top_groups` is exact.
    count = func.count(Article.id)
    rows = await db.execute(
        select(column, count)
        .where(*conditions)
        .group_by(column)
        .order_by(desc(count), column)
    )
    return [(str(key or ""), int(total)) for key, total in rows.all()]


async def build_catalog(
    db: AsyncSession,
    user: User,
    flt: CatalogFilter | None = None,
    *,
    list_limit: int | None = None,
    max_groups: int | None = None,
) -> CatalogResult:
    flt = flt or CatalogFilter()
    list_limit = list_limit if list_limit is not None else settings.CATALOG_LIST_LIMIT
    max_groups = max_groups if max_groups is not None else settings.CATALOG_MAX_GROUPS
    conditions = catalog_conditions(user, flt)
    result = CatalogResult(filter=flt)

    total = await db.execute(select(func.count(Article.id)).where(*conditions))
    result.total = int(total.scalar_one() or 0)
    if result.total == 0:
        return result

    for name, column in (
        ("by_type", Article.type),
        ("by_department", Article.dept),
        ("by_language", Article.language),
    ):
        kept, cut = top_groups(await _group(db, column, conditions), max_groups)
        setattr(result, name, kept)
        result.partial = result.partial or cut

    month = func.to_char(Article.created_at, "YYYY-MM")
    months = await db.execute(
        select(month, func.count(Article.id))
        .where(*conditions)
        .group_by(month)
        .order_by(desc(month))
        .limit(12)
    )
    result.by_month = [(str(key), int(count)) for key, count in months.all()]

    source_rows = await db.execute(
        select(
            DocumentSource.original_filename,
            DocumentSource.mime_type,
            _pages_expression(),
        )
        .join(Article, Article.id == DocumentSource.article_id)
        .where(*conditions)
        .limit(MAX_SOURCE_ROWS + 1)
    )
    sources = [tuple(row) for row in source_rows.all()]
    if len(sources) > MAX_SOURCE_ROWS:
        result.partial = True
        sources = sources[:MAX_SOURCE_ROWS]
    formats, result.pages = summarise_sources(sources)  # type: ignore[arg-type]
    result.by_format, cut = top_groups(
        ((name or OTHER_KEY, count) for name, count in formats.items()), max_groups
    )
    result.partial = result.partial or cut

    evidence = await db.execute(
        select(DocumentEvidence.kind, func.count(DocumentEvidence.id))
        .join(Article, Article.id == DocumentEvidence.article_id)
        .where(*conditions)
        .group_by(DocumentEvidence.kind)
    )
    result.evidence = {str(kind): int(count) for kind, count in evidence.all()}

    page_count = (
        select(func.coalesce(func.sum(_pages_expression()), 0))
        .where(DocumentSource.article_id == Article.id)
        .correlate(Article)
        .scalar_subquery()
    )
    recent = await db.execute(
        select(
            Article.id,
            Article.title,
            Article.type,
            Article.dept,
            Article.updated_at,
            page_count,
        )
        .where(*conditions)
        .order_by(desc(Article.updated_at), Article.title)
        .limit(max(0, list_limit))
    )
    result.recent = [
        CatalogDocument(
            article_id=str(row[0]),
            title=str(row[1] or ""),
            doc_type=str(row[2] or ""),
            department=str(row[3] or ""),
            updated_at=row[4],
            pages=int(row[5] or 0),
        )
        for row in recent.all()
    ]
    return result


# ── wording ───────────────────────────────────────────────────────────────────────────────

_TEXT = {
    "vi": {
        "other": "Khác",
        "none": "Hiện chưa có tài liệu nào{scope} mà bạn được phép xem.",
        "total": "Hiện có **{total}** tài liệu{scope} mà bạn được phép xem.",
        "by_type": "Theo loại",
        "by_department": "Theo phòng ban",
        "by_language": "Theo ngôn ngữ",
        "by_format": "Theo định dạng file",
        "by_month": "Số tài liệu nhập theo tháng (gần đây)",
        "scale": "Quy mô",
        "scale_value": "{pages} trang; đã trích {tables} bảng và {figures} hình",
        "recent": "Danh sách {n} tài liệu cập nhật gần nhất nằm ở bảng bên dưới; bấm vào tên để mở.",
        "note": "Số liệu tính lúc {time} UTC, chỉ gồm tài liệu đã xuất bản mà bạn có quyền đọc.",
        "partial": "Một số nhóm nhỏ được gộp vào mục “Khác”, hoặc số trang/định dạng chỉ là phần đã quét.",
        "scope_types": " loại {value}",
        "scope_depts": " của phòng ban {value}",
        "period_today": " nhập hôm nay",
        "period_week": " nhập trong tuần này",
        "period_month": " nhập trong tháng này",
        "period_year": " nhập trong năm nay",
        "period_days": " nhập trong {n} ngày qua",
        "table_summary": "Thống kê tài liệu",
        "table_recent": "Tài liệu cập nhật gần nhất",
        "col_group": "Nhóm",
        "col_value": "Giá trị",
        "col_count": "Số lượng",
        "col_title": "Tên tài liệu",
        "col_type": "Loại",
        "col_dept": "Phòng ban",
        "col_updated": "Cập nhật",
        "col_pages": "Số trang",
        "g_type": "Loại",
        "g_dept": "Phòng ban",
        "g_lang": "Ngôn ngữ",
        "g_format": "Định dạng",
        "g_month": "Tháng nhập",
        "g_evidence": "Đã trích",
        "g_pages": "Tổng số trang",
        "evidence_table": "bảng",
        "evidence_figure": "hình",
        "evidence_page_visual": "trang ảnh",
    },
    "en": {
        "other": "Other",
        "none": "There are no documents{scope} that you are allowed to see yet.",
        "total": "There are **{total}** documents{scope} that you are allowed to see.",
        "by_type": "By type",
        "by_department": "By department",
        "by_language": "By language",
        "by_format": "By file format",
        "by_month": "Documents added per month (recent)",
        "scale": "Scale",
        "scale_value": "{pages} pages; {tables} tables and {figures} figures extracted",
        "recent": "The {n} most recently updated documents are in the table below; click a title to open it.",
        "note": "Figures are as of {time} UTC and cover only published documents you may read.",
        "partial": "Some small groups are folded into “Other”, or the page/format figures cover only the part scanned.",
        "scope_types": " of type {value}",
        "scope_depts": " in {value}",
        "period_today": " added today",
        "period_week": " added this week",
        "period_month": " added this month",
        "period_year": " added this year",
        "period_days": " added in the last {n} days",
        "table_summary": "Document statistics",
        "table_recent": "Most recently updated documents",
        "col_group": "Group",
        "col_value": "Value",
        "col_count": "Count",
        "col_title": "Title",
        "col_type": "Type",
        "col_dept": "Department",
        "col_updated": "Updated",
        "col_pages": "Pages",
        "g_type": "Type",
        "g_dept": "Department",
        "g_lang": "Language",
        "g_format": "Format",
        "g_month": "Month added",
        "g_evidence": "Extracted",
        "g_pages": "Total pages",
        "evidence_table": "table",
        "evidence_figure": "figure",
        "evidence_page_visual": "page image",
    },
}


def _words(language: str) -> dict[str, str]:
    return _TEXT["en" if language == "en" else "vi"]


def _scope(flt: CatalogFilter, language: str) -> str:
    words = _words(language)
    parts: list[str] = []
    if flt.doc_types:
        parts.append(
            words["scope_types"].format(
                value=", ".join(escape_markdown(v) for v in flt.doc_types)
            )
        )
    if flt.departments:
        parts.append(
            words["scope_depts"].format(
                value=", ".join(escape_markdown(v) for v in flt.departments)
            )
        )
    period = flt.period
    if period.startswith("days:"):
        parts.append(words["period_days"].format(n=period.split(":", 1)[1]))
    elif period in {"today", "week", "month", "year"}:
        parts.append(words[f"period_{period}"])
    return "".join(parts)


def _label(key: str, words: dict[str, str]) -> str:
    return words["other"] if key == OTHER_KEY else key


def _line(label: str, groups: Sequence[tuple[str, int]], words: dict[str, str]) -> str:
    return f"**{label}:** " + " · ".join(
        f"{escape_markdown(_label(key, words)) or '—'} {count}" for key, count in groups
    )


def format_catalog_answer(result: CatalogResult, language: str) -> str:
    """The answer text, in the language the user wrote in. Deterministic."""
    words = _words(language)
    scope = _scope(result.filter, language)
    if result.total == 0:
        return words["none"].format(scope=scope)
    lines = [words["total"].format(total=result.total, scope=scope), ""]
    for key, groups in (
        ("by_type", result.by_type),
        ("by_department", result.by_department),
        ("by_format", result.by_format),
        ("by_language", result.by_language),
    ):
        if groups:
            lines.append(_line(words[key], groups, words))
    if result.by_month:
        lines.append(_line(words["by_month"], result.by_month, words))
    if result.pages or result.evidence:
        lines.append(
            f"**{words['scale']}:** "
            + words["scale_value"].format(
                pages=result.pages,
                tables=result.evidence.get("table", 0),
                figures=result.evidence.get("figure", 0),
            )
        )
    lines.append("")
    if result.recent:
        lines.append(words["recent"].format(n=len(result.recent)))
    lines.append(
        words["note"].format(time=result.generated_at.strftime("%Y-%m-%d %H:%M"))
    )
    if result.partial:
        lines.append(words["partial"])
    return "\n".join(lines)


#: Bounds on what a table artifact carries (also enforced by the frontend parser).
MAX_TABLE_ROWS = 200


def catalog_artifacts(result: CatalogResult, language: str) -> list[dict[str, Any]]:
    """Interactive tables for the answer: the statistics and the recent documents."""
    if result.total == 0:
        return []
    words = _words(language)
    summary_rows: list[list[Any]] = []
    for label_key, groups in (
        ("g_type", result.by_type),
        ("g_dept", result.by_department),
        ("g_format", result.by_format),
        ("g_lang", result.by_language),
        ("g_month", result.by_month),
    ):
        summary_rows.extend(
            [words[label_key], _label(key, words) or "—", count]
            for key, count in groups
        )
    for kind, count in sorted(result.evidence.items()):
        summary_rows.append(
            [words["g_evidence"], words.get(f"evidence_{kind}", kind), count]
        )
    summary_rows.append([words["g_pages"], "", result.pages])
    artifacts: list[dict[str, Any]] = [
        {
            "type": "table",
            "id": "catalog-summary",
            "title": words["table_summary"],
            "columns": [words["col_group"], words["col_value"], words["col_count"]],
            "rows": summary_rows[:MAX_TABLE_ROWS],
            "link_column": None,
            "row_links": None,
            "total_rows": len(summary_rows),
            "truncated": len(summary_rows) > MAX_TABLE_ROWS,
        }
    ]
    if result.recent:
        recent = result.recent[:MAX_TABLE_ROWS]
        artifacts.append(
            {
                "type": "table",
                "id": "catalog-recent",
                "title": words["table_recent"],
                "columns": [
                    words["col_title"],
                    words["col_type"],
                    words["col_dept"],
                    words["col_updated"],
                    words["col_pages"],
                ],
                "rows": [
                    [
                        item.title,
                        item.doc_type,
                        item.department,
                        item.updated_at.strftime("%Y-%m-%d") if item.updated_at else "",
                        item.pages,
                    ]
                    for item in recent
                ],
                "link_column": 0,
                "row_links": [{"article_id": item.article_id} for item in recent],
                "total_rows": len(result.recent),
                "truncated": len(result.recent) > MAX_TABLE_ROWS,
            }
        )
    return artifacts


def filter_for_question(
    question: str,
    doc_types: Sequence[str],
    departments: Sequence[str],
    now: datetime,
) -> CatalogFilter:
    """The filter a catalog question asked for: type and department from the plan, the
    period from the wording."""
    folded = fold_diacritics(" ".join(question.lower().split())) + " "
    since, until, period = parse_period(folded, now)
    return CatalogFilter(
        doc_types=list(doc_types),
        departments=list(departments),
        since=since,
        until=until,
        period=period,
    )
