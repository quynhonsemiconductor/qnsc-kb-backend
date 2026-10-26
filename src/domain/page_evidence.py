"""The figures and tables that sit on the pages an answer cites.

Retrieval only returns a figure or table when that figure is itself the best match, and a
figure's searchable text is little more than "Figure 1 on page 55". So a question about
"the learning curve" retrieves the paragraph that discusses it (page 55) and never the
picture beside it: the answer says what the figure shows and the reader cannot see it.

This closes that gap from the other side. Every citation already names an article and a
page; the evidence extracted from that page is attached to the citation, and the reader's
answer shows it beside the claim. Nothing here widens access: the evidence belongs to an
article the citation already came from, the query applies the same visibility predicate
the citation hydration uses, and the preview endpoint authorizes the parent article again
when the image is fetched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.domain.evidence_extraction import table_evidence_preview
from src.models.article import Article
from src.models.evidence import DocumentEvidence
from src.models.user import User
from src.repositories.article import ArticleRepository

#: Whole-page images are only worth showing for a page with no figure or table of its own.
_PRIORITY = {"table": 0, "figure": 0, "page_visual": 1}


def _own_evidence_id(item: Mapping[str, Any]) -> str:
    """The evidence an item already carries: a citation's flat field or a result's nested one."""
    return str(item.get("evidence_id") or (item.get("evidence") or {}).get("id") or "")


def pick_page_evidence(
    citations: Sequence[dict[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    *,
    per_citation: int,
    total: int,
    unique: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    """Which evidence rows each citation shows, by position in `citations`.

    Pure, so the judgement can be tested without a database. `rows` are evidence the query
    found for the cited pages, each with `id`, `article_id`, `kind`, `page_number`, `label`
    and (for tables) `structured_data`.

    * a citation that is itself a figure or table adds nothing: it already shows its own;
    * the same evidence is shown once per answer, at the first citation that reaches it
      (`unique=False` lists it for every item on its page, which is what the prompt needs:
      it is written before anyone knows which sources the answer will cite);
    * tables and figures come before a whole-page image, and a page image is dropped when
      the page has a table or figure;
    * `per_citation` and `total` bound what one claim and one answer can carry.
    """
    by_page: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_page.setdefault(
            (str(row["article_id"]), int(row["page_number"])), []
        ).append(row)

    shown: set[str] = {_own_evidence_id(item) for item in citations} - {""}
    chosen: dict[int, list[dict[str, Any]]] = {}
    count = 0
    for position, citation in enumerate(citations):
        if count >= total:
            break
        page = citation.get("page_number")
        article_id = citation.get("article_id")
        if _own_evidence_id(citation) or not page or not article_id:
            continue
        candidates = by_page.get((str(article_id), int(page)), [])
        has_primary = any(_PRIORITY.get(str(row["kind"]), 2) == 0 for row in candidates)
        picked: list[dict[str, Any]] = []
        for row in sorted(
            candidates,
            key=lambda r: (_PRIORITY.get(str(r["kind"]), 2), str(r["label"])),
        ):
            if len(picked) >= per_citation or count + len(picked) >= total:
                break
            if str(row["id"]) in shown:
                continue
            if row["kind"] == "page_visual" and has_primary:
                continue
            if unique:
                shown.add(str(row["id"]))
            picked.append(
                {
                    "evidence_id": str(row["id"]),
                    "kind": row["kind"],
                    "label": row["label"],
                    "page_number": int(row["page_number"]),
                    "preview_url": f"/api/v1/articles/{article_id}/evidence/{row['id']}/preview",
                    "table": (
                        table_evidence_preview(row.get("structured_data"))
                        if row["kind"] == "table"
                        else None
                    ),
                }
            )
        if picked:
            chosen[position] = picked
            count += len(picked)
    return chosen


async def attach_page_evidence(
    db: AsyncSession,
    user: User,
    citations: list[dict[str, Any]],
    *,
    unique: bool = True,
) -> None:
    """Add `page_evidence` to each citation (or retrieved result) whose page has figures or tables.

    Best-effort by design: a failure here costs the reader a picture, never the answer, so
    callers treat an exception as "no extra evidence".
    """
    if not settings.ANSWER_PAGE_EVIDENCE_ENABLED or not citations:
        return
    pairs = {
        (str(item["article_id"]), int(item["page_number"]))
        for item in citations
        if item.get("article_id")
        and item.get("page_number")
        and not _own_evidence_id(item)
    }
    if not pairs:
        return
    result = await db.execute(
        select(DocumentEvidence)
        .join(Article, Article.id == DocumentEvidence.article_id)
        .where(
            DocumentEvidence.article_id.in_({article for article, _ in pairs}),
            DocumentEvidence.page_number.in_({page for _, page in pairs}),
            Article.status == "published",
            Article.index_status == "ready",
            *ArticleRepository._authorized_article_filters(user),
        )
    )
    rows = [
        {
            "id": item.id,
            "article_id": item.article_id,
            "kind": item.kind,
            "page_number": item.page_number,
            "label": item.label,
            "structured_data": item.structured_data,
        }
        for item in result.scalars().all()
        if (str(item.article_id), int(item.page_number)) in pairs
    ]
    chosen = pick_page_evidence(
        citations,
        rows,
        per_citation=settings.ANSWER_PAGE_EVIDENCE_PER_CITATION,
        total=settings.ANSWER_PAGE_EVIDENCE_MAX if unique else len(citations) * 8,
        unique=unique,
    )
    for position, items in chosen.items():
        citations[position]["page_evidence"] = items
