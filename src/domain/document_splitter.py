"""Deterministic structure-aware document splitting for review candidates."""

from __future__ import annotations

import re
from typing import Any


# A review candidate is an approval unit, not a retrieval chunk.  Retrieval is
# split separately after approval, so making candidates as small as the 1,800
# character index parents turns every AI heading into an unnecessary article.
# Keep a normal document intact; only exceptionally long formatted documents
# need more than one review card.
_MAX_REVIEW_CANDIDATE_CHARS = 48_000


def _candidate_title(body: str, index: int, fallback: str) -> tuple[str, str | None]:
    for line in body.splitlines():
        clean = line.strip()
        if re.match(r"^#{1,6}\s+\S", clean):
            return (
                clean.lstrip("#").strip()[:255] or f"{fallback} — part {index}",
                clean.lstrip("#").strip()[:255],
            )
    return f"{fallback} — part {index}"[:255], None


def _large_document_candidates(title: str, text: str) -> list[dict[str, Any]]:
    """Split only a document that cannot sensibly be reviewed in one card.

    Prefer paragraph boundaries and never use the retrieval chunker's small
    parent size here.  Each result remains a substantial, lossless review
    unit; the reviewer can still manually split it when needed.
    """
    candidates: list[dict[str, Any]] = []
    start = 0
    index = 1
    while start < len(text):
        end = min(start + _MAX_REVIEW_CANDIDATE_CHARS, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + (_MAX_REVIEW_CANDIDATE_CHARS // 2):
                end = boundary
        body = text[start:end].strip()
        # A long unbroken line still has to make forward progress.
        if not body:
            end = min(start + _MAX_REVIEW_CANDIDATE_CHARS, len(text))
            body = text[start:end].strip()
        candidate_title, heading = _candidate_title(body, index, title)
        candidates.append(
            {
                "position": index,
                "title": candidate_title,
                "body_md": body,
                "source_start": start,
                "source_end": end,
                "heading": heading,
            }
        )
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        index += 1
    return candidates


def split_document_candidates(
    title: str, text: str, *, prefer_markdown_sections: bool = False, page_texts: list[dict] | None = None
) -> list[dict[str, Any]]:
    """Return ordered candidates with source character offsets.

    A candidate is normally the entire formatted document.  Its later index
    representation is chunked independently, so review must not inherit the
    small parent/child retrieval boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return []
    def position(start: int, end: int, heading: str | None) -> dict[str, Any]:
        result: dict[str, Any] = {"offset_start": start, "offset_end": end, "section": heading}
        if page_texts:
            cursor = 0
            first = last = None
            for page in page_texts:
                page_start = cursor
                page_end = cursor + len(str(page.get("text") or ""))
                if first is None and end > page_start:
                    first = page.get("page_number")
                if start < page_end:
                    last = page.get("page_number")
                cursor = page_end
            result["page_start"] = first
            result["page_end"] = last or first
        return result
    if len(clean_text) <= _MAX_REVIEW_CANDIDATE_CHARS:
        candidate_title, heading = _candidate_title(clean_text, 1, title)
        return [
            {
                "position": 1,
                "title": candidate_title,
                "body_md": clean_text,
                "source_start": 0,
                "source_end": len(clean_text),
                "heading": heading,
                "source_position": position(0, len(clean_text), heading),
            }
        ]
    # For unusually long documents, use broad review-sized ranges rather than
    # treating every Markdown heading as a separate knowledge article.  The
    # argument remains for call compatibility and has no effect on this rule.
    candidates = _large_document_candidates(title, clean_text)
    for item in candidates:
        item["source_position"] = position(item["source_start"], item["source_end"], item.get("heading"))
    return candidates


def splitter_metrics(documents: list[tuple[str, str]]) -> dict[str, Any]:
    """Produce a reproducible pilot report without an LLM call."""
    rows = []
    manual_correction_count = 0
    article_count = 0
    for name, text in documents:
        candidates = split_document_candidates(name, text)
        article_count += len(candidates)
        candidate_text = "\n".join(str(item["body_md"]) for item in candidates)
        original_headings = [
            line.strip().lstrip("#").strip()
            for line in text.splitlines()
            if re.match(r"^#{1,6}\s+\S", line.strip())
        ]
        preserved = all(heading in candidate_text for heading in original_headings)
        if not preserved:
            manual_correction_count += 1
        rows.append(
            {
                "document": name,
                "article_count": len(candidates),
                "headings_preserved": preserved,
            }
        )
    count = len(documents)
    return {
        "document_count": count,
        "article_count": article_count,
        "manual_correction_count": manual_correction_count,
        "manual_correction_share": (manual_correction_count / count) if count else 0.0,
        "rule": "one formatted review document up to 48,000 characters; exceptionally long documents split at broad paragraph boundaries",
        "documents": rows,
    }
