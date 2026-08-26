"""Description-led department suggestions for formatted split candidates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9_-]{2,}")
_STOP_WORDS = {
    "about",
    "and",
    "are",
    "for",
    "from",
    "into",
    "knowledge",
    "that",
    "the",
    "this",
    "with",
    "your",
    "document",
    "department",
}


def _major_sections(title: str, markdown: str) -> list[dict[str, Any]]:
    """Return broad AI-created ``##`` sections, never ``###`` fragments."""
    sections: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        is_major_boundary = re.match(r"^##(?!#)\s+\S", line.strip()) is not None
        if is_major_boundary and current and "\n".join(current).strip():
            # Keep the document H1/preamble with its first department section
            # so it cannot become an unrouted candidate of its own.
            has_preamble_content = any(
                value.strip() and not re.match(r"^#\s+\S", value.strip())
                for value in current
            )
            if sections or has_preamble_content:
                sections.append("\n".join(current).strip())
                current = []
        current.append(line)
    if current and "\n".join(current).strip():
        sections.append("\n".join(current).strip())
    if len(sections) < 2:
        sections = [markdown.strip()]

    candidates: list[dict[str, Any]] = []
    cursor = 0
    for position, body in enumerate(sections, start=1):
        start = markdown.find(body, cursor)
        start = cursor if start < 0 else start
        end = start + len(body)
        cursor = end
        heading = next(
            (
                line.lstrip("#").strip()[:255]
                for line in body.splitlines()
                if re.match(r"^#{1,6}\s+\S", line.strip())
            ),
            None,
        )
        candidates.append(
            {
                "position": position,
                "title": heading or (title if position == 1 else f"{title} — part {position}"),
                "body_md": body,
                "source_start": start,
                "source_end": end,
                "heading": heading,
            }
        )
    return candidates


def _tokens(value: str) -> Counter[str]:
    return Counter(
        token.lower()
        for token in _TOKEN_RE.findall(value or "")
        if token.lower() not in _STOP_WORDS
    )


def _proposal_name(title: str) -> str:
    cleaned = re.sub(r"[^\wÀ-ỹ -]+", " ", title).strip()
    words = cleaned.split()
    return " ".join(words[:6])[:100] or "New department"


def suggest_departments(
    title: str, body_md: str, departments: Iterable[Any]
) -> tuple[list[str], list[dict[str, Any]], dict[str, str] | None]:
    """Rank active departments from their names and short descriptions.

    This is intentionally deterministic: recommendation is available even when an
    LLM provider is disabled, and it never creates a department without a reviewer.
    """
    document_tokens = _tokens(f"{title} {title} {body_md}")
    ranked: list[tuple[int, Any]] = []
    for department in departments:
        reference_tokens = _tokens(
            f"{getattr(department, 'name', '')} {getattr(department, 'description', '')}"
        )
        score = sum(
            min(count, reference_tokens.get(token, 0))
            for token, count in document_tokens.items()
        )
        if score:
            ranked.append((score, department))
    ranked.sort(key=lambda item: (-item[0], item[1].name.lower()))
    suggestions = [
        {
            "department_id": str(department.id),
            "name": department.name,
            "description": department.description,
            "score": score,
        }
        for score, department in ranked[:3]
    ]
    selected_ids = [suggestions[0]["department_id"]] if suggestions else []
    proposed = None
    if not suggestions:
        proposed = {
            "name": _proposal_name(title),
            "description": f"Knowledge and procedures related to {title.strip()[:180] or 'this subject'}.",
        }
    return selected_ids, suggestions, proposed


def route_document_candidates(
    title: str, markdown: str, departments: Iterable[Any]
) -> list[dict[str, Any]]:
    """Split a formatted document only at a change of owning department.

    The formatter supplies broad ``##`` sections.  Adjacent sections assigned
    to the same department remain one review item; ``###`` headings never
    create an item.  This keeps the system a single-company knowledge base
    while producing separate drafts only for genuinely different departments.
    """
    department_list = list(departments)
    sections = _major_sections(title, markdown)
    routed = []
    for section in sections:
        ids, suggestions, proposed = suggest_departments(
            section["title"], section["body_md"], department_list
        )
        routed.append(
            {
                **section,
                "department_ids": ids,
                "department_suggestions": suggestions,
                "proposed_department": proposed,
            }
        )

    # If no section can be linked to an existing department, keep the document
    # intact and offer one reviewable new-department suggestion instead.
    if not any(item["department_ids"] for item in routed):
        ids, suggestions, proposed = suggest_departments(title, markdown, department_list)
        return [
            {
                "position": 1,
                "title": title[:255],
                "body_md": markdown.strip(),
                "source_start": 0,
                "source_end": len(markdown.strip()),
                "heading": None,
                "department_ids": ids,
                "department_suggestions": suggestions,
                "proposed_department": proposed,
            }
        ]

    grouped: list[dict[str, Any]] = []
    for item in routed:
        primary_id = item["department_ids"][0] if item["department_ids"] else None
        previous_primary = (
            grouped[-1]["department_ids"][0]
            if grouped and grouped[-1]["department_ids"]
            else None
        )
        if grouped and primary_id == previous_primary:
            grouped[-1]["body_md"] += "\n\n" + item["body_md"]
            grouped[-1]["source_end"] = item["source_end"]
            continue
        grouped.append(dict(item))

    for position, item in enumerate(grouped, start=1):
        item["position"] = position
    return grouped
