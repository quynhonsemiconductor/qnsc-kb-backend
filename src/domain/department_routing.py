"""Description-led department suggestions for formatted split candidates."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import structlog

_logger = structlog.get_logger()


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
            # Is the buffer a SECTION, or just the document's front matter?
            #
            # A buffer that already contains a `##` line holds a finished section and
            # must be emitted. A buffer that does not is preamble -- the H1, a version
            # line, a document code -- and belongs with the first real section rather
            # than becoming a candidate of its own that no department owns.
            #
            # This used to ask "does the preamble contain any non-H1 content?", which got
            # the common case backwards: `# Handbook` alone was correctly held, but
            # `# Handbook` + `Version 1.0` was emitted as a standalone unrouted candidate.
            # Testing the buffer for a `##` rather than inspecting only its first line
            # matters once the preamble has been merged into section one, because from
            # then on the buffer always OPENS with the H1.
            buffer_holds_section = any(
                re.match(r"^##(?!#)\s+\S", value.strip()) for value in current
            )
            if buffer_holds_section:
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
    routed = [
        {
            **section,
            **_assignment(
                *suggest_departments(section["title"], section["body_md"], department_list)
            ),
        }
        for section in sections
    ]
    return _finalize(title, markdown, department_list, routed)


def _assignment(
    ids: list[str], suggestions: list[dict[str, Any]], proposed: dict[str, str] | None
) -> dict[str, Any]:
    return {
        "department_ids": ids,
        "department_suggestions": suggestions,
        "proposed_department": proposed,
    }


def _body_without_headings(body_md: str) -> str:
    """Return a section body with its own heading lines removed."""
    return "\n".join(
        line
        for line in (body_md or "").splitlines()
        if not re.match(r"^#{1,6}\s+", line.strip())
    ).strip()


def _absorb_hollow_sections(routed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge sections that carry no content of their own into a neighbour.

    ``_major_sections`` starts a new section at every ``##`` with no size condition, so
    ``## Overview`` immediately followed by ``## Scope`` emits a candidate whose entire
    body is the four characters ``## A``. That is not a reviewable article, and it is not
    rare: a source over ``RESTRUCTURE_SECTION_CHARS`` is formatted in parts by separate
    LLM calls that never see each other, then concatenated, so adjacent headings appear at
    every seam.

    Emptiness is judged on non-heading content rather than a character count. A short
    section that says something real is a legitimate candidate; a heading with nothing
    under it is not, at any length.

    Runs BEFORE the department grouping on purpose. A hollow section has almost no tokens
    to rank, so its routing is close to arbitrary, and leaving it in place would let a
    spurious department assignment break a same-owner merge between its neighbours.
    """
    if len(routed) < 2:
        return routed
    merged: list[dict[str, Any]] = []
    # A hollow section with nothing before it cannot merge backwards; carry it forward
    # and prepend it to the next section that has content.
    pending: dict[str, Any] | None = None
    for item in routed:
        current = dict(item)
        if pending is not None:
            current["body_md"] = f"{pending['body_md']}\n\n{current['body_md']}"
            current["source_start"] = pending["source_start"]
            if pending.get("heading"):
                # The merged text opens with the held heading, so it is the identity a
                # reviewer sees; `_major_sections` would have picked the same one.
                current["heading"] = pending["heading"]
                current["title"] = pending["title"]
            pending = None
        if _body_without_headings(current["body_md"]):
            merged.append(current)
        elif merged:
            merged[-1]["body_md"] += "\n\n" + current["body_md"]
            merged[-1]["source_end"] = current["source_end"]
        else:
            pending = current
    if pending is not None:
        # Every section was hollow: keep the document rather than dropping it.
        merged.append(pending)
    return merged


def _finalize(
    title: str,
    markdown: str,
    department_list: list[Any],
    routed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse, merge and number routed sections.

    Shared by the keyword path and the LLM path deliberately. A second copy of the
    merge rule would be a second thing to keep in step, and the two would disagree the
    first time either changed.
    """
    routed = _absorb_hollow_sections(routed)
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


# ---------------------------------------------------------------------------
# LLM-assisted routing
#
# The ranking above matches a document against a department's name and description
# literally: no stemming, no synonyms, no meaning. `description` defaults to "", so on a
# tenant where nobody has written them every score is zero -- nothing routes, and every
# document instead proposes creating a new department named after its own filename.
# Even with descriptions written, a lecture on static timing analysis only reaches the
# right team if somebody thought to put "timing" in that team's description.
#
# So an LLM decides when one is configured, and the keyword ranking stays as the
# fallback: unchanged, still computed for every section, and still the answer whenever
# the model declines, errors, or is switched off.
#
# Departments are offered to the model as NUMBERS rather than UUIDs. A short integer is
# cheap to emit, survives a truncated reply, and cannot be half-hallucinated into
# something shaped like a real id. Every answer is checked against the list that was
# sent, and anything else falls back rather than being guessed at.
# ---------------------------------------------------------------------------

#: Enough of a section to recognise its subject; the rest is prompt cost.
_LLM_BODY_CHARS = 1_200
#: Past this a document is being re-outlined, not routed, and the reply grows unbounded.
_LLM_MAX_SECTIONS = 12

_ROUTING_SYSTEM_PROMPT = (
    "You assign sections of a document to the department that owns them in a company "
    "knowledge base. Reply with JSON only, no prose: "
    '{"assignments": [{"section": <int>, "department": <int>}]}. '
    "Use only the department numbers given. Use 0 when no listed department is a "
    "sensible owner. Include every section exactly once."
)


def _routing_prompt(
    title: str, sections: list[dict[str, Any]], departments: list[Any]
) -> str:
    lines = ["DEPARTMENTS:"]
    for index, department in enumerate(departments, start=1):
        description = (getattr(department, "description", "") or "").strip()
        suffix = f" - {description[:300]}" if description else ""
        lines.append(f"{index}. {getattr(department, 'name', '')}{suffix}")
    lines += ["", f"DOCUMENT: {title}", "", "SECTIONS:"]
    for index, section in enumerate(sections, start=1):
        lines.append(f"--- section {index}: {section.get('title') or ''}")
        lines.append((section.get("body_md") or "")[:_LLM_BODY_CHARS])
    return "\n".join(lines)


def _parse_assignments(
    reply: str, section_count: int, department_count: int
) -> dict[int, int]:
    """Read the reply, keeping only assignments that name things that exist.

    Tolerant about how the JSON is wrapped -- models fence it, or add a sentence -- and
    strict about what it says. An out-of-range number is dropped rather than clamped:
    a wrong department is worse than no suggestion, because a reviewer reads a filled-in
    field as something the system worked out.
    """
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    payload = json.loads(text[start : end + 1])

    chosen: dict[int, int] = {}
    for entry in payload.get("assignments") or []:
        if not isinstance(entry, dict):
            continue
        section, department = entry.get("section"), entry.get("department")
        # bool is an int subclass, and True would silently mean department 1.
        if type(section) is not int or type(department) is not int:
            continue
        # 0 is a real answer meaning "no owner"; it leaves the section to the fallback.
        if 1 <= section <= section_count and 1 <= department <= department_count:
            chosen[section] = department
    return chosen


#: Why the LLM contributed nothing to a routing decision. `off` and `not_applicable` are
#: normal; `unavailable` and `unreadable` mean the model was asked and could not answer,
#: and a provider that is permanently broken sits in one of those forever. They used to
#: be the same empty dict as "the model declined", so nothing distinguished a working
#: keyword fallback from a routing feature that had silently stopped existing.
_ROUTING_OFF = "off"
_ROUTING_NOT_APPLICABLE = "not_applicable"
_ROUTING_UNAVAILABLE = "unavailable"
_ROUTING_UNREADABLE = "unreadable"
_ROUTING_ANSWERED = "answered"


@dataclass(frozen=True)
class _RoutingOutcome:
    """What the LLM routing attempt produced, and why it produced that.

    `assignments` maps 1-based section number to a Department. Empty is a valid answer --
    the model can decline every section -- so `state` is what tells the two apart.
    """

    state: str
    assignments: dict[int, Any]


async def _llm_section_departments(
    title: str, sections: list[dict[str, Any]], departments: list[Any]
) -> _RoutingOutcome:
    """Ask the LLM which department owns each section, and say what came back.

    Never raises. A routing suggestion is something a reviewer sees and can change; it
    must not be able to fail a document import, which this codebase has already paid
    for once. Not raising is not the same as not reporting, though: an unavailable
    provider is named in the outcome and logged at warning level, because "the model had
    no opinion" and "the model could not be reached" look identical in the result and
    only one of them is somebody's job to fix.
    """
    from src.core.config import settings

    if not settings.DEPARTMENT_ROUTING_LLM_ENABLED:
        return _RoutingOutcome(_ROUTING_OFF, {})
    if not departments or not sections or len(sections) > _LLM_MAX_SECTIONS:
        return _RoutingOutcome(_ROUTING_NOT_APPLICABLE, {})
    try:
        from src.domain.llm_client import complete, resolve_provider

        if resolve_provider() is None:
            _logger.warning(
                "LLM department routing unavailable: no provider configured; "
                "using keyword ranking",
                section_count=len(sections),
            )
            return _RoutingOutcome(_ROUTING_UNAVAILABLE, {})
        reply, _tokens, _model, _provider = await complete(
            [
                {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _routing_prompt(title, sections, departments),
                },
            ],
            timeout=settings.DEPARTMENT_ROUTING_LLM_TIMEOUT,
            # Hidden reasoning would spend the whole budget before the JSON appears.
            thinking=False,
            max_tokens=64 + 24 * len(sections),
        )
    except Exception:
        _logger.warning(
            "LLM department routing unavailable; using keyword ranking", exc_info=True
        )
        return _RoutingOutcome(_ROUTING_UNAVAILABLE, {})

    try:
        chosen = _parse_assignments(reply, len(sections), len(departments))
    except Exception:
        _logger.warning("LLM department routing returned unreadable JSON", exc_info=True)
        return _RoutingOutcome(_ROUTING_UNREADABLE, {})
    return _RoutingOutcome(
        _ROUTING_ANSWERED,
        {section: departments[index - 1] for section, index in chosen.items()},
    )


async def route_document_candidates_llm(
    title: str, markdown: str, departments: Iterable[Any]
) -> list[dict[str, Any]]:
    """Route with the LLM when one is configured, and exactly as before when not.

    The keyword ranking is computed for every section either way: it supplies the
    reviewer's alternatives and the new-department proposal, and it is the whole answer
    whenever the model has nothing to say.
    """
    department_list = list(departments)
    sections = _major_sections(title, markdown)
    baseline = [
        suggest_departments(section["title"], section["body_md"], department_list)
        for section in sections
    ]
    outcome = await _llm_section_departments(title, sections, department_list)
    chosen = outcome.assignments

    routed: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        ids, suggestions, proposed = baseline[index]
        department = chosen.get(index + 1)
        if department is not None:
            department_id = str(department.id)
            ids = [department_id]
            # The model named a real owner, so there is nothing left to propose creating.
            proposed = None
            existing = next(
                (item for item in suggestions if item["department_id"] == department_id),
                None,
            )
            suggestions = [
                existing
                or {
                    "department_id": department_id,
                    "name": getattr(department, "name", ""),
                    "description": getattr(department, "description", ""),
                    # No keyword overlap is exactly why the LLM was worth asking.
                    "score": 0,
                }
            ] + [
                item for item in suggestions if item["department_id"] != department_id
            ]
        routed.append(
            {**section, **_assignment(ids, suggestions[:3], proposed)}
        )
    return _finalize(title, markdown, department_list, routed)
