"""A readable view of how an article was cut into passages, with the cuts that look wrong.

RAGFlow lets a person look at the chunking and step in. When an answer misses something it
should have found, the first question is whether the text was cut sensibly: a table split
across passages, a scan that produced fragments, a passage embedded by an older model.
This turns the stored parent and child chunks into one structure for that screen and marks
what is suspicious; it changes nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

TINY_CHARS = 40
HUGE_CHARS = 3000
PARENT_TEXT_CHARS = 2400

FLAGS = ("tiny", "huge", "duplicate", "stale_embedding", "old_chunking")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def build_chunk_view(
    parents: list[Any],
    children: list[Any],
    embedding_version: str,
    chunking_version: str,
) -> dict[str, Any]:
    counts = Counter(_norm(c.chunk_text) for c in children)
    by_parent: dict[str, list[dict[str, Any]]] = {}
    flag_totals: Counter[str] = Counter()
    for child in sorted(children, key=lambda c: c.chunk_index):
        text = child.chunk_text or ""
        flags = []
        if len(text.strip()) < TINY_CHARS:
            flags.append("tiny")
        if len(text) > HUGE_CHARS:
            flags.append("huge")
        if counts[_norm(text)] > 1 and text.strip():
            flags.append("duplicate")
        if child.embedding_version != embedding_version:
            flags.append("stale_embedding")
        if child.chunking_version != chunking_version:
            flags.append("old_chunking")
        flag_totals.update(flags)
        by_parent.setdefault(str(child.parent_chunk_id), []).append(
            {
                "id": str(child.id),
                "index": child.chunk_index,
                "text": text,
                "chars": len(text),
                "page_number": child.page_number,
                "heading": child.heading,
                "flags": flags,
            }
        )

    def first_index(parent: Any) -> int:
        kids = by_parent.get(str(parent.id))
        return kids[0]["index"] if kids else 10**9

    parent_views = [
        {
            "id": str(parent.id),
            "heading": parent.heading,
            "section_ref": parent.section_ref,
            "page_number": parent.page_number,
            "chunk_type": parent.chunk_type,
            "chars": len(parent.text or ""),
            "text": (parent.text or "")[:PARENT_TEXT_CHARS],
            "children": by_parent.get(str(parent.id), []),
        }
        for parent in sorted(parents, key=first_index)
    ]
    total = sum(c["chars"] for kids in by_parent.values() for c in kids)
    child_count = sum(len(kids) for kids in by_parent.values())
    return {
        "summary": {
            "parents": len(parent_views),
            "children": child_count,
            "avg_chars": round(total / child_count) if child_count else 0,
            "flags": {flag: flag_totals.get(flag, 0) for flag in FLAGS},
            "embedding_version": embedding_version,
            "chunking_version": chunking_version,
        },
        "parents": parent_views,
    }
