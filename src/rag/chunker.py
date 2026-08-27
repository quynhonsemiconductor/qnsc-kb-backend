"""Structure-aware parent/child chunking for the QNSC indexer."""
from __future__ import annotations

import re
from typing import Any


_HEADING_RE = re.compile(r"^(?:#{1,6}\s+|[A-Z][A-Z0-9 /&:()_-]{2,}:\s*$)")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


def _is_heading(line: str) -> bool:
    clean = line.strip()
    return bool(clean and _HEADING_RE.match(clean))


def _is_table(lines: list[str]) -> bool:
    pipe_rows = sum("|" in line for line in lines)
    return len(lines) >= 2 and pipe_rows >= 2


def _is_list(lines: list[str]) -> bool:
    return len(lines) >= 2 and sum(bool(_LIST_RE.match(line)) for line in lines) >= 2


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _blocks(text: str, heading: str | None = None) -> list[dict[str, str]]:
    """Split into logical blocks without breaking tables or lists."""
    blocks: list[dict[str, str]] = []
    active_heading = (heading or "").strip() or None
    for raw in re.split(r"\n\s*\n", text or ""):
        clean = raw.strip()
        if not clean:
            continue
        lines = [line.rstrip() for line in clean.splitlines() if line.strip()]
        if not lines:
            continue
        first = lines[0].strip()
        if _is_heading(first):
            active_heading = first.lstrip("#").strip().rstrip(":")[:255]
            blocks.append({"text": clean, "type": "heading", "heading": active_heading})
        elif _is_table(lines):
            blocks.append({"text": "\n".join(lines), "type": "table", "heading": active_heading or ""})
        elif _is_list(lines):
            blocks.append({"text": "\n".join(lines), "type": "list", "heading": active_heading or ""})
        else:
            for sentence in _split_sentences(clean):
                blocks.append({"text": sentence, "type": "text", "heading": active_heading or ""})
    return blocks


def _split_large_block(block: dict[str, str], size: int) -> list[dict[str, str]]:
    text = block["text"]
    if len(text) <= size:
        return [block]
    lines = text.splitlines()
    if block["type"] in {"table", "list"}:
        pieces: list[dict[str, str]] = []
        current: list[str] = []
        for line in lines:
            if current and len("\n".join(current + [line])) > size:
                pieces.append({**block, "text": "\n".join(current)})
                current = []
            current.append(line)
        if current:
            pieces.append({**block, "text": "\n".join(current)})
        return pieces
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [{**block, "text": text[index:index + size]} for index in range(0, len(text), size)]
    pieces = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > size:
            pieces.append({**block, "text": current})
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append({**block, "text": current})
    return pieces


def _pack(blocks: list[dict[str, str]], size: int, overlap: int = 0) -> list[dict[str, str]]:
    packed: list[dict[str, str]] = []
    current: list[dict[str, str]] = []
    current_length = 0
    for block in blocks:
        for piece in _split_large_block(block, size):
            proposed = current_length + len(piece["text"]) + (1 if current else 0)
            if current and proposed > size:
                packed.append({
                    "text": "\n\n".join(item["text"] for item in current).strip(),
                    "type": "table" if any(item["type"] == "table" for item in current) else ("list" if any(item["type"] == "list" for item in current) else ("heading" if any(item["type"] == "heading" for item in current) else "text")),
                    "heading": next((item["heading"] for item in reversed(current) if item["heading"]), ""),
                })
                overlap_text = packed[-1]["text"][-overlap:] if overlap and packed[-1]["type"] == "text" else ""
                current = ([{"text": overlap_text, "type": "text", "heading": packed[-1]["heading"]}] if overlap_text else [])
                current_length = len(overlap_text)
            current.append(piece)
            current_length += len(piece["text"]) + (1 if current_length else 0)
    if current:
        packed.append({
            "text": "\n\n".join(item["text"] for item in current).strip(),
            "type": "table" if any(item["type"] == "table" for item in current) else ("list" if any(item["type"] == "list" for item in current) else ("heading" if any(item["type"] == "heading" for item in current) else "text")),
            "heading": next((item["heading"] for item in reversed(current) if item["heading"]), ""),
        })
    return [item for item in packed if item["text"]]


def sliding_chunks(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    """Return retrieval chunks while preserving logical table/list blocks."""
    return [item["text"] for item in _pack(_blocks(text), size=size, overlap=overlap)]


def create_parent_child_chunks(text: str, heading: str | None = None) -> list[dict[str, Any]]:
    """Create structure-aware parent chunks and smaller retrieval children.

    The CHILD size is bounded by the embedding model's context window, not by taste.
    `_pack` measures characters; the model measures tokens, and local_onnx calls
    `tokenizer.enable_truncation(max_length=EMBEDDING_MAX_TOKENS)` — so past the window
    the tail is dropped silently, with no error and no log line. MiniLM-L12-v2 allows
    128 tokens. At the previous 500 the tail of most chunks was embedded as nothing at
    all, worst in Vietnamese, where diacritics cost extra tokens.

    THE BUDGET IS `size + overlap`, NOT `size`. _pack flushes BEFORE the piece that would
    overflow, then seeds the next chunk with the last `overlap` characters of the one it
    just closed and appends that piece regardless — so a chunk reaches size + overlap + 2
    (the "

" join). Setting size alone and ignoring overlap still overflows: at
    size=300/overlap=100 this emitted 401 characters, measured.

    250 + 60 + 2 = 312 characters, about 125 tokens at a deliberately pessimistic 2.5
    characters per token for Vietnamese, which fits 128 with the two special tokens.

    Small children are the DESIGN, not a concession: the child exists to be matched
    precisely, and the 1800-character parent above is what supplies context afterwards.
    Raise these only alongside a model with a longer window — bge-m3 allows 8192.
    """
    parents = _pack(_blocks(text, heading=heading), size=1800, overlap=250)
    return [
        {
            "parent_index": index,
            "parent_text": parent["text"],
            "chunk_type": parent["type"],
            "heading": parent["heading"] or heading or None,
            "children": sliding_chunks(parent["text"], size=250, overlap=60),
        }
        for index, parent in enumerate(parents)
    ]
