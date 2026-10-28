"""Collapse a passage that several documents repeat into one result.

Companies copy the same clause into many files. Searched as-is, one passage then takes ten of
the slots meant for ten different passages, and the answer sees the same text ten times.

Two passages are the same when their text is equal after folding case,
punctuation and whitespace (exact), or when their embeddings are almost identical (near). Only
passages from DIFFERENT articles are merged: a repeat inside one article is that article's own
structure, not a copy. The result that stays is the best one, and it remembers where else the
passage appears, so the citation can still name every source.

Which copy stays: the higher score, and between near-equal scores the more recently reviewed,
then the more recently updated article. Everything here only removes redundancy from a list the
caller already had the right to see; it never adds a passage.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any, Sequence

import numpy as np

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)

#: Scores closer than this are a tie, so recency decides. Matches the 2-decimal precision the
#: ranking scores are compared at elsewhere.
_SCORE_TIE = 2


def passage_key(text: str) -> str:
    """The text with case, punctuation and spacing folded away."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _NON_WORD.sub(" ", folded).strip()


def _timestamp(value: Any) -> float:
    return value.timestamp() if isinstance(value, datetime) else 0.0


def _preference(chunk: Any, score: float) -> tuple[float, float, float]:
    article = getattr(chunk, "article", None)
    return (
        round(float(score), _SCORE_TIE),
        _timestamp(getattr(article, "last_reviewed", None)),
        _timestamp(getattr(article, "updated_at", None)),
    )


def _vector(chunk: Any) -> np.ndarray | None:
    raw = getattr(chunk, "embedding", None)
    if raw is None:
        return None
    try:
        vector = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def _source(chunk: Any) -> dict[str, str | None]:
    article = chunk.article
    return {
        "chunk_id": str(chunk.id),
        "article_id": str(article.id),
        "title": article.title,
        "dept": article.dept,
    }


def collapse_duplicates(
    ranked: Sequence[tuple[Any, float]],
    *,
    similarity: float,
    min_chars: int,
) -> tuple[list[tuple[Any, float]], dict[str, list[dict[str, str | None]]]]:
    """Return `ranked` without cross-article copies, and where each kept chunk also appears.

    Order is the order of `ranked`: a copy that wins takes the place of the one it replaces.
    `similarity` is a cosine in (0, 1]; a value above 1 turns near matching off and leaves the
    exact match. Passages shorter than `min_chars` (headings, one-line rows) are never merged:
    short text repeats by coincidence.
    """
    kept: list[tuple[Any, float]] = []
    also_in: dict[str, list[dict[str, str | None]]] = {}
    by_key: dict[str, int] = {}
    matrix: np.ndarray | None = None
    rows: list[int] = []  # position in `kept` of each row of `matrix`
    used = 0

    for chunk, score in ranked:
        article = getattr(chunk, "article", None)
        text = getattr(chunk, "chunk_text", "") or ""
        # Evidence chunks (figures, tables) are not text; only plain text is compared.
        eligible = (
            article is not None
            and getattr(chunk, "evidence_id", None) is None
            and len(text.strip()) >= min_chars
        )
        match: int | None = None
        key = ""
        vector: np.ndarray | None = None
        if eligible:
            key = passage_key(text)
            candidate = by_key.get(key)
            if candidate is not None and kept[candidate][0].article.id != article.id:
                match = candidate
            elif similarity <= 1.0:
                vector = _vector(chunk)
                if vector is not None and matrix is not None and used:
                    if matrix.shape[1] == vector.size:
                        sims = matrix[:used] @ vector
                        for position in np.argsort(-sims):
                            if float(sims[position]) < similarity:
                                break
                            row_index = rows[int(position)]
                            if kept[row_index][0].article.id != article.id:
                                match = row_index
                                break

        if match is None:
            kept.append((chunk, score))
            index = len(kept) - 1
            if eligible:
                by_key.setdefault(key, index)
                if vector is None and similarity <= 1.0:
                    vector = _vector(chunk)
                if vector is not None:
                    if matrix is None:
                        matrix = np.zeros((len(ranked), vector.size), dtype=np.float32)
                    if matrix.shape[1] == vector.size:
                        matrix[used] = vector
                        rows.append(index)
                        used += 1
            continue

        current, current_score = kept[match]
        sources = also_in.setdefault(str(current.id), [])
        if _preference(chunk, score) > _preference(current, current_score):
            # The newcomer is the better copy: it takes the slot, the old one is listed.
            carried = also_in.pop(str(current.id), [])
            kept[match] = (chunk, score)
            also_in[str(chunk.id)] = carried + [_source(current)]
            # The slot's vector now stands for the new chunk; refresh it for later matches.
            new_vector = _vector(chunk)
            if new_vector is not None and matrix is not None and match in rows:
                if matrix.shape[1] == new_vector.size:
                    matrix[rows.index(match)] = new_vector
        else:
            sources.append(_source(chunk))

    # One entry per distinct article, and none for the kept chunk's own article.
    cleaned: dict[str, list[dict[str, str | None]]] = {}
    for chunk, _score in kept:
        entries = also_in.get(str(chunk.id)) or []
        seen = {str(chunk.article.id)}
        unique: list[dict[str, str | None]] = []
        for entry in entries:
            if entry["article_id"] in seen:
                continue
            seen.add(str(entry["article_id"]))
            unique.append(entry)
        if unique:
            cleaned[str(chunk.id)] = unique
    return kept, cleaned
