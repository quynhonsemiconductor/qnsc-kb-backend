"""Small deterministic reranker used when a cross-encoder is unavailable."""
from __future__ import annotations

import re
from typing import NamedTuple, Sequence


# These words add little retrieval signal. Keeping them out of lexical
# coverage prevents a query such as "What is CTS?" from ranking generic text
# containing "what/is" above the passage containing the important term CTS.
STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "can", "do", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
    "là", "và", "có", "cho", "của", "để", "gì", "nào", "như", "về", "tôi",
}

REFERENCE_MARKERS = (
    "references", "reference", "helpful documents", "sources", "bibliography",
    "tài liệu tham khảo", "nguồn tham khảo", "http://", "https://", "www.",
)

DEFINITION_QUERY_MARKERS = (
    "what is", "what are", "define", "definition", "meaning", "explain",
    "là gì", "định nghĩa", "có nghĩa là", "giải thích", "khái niệm",
)

DEFINITION_PATTERNS = (
    r"\b(?:is|are|means|refers to|defined as|describes)\b",
    r"\b(?:là|được gọi là|có nghĩa là|dùng để chỉ|được định nghĩa là)\b",
)


# Compiled once at import. These run against every candidate passage on every search.
DEFINITION_PATTERNS_RE = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in DEFINITION_PATTERNS
)
URL_RE = re.compile(r"https?://|www\.")
TOKEN_RE = re.compile(r"[\w'-]+")


def normalize_query(query: str) -> str:
    """Remove low-signal question words before keyword/vector retrieval."""
    tokens = [
        token for token in re.findall(r"[\w'-]+", (query or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]
    # An all-stopword input has no retrieval signal. Returning the original
    # query here caused generic words such as "what is" to retrieve arbitrary
    # documents through vector similarity.
    return " ".join(tokens)


def is_definition_query(query: str) -> bool:
    normalized = " ".join((query or "").lower().split())
    return any(marker in normalized for marker in DEFINITION_QUERY_MARKERS)


class PreparedQuery(NamedTuple):
    """The query-side half of a score, computed once instead of once per candidate.

    Scoring runs for every candidate in the pool against the SAME query, and all of this
    was recomputed each time: the query was tokenised three times over (in
    normalize_query, again for `terms`, and again to rebuild a string that was already
    exactly `normalized`), plus a separate lowercase pass for the definition check.
    """

    normalized: str
    terms: frozenset[str]
    is_definition: bool


def prepare_query(query: str) -> PreparedQuery:
    normalized = normalize_query(query)
    return PreparedQuery(
        normalized=normalized,
        # normalize_query already lower-cased and already split on this exact pattern,
        # so its output re-tokenises to itself.
        terms=frozenset(normalized.split()),
        is_definition=is_definition_query(query),
    )


def score_prepared_text(
    prepared: PreparedQuery, text: str, title: str = "", section: str = ""
) -> float:
    """Score a passage against an already-prepared query."""
    terms = prepared.terms
    passage = " ".join(str(value or "") for value in (text, title, section)).lower()
    text_tokens = TOKEN_RE.findall(passage)
    normalized_text = " ".join(text_tokens)
    # A whole-token set intersection. The old form ran one anchored regex per term over
    # the whole passage; because `normalized_text` is tokens joined by single spaces and
    # every token character is itself in the token class, those lookarounds could only
    # ever match a complete token — so this is the same predicate without the scan.
    matched = len(terms & set(text_tokens))
    score = matched / max(len(terms), 1)
    if len(prepared.normalized) > 2 and prepared.normalized in normalized_text:
        score += 0.35

    if prepared.is_definition:
        if any(pattern.search(passage) for pattern in DEFINITION_PATTERNS_RE):
            score += 0.75
        if any(marker in passage for marker in REFERENCE_MARKERS):
            score -= 1.0
        # A passage dominated by URLs is reference material, even when it
        # repeats the subject name many times.
        if len(URL_RE.findall(passage)) >= 2:
            score -= 0.5
    elif any(marker in passage for marker in REFERENCE_MARKERS):
        score -= 0.25
    return score


def score_retrieval_text(query: str, text: str, title: str = "", section: str = "") -> float:
    """Score how well a passage answers the query, not just contains its terms."""
    return score_prepared_text(prepare_query(query), text, title, section)


def retrieval_score(query: str, chunk: object, prepared: PreparedQuery | None = None) -> float:
    """Score one retrieved chunk. THE definition of a chunk's relevance.

    Extracted because three call sites — the reranker, the relevance threshold, and the
    score reported in the response — each rebuilt these arguments by hand and each paid
    for the scoring again. `score_retrieval_text` costs ~0.9 ms, so a search scored every
    chunk three times over.
    """
    parent = getattr(chunk, "parent_chunk", None)
    # The child passage is the precise retrieval unit. Parent text is only
    # a fallback because it often contains repeated slides/references.
    text = getattr(chunk, "chunk_text", "") or getattr(parent, "text", "")
    return score_prepared_text(
        prepared if prepared is not None else prepare_query(query),
        text,
        getattr(getattr(chunk, "article", None), "title", ""),
        getattr(parent, "section_ref", "") if parent else "",
    )


def rerank_chunks_with_scores(
    query: str, chunks: Sequence[object], limit: int = 5
) -> list[tuple[object, float]]:
    """Rerank and hand back the scores, so no caller has to recompute them."""
    prepared = prepare_query(query)
    scored: list[tuple[float, int, object]] = []
    for position, chunk in enumerate(chunks):
        scored.append((retrieval_score(query, chunk, prepared), -position, chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(item[2], item[0]) for item in scored[:limit]]


def rerank_chunks(query: str, chunks: Sequence[object], limit: int = 5) -> list[object]:
    return [chunk for chunk, _score in rerank_chunks_with_scores(query, chunks, limit)]
