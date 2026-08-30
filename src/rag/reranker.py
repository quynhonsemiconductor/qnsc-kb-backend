"""Small deterministic reranker used when a cross-encoder is unavailable."""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple, Sequence


# Vietnamese is routinely typed without diacritics — "CTS la gi" for "CTS là gì" — and
# every lexical comparison below has to survive that. Folding is done with a translation
# table built once at import: one pass per string, no per-call NFD allocation, on a path
# that runs for every candidate in the pool on every search.
#
# `đ`/`Đ` are handled explicitly because they are NOT decomposable — NFD leaves them
# whole, so stripping combining marks alone would fold "được" to "đuoc" and never match
# a user's "duoc". PostgreSQL's unaccent() maps them to `d`, which migration 58 wired
# into both the FTS index and its query, so matching that behaviour here keeps the
# lexical and reranking legs in agreement.
def _build_fold_table() -> dict[int, str]:
    table = {ord("đ"): "d", ord("Đ"): "d"}
    for codepoint in range(0x00C0, 0x1EFA):
        char = chr(codepoint)
        decomposed = unicodedata.normalize("NFD", char)
        stripped = "".join(
            part for part in decomposed if unicodedata.category(part) != "Mn"
        )
        if stripped and stripped != char:
            table[codepoint] = stripped.lower()
    return table


_FOLD_TABLE = _build_fold_table()


def fold_diacritics(value: str) -> str:
    """Lowercase and strip Vietnamese/Latin diacritics, as PostgreSQL unaccent() does."""
    return (value or "").lower().translate(_FOLD_TABLE)


# These words add little retrieval signal. Keeping them out of lexical
# coverage prevents a query such as "What is CTS?" from ranking generic text
# containing "what/is" above the passage containing the important term CTS.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "give", "how", "i", "in", "is", "it", "list", "me",
    "my", "need", "of", "on", "or", "please", "provide", "show", "so", "some",
    "tell", "that", "the", "them", "there", "these", "this", "those", "to",
    "us", "use", "used", "using", "want", "we", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
    # Vietnamese. The list here used to hold eleven words and none of the ones a
    # person actually opens a request with, which is why a politely phrased question
    # scored five times lower than the same question typed as two keywords.
    "à", "ạ", "ai", "bạn", "bằng", "các", "cách", "cần", "cho", "chúng", "có",
    "của", "cung", "cấp", "danh", "dùng", "dụng", "gì", "giúp", "hãy", "khi",
    "không", "là", "làm", "liệt", "kê", "một", "muốn", "nào", "này", "nêu",
    "như", "những", "ở", "ra", "rằng", "sách", "sao", "sẽ", "sử", "thì",
    "tôi", "trong", "và", "vậy", "về", "với", "được", "đó", "để", "đưa",
}

# What the code actually tests against. Derived, not hand-maintained: a second
# hand-written list of unaccented forms would drift from the one above on the first edit.
STOPWORDS_FOLDED = frozenset(fold_diacritics(word) for word in STOPWORDS) | frozenset(
    STOPWORDS
)

REFERENCE_MARKERS = (
    "references", "reference", "helpful documents", "sources", "bibliography",
    "tài liệu tham khảo", "nguồn tham khảo", "http://", "https://", "www.",
)

DEFINITION_QUERY_MARKERS = (
    "what is", "what are", "define", "definition", "meaning", "explain",
    "là gì", "định nghĩa", "có nghĩa là", "giải thích", "khái niệm",
)

# Matched against folded text, so the markers must be folded too.
REFERENCE_MARKERS_FOLDED = tuple(
    fold_diacritics(marker) for marker in REFERENCE_MARKERS
)
DEFINITION_QUERY_MARKERS_FOLDED = tuple(
    fold_diacritics(marker) for marker in DEFINITION_QUERY_MARKERS
)

DEFINITION_PATTERNS = (
    r"\b(?:is|are|means|refers to|defined as|describes)\b",
    r"\b(?:la|duoc goi la|co nghia la|dung de chi|duoc dinh nghia la)\b",
)


# Compiled once at import. These run against every candidate passage on every search.
# The Vietnamese alternatives are written pre-folded because the passage they are
# matched against has been folded.
DEFINITION_PATTERNS_RE = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in DEFINITION_PATTERNS
)
URL_RE = re.compile(r"https?://|www\.")
TOKEN_RE = re.compile(r"[\w'-]+")


def normalize_query(query: str) -> str:
    """Remove low-signal question words before keyword/vector retrieval.

    Folded first, so "cung cap cho toi" is recognised as the same set of stopwords as
    "cung cấp cho tôi". Untyped diacritics used to leave every one of those words in the
    scored term set, where they could only ever sit in the denominator.
    """
    tokens = [
        token for token in TOKEN_RE.findall(fold_diacritics(query))
        if len(token) > 1 and token not in STOPWORDS_FOLDED
    ]
    # An all-stopword input has no retrieval signal. Returning the original
    # query here caused generic words such as "what is" to retrieve arbitrary
    # documents through vector similarity.
    return " ".join(tokens)


def is_definition_query(query: str) -> bool:
    """Whether the user asked what something IS.

    Folded, because "CTS la gi" is the same question as "CTS là gì" and used to miss the
    marker list entirely — losing the definition bonus on exactly the queries it exists
    to serve.
    """
    normalized = " ".join(fold_diacritics(query).split())
    return any(marker in normalized for marker in DEFINITION_QUERY_MARKERS_FOLDED)


def corpus_terms(passages: Sequence[str]) -> frozenset[str]:
    """Every token that appears anywhere in a candidate pool.

    Folded to the same form as the query terms it is intersected with; otherwise an
    accented passage token could never cancel an unaccented query term.
    """
    found: set[str] = set()
    for passage in passages:
        found.update(TOKEN_RE.findall(fold_diacritics(passage)))
    return frozenset(found)


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


def prepare_query(query: str, findable: frozenset[str] | None = None) -> PreparedQuery:
    """Prepare the query side of a score.

    `findable` is every token present somewhere in the candidate pool. Query terms that
    appear in NO candidate are dropped from the scored set, because the score is
    `matched / len(terms)` and those terms can only ever sit in the denominator.

    That denominator was the whole query, so relevance shrank with the length of the
    question rather than with anything about the passage. Against the same text:

        "RTL Generator"                                    2 terms -> 1.350  answered
        "Vậy cung cấp cho tôi các Tool dùng RTL Generator"  8 terms -> 0.250  refused

    Both matched the same two terms. The second was refused for being politely phrased,
    and the effect is far worse in Vietnamese, where a request opens with several words
    that no English technical document will ever contain.

    Ranking is unaffected -- the denominator is constant across candidates for one query
    -- so this only changes the absolute value, which is what the confidence thresholds
    read. If nothing at all is findable the full set is kept, so a query about content
    that genuinely is not there still scores zero and is still refused.
    """
    normalized = normalize_query(query)
    # normalize_query already lower-cased and already split on this exact pattern,
    # so its output re-tokenises to itself.
    terms = frozenset(normalized.split())
    if findable is not None:
        scored_terms = terms & findable
        if scored_terms:
            terms = scored_terms
    return PreparedQuery(
        normalized=normalized,
        terms=terms,
        is_definition=is_definition_query(query),
    )


def score_prepared_text(
    prepared: PreparedQuery, text: str, title: str = "", section: str = ""
) -> float:
    """Score a passage against an already-prepared query."""
    terms = prepared.terms
    # Folded, not merely lower-cased: the query terms are folded, so an accented passage
    # token would otherwise never match the unaccented term a user actually typed.
    passage = fold_diacritics(" ".join(str(value or "") for value in (text, title, section)))
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
        if any(marker in passage for marker in REFERENCE_MARKERS_FOLDED):
            score -= 1.0
        # A passage dominated by URLs is reference material, even when it
        # repeats the subject name many times.
        if len(URL_RE.findall(passage)) >= 2:
            score -= 0.5
    elif any(marker in passage for marker in REFERENCE_MARKERS_FOLDED):
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


def chunk_passage(chunk: object) -> str:
    """The text a chunk is scored on. One definition, used by scoring and by the pool."""
    parent = getattr(chunk, "parent_chunk", None)
    return " ".join(
        str(value or "")
        for value in (
            getattr(chunk, "chunk_text", "") or getattr(parent, "text", ""),
            getattr(getattr(chunk, "article", None), "title", ""),
            getattr(parent, "section_ref", "") if parent else "",
        )
    )


def prepare_query_for_chunks(query: str, chunks: Sequence[object]) -> PreparedQuery:
    """Prepare a query against the pool it will be scored over."""
    return prepare_query(query, corpus_terms([chunk_passage(chunk) for chunk in chunks]))


def rerank_chunks_with_scores(
    query: str, chunks: Sequence[object], limit: int = 5
) -> list[tuple[object, float]]:
    """Rerank and hand back the scores, so no caller has to recompute them."""
    prepared = prepare_query_for_chunks(query, chunks)
    scored: list[tuple[float, int, object]] = []
    for position, chunk in enumerate(chunks):
        scored.append((retrieval_score(query, chunk, prepared), -position, chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(item[2], item[0]) for item in scored[:limit]]


def rerank_chunks(query: str, chunks: Sequence[object], limit: int = 5) -> list[object]:
    return [chunk for chunk, _score in rerank_chunks_with_scores(query, chunks, limit)]
