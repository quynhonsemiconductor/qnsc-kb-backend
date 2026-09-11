"""Query-complexity routing: deciding when one hybrid_search pass is not enough.

A comparison question ("so sánh SOP A và SOP B") retrieves worse than two separate
questions would, because the single embedding for the whole sentence sits somewhere
between the two topics instead of close to either one -- the same reason a centroid
embedding in `article_linking.py` represents a whole article rather than any one
paragraph well. `search_service.py` uses this to run one hybrid_search per detected
subject instead of one for the whole query, but only when comparison intent is actually
signalled: splitting on "and" for every query would fragment plenty of ordinary
questions ("quy trình đăng ký và phê duyệt") that were never asking to compare two things.
"""
from __future__ import annotations

import re

from src.rag.reranker import fold_diacritics

# Matched against folded text. Deliberately narrow: each phrase names an actual
# comparison, not just conjunction ("và" alone is far too common to use as a signal on
# its own -- see the module docstring).
COMPARISON_QUERY_MARKERS = (
    "so sanh", "khac nhau", "khac biet", "hay la", "nen chon",
    "compare", "comparison", "difference between", "differences between",
    "versus",
)

COMPARISON_QUERY_MARKERS_FOLDED = tuple(
    fold_diacritics(marker) for marker in COMPARISON_QUERY_MARKERS
)

#: A bare "vs"/"vs." needs word boundaries -- unlike the phrases above, "vs" alone
#: appears inside ordinary words in some encodings/typos, so it is checked separately
#: with \b rather than folded into the plain substring list.
_VS_RE = re.compile(r"\bvs\.?\b", re.IGNORECASE)

#: Conjunctions a comparison question typically lists its subjects with. Checked only
#: once comparison intent is already established by a marker above. "và" is matched as
#: the accented word directly -- unlike the marker list, this runs against the ORIGINAL
#: query (so the returned subjects keep their diacritics), and the plain-ASCII "va" a
#: diacritic-free query would use is covered by its own alternative rather than folding
#: the whole query and losing every accent in the parts being returned.
_SPLIT_RE = re.compile(r"\s+(?:và|va|and|hay|or|vs\.?|versus)\s+", re.IGNORECASE)

MAX_SUBJECTS = 3
MIN_SUBJECT_LENGTH = 2


def is_comparison_query(query: str) -> bool:
    """Whether `query` is asking to compare two or more things, not just naming them.

    Deliberately conservative: a false negative costs a bit of retrieval quality on one
    query (falls back to ordinary single-pass search), but a false positive fragments an
    ordinary question into nonsense sub-queries. Erring toward missing some comparison
    questions is the safer failure mode.
    """
    normalized = fold_diacritics(query or "")
    if any(marker in normalized for marker in COMPARISON_QUERY_MARKERS_FOLDED):
        return True
    return bool(_VS_RE.search(query or ""))


#: How close to the top score a result has to be to count as "also a leading answer"
#: rather than a distant runner-up. Reuses the idea behind similarity.py's own
#: MATCH_THRESHOLD -- a relative band, not an absolute score, because the meaning of a
#: given absolute score already depends on the reranker in front of it.
AMBIGUITY_SCORE_MARGIN = 0.85
MIN_RESULTS_FOR_AMBIGUITY_CHECK = 2


def detect_ambiguous_departments(results: list[dict], *, top_n: int = 5) -> list[str] | None:
    """Distinct departments among the near-top results, if more than one -- else None.

    Used to decide whether a question should be answered directly or asked back as a
    clarification ("did you mean the Finance or the HR policy?"): when the best-scoring
    results are split across departments with no clear single winner, picking one to
    answer from is a guess the reader did not ask for. A single department among the
    near-top results, or too few results to judge, both return None -- ambiguity
    detection needs something to be ambiguous BETWEEN.
    """
    scored = sorted(
        (item for item in results[:top_n] if item.get("dept")),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    if len(scored) < MIN_RESULTS_FOR_AMBIGUITY_CHECK:
        return None
    top_score = float(scored[0].get("score") or 0.0)
    if top_score <= 0:
        return None
    near_top = [
        item
        for item in scored
        if float(item.get("score") or 0.0) >= top_score * AMBIGUITY_SCORE_MARGIN
    ]
    # Order of first appearance, not alphabetical: the highest-scoring department's
    # name is offered first, matching how a reader would expect "most likely" to lead.
    departments: list[str] = []
    for item in near_top:
        name = str(item["dept"])
        if name not in departments:
            departments.append(name)
    return departments if len(departments) > 1 else None


def split_comparison_subjects(query: str) -> list[str]:
    """Split a comparison query into its distinct subjects, best-effort.

    Only meaningful to call after `is_comparison_query` returns True. Returns at most
    `MAX_SUBJECTS` non-trivial parts; if splitting does not actually separate the query
    into multiple usable pieces (no recognised conjunction, or a part too short to be a
    real subject), returns a single-item list containing the original query unchanged --
    the caller should treat that as "could not decompose" and fall back to one search.
    """
    parts = [part.strip() for part in _SPLIT_RE.split(query or "") if part.strip()]
    usable = [part for part in parts if len(part) >= MIN_SUBJECT_LENGTH]
    if len(usable) < 2:
        return [query.strip()] if query and query.strip() else []
    return usable[:MAX_SUBJECTS]
