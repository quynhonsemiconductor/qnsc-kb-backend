"""Vietnamese typed without diacritics must score the same as typed with them.

Vietnamese input routinely arrives unaccented — "CTS la gi" for "CTS là gì" — and the
reranker's STOPWORDS list held only the accented forms. That cost the same question twice
over, MEASURED before this fix:

    'CTS la gi'   score=0.333   is_definition_query=False
    'CTS là gì'   score=2.100   is_definition_query=True

`la` and `gi` were not recognised as stopwords, so they stayed in the scored term set where
they could only sit in the denominator; and `is_definition_query` missed its marker list,
losing the +0.75 definition bonus. A 6.3x penalty for omitting diacritics.

That matters because RAG_MIN_RELEVANCE_SCORE is an ABSOLUTE floor (0.12): the penalty
pushes correct-but-borderline passages under it, and the system then refuses to answer
rather than returning a weak hit.

The repo already folds accents on the lexical leg — migration 20260816_58 wraps both the
FTS index and its query in immutable_unaccent — so this brings the reranker into agreement
with retrieval rather than inventing a convention.
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from src.rag.reranker import (
    DEFINITION_QUERY_MARKERS,
    REFERENCE_MARKERS,
    STOPWORDS,
    STOPWORDS_FOLDED,
    fold_diacritics,
    is_definition_query,
    normalize_query,
    score_retrieval_text,
)

CTS_PASSAGE = (
    "Clock Tree Synthesis (CTS) is the process of distributing the clock signal to all "
    "sequential elements in the design, balancing skew and insertion delay."
)

#: (unaccented, accented) forms of the same real query.
EQUIVALENT_QUERIES = [
    ("CTS la gi", "CTS là gì"),
    ("dinh nghia cua clock tree synthesis", "định nghĩa của clock tree synthesis"),
    ("giai thich skew", "giải thích skew"),
    (
        "cung cap cho toi cac tool dung RTL Generator",
        "cung cấp cho tôi các tool dùng RTL Generator",
    ),
]


@pytest.mark.parametrize("plain,accented", EQUIVALENT_QUERIES)
def test_the_same_question_scores_the_same_with_or_without_diacritics(plain, accented):
    assert score_retrieval_text(plain, CTS_PASSAGE) == pytest.approx(
        score_retrieval_text(accented, CTS_PASSAGE)
    )


@pytest.mark.parametrize("plain,accented", EQUIVALENT_QUERIES)
def test_retrieval_terms_are_identical_with_or_without_diacritics(plain, accented):
    """Both forms must reach the vector and keyword legs as the same query."""
    assert normalize_query(plain) == normalize_query(accented)


def test_an_unaccented_definition_question_is_still_a_definition_question():
    """This is the half that cost 0.75 outright, not just a denominator."""
    assert is_definition_query("CTS la gi")
    assert is_definition_query("dinh nghia cua CTS")
    assert is_definition_query("giai thich CTS")


def test_the_unaccented_definition_question_clears_the_refusal_floor():
    """The regression that made the system refuse rather than answer weakly."""
    score = score_retrieval_text("CTS la gi", CTS_PASSAGE)

    assert score > settings.RAG_MIN_RELEVANCE_SCORE
    # It used to score 0.333 against an accented 2.100.
    assert score == pytest.approx(score_retrieval_text("CTS là gì", CTS_PASSAGE))


def test_unaccented_vietnamese_stopwords_are_dropped_from_the_scored_terms():
    """A politely phrased unaccented request must not be diluted by its own politeness."""
    assert normalize_query("cung cap cho toi cac tool") == "tool"


def test_the_folded_stopword_set_is_derived_from_the_accented_one():
    """Two hand-maintained lists would drift apart on the first edit."""
    for word in STOPWORDS:
        assert fold_diacritics(word) in STOPWORDS_FOLDED


def test_folding_matches_postgres_unaccent_on_the_vietnamese_d_stroke():
    """`đ` is NOT decomposable, so stripping combining marks alone leaves it behind.

    PostgreSQL's unaccent() maps it to `d`, and migration 58 put unaccent on both sides
    of the FTS leg. Folding "được" to "đuoc" here would silently disagree with retrieval.
    Verified against a live pgvector 0.8.6 server: unaccent('được') = 'duoc',
    unaccent('Để') = 'De', unaccent('định nghĩa') = 'dinh nghia'.
    """
    assert fold_diacritics("được") == "duoc"
    assert fold_diacritics("Để") == "de"
    assert fold_diacritics("định nghĩa") == "dinh nghia"
    assert fold_diacritics("máy tính") == "may tinh"


def test_folding_leaves_technical_identifiers_alone():
    """Tool names and acronyms are the terms that actually carry retrieval signal."""
    assert fold_diacritics("CTS") == "cts"
    assert fold_diacritics("-max_delay") == "-max_delay"
    assert fold_diacritics("Innovus 21.1") == "innovus 21.1"


def test_the_marker_lists_used_at_match_time_are_folded():
    """Markers are compared against a folded passage, so unfolded ones could never hit."""
    from src.rag.reranker import (
        DEFINITION_QUERY_MARKERS_FOLDED,
        REFERENCE_MARKERS_FOLDED,
    )

    assert len(DEFINITION_QUERY_MARKERS_FOLDED) == len(DEFINITION_QUERY_MARKERS)
    assert len(REFERENCE_MARKERS_FOLDED) == len(REFERENCE_MARKERS)
    for marker in DEFINITION_QUERY_MARKERS_FOLDED + REFERENCE_MARKERS_FOLDED:
        assert marker == fold_diacritics(marker)


def test_an_unaccented_reference_section_is_still_penalised():
    """The penalty side has to fold too, or reference pages outrank real answers."""
    references = "Tai lieu tham khao: see the vendor manual for CTS details."

    assert score_retrieval_text("CTS la gi", references) < score_retrieval_text(
        "CTS la gi", CTS_PASSAGE
    )


def test_a_genuinely_irrelevant_passage_still_scores_zero():
    """Folding must not manufacture matches that are not there."""
    unrelated = "The cafeteria serves lunch between 11:30 and 13:00 on weekdays."

    assert score_retrieval_text("CTS la gi", unrelated) <= 0.0
