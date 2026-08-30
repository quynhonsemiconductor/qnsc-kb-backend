"""Asking politely must not be why the knowledge base refuses to answer.

Reported: the same content, asked two ways.

    "RTL Generator"                                    answered, 7 sources
    "Vậy cung cấp cho tôi các Tool dùng RTL Generator"  "Tôi không tìm thấy đủ thông tin..."

Relevance was `matched / len(terms)` over EVERY term in the query, so the denominator was
the length of the question. Against one passage that measured 1.350 and 0.250: both
matched the same two terms, and the second fell under RAG_MIN_CONTEXT_SCORE (0.35) purely
for having six more words around them.

It was far worse in Vietnamese. The stopword list held eleven Vietnamese words and none
of the ones a person actually opens a request with -- vậy, các, cung cấp, dùng, hãy,
giúp, muốn -- so each one became denominator mass that no English technical document
could ever match.

Two changes. The stopword list now covers how people actually ask, and query terms that
appear in no candidate at all are dropped from the scored set rather than left to sit in
the denominator.

The guard this scoring exists for is unchanged and asserted below: a query about content
that genuinely is not there still scores zero and is still refused.
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from src.rag.reranker import (
    corpus_terms,
    normalize_query,
    prepare_query,
    score_prepared_text,
)

PASSAGE = (
    "RTL Generator. PeakRDL + PeakRDL-regblock: SystemRDL/IP-XACT to synthesizable "
    "SystemVerilog CSR, C header, UVM RAL, HTML. RgGen: CSR generator supporting "
    "SV/Verilog, UVM RAL, C header, APB/AXI-Lite/Wishbone."
)
TITLE = "RTL Generator"

TERSE = "RTL Generator"
POLITE = "Vậy cung cấp cho tôi các Tool dùng RTL Generator"


def _score(query: str, passage: str = PASSAGE, title: str = TITLE) -> float:
    return score_prepared_text(
        prepare_query(query, corpus_terms([f"{passage} {title}"])), passage, title
    )


def test_the_reported_question_is_now_answered():
    """The exact pair from the report."""
    assert _score(POLITE) >= settings.RAG_MIN_CONTEXT_SCORE


def test_both_phrasings_reach_the_same_conclusion():
    """They matched the same terms in the same passage, so they must not disagree about
    whether the knowledge base can answer."""
    terse, polite = _score(TERSE), _score(POLITE)
    passes = settings.RAG_MIN_CONTEXT_SCORE
    assert (terse >= passes) == (polite >= passes)


def test_padding_a_question_no_longer_collapses_its_score():
    """Before: 1.350 -> 0.250, a 5x drop from six extra words."""
    assert _score(POLITE) >= _score(TERSE) * 0.6


@pytest.mark.parametrize(
    "word", ["vậy", "các", "cung", "cấp", "dùng", "hãy", "giúp", "muốn", "những", "với"]
)
def test_ordinary_vietnamese_request_words_carry_no_weight(word):
    """These are how a request opens. None of them were filtered."""
    assert word not in normalize_query(f"{word} RTL Generator").split()


@pytest.mark.parametrize("word", ["please", "provide", "show", "list", "using", "need"])
def test_ordinary_english_request_words_carry_no_weight(word):
    assert word not in normalize_query(f"{word} RTL Generator").split()


def test_a_term_no_document_contains_is_not_counted_against_the_passage():
    """It can only ever sit in the denominator: nothing can match it, so keeping it just
    scales relevance down by how much the asker said."""
    pool = corpus_terms([PASSAGE])
    with_extra = prepare_query("RTL Generator pricing licence roadmap", pool)
    assert with_extra.terms == frozenset({"rtl", "generator"})


def test_a_question_about_absent_content_is_still_refused():
    """The guard this scoring exists for. If nothing in the query is findable the full
    term set is kept, so the score stays at zero rather than dividing by nothing."""
    pool = corpus_terms([PASSAGE])
    prepared = prepare_query("chính sách nghỉ phép thai sản", pool)

    assert prepared.terms, "an unmatchable query must keep its terms, not become empty"
    assert score_prepared_text(prepared, PASSAGE, TITLE) < settings.RAG_MIN_CONTEXT_SCORE


def test_a_partly_answerable_question_is_scored_on_the_part_that_exists():
    """Asking about two things, one of which the corpus covers, should still answer
    about that one rather than refusing over the half that is missing."""
    assert _score("RTL Generator và chính sách nghỉ phép") >= settings.RAG_MIN_CONTEXT_SCORE


def test_ranking_is_unchanged_by_verbosity():
    """The denominator is constant across candidates for one query, so this only moves
    the absolute value -- which is what the thresholds read."""
    relevant = PASSAGE
    unrelated = "Quy trình xin nghỉ phép và bảng lương nhân viên."
    pool = corpus_terms([relevant, unrelated])

    for query in (TERSE, POLITE):
        prepared = prepare_query(query, pool)
        assert score_prepared_text(prepared, relevant, TITLE) > score_prepared_text(
            prepared, unrelated
        )


def test_scoring_without_a_pool_still_works():
    """prepare_query is called with one argument in older paths and in tests."""
    assert prepare_query(TERSE).terms == frozenset({"rtl", "generator"})
