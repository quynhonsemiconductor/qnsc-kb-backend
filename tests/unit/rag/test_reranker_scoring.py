"""Term matching is whole-token, and preparing a query must not change any score.

The per-term anchored regex scan was replaced by a set intersection over the passage's
tokens. That is only equivalent because a token consists entirely of token characters, so
an anchored match could never land inside one. These cases are where that reasoning would
break if it were wrong.
"""
from __future__ import annotations

from src.rag.reranker import (
    prepare_query,
    score_prepared_text,
    score_retrieval_text,
)


def test_a_term_does_not_match_part_of_a_longer_hyphenated_token():
    """Term coverage is whole-token: "abc" is not covered by "abc-def".

    Compared rather than asserted to be zero, because the separate phrase bonus IS a
    plain substring test and fires for both — it always has.
    """
    assert score_retrieval_text("abc", "abc ghi") > score_retrieval_text(
        "abc", "abc-def ghi"
    )


def test_a_hyphenated_term_matches_the_whole_token():
    assert score_retrieval_text("abc-def", "xyz abc-def ghi") > 0.0


def test_an_apostrophe_is_part_of_the_token():
    assert score_retrieval_text("o'clock", "meet at o'clock sharp") > score_retrieval_text(
        "clock", "meet at o'clock sharp"
    )


def test_punctuation_between_terms_does_not_prevent_a_match():
    assert score_retrieval_text("nghi phep", "quy dinh: nghi phep, hang nam.") > 0.0


def test_a_query_with_no_signal_scores_nothing_rather_than_dividing_by_zero():
    assert score_retrieval_text("la va co", "bat ky noi dung nao") == 0.0
    assert score_retrieval_text("", "bat ky noi dung nao") == 0.0


def test_preparing_the_query_gives_the_same_score_as_scoring_it_directly():
    """rerank_chunks prepares once and reuses it; that must be a pure speedup."""
    cases = [
        ("what is CTS?", "CTS is the central tracking system", "CTS", "Section 1"),
        ("CTS la gi", "Xem them references: https://a.io www.b.vn", "Tai lieu", ""),
        ("chinh sach nghi phep", "chinh sach nghi phep hang nam", "Chinh sach", "2.1"),
        ("define an-toan", "an-toan means safety", "An toan", ""),
    ]
    for query, text, title, section in cases:
        prepared = prepare_query(query)
        assert score_prepared_text(prepared, text, title, section) == score_retrieval_text(
            query, text, title, section
        )


def test_a_prepared_query_is_reusable_without_drifting():
    prepared = prepare_query("chinh sach nghi phep")
    first = score_prepared_text(prepared, "chinh sach nghi phep hang nam", "T", "")
    for _ in range(20):
        assert score_prepared_text(prepared, "chinh sach nghi phep hang nam", "T", "") == first
