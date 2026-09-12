"""Comparison-intent detection must be conservative: missing one costs a bit of
retrieval quality on one query, but a false positive fragments an ordinary question
into nonsense sub-queries. Every assertion here is really checking which side of that
trade-off a given input falls on.
"""
from __future__ import annotations

import pytest

from src.rag.query_router import (
    detect_ambiguous_departments,
    is_comparison_query,
    needs_deep_retrieval,
    split_comparison_subjects,
)


@pytest.mark.parametrize(
    "query",
    [
        "So sánh SOP-114 và SOP-118",
        "so sanh SOP-114 va SOP-118",  # typed without diacritics
        "What's the difference between Fusion Compiler and Innovus CCOpt?",
        "CTS vs TritonCTS, which should we use?",
        "SOP-114 khác nhau ở điểm nào so với SOP-118?",
    ],
)
def test_recognises_real_comparison_questions(query):
    assert is_comparison_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "Quy trình đăng ký và phê duyệt nghỉ phép",
        "What is CTS?",
        "List the tools used for RTL generation and simulation",
        "How do I request access and reset my password?",
    ],
)
def test_does_not_flag_ordinary_questions_that_merely_contain_and(query):
    assert not is_comparison_query(query)


def test_splits_a_two_subject_comparison():
    result = split_comparison_subjects("So sánh SOP-114 và SOP-118")
    assert result == ["So sánh SOP-114", "SOP-118"]


def test_splits_an_english_versus_comparison():
    result = split_comparison_subjects("Fusion Compiler vs Innovus CCOpt")
    assert result == ["Fusion Compiler", "Innovus CCOpt"]


def test_caps_at_max_subjects():
    # Single-letter stand-ins would trip MIN_SUBJECT_LENGTH and defeat the point of this
    # test (falling back to "could not split" for an unrelated reason), so each subject
    # here is a realistic multi-character name.
    result = split_comparison_subjects("so sánh Alpha và Beta và Gamma và Delta và Epsilon")
    assert len(result) == 3


def test_falls_back_to_the_whole_query_when_it_cannot_split():
    # Comparison intent ("khác nhau") is present but there is no conjunction to split on.
    result = split_comparison_subjects("SOP-114 khác nhau ở điểm nào?")
    assert result == ["SOP-114 khác nhau ở điểm nào?"]


def test_empty_query_splits_to_nothing():
    assert split_comparison_subjects("") == []
    assert split_comparison_subjects("   ") == []


def _result(dept, score):
    return {"dept": dept, "score": score}


def test_a_single_clear_winner_is_not_ambiguous():
    results = [_result("Finance", 0.9), _result("Finance", 0.5), _result("HR", 0.2)]
    assert detect_ambiguous_departments(results) is None


def test_two_departments_near_the_top_score_is_ambiguous():
    results = [_result("Finance", 0.9), _result("HR", 0.88)]
    assert detect_ambiguous_departments(results) == ["Finance", "HR"]


def test_a_distant_second_department_is_not_ambiguous():
    results = [_result("Finance", 0.9), _result("HR", 0.3)]
    assert detect_ambiguous_departments(results) is None


def test_the_leading_department_is_listed_first():
    results = [_result("HR", 0.85), _result("Finance", 0.9)]
    assert detect_ambiguous_departments(results) == ["Finance", "HR"]


def test_duplicate_departments_are_not_double_counted():
    results = [_result("Finance", 0.9), _result("Finance", 0.89), _result("HR", 0.87)]
    assert detect_ambiguous_departments(results) == ["Finance", "HR"]


def test_fewer_than_two_scored_results_is_not_ambiguous():
    assert detect_ambiguous_departments([_result("Finance", 0.9)]) is None
    assert detect_ambiguous_departments([]) is None


def test_results_without_a_department_are_ignored():
    results = [{"score": 0.9}, _result("HR", 0.85)]
    assert detect_ambiguous_departments(results) is None


def test_zero_top_score_is_not_ambiguous():
    results = [_result("Finance", 0.0), _result("HR", 0.0)]
    assert detect_ambiguous_departments(results) is None


def test_only_the_top_n_results_are_considered():
    # HR's high score sits outside the default top_n window, so it never enters the
    # near-top comparison against Finance.
    results = [_result("Finance", 0.9)] + [_result("Other", 0.1)] * 5 + [_result("HR", 0.89)]
    assert detect_ambiguous_departments(results, top_n=5) is None


# --- needs_deep_retrieval: gates the (opt-in) cross-encoder, never the base pipeline ---


def test_a_comparison_question_needs_deep_retrieval():
    assert needs_deep_retrieval("So sánh SOP-114 và SOP-118")


def test_a_plain_factual_question_does_not_need_deep_retrieval():
    assert not needs_deep_retrieval("What is CTS?")


def test_a_single_question_mark_does_not_need_deep_retrieval():
    assert not needs_deep_retrieval("How do I request access?")


def test_two_distinct_questions_need_deep_retrieval():
    assert needs_deep_retrieval("What is the leave policy? How do I submit a request?")


def test_empty_query_does_not_need_deep_retrieval():
    assert not needs_deep_retrieval("")
