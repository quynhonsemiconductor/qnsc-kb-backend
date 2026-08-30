"""The benchmark gate reads these numbers, so the scorer itself needs defending.

Each test below encodes a property that, if it broke, would silently move the
reported MLQA/ViQuAD score without any other symptom.
"""
from __future__ import annotations

from src.rag.squad_metrics import (
    aggregate,
    exact_match,
    normalize_answer,
    score_answer,
    token_f1,
)


def test_exact_match_ignores_punctuation_case_and_articles() -> None:
    assert exact_match("The Sun.", "sun", "en") == 1.0
    assert exact_match("a policy", "policy", "en") == 1.0


def test_vietnamese_articles_are_not_stripped() -> None:
    # "một" is a numeral, not an article: removing it would equate "one day"
    # with "day". English article stripping must not leak into Vietnamese.
    assert normalize_answer("một ngày", "vi") == "một ngày"


def test_diacritics_are_preserved() -> None:
    # Folding would make these score as identical, but they are different words.
    assert exact_match("hoà", "hoa", "vi") == 0.0
    assert normalize_answer("Tổ chức", "vi") == "tổ chức"


def test_short_vietnamese_tokens_survive_normalisation() -> None:
    # The dashboard metric in evaluator.py drops tokens of <=3 characters, which
    # deletes real Vietnamese content words. This scorer must not.
    assert normalize_answer("vua Hồ", "vi") == "vua hồ"
    assert token_f1("vua Hồ", "vua Hồ", "vi") == 1.0


def test_token_f1_penalises_padding() -> None:
    # Recall-only scoring would give this 1.0; F1 must not, or a verbose answer
    # that happens to contain the span would score as a perfect extraction.
    score = token_f1("the answer is clearly the sun and nothing else", "the sun", "en")
    assert 0.0 < score < 0.5


def test_token_f1_is_symmetric_in_overlap() -> None:
    assert token_f1("sun", "the sun", "en") == token_f1("the sun", "sun", "en")


def test_no_overlap_scores_zero() -> None:
    assert token_f1("moon", "sun", "en") == 0.0


def test_unanswerable_requires_abstention() -> None:
    # ViQuAD 2.0 unanswerables: empty reference list.
    assert score_answer("", [], "vi") == (1.0, 1.0)
    assert score_answer("bất kỳ câu trả lời", [], "vi") == (0.0, 0.0)


def test_answerable_question_scores_zero_when_abstaining() -> None:
    # The mirror of the case above: silence is wrong when an answer exists.
    assert score_answer("", ["tổ chức"], "vi") == (0.0, 0.0)


def test_score_takes_best_of_several_references() -> None:
    exact, f1 = score_answer("Hà Nội", ["Sài Gòn", "Hà Nội"], "vi")
    assert exact == 1.0
    assert f1 == 1.0


def test_aggregate_reports_percentages() -> None:
    result = aggregate([(1.0, 1.0), (0.0, 0.5)])
    assert result["exact_match"] == 50.0
    assert result["f1"] == 75.0
    assert result["count"] == 2


def test_aggregate_of_nothing_is_not_a_crash() -> None:
    assert aggregate([]) == {"exact_match": 0.0, "f1": 0.0, "count": 0}
