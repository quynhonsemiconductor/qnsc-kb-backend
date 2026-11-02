"""The word lists that gate agent work match whole words, not fragments of other words.

They were `marker in text`. With accents folded, "chinh sach" (policy) contains "hinh"
(picture) and "kinh doanh" (business) contains "anh ", so the most common phrase in this
domain read as a question about a figure; "paragraph" contains "graph", "meaning" contains
"mean", "account" contains "count".
"""

from __future__ import annotations

import pytest

from src.domain.agent.plan import (
    is_chart_question,
    is_figure_question,
    is_simple_question,
)
from src.domain.ai_service import _asks_for_analytics, _is_visual_question


@pytest.mark.parametrize(
    "question",
    [
        "Chính sách nghỉ phép là gì?",
        "Quy định kinh doanh của công ty",
        "Thông tin chính thức về lương",
        "What does paragraph 3 say about leave?",
        "How do I configure the badge reader?",
    ],
)
def test_ordinary_questions_are_neither_figure_nor_chart_questions_and_stay_simple(
    question,
):
    assert not is_figure_question(question)
    assert not is_chart_question(question)
    assert is_simple_question(question, has_history=False)


@pytest.mark.parametrize(
    ("question", "figure", "chart"),
    [
        ("Cho tôi xem hình quy trình sản xuất", True, False),
        ("Cho toi xem hinh quy trinh san xuat", True, False),
        ("Giai đoạn nào trong sơ đồ này?", True, False),
        ("Which stage of the diagram is risky?", True, False),
        ("Vẽ biểu đồ doanh thu theo quý", False, True),
        ("Show the revenue trend as a chart", False, True),
        ("Plot the quarterly numbers", False, True),
        ("Visualise the headcount", False, True),
    ],
)
def test_real_figure_and_chart_wording_is_still_recognised(question, figure, chart):
    assert is_figure_question(question) is figure
    assert is_chart_question(question) is chart
    # The fast path must never swallow a question carrying this wording.
    assert not is_simple_question(question, has_history=False)


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the country account",  # "sum", "count" inside other words
        "What is the meaning of CTS?",  # "mean" inside "meaning"
        "Explain the cost-benefit analysis",
    ],
)
def test_analytics_wording_inside_other_words_is_not_an_aggregate_request(question):
    assert not _asks_for_analytics(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the average Revenue?",
        "Tổng số ngày phép",
        "How many sick days are there?",
        "Bao nhiêu ngày nghỉ?",
        "the highest value",
        "the sum",
    ],
)
def test_real_aggregate_wording_is_still_recognised(question):
    assert _asks_for_analytics(question)


@pytest.mark.parametrize(
    "question",
    ["How do I configure the badge reader?", "What is a configuration item?"],
)
def test_visual_wording_inside_other_words_is_not_a_visual_question(question):
    assert not _is_visual_question(question)


@pytest.mark.parametrize(
    "question",
    ["Show the chart", "What does the figure show?", "Hình quy trình", "Xem biểu đồ"],
)
def test_real_visual_wording_is_still_a_visual_question(question):
    assert _is_visual_question(question)
