"""A Vietnamese follow-up must be recognised as one.

`_needs_query_rewrite` decides whether a question refers back to the conversation and
should be searched with that context. Its markers were entirely English -- "what about",
"how about", and the pronouns that/those/it/they/them/also/more -- so in a Vietnamese
deployment it never fired, and a follow-up never received the context the step exists to
supply.

That step is also not an AI rewrite, despite reading like one: it concatenates the last
three user turns onto the question. Under the old scoring that made things worse rather
than better, because relevance was matched/len(terms) and concatenation inflated the
denominator -- measured at 0.667 -> 0.200 on one passage, dropping it under the 0.35
confidence gate. The step meant to rescue follow-ups was causing the refusal it existed
to prevent. That is fixed separately; firing it correctly is only safe because of it.

The word cap is raised for the same reason the markers were missing. Vietnamese is
written in syllables, so "cung cấp cho tôi" counts four words where English counts two,
and a cap chosen against English sentences was much tighter in Vietnamese than intended.
"""
from __future__ import annotations

import pytest

from src.domain.ai_service import _needs_query_rewrite


class _Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


HISTORY = [_Message("user", "RTL Generator là gì?"), _Message("assistant", "...")]


@pytest.mark.parametrize(
    "question",
    [
        "Vậy cung cấp cho tôi các Tool dùng RTL Generator",  # the reported question
        "Còn CSR thì sao?",
        "Thế còn bus interconnect?",
        "Cái đó dùng ở đâu?",
        "Nó có hỗ trợ AXI không?",
        "Cho tôi thêm ví dụ",
        "Tiếp theo là gì?",
        "Ngoài ra còn gì nữa?",
    ],
)
def test_a_vietnamese_follow_up_is_recognised(question):
    assert _needs_query_rewrite(question, HISTORY)


@pytest.mark.parametrize(
    "question",
    ["So what about those tools", "How about that one", "Tell me more", "And next?"],
)
def test_english_follow_ups_still_work(question):
    assert _needs_query_rewrite(question, HISTORY)


@pytest.mark.parametrize(
    "question",
    [
        "RTL Generator",
        "Danh sách công cụ tổng hợp logic",
        "Quy trình xin nghỉ phép của công ty là gì và cần những giấy tờ nào kèm theo đơn xin",
    ],
)
def test_a_fresh_question_is_not_treated_as_a_follow_up(question):
    """Searching a new subject with the previous subject attached is what buries it."""
    assert not _needs_query_rewrite(question, HISTORY)


def test_the_first_question_of_a_conversation_is_never_a_follow_up():
    assert not _needs_query_rewrite("Cái đó là gì?", [])


def test_a_long_question_is_not_a_follow_up_however_it_starts():
    """A question that states its own subject at length does not need the previous one,
    and attaching it would bury the subject actually being asked about."""
    question = "Vậy " + " ".join(["từ"] * 20)
    assert not _needs_query_rewrite(question, HISTORY)
