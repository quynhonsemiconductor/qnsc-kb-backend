"""Which questions are about the collection itself, decided from the wording alone.

A question like "how many documents do we have" cannot be answered by retrieving passages:
the nearest passages are a sample of the text, not a census of the documents. These tests pin
the line between that and a question that merely mentions documents. The line is drawn
conservatively on purpose: `None` sends a question to the planner, which is how it was
always handled, so a miss costs a model call while a false hit answers the wrong question.
"""

from __future__ import annotations

import pytest

from src.domain.agent.plan import (
    INTENTS,
    detect_corpus_intent,
    match_departments,
    match_doc_types,
    sanitize_plan,
)

CATALOG = [
    "Thống kê toàn bộ tài liệu hiện có",
    "thống kê tài liệu",
    "Thống kê tất cả các tài liệu",
    "Có bao nhiêu tài liệu?",
    "Có bao nhiêu SOP của phòng Kỹ thuật?",
    "Hiện có bao nhiêu văn bản trong kho tri thức",
    "Số lượng tài liệu hiện nay là bao nhiêu",
    "Liệt kê tài liệu mới nhất",
    "Liệt kê toàn bộ tài liệu",
    "Cho tôi xem danh sách tài liệu",
    "cho mình biết danh sách văn bản",
    "Có những tài liệu nào?",
    "Tài liệu mới nhất là gì",
    "Tài liệu vừa cập nhật",
    "Có bao nhiêu tài liệu nhập tháng này",
    "Tổng quan về kho tri thức",
    "How many documents are there?",
    "How many policies do we have",
    "List all documents",
    "show me the documents",
    "Document statistics please",
    "What documents do we have?",
]

CORPUS = [
    "Tóm tắt tất cả các chính sách nhân sự",
    "Tổng hợp các quy trình",
    "tóm tắt toàn bộ tài liệu",
    "Tất cả các tài liệu nói về an toàn đề cập gì",
    "Summarize all the policies",
    "summarise every document",
    # an assessment of the content as a whole, not only a summary
    "CHo tôi insight về toàn bộ tài liệu thuộc phòng ban public2 của tôi",
    "Insight về tất cả tài liệu",
    "Phân tích toàn bộ các tài liệu của phòng Kỹ thuật",
    "Đánh giá tất cả chính sách hiện có",
    "Nhận xét các tài liệu nhân sự",
    "Give me insights on all documents in public2",
    "Overview of every policy",
]

# Mentions documents, statistics or numbers, but is not about the collection.
NOT_COLLECTION = [
    # a topic: counting rows cannot answer it
    "Có bao nhiêu tài liệu về an toàn hóa chất",
    "How many documents about onboarding",
    "thống kê tài liệu liên quan đến mua hàng",
    # figures inside a document
    "Thống kê số liệu trong bảng doanh thu Q3",
    "Tổng doanh thu của tài liệu báo cáo là bao nhiêu",
    "Tỷ lệ lỗi trong file kiểm tra là bao nhiêu",
    "What is the revenue in the table",
    # ordinary questions
    "Quy trình nghỉ phép là gì",
    "Thời hạn nộp hồ sơ là bao lâu?",
    "Tài liệu này nói gì về bảo mật",
    "Giải thích tài liệu hướng dẫn cài đặt",
    "What is the leave policy?",
    # one document, or an "evaluate" without a quantifier
    "Đánh giá tài liệu này giúp tôi",
    "Phân tích lỗi trong tài liệu hướng dẫn",
    "Insight về tài liệu AXI4",
    "Review this document",
    "",
]


@pytest.mark.parametrize("question", CATALOG)
def test_a_question_about_the_collection_is_a_catalog_question(question):
    assert detect_corpus_intent(question) == "catalog"


@pytest.mark.parametrize("question", CORPUS)
def test_a_question_that_needs_many_documents_is_a_corpus_question(question):
    assert detect_corpus_intent(question) == "corpus"


@pytest.mark.parametrize("question", NOT_COLLECTION)
def test_other_questions_are_left_to_the_planner(question):
    assert detect_corpus_intent(question) is None


def test_the_check_ignores_case_diacritics_and_spacing():
    assert detect_corpus_intent("  THỐNG   KÊ  tài   liệu ") == "catalog"
    assert detect_corpus_intent("thong ke toan bo tai lieu hien co") == "catalog"


def test_the_planner_may_also_choose_the_new_intents():
    assert {"catalog", "corpus"} <= set(INTENTS)
    assert sanitize_plan({"intent": "catalog"}, "q", []).intent == "catalog"
    assert sanitize_plan({"intent": "corpus"}, "q", []).intent == "corpus"
    # an invented intent still falls back
    assert sanitize_plan({"intent": "wander"}, "q", []).intent == "other"


@pytest.mark.parametrize(
    ("folded", "expected"),
    [
        ("co bao nhieu sop ", ["SOP"]),
        ("so luong chinh sach hien co ", ["POLICY"]),
        ("liet ke sop va chinh sach ", ["SOP", "POLICY"]),
        ("how many policies ", ["POLICY"]),
        ("co bao nhieu quy trinh ", []),  # ambiguous word: never guessed
        ("co bao nhieu tai lieu ", []),
        ("co bao nhieu soptimal ", []),  # whole words only
    ],
)
def test_a_document_type_is_taken_only_from_an_unambiguous_word(folded, expected):
    assert match_doc_types(folded) == expected


def test_departments_are_matched_by_name_without_diacritics():
    departments = ["Kỹ thuật", "IT", "IT Ops", "Nhân sự"]
    assert match_departments("sop cua phong ky thuat ", departments) == ["Kỹ thuật"]
    assert set(match_departments("tai lieu cua nhan su va ky thuat ", departments)) == {
        "Nhân sự",
        "Kỹ thuật",
    }
    assert match_departments("tai lieu cua phong khac ", departments) == []


def test_a_longer_department_name_hides_the_shorter_one_inside_it():
    assert match_departments("tai lieu cua it ops ", ["IT", "IT Ops"]) == ["IT Ops"]
    assert match_departments("tai lieu cua it ", ["IT", "IT Ops"]) == ["IT"]


def test_a_department_name_inside_another_word_does_not_match():
    assert match_departments("submitting items ", ["IT"]) == []
    assert match_departments("kiem tra it em ", ["IT"]) == ["IT"]
