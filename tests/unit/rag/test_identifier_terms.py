"""Which words of a question name one specific thing."""

from __future__ import annotations

import pytest

from src.rag.identifier_terms import MAX_IDENTIFIER_TERMS, identifier_terms


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Thanh ghi PADOUTSET của GPIO nằm ở offset nào?", ["padoutset", "gpio"]),
        (
            "Offset của thanh ghi CONFIGOPTS trong SPI Host là bao nhiêu?",
            ["configopts", "spi"],
        ),
        ("Thanh ghi CFG_REG_LO của timer ở offset nào?", ["cfg_reg_lo"]),
        ("Địa chỉ base của I2C là gì?", ["i2c"]),
        ("UART0 nằm ở địa chỉ nào?", ["uart0"]),
        ("Lệnh debug fetch ở 0x2000_1080 thì sao?", ["0x2000_1080"]),
        ("what does i_dbg_req do", ["i_dbg_req"]),
        ("PADOUTSET", ["padoutset"]),
    ],
)
def test_the_name_in_a_question_is_found(question, expected):
    assert identifier_terms(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Nhân viên được nghỉ phép bao nhiêu ngày một năm?",
        "Còn nếu nộp muộn thì sao?",
        "Quy định năm 2026 là gì, có 12 ngày không?",
        "Chính sách bảo hiểm sức khỏe của công ty như thế nào?",
        "",
    ],
)
def test_ordinary_questions_have_none_so_the_leg_is_skipped(question):
    assert identifier_terms(question) == []


def test_numbers_on_their_own_are_not_names():
    assert identifier_terms("Mua 150 triệu đồng thì cần 3 báo giá") == []


def test_a_question_typed_in_capitals_is_not_a_list_of_names():
    assert identifier_terms("WHAT IS THE RESET VALUE OF THE REGISTER") == []


def test_repeats_collapse_and_case_and_accents_fold():
    assert identifier_terms("GPIO, gpio và Gpio rồi PADOUT") == ["gpio", "padout"]


def test_a_stray_underscore_is_not_an_identifier():
    assert identifier_terms("giá trị _ này") == []


def test_short_all_name_queries_are_kept():
    assert identifier_terms("GPIO PADOUTSET") == ["gpio", "padoutset"]
    assert identifier_terms("GPIO PADOUTSET offset") == ["gpio", "padoutset"]
    assert identifier_terms("GPIO PADOUTSET CLR") == ["gpio", "padoutset", "clr"]


def test_the_list_is_bounded():
    question = "tìm các thanh ghi sau trong tài liệu " + " ".join(
        f"REG_{n}" for n in range(30)
    )
    assert len(identifier_terms(question)) == MAX_IDENTIFIER_TERMS
