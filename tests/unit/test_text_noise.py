"""Retrieval-side noise removal: what it strips, and — mostly — what it must not.

Deleting real content is the worse failure. A passage that never gets indexed cannot be
found and raises no error, so most of these tests pin the cases where a line only LOOKS
like furniture and has to survive.
"""
from __future__ import annotations

from src.domain.text_noise import (
    BOILERPLATE_MAX_LINE_CHARS,
    detect_boilerplate,
    is_navigation_line,
    strip_noise,
)


def _paged(*bodies: str) -> list[tuple[int, str]]:
    return [(index, body) for index, body in enumerate(bodies, start=1)]


def test_a_running_header_is_detected_across_pages_despite_its_page_number():
    """"Trang 1/3" and "Trang 2/3" are the same piece of furniture."""
    pages = _paged(
        "Cong ty ABC - Quy dinh noi bo | Trang 1/3\nDieu 1. Pham vi ap dung",
        "Cong ty ABC - Quy dinh noi bo | Trang 2/3\nDieu 2. Thoi gian lam viec",
        "Cong ty ABC - Quy dinh noi bo | Trang 3/3\nDieu 3. Nghi phep hang nam",
    )

    boilerplate = detect_boilerplate(pages)

    assert boilerplate, "the repeated header must be recognised"
    cleaned = strip_noise(pages[0][1], boilerplate)
    assert "Cong ty ABC" not in cleaned
    assert cleaned == "Dieu 1. Pham vi ap dung"


def test_content_that_happens_to_repeat_on_two_pages_is_kept():
    """Two pages is not evidence. A heading reused once must not be deleted."""
    pages = _paged("Quy dinh chung\nNoi dung mot", "Quy dinh chung\nNoi dung hai")

    assert detect_boilerplate(pages) == frozenset()


def test_a_line_on_a_minority_of_pages_is_kept():
    pages = _paged(
        "Phu luc A\nNoi dung mot",
        "Noi dung hai",
        "Noi dung ba",
        "Noi dung bon",
        "Phu luc A\nNoi dung nam",
    )

    assert detect_boilerplate(pages) == frozenset()


def test_a_long_repeated_line_is_treated_as_content_not_furniture():
    """A recurring clause is content; headers and footers are short."""
    clause = "Dieu khoan nay " + "rat dai va chi tiet " * 8
    assert len(clause) > BOILERPLATE_MAX_LINE_CHARS
    pages = _paged(f"{clause}\nMot", f"{clause}\nHai", f"{clause}\nBa")

    assert detect_boilerplate(pages) == frozenset()


def test_a_line_repeated_many_times_on_a_single_page_is_not_furniture():
    """That is one page's formatting artefact, not document furniture."""
    pages = _paged("dong lap\ndong lap\ndong lap\ndong lap\nNoi dung", "Mot", "Hai")

    assert detect_boilerplate(pages) == frozenset()


def test_too_few_pages_disables_detection_entirely():
    assert detect_boilerplate([]) == frozenset()
    assert detect_boilerplate(None) == frozenset()
    assert detect_boilerplate(_paged("Header\nMot", "Header\nHai")) == frozenset()


def test_blank_pages_do_not_count_toward_the_page_total():
    """Otherwise a header on every REAL page fails the ratio because of empty ones."""
    pages = _paged("Header X\nMot", "", "   ", "Header X\nHai", "Header X\nBa")

    assert detect_boilerplate(pages), "three real pages all carry the header"


def test_page_labels_are_navigation():
    for line in ("7", "- 7 -", "[7]", "7/45", "Trang 7", "Trang 7/45", "Page 7 of 45", "p. 7"):
        assert is_navigation_line(line), line


def test_numbered_headings_and_clauses_are_not_navigation():
    """The label has to be the WHOLE line; anything with words carries signal."""
    for line in (
        "1. Gioi thieu",
        "Dieu 7 quy dinh ve nghi phep",
        "7.",
        "Trang bi bao ho lao dong",
        "2024 la nam ban hanh",
        "Muc 3 - An toan",
    ):
        assert not is_navigation_line(line), line


def test_table_of_contents_rows_are_navigation():
    assert is_navigation_line("Chuong 1 .......... 12")
    assert is_navigation_line("An toan lao dong ····· 5")


def test_a_sentence_ending_in_a_number_is_not_a_contents_row():
    assert not is_navigation_line("Tong so nhan vien la 120")
    assert not is_navigation_line("Xem muc 3.2 va 4")


def test_stripping_keeps_paragraph_structure():
    text = "Doan mot.\n\nDoan hai.\n\n7\n\nDoan ba."

    assert strip_noise(text) == "Doan mot.\n\nDoan hai.\n\nDoan ba."


def test_stripping_nothing_leaves_the_text_alone():
    text = "Dieu 1. Pham vi\n\nDieu 2. Doi tuong ap dung"

    assert strip_noise(text) == text
    assert strip_noise("") == ""


def test_a_page_of_pure_boilerplate_becomes_empty():
    """Correct: there is nothing on it worth retrieving. The caller guards the document."""
    pages = _paged("Header X\nMot", "Header X\nHai", "Header X")
    boilerplate = detect_boilerplate(pages)

    assert strip_noise("Header X", boilerplate) == ""
