"""A sentinel written on the same line as its text, and the one the prompt never asked for.

deepseek-flash answers `<<<GROUNDED>>>Không tìm thấy...` and `<<<EXTENDED>>>Trong các SoC...` with the text
right after the marker, and ends with `<<<END>>>`. The parser wanted each on a line of its own, found no
boundary, and showed the markers; the general-knowledge text, which is uncited by definition, was left
inside "Answer from the Knowledge Base". Shapes below are taken from real answers.
"""

from __future__ import annotations

import pytest

from src.rag.answer_sections import IncrementalAnswerStream, split_answer_sections

INLINE = (
    "<<<GROUNDED>>>Thanh ghi **CFG** của PWM nằm ở offset **M+0x04** [C18].\n"
    "\n"
    "<<<EXTENDED>>>Trong thực tế, các khối PWM thường có một thanh ghi cấu hình.\n"
    "<<<END>>>"
)
GROUNDED_TEXT = "Thanh ghi **CFG** của PWM nằm ở offset **M+0x04** [C18]."


def _drain(raw: str, size: int) -> str:
    stream = IncrementalAnswerStream()
    pieces = [raw[i : i + size] for i in range(0, len(raw), size)]
    return "".join(stream.feed(piece) for piece in pieces) + stream.finish()


def test_the_sections_are_separated_and_no_marker_is_left():
    grounded, extended = split_answer_sections(INLINE)
    assert grounded == GROUNDED_TEXT
    assert extended == "Trong thực tế, các khối PWM thường có một thanh ghi cấu hình."
    assert "<<<" not in grounded + extended


def test_general_knowledge_does_not_stay_in_the_grounded_section():
    grounded, _ = split_answer_sections(INLINE)
    assert "Trong thực tế" not in grounded


def test_a_marker_followed_by_a_space_works_too():
    grounded, extended = split_answer_sections(
        "<<<GROUNDED>>> Có 12 ngày [C1].\n<<<EXTENDED>>> Thông lệ chung."
    )
    assert (grounded, extended) == ("Có 12 ngày [C1].", "Thông lệ chung.")


def test_the_ordinary_shape_is_unchanged():
    raw = "<<<GROUNDED>>>\nCó 12 ngày [C1].\n<<<EXTENDED>>>\nThông lệ chung.\n"
    assert split_answer_sections(raw) == ("Có 12 ngày [C1].", "Thông lệ chung.")


def test_a_refusal_with_an_extended_part():
    grounded, extended = split_answer_sections(
        "<<<GROUNDED>>>Không tìm thấy trong Knowledge Base.\n<<<EXTENDED>>>Thông lệ chung."
    )
    assert grounded == "Không tìm thấy trong Knowledge Base."
    assert extended == "Thông lệ chung."


def test_end_alone_is_dropped():
    assert split_answer_sections("Có 12 ngày [C1].\n<<<END>>>") == (
        "Có 12 ngày [C1].",
        "",
    )


def test_a_marker_quoted_inside_a_code_fence_is_content():
    raw = "<<<GROUNDED>>>\n```\n<<<GROUNDED>>>text\n```\n"
    grounded, _ = split_answer_sections(raw)
    assert "<<<GROUNDED>>>text" in grounded


def test_a_marker_mentioned_mid_sentence_is_content():
    raw = "<<<GROUNDED>>>\nPrompt dùng <<<GROUNDED>>>text để đánh dấu [C1]."
    grounded, _ = split_answer_sections(raw)
    assert "<<<GROUNDED>>>text" in grounded


@pytest.mark.parametrize("size", [1, 2, 3, 7, 4096])
def test_the_stream_never_shows_a_marker_and_stops_at_extended(size):
    streamed = _drain(INLINE, size)
    assert "<<<" not in streamed
    assert "GROUNDED" not in streamed and "END" not in streamed
    assert "Trong thực tế" not in streamed
    assert streamed.strip() == GROUNDED_TEXT


@pytest.mark.parametrize("size", [1, 5, 4096])
def test_the_stream_matches_the_final_grounded_text(size):
    assert _drain(INLINE, size).strip() == split_answer_sections(INLINE)[0]


def test_text_after_the_inline_marker_streams_as_it_arrives():
    stream = IncrementalAnswerStream()
    assert stream.feed("<<<GROUNDED>>>Có 12") == "Có 12"
