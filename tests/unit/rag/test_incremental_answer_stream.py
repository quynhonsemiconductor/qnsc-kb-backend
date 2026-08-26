"""Tokens must reach the reader while the provider is still generating.

The provider streams, but the raw stream carries `<<<GROUNDED>>>` / `<<<EXTENDED>>>`
section markers. The old answer to that was to forward nothing and replace the message
atomically at the end, which turned a streaming provider into a spinner followed by a wall
of text. IncrementalAnswerStream releases each token the moment it cannot be part of a
sentinel line — so these tests are mostly about what must NEVER escape.
"""
from __future__ import annotations

import pytest

from src.rag.answer_sections import IncrementalAnswerStream, split_answer_sections


def _drain(raw: str, chunk_size: int) -> str:
    """Feed `raw` the way a provider does: arbitrary chunks that split sentinels."""
    stream = IncrementalAnswerStream()
    pieces = [raw[index : index + chunk_size] for index in range(0, len(raw), chunk_size)]
    return "".join(stream.feed(piece) for piece in pieces) + stream.finish()


ANSWER = (
    "<<<GROUNDED>>>\n"
    "Cong ty cho phep 12 ngay phep mot nam [C1].\n"
    "\n"
    "Xem them muc 3.\n"
    "<<<EXTENDED>>>\n"
    "Noi dung ngoai le khong duoc trich dan.\n"
)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 13, 64, 4096])
def test_the_result_does_not_depend_on_how_the_provider_chunks_it(chunk_size):
    """A sentinel split across two network reads must still be recognised."""
    assert _drain(ANSWER, chunk_size) == _drain(ANSWER, 1)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 4096])
def test_no_sentinel_or_fragment_of_one_ever_reaches_the_reader(chunk_size):
    streamed = _drain(ANSWER, chunk_size)

    assert "<<<" not in streamed
    assert "GROUNDED" not in streamed
    assert "EXTENDED" not in streamed


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 4096])
def test_the_extended_section_is_never_streamed(chunk_size):
    """It is rendered under its own heading behind a divider, after the divider exists."""
    assert "khong duoc trich dan" not in _drain(ANSWER, chunk_size)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 4096])
def test_what_streams_is_exactly_what_the_parser_calls_grounded(chunk_size):
    """The replace event that follows must not change the text the reader already saw."""
    grounded, _extended = split_answer_sections(ANSWER)

    assert _drain(ANSWER, chunk_size).strip() == grounded.strip()


def test_a_sentinel_inside_a_code_fence_is_content_not_a_boundary():
    raw = (
        "<<<GROUNDED>>>\n"
        "Vi du:\n"
        "```python\n"
        "marker = '<<<EXTENDED>>>'\n"
        "```\n"
        "Ket thuc.\n"
    )
    streamed = _drain(raw, 3)

    assert "marker = '<<<EXTENDED>>>'" in streamed, "fenced text is content"
    assert "Ket thuc." in streamed, "the fenced marker must not stop the stream"


def test_unmarked_output_streams_in_full():
    """split_answer_sections treats output with no sentinel as grounded; so does this."""
    raw = "Tra loi truc tiep khong co sentinel.\nDong hai.\n"

    assert _drain(raw, 4).strip() == raw.strip()


def test_a_trailing_line_without_a_newline_is_flushed():
    assert _drain("<<<GROUNDED>>>\nKhong co xuong dong cuoi", 3) == (
        "Khong co xuong dong cuoi"
    )


def test_a_trailing_sentinel_without_a_newline_is_not_flushed():
    assert _drain("<<<GROUNDED>>>\nNoi dung.\n<<<EXTENDED>>>", 3) == "Noi dung.\n"


def test_blank_lines_are_preserved_so_markdown_still_renders():
    assert "\n\n" in _drain(ANSWER, 3)


def test_ordinary_text_is_released_without_waiting_for_the_line_to_end():
    """The point of the whole exercise: partial lines go out as they arrive."""
    stream = IncrementalAnswerStream()

    assert stream.feed("Cong ty ") == "Cong ty "
    assert stream.feed("cho phep") == "cho phep"


def test_only_a_plausible_sentinel_start_is_briefly_withheld():
    stream = IncrementalAnswerStream()

    assert stream.feed("<") == "", "could still become <<<GROUNDED>>>"
    assert stream.feed("d") == "<d", "cannot: released immediately, nothing lost"


def test_the_stream_stays_stopped_after_the_extended_marker():
    stream = IncrementalAnswerStream()
    stream.feed("<<<GROUNDED>>>\nMot.\n<<<EXTENDED>>>\n")

    assert stream.stopped is True
    assert stream.feed("them noi dung") == ""
    assert stream.finish() == ""


def test_a_sentinel_with_the_wrong_bracket_count_is_still_a_boundary():
    """glm-4.5-flash writes `<<<GROUNDED>>` — two closing angles.

    Matched exactly, no boundary was found at all: the raw sentinel appeared in the
    answer and the uncited EXTENDED section was folded into the grounded one, which is
    the single distinction the two-section design exists to keep.
    """
    from src.rag.answer_sections import split_answer_sections

    raw = "<<<GROUNDED>>\nCâu trả lời có nguồn [C4]\n\n<<<EXTENDED>>\nKiến thức chung."

    grounded, extended = split_answer_sections(raw)

    assert grounded == "Câu trả lời có nguồn [C4]"
    assert extended == "Kiến thức chung."
    assert "<<<" not in grounded and "<<<" not in extended


def test_a_malformed_sentinel_is_never_streamed_to_the_reader():
    from src.rag.answer_sections import IncrementalAnswerStream

    stream = IncrementalAnswerStream()
    shown = "".join(stream.feed(character) for character in "<<GROUNDED>>\nXin chào")
    shown += stream.finish()

    assert shown == "Xin chào"
