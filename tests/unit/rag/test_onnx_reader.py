"""Reader behaviour that must hold regardless of which weights are loaded.

These run only when a reader export is present: they exercise real inference, and
a mocked session would test the mock. Skipping is honest; asserting against a
stub would not be.
"""
from __future__ import annotations

import pathlib

import pytest

from src.core.config import settings

reader_dir = pathlib.Path(settings.READER_ONNX_DIR or "")
pytestmark = pytest.mark.skipif(
    not (reader_dir / settings.READER_ONNX_FILE).is_file()
    or not (reader_dir / "tokenizer.json").is_file(),
    reason=f"no ONNX reader export at READER_ONNX_DIR={reader_dir}",
)

ENGLISH_PASSAGE = (
    "The Treaty of Versailles was signed on 28 June 1919 in the Hall of Mirrors. "
    "It formally ended the state of war between Germany and the Allied Powers."
)
VIETNAMESE_PASSAGE = (
    "Hà Nội là thủ đô của Việt Nam. Thành phố này có dân số khoảng 8,4 triệu người "
    "và nằm bên bờ sông Hồng."
)


@pytest.fixture(scope="module")
def reader():
    from src.lib.reader import resolve_reader

    instance = resolve_reader()
    instance.warm_up()
    return instance


def test_extracts_an_english_span(reader) -> None:
    span = reader.read("When was the Treaty of Versailles signed?", [ENGLISH_PASSAGE])
    assert "28 June 1919" in span.text


def test_extracts_a_vietnamese_span(reader) -> None:
    span = reader.read("Thủ đô của Việt Nam là thành phố nào?", [VIETNAMESE_PASSAGE])
    assert "Hà Nội" in span.text


def test_answer_comes_from_the_passage_not_the_question(reader) -> None:
    # The decoder masks question tokens; without that mask the argmax can land
    # inside the question and return a fragment of it.
    span = reader.read("Where is Hanoi located in Vietnam?", [VIETNAMESE_PASSAGE])
    assert span.text
    assert span.text in VIETNAMESE_PASSAGE


def test_offsets_locate_the_answer_in_the_original_text(reader) -> None:
    span = reader.read("When was the Treaty of Versailles signed?", [ENGLISH_PASSAGE])
    assert ENGLISH_PASSAGE[span.start : span.end].strip() == span.text


def test_picks_the_passage_that_holds_the_answer(reader) -> None:
    span = reader.read(
        "Thủ đô của Việt Nam là thành phố nào?",
        [ENGLISH_PASSAGE, VIETNAMESE_PASSAGE],
    )
    assert span.passage_index == 1
    assert "Hà Nội" in span.text


def test_unrelated_passage_scores_below_its_own_null(reader) -> None:
    # A SQuAD2-trained model should prefer no-answer here. The score is
    # null-relative, so "prefers abstention" means score < 0.
    span = reader.read("What is the boiling point of mercury?", [VIETNAMESE_PASSAGE])
    assert span.score < 0


def test_empty_passage_list_abstains(reader) -> None:
    span = reader.read("anything at all?", [])
    assert span.is_abstention


def test_blank_passages_abstain(reader) -> None:
    span = reader.read("anything at all?", ["", "   "])
    assert span.is_abstention
