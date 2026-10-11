"""The LLM pass must not be paid for structure the extractor already produced.

`restructure_document` regenerates the WHOLE document: the document in prompt tokens,
the document again in output tokens, and minutes of wall clock (a 26,633-character
upload measured 304 s against a 300 s timeout). What downstream actually consumes from
it is Markdown structure — `split_into_chunks` splits on ##/### and the reviewer reads
the headings and tables.

MarkItDown's pdfminer path emits NO headings for PDFs, so for the dominant upload
format the model was the only source of structure and the cost looked unavoidable. With
PyMuPDF4LLM supplying real headings (see test_pdf_markdown_structure.py), a structured
document arrives with that work already done, and running the model redoes it.

These tests pin the skip decision itself: when it fires, when it must NOT fire, and that
it never calls the provider or mangles the Markdown it passes through.
"""

from __future__ import annotations

import asyncio

import pytest

from src.core.config import settings
from src.domain import content_restructure


def _run(coro):
    return asyncio.run(coro)


def _structured_document(sections: int = 6) -> str:
    body = ["# Wafer inspection procedure", "", "Scope applies to every lot.", ""]
    for index in range(1, sections + 1):
        body += [
            f"## {index}. Section {index}",
            "",
            f"Threshold {index} requires 2.5% tolerance and 725 um thickness.",
            "",
        ]
    return "\n".join(body)


@pytest.fixture
def _no_provider(monkeypatch):
    """Any call to the model is a failure of these tests, so make one impossible."""

    async def _explode(*_args, **_kwargs):
        raise AssertionError("the LLM must not be called for a structured document")

    monkeypatch.setattr(content_restructure, "complete", _explode)
    monkeypatch.setattr(
        content_restructure, "resolve_provider", lambda *_a, **_k: _Provider()
    )


class _Provider:
    model = "test-model"
    name = "test"


def test_a_structured_document_skips_the_model_entirely(_no_provider):
    source = _structured_document()

    result = _run(content_restructure.restructure_document("Procedure", source))

    assert result.status == "already_structured"
    assert result.model == "none"
    assert result.error is None


def test_the_passed_through_body_is_the_source_markdown_verbatim(_no_provider):
    """_fallback_text() rewrites layout markers and would flatten real Markdown tables,
    so the skip path must NOT route through it."""
    source = _structured_document(3) + "\n".join(
        [
            "## 4. Specification table",
            "",
            "| Parameter | Min | Max | Unit |",
            "|---|---|---|---|",
            "| Thickness | 710 | 740 | um |",
            "| Resistivity | 8.5 | 11.5 | ohm-cm |",
            "",
        ]
    )

    result = _run(content_restructure.restructure_document("Procedure", source))

    assert result.status == "already_structured"
    assert "| Thickness | 710 | 740 | um |" in result.body_md
    assert result.body_md == source.strip()


def test_the_skip_preserves_every_number(_no_provider):
    source = _structured_document()

    result = _run(content_restructure.restructure_document("Procedure", source))

    assert result.report.numeric_coverage == 1.0
    assert result.report.missing_numeric_tokens == []


def test_the_skip_still_produces_chunk_boundaries(_no_provider):
    """The point of keeping the structure is that it chunks; assert it actually does."""
    result = _run(
        content_restructure.restructure_document("Procedure", _structured_document(6))
    )

    assert result.chunks is not None
    # 6 '##' sections plus the '#' title preamble.
    assert len(result.chunks) == 7
    assert result.chunks[1].startswith("## 1. Section 1")


def test_a_flat_document_is_still_sent_to_the_model(monkeypatch):
    """The regression this must not cause: a document with no headings is exactly the
    case the LLM pass exists for."""
    called = {}
    source = "Threshold is 2.5% at 725 um. " * 40

    async def _complete(messages, **_kwargs):
        called["yes"] = True
        # Return the source under a heading, so the real preservation checks in
        # _restructure_single_document pass and the status reflects the CALL rather
        # than a rejection of this stub's output.
        return f"# Formatted\n\n## Section\n\n{source}", 0, "m", "p"

    monkeypatch.setattr(content_restructure, "complete", _complete)
    monkeypatch.setattr(
        content_restructure, "resolve_provider", lambda *_a, **_k: _Provider()
    )

    result = _run(content_restructure.restructure_document("Flat", source))

    assert called.get("yes") is True
    assert result.status == "llm"


def test_a_title_only_document_is_not_called_structured():
    """A single '#' creates no chunk boundary, so it is not the structure we need."""
    assert not content_restructure.is_already_structured(
        "# Only a title\n\n" + "Body text without any section headings. " * 40
    )


def test_sparse_headings_in_a_long_document_are_not_enough():
    """The absolute count alone would accept a 100-page document whose only headings
    sit on page one."""
    source = (
        "## A\n\ntext\n\n## B\n\ntext\n\n## C\n\n"
        + "Unstructured body text that runs on and on. " * 400
    )

    assert content_restructure._chunk_heading_count(source) >= 3
    assert not content_restructure.is_already_structured(source)


def test_a_short_note_with_one_heading_is_not_enough():
    """The density alone would accept this, which is why the count is also required."""
    source = "## Note\n\nShort remark."

    assert content_restructure._chunk_heading_count(source) == 1
    assert not content_restructure.is_already_structured(source)


def test_bold_label_lines_are_not_mistaken_for_structure():
    """The false positive to guard: a form or slide deck whose every line is short and
    bold.

    PyMuPDF4LLM requires a span to be larger AND bolder than body text before it emits
    a `##`, so a document of uniformly bold labels converts to `**Label:**` lines and
    yields zero chunk headings — verified against a generated 42-label form (0 headings
    in 996 characters). Such a document genuinely needs the model, and counting bold as
    structure would skip the one pass that could help it.
    """
    source = "\n\n".join(f"**Truong du lieu {index}:**" for index in range(42))

    assert content_restructure._chunk_heading_count(source) == 0
    assert not content_restructure.is_already_structured(source)


def test_headings_inside_fenced_code_do_not_count():
    source = (
        "```markdown\n## Not a heading\n## Nor this\n## Nor this one\n```\n\n"
        + "Body text. " * 60
    )

    assert content_restructure._chunk_heading_count(source) == 0
    assert not content_restructure.is_already_structured(source)


def test_the_skip_can_be_turned_off(monkeypatch):
    """An operator must be able to force the model back on without a code change."""
    monkeypatch.setattr(settings, "RESTRUCTURE_SKIP_WHEN_STRUCTURED", False)

    assert not content_restructure.is_already_structured(_structured_document())


def test_disabled_restructuring_still_wins_over_the_skip(monkeypatch):
    """`enabled=False` is an explicit instruction and must not be reported as a skip."""
    result = _run(
        content_restructure.restructure_document(
            "Procedure", _structured_document(), enabled=False
        )
    )

    assert result.status == "disabled"
