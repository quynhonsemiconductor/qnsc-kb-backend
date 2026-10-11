"""A PDF must arrive with headings, because headings are what avoid the LLM pass.

MarkItDown reads PDFs through pdfminer, which returns a flat text stream. On a
structured 12-section specification it produced 72 Markdown table rows and ZERO
headings. `split_into_chunks` splits on ##/### boundaries, so for PDFs — the dominant
upload format — there was nothing to split on, and the LLM restructuring pass was the
only thing that could create any structure. That is why every document cost a whole
document's tokens and minutes of wall clock.

PyMuPDF4LLM reads the same bytes through PyMuPDF's layout model, which knows font size
and weight, so it emits real `#`/`##` headings AND the tables: 49 headings on that same
document in 0.29 s, against ~300 s for the model.

These tests pin the ORDERING (PDFs prefer PyMuPDF4LLM, Office formats stay with
MarkItDown) and the failure behaviour, because this is an enhancement layer and the page
extractor must remain the floor.
"""

from __future__ import annotations

import pytest

from src.core.config import settings
from src.domain import source_extraction


@pytest.fixture(autouse=True)
def _clear_cache():
    source_extraction._markitdown.cache_clear()
    yield
    source_extraction._markitdown.cache_clear()


#: Long enough to clear the 80-character floor in extract_source_markdown, which
#: rejects a suspiciously short conversion against the page text.
_PAGE_TEXT = (
    "Threshold is 2.5% at 725 um. The reviewer confirms every lot before packaging. "
    "Resistivity must fall between 8.5 and 11.5 ohm-cm for the lot to be accepted."
)


def test_a_pdf_prefers_the_structure_aware_converter(monkeypatch):
    monkeypatch.setattr(
        source_extraction,
        "_convert_pdf_with_pymupdf4llm",
        lambda _data: f"# Title\n\n## Section\n\n{_PAGE_TEXT}",
    )
    monkeypatch.setattr(
        source_extraction,
        "_convert_with_markitdown",
        lambda *_a, **_k: pytest.fail("MarkItDown must not be asked for a PDF first"),
    )

    output = source_extraction.extract_source_markdown(
        "spec.pdf",
        b"%PDF-1.4",
        pages=[{"page_number": 1, "text": _PAGE_TEXT}],
    )

    assert output.startswith("# Title")
    assert "## Section" in output


def test_an_office_document_still_goes_to_markitdown(monkeypatch):
    """PyMuPDF4LLM is PDF-only. DOCX already yields headings through MarkItDown, which
    reads the document's own heading styles."""
    monkeypatch.setattr(
        source_extraction,
        "_convert_pdf_with_pymupdf4llm",
        lambda _data: pytest.fail("the PDF converter must not see a .docx"),
    )
    monkeypatch.setattr(
        source_extraction,
        "_convert_with_markitdown",
        lambda *_a, **_k: f"# Title\n\n## Section\n\n{_PAGE_TEXT}",
    )

    output = source_extraction.extract_source_markdown(
        "spec.docx",
        b"PK\x03\x04",
        pages=[{"page_number": 1, "text": _PAGE_TEXT}],
    )

    assert "## Section" in output


def test_a_pdf_converter_failure_falls_back_to_markitdown(monkeypatch):
    """Two converters means two chances before the page extractor; the second must
    actually be tried."""
    monkeypatch.setattr(
        source_extraction, "_convert_pdf_with_pymupdf4llm", lambda _data: ""
    )
    monkeypatch.setattr(
        source_extraction,
        "_convert_with_markitdown",
        lambda *_a, **_k: f"Flat but complete. {_PAGE_TEXT}",
    )

    output = source_extraction.extract_source_markdown(
        "spec.pdf",
        b"%PDF-1.4",
        pages=[{"page_number": 1, "text": _PAGE_TEXT}],
    )

    assert "Flat but complete" in output


def test_a_scanned_pdf_keeps_its_ocr_text_verbatim():
    """A scan has NO text layer, so the OCR'd pages are the only copy of the content.

    Both converters read the PDF's text layer and return "" for an image-only page
    (verified against a generated scan: 0 characters from each). That is the safe
    answer, and this pins the consequence — the OCR text must come back byte-identical
    rather than replaced by an empty conversion.
    """
    ocr_text = "Dong van ban scan: nguong 2.5% va do day 725 um. " * 12

    output = source_extraction.extract_source_markdown(
        "scanned.pdf",
        b"%PDF-1.4",
        pages=[{"page_number": 1, "text": ocr_text}],
    )

    assert output == ocr_text.strip()


def test_a_scanned_pdf_is_not_mistaken_for_a_structured_one():
    """OCR output carries no font metadata, so it yields no headings and must still be
    sent to the model — the skip must not strand a scan with no reading view."""
    from src.domain.content_restructure import is_already_structured

    ocr_text = "Dong van ban scan: nguong 2.5% va do day 725 um. " * 12

    assert not is_already_structured(ocr_text)


def test_both_converters_failing_still_returns_the_page_text(monkeypatch):
    """The floor: an upload never fails because an enhancement layer did."""
    monkeypatch.setattr(
        source_extraction, "_convert_pdf_with_pymupdf4llm", lambda _data: ""
    )
    monkeypatch.setattr(
        source_extraction, "_convert_with_markitdown", lambda *_a, **_k: ""
    )

    output = source_extraction.extract_source_markdown(
        "spec.pdf",
        b"%PDF-1.4",
        pages=[{"page_number": 1, "text": _PAGE_TEXT}],
    )

    assert output == _PAGE_TEXT


def test_an_unimportable_converter_does_not_raise(monkeypatch):
    """Same posture as _markitdown(): a missing optional dependency is logged, not
    propagated into the upload request."""
    import builtins

    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):
        if name.startswith("pymupdf4llm"):
            raise ImportError("No module named 'pymupdf4llm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)

    assert source_extraction._convert_pdf_with_pymupdf4llm(b"%PDF-1.4") == ""


def test_a_conversion_failure_returns_empty_rather_than_raising(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "pymupdf4llm",
        type(
            "m",
            (),
            {
                "to_markdown": staticmethod(
                    lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad pdf"))
                )
            },
        ),
    )

    assert source_extraction._convert_pdf_with_pymupdf4llm(b"not really a pdf") == ""


def test_the_page_ceiling_is_not_swallowed_as_a_converter_fault(monkeypatch):
    """A document over MAX_SOURCE_PAGES is a real rejection. Returning "" here would
    silently hand an oversized file to the next converter instead."""
    import sys

    monkeypatch.setattr(settings, "MAX_SOURCE_PAGES", 2)

    class _Document:
        page_count = 5000

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        type("m", (), {"open": staticmethod(lambda **_k: _Document())}),
    )
    monkeypatch.setitem(
        sys.modules,
        "pymupdf4llm",
        type("m", (), {"to_markdown": staticmethod(lambda *_a, **_k: "x")}),
    )

    with pytest.raises(source_extraction.SourceExtractionError):
        source_extraction._convert_pdf_with_pymupdf4llm(b"%PDF-1.4")


def test_the_converter_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(settings, "PYMUPDF4LLM_ENABLED", False)

    assert source_extraction._convert_pdf_with_pymupdf4llm(b"%PDF-1.4") == ""


def test_a_bold_heading_becomes_a_plain_heading():
    """PyMuPDF4LLM reports heading weight as `**...**` on top of the `##`. The
    asterisks are not cosmetic: _major_sections reads a heading as a section TITLE and
    shows it to the reviewer, so they would surface as `**2. Equipment**` in the review
    UI and in every routed candidate."""
    source = "# **Procedure**\n\n## **2. Equipment**\n\nBody text stays as it is."

    output = source_extraction._strip_heading_emphasis(source)

    assert "# Procedure" in output
    assert "## 2. Equipment" in output
    assert "**" not in output.splitlines()[0]


def test_emphasis_inside_body_text_is_left_alone():
    """Bold in a paragraph is content, not a heading artifact."""
    source = "## Section\n\nThe limit is **2.5%** and must not be exceeded."

    output = source_extraction._strip_heading_emphasis(source)

    assert "**2.5%**" in output


def test_a_partially_bold_heading_is_not_mangled():
    """Only a wrapper around the WHOLE heading is redundant. A heading that emphasises
    one word is making a distinction, so leave it."""
    source = "## 2. Equipment — **mandatory**"

    assert source_extraction._strip_heading_emphasis(source) == source


def test_stripping_emphasis_keeps_table_rows_intact():
    """The tables are the other half of what this converter is here for."""
    source = (
        "## **Specification**\n\n| Parameter | Min |\n|---|---|\n| Thickness | 710 |"
    )

    output = source_extraction._strip_heading_emphasis(source)

    assert "| Thickness | 710 |" in output
    assert "## Specification" in output
