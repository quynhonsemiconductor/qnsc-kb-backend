"""Reading a .docx: document order, headings, no table of contents, and packages with missing parts.

Measured on a real 22-file specification set (register maps, block diagrams):

* two files failed outright, "Could not process uploaded source", because the package had a
  relationship to a font part (`fonts/font1.odttf`) it did not contain: python-docx raises KeyError;
* the rest were read paragraphs first and every table afterwards, so each register table ended up after
  the glossary at the very end of the text, away from the section (and the heading) it belongs to.
"""

from __future__ import annotations

import io
import zipfile

import pytest

docx = pytest.importorskip("docx")

from src.domain.source_extraction import _extract_docx  # noqa: E402


def _build(builder) -> bytes:
    document = docx.Document()
    builder(document)
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _spec(document) -> None:
    document.add_heading("GPIO", level=1)
    document.add_paragraph("Overview text.")
    document.add_heading("Register Map", level=2)
    document.add_paragraph("Offsets are from the bank base.")
    table = document.add_table(rows=2, cols=3)
    for cell, text in zip(table.rows[0].cells, ("Register", "Offset", "Access")):
        cell.text = text
    for cell, text in zip(table.rows[1].cells, ("PADDIR", "0x00", "RW")):
        cell.text = text
    document.add_heading("Acronyms", level=2)
    document.add_paragraph("IP is a core.")


def test_a_table_stays_under_the_heading_above_it_and_headings_are_markdown():
    text = _extract_docx(_build(_spec))
    assert "# GPIO" in text and "## Register Map" in text and "## Acronyms" in text
    register_map = text.index("## Register Map")
    row = text.index("PADDIR | 0x00 | RW")
    acronyms = text.index("## Acronyms")
    assert register_map < row < acronyms, text


def test_the_table_of_contents_is_not_read_as_content():
    def build(document):
        styles = document.styles
        toc = styles.add_style("TOC 1", 1)
        document.add_paragraph("1. Overview 6", style=toc)
        document.add_heading("Overview", level=1)
        document.add_paragraph("The body.")

    text = _extract_docx(_build(build))
    assert "Overview 6" not in text
    assert "# Overview" in text and "The body." in text


def test_a_merged_cell_is_read_once():
    def build(document):
        table = document.add_table(rows=1, cols=3)
        merged = table.rows[0].cells[0].merge(table.rows[0].cells[1])
        merged.text = "Both"
        table.rows[0].cells[2].text = "Last"

    assert "Both | Last" in _extract_docx(_build(build))


def _with_relationship_to_a_missing_part(raw: bytes) -> bytes:
    """The shape the real files have: a font relationship whose part is not in the package."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(
        out, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            blob = source.read(info.filename)
            if info.filename == "word/_rels/document.xml.rels":
                blob = blob.replace(
                    b"</Relationships>",
                    b'<Relationship Id="rIdFont9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font" Target="fonts/font1.odttf"/></Relationships>',
                )
            target.writestr(info, blob)
    return out.getvalue()


def test_a_package_with_a_relationship_to_a_missing_part_is_still_read():
    broken = _with_relationship_to_a_missing_part(_build(_spec))
    # Control: this is the failure the upload used to end in.
    with pytest.raises(KeyError):
        docx.Document(io.BytesIO(broken))
    text = _extract_docx(broken)
    assert "PADDIR | 0x00 | RW" in text and "## Register Map" in text


def test_the_repair_removes_only_the_dangling_relationship():
    from src.domain.source_extraction import _drop_dangling_relationships

    def build(document):
        document.add_paragraph("Body.")
        document.part.relate_to(
            "https://example.test/spec",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
            is_external=True,
        )

    repaired = _drop_dangling_relationships(
        _with_relationship_to_a_missing_part(_build(build))
    )
    with zipfile.ZipFile(io.BytesIO(repaired)) as package:
        rels = package.read("word/_rels/document.xml.rels").decode("utf-8")
    assert "font1.odttf" not in rels
    assert (
        "https://example.test/spec" in rels
    ), "an external relationship has no part to be missing"
    assert "styles.xml" in rels, "a relationship whose part exists must stay"
    assert "Body." in _extract_docx(repaired)
