"""Editing text inside a stored original must change that text and nothing else.

The risks are the ones a find-and-replace on a binary format always has: a match that Word
split across several runs, a spreadsheet number that is not text, a PDF whose old text must
actually leave the file (not just be covered), accents, and a refusal rather than a pretence
for a scan.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from src.domain.source_edit import (
    MAX_REPLACEMENTS,
    Replacement,
    SourceEditError,
    apply_source_edits,
    supports_editing,
    validate_replacements,
)

pymupdf = pytest.importorskip("pymupdf")
docx = pytest.importorskip("docx")
openpyxl = pytest.importorskip("openpyxl")
pptx = pytest.importorskip("pptx")
pymupdf_fonts = pytest.importorskip("pymupdf_fonts")


def edit(filename, data, *pairs, write=True):
    return apply_source_edits(
        filename, data, [Replacement(a, b) for a, b in pairs], write=write
    )


# ── validation ─────────────────────────────────────────────────────────────────────────────


def test_rejects_an_empty_list_an_empty_find_or_a_no_op():
    with pytest.raises(SourceEditError):
        validate_replacements([])
    with pytest.raises(SourceEditError):
        validate_replacements([Replacement("", "x")])
    with pytest.raises(SourceEditError):
        validate_replacements([Replacement("same", "same")])


def test_rejects_too_many_or_too_long():
    with pytest.raises(SourceEditError):
        validate_replacements(
            [Replacement(f"a{i}", "b") for i in range(MAX_REPLACEMENTS + 1)]
        )
    with pytest.raises(SourceEditError):
        validate_replacements([Replacement("a" * 501, "b")])


def test_pictures_are_refused_with_a_reason():
    assert not supports_editing("scan.png") and supports_editing("a.docx")
    with pytest.raises(SourceEditError, match="pixels"):
        edit("scan.tif", b"II*\x00", ("a", "b"))


def test_an_unknown_extension_is_refused():
    with pytest.raises(SourceEditError):
        edit("a.rtf", b"{}", ("a", "b"))


# ── plain text ─────────────────────────────────────────────────────────────────────────────


def test_text_file_replacement_counts_and_keeps_a_bom():
    data = "﻿Lương 1234 và 1234".encode("utf-8")
    outcome = edit("a.txt", data, ("1234", "1244"))
    assert outcome.counts == [2]
    assert outcome.data == "﻿Lương 1244 và 1244".encode("utf-8")


def test_text_dry_run_changes_nothing():
    outcome = edit("a.md", b"abc abc", ("abc", "x"), write=False)
    assert outcome.counts == [2] and outcome.data is None


def test_non_utf8_text_is_refused():
    with pytest.raises(SourceEditError):
        edit("a.txt", b"\xff\xfe\x00bad", ("a", "b"))


# ── Word ───────────────────────────────────────────────────────────────────────────────────


def _docx() -> bytes:
    document = docx.Document()
    document.add_paragraph("Mức lương là 1234 đồng mỗi tháng.")
    split = document.add_paragraph()
    split.add_run("Mã số 12").bold = True
    split.add_run("34 được cấp").italic = True  # "1234" is split across two runs
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Khoản"
    table.cell(0, 1).text = "1234"
    document.sections[0].footer.paragraphs[0].text = "Trang 1234"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _docx_texts(data: bytes) -> list[str]:
    document = docx.Document(io.BytesIO(data))
    texts = [p.text for p in document.paragraphs]
    texts += [c.text for t in document.tables for r in t.rows for c in r.cells]
    texts += [p.text for p in document.sections[0].footer.paragraphs]
    return texts


def test_docx_changes_every_occurrence_including_split_runs_tables_and_footers():
    outcome = edit("a.docx", _docx(), ("1234", "1244"))
    assert outcome.counts == [4]
    texts = _docx_texts(outcome.data)
    assert "Mức lương là 1244 đồng mỗi tháng." in texts
    assert "Mã số 1244 được cấp" in texts
    assert "1244" in texts and "Trang 1244" in texts
    assert not any("1234" in t for t in texts)


def test_docx_split_run_keeps_the_formatting_of_the_run_it_started_in():
    outcome = edit("a.docx", _docx(), ("1234", "1244"))
    paragraph = next(
        p
        for p in docx.Document(io.BytesIO(outcome.data)).paragraphs
        if p.text.startswith("Mã số")
    )
    assert paragraph.runs[0].bold is True
    assert "1244" in paragraph.runs[0].text


def test_docx_keeps_accents_and_other_parts_byte_for_byte():
    original = _docx()
    outcome = edit("a.docx", original, ("đồng", "VNĐ"))
    assert "Mức lương là 1234 VNĐ mỗi tháng." in _docx_texts(outcome.data)
    before, after = zipfile.ZipFile(io.BytesIO(original)), zipfile.ZipFile(
        io.BytesIO(outcome.data)
    )
    assert before.namelist() == after.namelist()
    for name in before.namelist():
        if name != "word/document.xml":
            assert before.read(name) == after.read(name), name


def test_docx_dry_run_counts_without_writing():
    outcome = edit("a.docx", _docx(), ("1234", "1244"), write=False)
    assert outcome.counts == [4] and outcome.data is None


def test_docx_reports_zero_for_text_that_is_not_there():
    assert edit("a.docx", _docx(), ("khong-co", "x"), write=False).counts == [0]


def test_docx_match_is_case_and_accent_sensitive():
    assert edit("a.docx", _docx(), ("DONG", "x"), write=False).counts == [0]
    assert edit("a.docx", _docx(), ("dong", "x"), write=False).counts == [0]


def test_docx_replacements_apply_in_order_to_the_previous_result():
    outcome = edit("a.docx", _docx(), ("1234", "9999"), ("9999", "1244"))
    assert outcome.counts == [4, 4]
    assert "Trang 1244" in _docx_texts(outcome.data)


def test_docx_replacement_may_contain_markup_characters_safely():
    outcome = edit("a.docx", _docx(), ("1234", "<b>&1244</b>"))
    assert "Mức lương là <b>&1244</b> đồng mỗi tháng." in _docx_texts(outcome.data)


def test_a_corrupt_office_file_is_refused():
    with pytest.raises(SourceEditError, match="damaged"):
        edit("a.docx", b"not a zip", ("a", "b"))


# ── Excel ──────────────────────────────────────────────────────────────────────────────────


def _xlsx() -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "Doanh thu 1234"
    sheet["B1"] = 1234
    sheet["C1"] = 12340
    sheet["D1"] = "=B1*2"
    sheet["A2"] = "Ghi chú"
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_xlsx_changes_text_and_exact_numbers_but_not_formulas_or_other_numbers():
    outcome = edit("a.xlsx", _xlsx(), ("1234", "1244"))
    sheet = openpyxl.load_workbook(io.BytesIO(outcome.data)).active
    assert sheet["A1"].value == "Doanh thu 1244"
    assert sheet["B1"].value == 1244
    assert (
        sheet["C1"].value == 12340
    )  # contains 1234 as digits but is a different number
    assert sheet["D1"].value == "=B1*2"
    assert outcome.counts == [2]


def test_xlsx_number_is_not_replaced_by_non_numeric_text():
    outcome = edit("a.xlsx", _xlsx(), ("12340", "abc"))
    assert openpyxl.load_workbook(io.BytesIO(outcome.data)).active["C1"].value == 12340
    assert outcome.counts == [0]


# ── PowerPoint ─────────────────────────────────────────────────────────────────────────────


def _pptx() -> bytes:
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Quy trình 1234"
    box = slide.shapes.add_textbox(0, 0, 100, 100).text_frame
    box.text = "Chi phí: "
    run = box.paragraphs[0].add_run()
    run.text = "12"
    run.font.bold = True
    box.paragraphs[0].add_run().text = "34"
    slide.notes_slide.notes_text_frame.text = "Nhớ số 1234"
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def test_pptx_changes_titles_split_runs_and_notes():
    outcome = edit("a.pptx", _pptx(), ("1234", "1244"))
    assert outcome.counts == [3]
    deck = pptx.Presentation(io.BytesIO(outcome.data))
    slide = deck.slides[0]
    texts = [shape.text_frame.text for shape in slide.shapes if shape.has_text_frame]
    assert "Quy trình 1244" in texts
    assert "Chi phí: 1244" in texts
    assert slide.notes_slide.notes_text_frame.text == "Nhớ số 1244"


# ── PDF ────────────────────────────────────────────────────────────────────────────────────


def _pdf(pages=("Số hiệu 1234 của Công ty", "Trang hai 1234")) -> bytes:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        # Noto Sans, so the Vietnamese in the fixture is really in the file.
        page.insert_font(fontname="notos", fontbuffer=pymupdf_fonts.myfont("notos"))
        page.insert_text((72, 100), text, fontname="notos", fontsize=14)
    return document.tobytes()


def _pdf_text(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def test_pdf_removes_the_old_text_and_shows_the_new():
    outcome = edit("a.pdf", _pdf(("Số hiệu 1234 của Công ty",)), ("1234", "1244"))
    text = _pdf_text(outcome.data)
    assert "1244" in text
    assert "1234" not in text  # truly gone from the file, not merely covered
    assert outcome.counts == [1]


def test_pdf_changes_every_page():
    outcome = edit("a.pdf", _pdf(), ("1234", "1244"))
    assert outcome.counts == [2]
    assert _pdf_text(outcome.data).count("1244") == 2


def test_pdf_can_write_vietnamese_with_diacritics():
    outcome = edit("a.pdf", _pdf(("Số hiệu 1234",)), ("1234", "Bảy mươi ếợ"))
    assert "Bảy mươi ếợ" in _pdf_text(outcome.data)


def test_pdf_keeps_the_text_around_the_change():
    outcome = edit("a.pdf", _pdf(("Số hiệu 1234 của Công ty",)), ("1234", "1244"))
    text = _pdf_text(outcome.data)
    assert "Số hiệu" in text and "của Công ty" in text


def test_pdf_dry_run_counts_and_does_not_modify():
    original = _pdf()
    outcome = edit("a.pdf", original, ("1234", "1244"), write=False)
    assert outcome.counts == [2] and outcome.data is None


def test_pdf_notes_the_font_change():
    outcome = edit("a.pdf", _pdf(), ("1234", "1244"))
    assert any("Noto Sans" in note for note in outcome.notes)


def test_pdf_with_no_text_layer_is_refused_as_a_scan():
    document = pymupdf.open()
    page = document.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pixmap.set_rect(pixmap.irect, (200, 200, 200))
    page.insert_image(pymupdf.Rect(0, 0, 200, 200), pixmap=pixmap)
    with pytest.raises(SourceEditError, match="scan"):
        edit("a.pdf", document.tobytes(), ("a", "b"))


def test_a_mixed_pdf_edits_text_pages_and_says_which_were_left():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "So 1234", fontname="helv", fontsize=14)
    blank = document.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pixmap.set_rect(pixmap.irect, (10, 10, 10))
    blank.insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=pixmap)
    outcome = edit("a.pdf", document.tobytes(), ("1234", "1244"))
    assert outcome.counts == [1]
    assert any("scan" in note for note in outcome.notes)


def test_a_corrupt_pdf_is_refused():
    with pytest.raises(SourceEditError):
        edit("a.pdf", b"%PDF-1.4 garbage", ("a", "b"))


def test_pdf_longer_text_moves_the_rest_of_the_line_instead_of_overlapping_it():
    original = _pdf(("Hạn nộp 12/03 cho tất cả nhân viên.",))
    outcome = edit("a.pdf", original, ("12/03", "31/12/2026"))
    text = _pdf_text(outcome.data)
    assert "31/12/2026" in text and "12/03" not in text
    assert (
        "cho tất cả nhân viên." in text
    )  # the tail survived, drawn after the new text
    assert any("moved along the line" in note for note in outcome.notes)
    # Nothing is drawn on top of anything else: every word box is disjoint from the others.
    with pymupdf.open(stream=outcome.data, filetype="pdf") as document:
        boxes = [pymupdf.Rect(w[:4]) for w in document[0].get_text("words")]
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            assert not (first & second).get_area() > 1, (first, second)


def test_pdf_two_changes_on_one_line_are_both_made_and_counted():
    outcome = edit(
        "a.pdf",
        _pdf(("Từ 12/03 đến 15/03 hằng năm",)),
        ("12/03", "01/01/2027"),
        ("15/03", "02"),
    )
    assert outcome.counts == [1, 1]
    text = _pdf_text(outcome.data)
    assert (
        "01/01/2027" in text
        and "02" in text
        and "12/03" not in text
        and "15/03" not in text
    )
