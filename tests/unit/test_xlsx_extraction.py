"""Reading a .xlsx keeps the columns, one table per block, and one line per row.

Measured on a real pin list and bus-generator workbook: 11 rows had an empty cell in the middle (the
following cells shifted left), 75 cells held a line break (the row was cut into several lines), and the
blank rows between stacked tables were lost.
"""

from __future__ import annotations

import io

import pytest

openpyxl = pytest.importorskip("openpyxl")

from src.domain.source_extraction import _extract_xlsx  # noqa: E402


def _book(build) -> bytes:
    workbook = openpyxl.Workbook()
    build(workbook)
    out = io.BytesIO()
    workbook.save(out)
    return out.getvalue()


def test_an_empty_cell_inside_a_row_keeps_the_columns_aligned():
    def build(workbook):
        sheet = workbook.active
        sheet.title = "Pins"
        sheet.append(["Pin", "Alt", "Note"])
        sheet.append(["PIN5", None, "GPIO0_7"])

    lines = _extract_xlsx(_book(build)).splitlines()
    assert "Pin | Alt | Note" in lines
    assert "PIN5 | | GPIO0_7" in lines, lines


def test_trailing_empty_cells_are_not_padded():
    def build(workbook):
        sheet = workbook.active
        sheet.append(["Only", "two", None, None])

    assert "Only | two" in _extract_xlsx(_book(build)).splitlines()


def test_a_line_break_inside_a_cell_does_not_split_the_row():
    def build(workbook):
        sheet = workbook.active
        sheet.append(
            ["Data Width", 32, "Bit-width of the APB data bus.\nFixed at 32-bit."]
        )

    assert (
        "Data Width | 32 | Bit-width of the APB data bus. Fixed at 32-bit."
        in _extract_xlsx(_book(build)).splitlines()
    )


def test_a_blank_row_separates_two_tables_into_two_blocks():
    def build(workbook):
        sheet = workbook.active
        sheet.append(["A", "B"])
        sheet.append(["1", "2"])
        sheet.append([])
        sheet.append(["C", "D"])
        sheet.append(["3", "4"])

    blocks = [b for b in _extract_xlsx(_book(build)).split("\n\n") if "|" in b]
    assert blocks == ["A | B\n1 | 2", "C | D\n3 | 4"], blocks


def test_each_sheet_gets_its_own_heading_block():
    def build(workbook):
        workbook.active.title = "CFG"
        workbook.active.append(["k", "v"])
        workbook.create_sheet("MM").append(["x", "y"])

    text = _extract_xlsx(_book(build))
    assert "\n\n## CFG\n\nk | v" in "\n\n" + text
    assert "\n\n## MM\n\nx | y" in text
