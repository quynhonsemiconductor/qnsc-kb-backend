import csv
import io

import pytest

from src.core.config import settings
from src.domain.evidence_extraction import (
    extract_document_evidence,
    table_evidence_preview,
)


def test_csv_becomes_bounded_searchable_table(monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_MAX_TABLE_ROWS", 2)
    data = "Quarter,Revenue\nQ1,100\nQ2,200\nQ3,300\n".encode()

    evidence = extract_document_evidence("revenue.csv", data)

    assert len(evidence) == 1
    table = evidence[0]
    assert table["kind"] == "table"
    assert table["structured_data"] == {
        "columns": ["Quarter", "Revenue"],
        "rows": [["Q1", "100"], ["Q2", "200"]],
    }
    assert "Revenue" in table["search_text"]
    assert len(table["content_hash"]) == 64


def test_plain_text_does_not_create_fake_visual_evidence():
    assert extract_document_evidence("notes.txt", b"plain text") == []
    assert extract_document_evidence("notes.md", b"# Markdown") == []


def test_image_is_one_visual_artifact_without_decoding_or_remote_calls():
    evidence = extract_document_evidence("diagram.png", b"server-validated-image-bytes")

    assert evidence == [
        {
            "kind": "page_visual",
            "page_number": 1,
            "label": "Uploaded image",
            "search_text": "Uploaded image",
            "structured_data": None,
            "content_hash": evidence[0]["content_hash"],
            "extraction_version": "evidence-v1",
        }
    ]


def test_xlsx_reads_cached_values_only_and_bounds_rows(monkeypatch):
    Workbook = pytest.importorskip("openpyxl").Workbook
    monkeypatch.setattr(settings, "EVIDENCE_MAX_TABLE_ROWS", 2)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Forecast"
    sheet.append(["Month", "Units"])
    sheet.append(["Jan", 10])
    sheet.append(["Feb", 20])
    sheet.append(["Mar", "=SUM(B2:B3)"])
    output = io.BytesIO()
    workbook.save(output)

    evidence = extract_document_evidence("forecast.xlsx", output.getvalue())

    assert evidence[0]["label"] == "Worksheet Forecast"
    assert evidence[0]["structured_data"]["rows"] == [["Jan", "10"], ["Feb", "20"]]


def test_citation_table_preview_is_bounded_and_reports_total(monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_TABLE_PREVIEW_ROWS", 2)

    preview = table_evidence_preview(
        {
            "columns": ["Quarter", "Revenue"],
            "rows": [["Q1", "100"], ["Q2", "200"], ["Q3", "300"]],
        }
    )

    assert preview == {
        "columns": ["Quarter", "Revenue"],
        "rows": [["Q1", "100"], ["Q2", "200"]],
        "total_rows": 3,
        "truncated": True,
    }


def test_table_columns_and_cells_are_bounded(monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_MAX_TABLE_COLUMNS", 2)
    monkeypatch.setattr(settings, "EVIDENCE_MAX_CELL_CHARS", 4)
    data = io.StringIO()
    csv.writer(data).writerows([["LongHeader", "B", "C"], ["abcdef", "2", "3"]])

    table = extract_document_evidence("bounded.csv", data.getvalue().encode())[0]

    assert table["structured_data"] == {
        "columns": ["Long", "B"],
        "rows": [["abcd", "2"]],
    }
