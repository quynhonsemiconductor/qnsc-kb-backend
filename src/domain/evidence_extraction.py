"""Bounded, local extraction of tables and visual document evidence."""
from __future__ import annotations

import csv
import hashlib
import io
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from src.core.config import settings

EXTRACTION_VERSION = "evidence-v1"


def _clean_cell(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[
        : settings.EVIDENCE_MAX_CELL_CHARS
    ]


def _table_spec(
    rows: Iterable[Iterable[Any]], *, page_number: int = 1, label: str = "Table"
) -> dict[str, Any] | None:
    # Consume at most header + configured data rows. CSV and Office parsers may expose
    # very large iterables, and converting the entire source to a list before slicing
    # defeats the memory bound this function exists to enforce.
    cleaned = [
        [
            _clean_cell(cell)
            for cell in islice(row, settings.EVIDENCE_MAX_TABLE_COLUMNS)
        ]
        for row in islice(rows, settings.EVIDENCE_MAX_TABLE_ROWS + 1)
    ]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return None
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    first = cleaned[0]
    columns = [value or f"Column {index + 1}" for index, value in enumerate(first)]
    data_rows = cleaned[1:]
    preview_lines = [" | ".join(columns), *[" | ".join(row) for row in data_rows]]
    search_text = f"{label}\n" + "\n".join(preview_lines)
    structured = {"columns": columns, "rows": data_rows}
    digest = hashlib.sha256(search_text.encode("utf-8")).hexdigest()
    return {
        "kind": "table",
        "page_number": page_number,
        "label": label[:255],
        "search_text": search_text[: settings.EVIDENCE_MAX_SEARCH_TEXT_CHARS],
        "structured_data": structured,
        "content_hash": digest,
        "extraction_version": EXTRACTION_VERSION,
    }


def _pdf_evidence(data: bytes) -> list[dict[str, Any]]:
    import pymupdf

    document = pymupdf.open(stream=data, filetype="pdf")
    specs: list[dict[str, Any]] = []
    try:
        for page_index, page in enumerate(document, start=1):
            if len(specs) >= settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT:
                break
            try:
                finder = getattr(page, "find_tables", None)
                tables = finder().tables if callable(finder) else []
            except Exception:
                tables = []
            for table_index, table in enumerate(tables, start=1):
                spec = _table_spec(
                    table.extract() or [],
                    page_number=page_index,
                    label=f"Table {table_index} on page {page_index}",
                )
                if spec:
                    rectangle = getattr(table, "bbox", None)
                    if rectangle:
                        spec["bounding_box"] = [float(value) for value in rectangle]
                    spec["page_width"] = float(page.rect.width)
                    spec["page_height"] = float(page.rect.height)
                    specs.append(spec)
                if len(specs) >= settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT:
                    break
            if len(specs) >= settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT:
                break
            # A page containing raster images or vector drawings remains useful visual
            # evidence even when its text has already been extracted for ordinary RAG.
            if page.get_images(full=True) or page.get_drawings():
                text = " ".join(page.get_text("text").split())
                label = f"Visual evidence on page {page_index}"
                search_text = f"{label}\n{text}".strip()
                specs.append(
                    {
                        "kind": "page_visual",
                        "page_number": page_index,
                        "label": label,
                        "search_text": search_text[
                            : settings.EVIDENCE_MAX_SEARCH_TEXT_CHARS
                        ],
                        "structured_data": None,
                        "bounding_box": [
                            float(page.rect.x0),
                            float(page.rect.y0),
                            float(page.rect.x1),
                            float(page.rect.y1),
                        ],
                        "page_width": float(page.rect.width),
                        "page_height": float(page.rect.height),
                        "content_hash": hashlib.sha256(
                            f"page:{page_index}:{search_text}".encode("utf-8")
                        ).hexdigest(),
                        "extraction_version": EXTRACTION_VERSION,
                    }
                )
        return specs[: settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT]
    finally:
        document.close()


def _xlsx_evidence(data: bytes) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    specs: list[dict[str, Any]] = []
    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(
                max_row=settings.EVIDENCE_MAX_TABLE_ROWS + 1,
                max_col=settings.EVIDENCE_MAX_TABLE_COLUMNS,
                values_only=True,
            )
            spec = _table_spec(rows, label=f"Worksheet {sheet.title}")
            if spec:
                specs.append(spec)
            if len(specs) >= settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT:
                break
    finally:
        workbook.close()
    return specs


def _docx_evidence(data: bytes) -> list[dict[str, Any]]:
    from docx import Document

    document = Document(io.BytesIO(data))
    specs: list[dict[str, Any]] = []
    for index, table in enumerate(document.tables, start=1):
        spec = _table_spec(
            ([cell.text for cell in row.cells] for row in table.rows),
            label=f"Table {index}",
        )
        if spec:
            specs.append(spec)
        if len(specs) >= settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT:
            break
    return specs


def _pptx_evidence(data: bytes) -> list[dict[str, Any]]:
    from pptx import Presentation

    presentation = Presentation(io.BytesIO(data))
    specs: list[dict[str, Any]] = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        for shape in slide.shapes:
            if not getattr(shape, "has_table", False):
                continue
            spec = _table_spec(
                ([cell.text for cell in row.cells] for row in shape.table.rows),
                page_number=slide_index,
                label=f"Table on slide {slide_index}",
            )
            if spec:
                specs.append(spec)
            if len(specs) >= settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT:
                return specs
    return specs


def extract_document_evidence(filename: str, data: bytes) -> list[dict[str, Any]]:
    """Return searchable evidence specs without executing document-controlled content."""
    if not settings.EVIDENCE_EXTRACTION_ENABLED or not data:
        return []
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return _pdf_evidence(data)
    if extension in {".xlsx", ".xlsm"}:
        return _xlsx_evidence(data)
    if extension == ".docx":
        return _docx_evidence(data)
    if extension == ".pptx":
        return _pptx_evidence(data)
    if extension == ".csv":
        rows = csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace")))
        spec = _table_spec(rows, label="CSV table")
        return [spec] if spec else []
    if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        text = ""
        label = "Uploaded image"
        return [
            {
                "kind": "page_visual",
                "page_number": 1,
                "label": label,
                "search_text": text or label,
                "structured_data": None,
                "content_hash": hashlib.sha256(data).hexdigest(),
                "extraction_version": EXTRACTION_VERSION,
            }
        ]
    return []


def table_evidence_preview(structured_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a bounded citation projection without copying the full table into SSE."""
    if not isinstance(structured_data, dict):
        return None
    columns = structured_data.get("columns")
    rows = structured_data.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None
    preview_rows = rows[: settings.EVIDENCE_TABLE_PREVIEW_ROWS]
    return {
        "columns": columns,
        "rows": preview_rows,
        "total_rows": len(rows),
        "truncated": len(preview_rows) < len(rows),
    }


def render_evidence_preview(
    filename: str, data: bytes, page_number: int, bounding_box: list[float] | None
) -> bytes:
    """Render one authorized PDF region or image to a bounded inert PNG."""
    from PIL import Image

    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        import pymupdf

        document = pymupdf.open(stream=data, filetype="pdf")
        try:
            if page_number < 1 or page_number > document.page_count:
                raise ValueError("Evidence page is not available")
            page = document[page_number - 1]
            clip = pymupdf.Rect(bounding_box) if bounding_box else page.rect
            clip &= page.rect
            if clip.is_empty:
                raise ValueError("Evidence region is invalid")
            scale = min(
                settings.EVIDENCE_PREVIEW_MAX_PIXELS / max(clip.width, clip.height),
                2.0,
            )
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(max(scale, 0.5), max(scale, 0.5)),
                clip=clip,
                alpha=False,
            )
            result = pixmap.tobytes("png")
        finally:
            document.close()
    elif extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            image.thumbnail(
                (
                    settings.EVIDENCE_PREVIEW_MAX_PIXELS,
                    settings.EVIDENCE_PREVIEW_MAX_PIXELS,
                )
            )
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            result = output.getvalue()
    else:
        raise ValueError("This evidence format has no visual preview")
    if len(result) > settings.EVIDENCE_PREVIEW_MAX_BYTES:
        raise ValueError("Evidence preview exceeds the configured size limit")
    return result
