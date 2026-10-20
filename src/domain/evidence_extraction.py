"""Bounded, local extraction of tables and visual document evidence."""

from __future__ import annotations

import csv
import hashlib
import io
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from src.core.config import settings

# v2 crops visual evidence to each figure instead of to the whole page. Rows written by
# v1 carry a page-sized bounding box and are not upgraded in place: `rebuild_article_index`
# deletes and recreates an article's evidence, so a reindex is what replaces them.
EXTRACTION_VERSION = "evidence-v2"


def select_page_figures(
    placements: Sequence[tuple[int, tuple[float, float, float, float]]],
    *,
    page_width: float,
    page_height: float,
    page_count: int,
    xref_page_counts: Mapping[int, int] | None = None,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Choose which embedded images on one page are worth citing as figures.

    `placements` is `(xref, (x0, y0, x1, y1))` in page points, as PyMuPDF reports them;
    the same xref may appear more than once on a page. Returns the kept placements in
    reading order, each as a single rect — repeats of one xref are unioned, because two
    rects for one image are almost always a tiled or double-drawn placement of the same
    artwork rather than two separate figures.

    Deliberately pure: no PyMuPDF types cross this boundary, so the judgement that
    decides what a reviewer sees can be tested without a PDF engine.

    Three ways to be decoration, and an image only has to match one:
      * too small a share of the page (an icon),
      * too small in absolute points on either side (a rule or divider),
      * present on too much of the document (a logo, a letterhead, a watermark).
    """
    page_area = max(page_width, 0.0) * max(page_height, 0.0)
    if page_area <= 0 or not placements:
        return []

    counts = xref_page_counts or {}
    template_pages = max(
        settings.EVIDENCE_FIGURE_TEMPLATE_MIN_PAGES,
        int(page_count * settings.EVIDENCE_FIGURE_TEMPLATE_PAGE_FRACTION),
    )

    unioned: dict[int, tuple[float, float, float, float]] = {}
    order: list[int] = []
    for xref, rect in placements:
        x0, y0, x1, y1 = (float(value) for value in rect)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        previous = unioned.get(xref)
        if previous is None:
            unioned[xref] = (x0, y0, x1, y1)
            order.append(xref)
        else:
            unioned[xref] = (
                min(previous[0], x0),
                min(previous[1], y0),
                max(previous[2], x1),
                max(previous[3], y1),
            )

    kept: list[tuple[int, tuple[float, float, float, float]]] = []
    for xref in order:
        x0, y0, x1, y1 = unioned[xref]
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            continue
        if (width * height) / page_area < settings.EVIDENCE_FIGURE_MIN_PAGE_FRACTION:
            continue
        if min(width, height) < settings.EVIDENCE_FIGURE_MIN_POINTS:
            continue
        # A big image is still decoration when the document repeats it. Checked last
        # because it is the only signal that needs to know about the other pages.
        if page_count > 1 and counts.get(xref, 1) >= template_pages:
            continue
        kept.append((xref, (x0, y0, x1, y1)))

    # Reading order, so "Figure 1" is the top-most figure rather than whichever xref the
    # PDF happened to list first.
    kept.sort(key=lambda item: (round(item[1][1], 1), round(item[1][0], 1)))
    return kept


def figure_caption_text(
    lines: Sequence[tuple[tuple[float, float, float, float], str]],
    figure_rect: tuple[float, float, float, float],
    *,
    page_height: float,
) -> str:
    """Return nearby text for a figure, for retrieval rather than for display.

    A cropped figure carries no words, so without this a figure is unsearchable: the
    page's text was the only reason the old whole-page evidence could be retrieved at
    all. Lines are taken from a band just above and below the image, which is where a
    caption sits, and from inside it, which is where a diagram's own labels sit.
    """
    margin = max(0.0, page_height) * settings.EVIDENCE_FIGURE_CAPTION_MARGIN
    x0, y0, x1, y1 = figure_rect
    picked: list[tuple[float, str]] = []
    for (lx0, ly0, lx1, ly1), text in lines:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            continue
        vertically_near = ly1 >= y0 - margin and ly0 <= y1 + margin
        horizontally_overlapping = lx1 >= x0 and lx0 <= x1
        if vertically_near and horizontally_overlapping:
            picked.append((ly0, cleaned))
    picked.sort(key=lambda item: item[0])
    return " ".join(text for _top, text in picked)


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
        [_clean_cell(cell) for cell in islice(row, settings.EVIDENCE_MAX_TABLE_COLUMNS)]
        for row in islice(rows, settings.EVIDENCE_MAX_TABLE_ROWS + 1)
    ]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return None
    # openpyxl may yield every column up to `max_col` even when a sheet uses only the
    # first few, while other versions stop at the worksheet's actual dimension. Trim the
    # common empty tail so extraction is stable across versions and does not serialize
    # dozens of phantom columns into each row.
    width = max(
        max((index + 1 for index, value in enumerate(row) if value), default=0)
        for row in cleaned
    )
    if width == 0:
        return None
    cleaned = [row[:width] + [""] * max(0, width - len(row)) for row in cleaned]
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


def _page_image_placements(
    page: Any,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Every embedded image placement on one page, as `(xref, rect)` in points."""
    placements: list[tuple[int, tuple[float, float, float, float]]] = []
    try:
        images = page.get_images(full=True)
    except Exception:
        return placements
    for image in images:
        xref = int(image[0])
        try:
            rects = list(page.get_image_rects(xref))
        except Exception:
            # An image the page references but does not place (or that this build cannot
            # locate) has no region to crop, so it cannot become a figure. The page_visual
            # fallback still covers the page.
            continue
        for rect in rects:
            placements.append(
                (xref, (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))
            )
    return placements


def _page_text_lines(
    page: Any,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Text lines with their rectangles, for figure captions."""
    lines: list[tuple[tuple[float, float, float, float], str]] = []
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return lines
    for block in blocks:
        for line in block.get("lines", []) or []:
            text = "".join(span.get("text", "") for span in line.get("spans", []) or [])
            box = line.get("bbox")
            if not text.strip() or not box or len(box) != 4:
                continue
            lines.append(
                (
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    text,
                )
            )
    return lines


def _pdf_evidence(data: bytes) -> list[dict[str, Any]]:
    import pymupdf

    document = pymupdf.open(stream=data, filetype="pdf")
    specs: list[dict[str, Any]] = []
    try:
        # One pass to learn which images repeat across the document, because a logo can
        # only be recognised by being everywhere. Cheap: `get_images` reads the page's
        # resource list, it does not decode pixels.
        xref_page_counts: dict[int, int] = {}
        for page in cast(Iterable[Any], document):
            seen_on_page: set[int] = set()
            try:
                images = page.get_images(full=True)
            except Exception:
                continue
            for image in images:
                seen_on_page.add(int(image[0]))
            for xref in seen_on_page:
                xref_page_counts[xref] = xref_page_counts.get(xref, 0) + 1

        for page_index, page in enumerate(cast(Iterable[Any], document), start=1):
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
            # Each real figure, cropped to itself. The whole page used to be stored as
            # one artifact, which meant a citation showed the figure plus everything
            # else that happened to share the page.
            figures = select_page_figures(
                _page_image_placements(page),
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                page_count=document.page_count,
                xref_page_counts=xref_page_counts,
            )
            if figures:
                lines = _page_text_lines(page)
                for figure_index, (xref, rect) in enumerate(figures, start=1):
                    label = f"Figure {figure_index} on page {page_index}"
                    caption = figure_caption_text(
                        lines, rect, page_height=float(page.rect.height)
                    )
                    search_text = f"{label}\n{caption}".strip()
                    specs.append(
                        {
                            "kind": "figure",
                            "page_number": page_index,
                            "label": label,
                            "search_text": search_text[
                                : settings.EVIDENCE_MAX_SEARCH_TEXT_CHARS
                            ],
                            "structured_data": None,
                            "bounding_box": [float(value) for value in rect],
                            # The page's own size, not the figure's: the renderer
                            # intersects the clip with `page.rect`, and the stored
                            # dimensions are what the UI maps the box against.
                            "page_width": float(page.rect.width),
                            "page_height": float(page.rect.height),
                            # Keyed on the image and its placement, so two figures on one
                            # page never collide and the hash is stable across runs. The
                            # caption is excluded: re-running extraction on the same PDF
                            # must produce the same identity.
                            "content_hash": hashlib.sha256(
                                "figure:{}:{}:{}".format(
                                    page_index,
                                    xref,
                                    ",".join(f"{value:.2f}" for value in rect),
                                ).encode("utf-8")
                            ).hexdigest(),
                            "extraction_version": EXTRACTION_VERSION,
                        }
                    )
                    if len(specs) >= settings.EVIDENCE_MAX_ARTIFACTS_PER_DOCUMENT:
                        break
            elif page.get_drawings():
                # Vector artwork — a chart or diagram drawn rather than embedded — has no
                # image rect to crop to, so the page remains the only region available.
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


def table_evidence_preview(
    structured_data: dict[str, Any] | None,
) -> dict[str, Any] | None:
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
        with Image.open(io.BytesIO(data)) as source_image:
            image = source_image.convert("RGB")
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
