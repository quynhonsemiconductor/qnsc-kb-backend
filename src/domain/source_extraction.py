"""Extract searchable text from uploaded knowledge sources."""

from __future__ import annotations

import csv
import io
import json
import os
import posixpath
import re
import socket
import struct
import subprocess
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
import structlog
from src.core.config import settings

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".xlsm",
    ".pptx",
    ".txt",
    ".md",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


class SourceExtractionError(ValueError):
    pass


def _scan_with_clamd(data: bytes) -> None:
    """Scan bytes through ClamAV's TCP INSTREAM protocol without temp files."""
    host = settings.MALWARE_SCANNER_HOST
    if not host:
        raise SourceExtractionError("Malware scanning is unavailable")
    try:
        with socket.create_connection(
            (host, settings.MALWARE_SCANNER_PORT), timeout=10
        ) as client:
            client.settimeout(30)
            client.sendall(b"zINSTREAM\0")
            for offset in range(0, len(data), 1024 * 1024):
                chunk = data[offset : offset + 1024 * 1024]
                client.sendall(struct.pack("!I", len(chunk)) + chunk)
            client.sendall(struct.pack("!I", 0))
            response = client.recv(4096).decode("utf-8", errors="replace")
    except OSError as exc:
        raise SourceExtractionError("Malware scanning is unavailable") from exc
    if "OK" not in response or "FOUND" in response:
        raise SourceExtractionError("The uploaded file failed malware scanning")


def _page(number: int, text: str) -> dict[str, Any]:
    return {"page_number": number, "text": _clean(text)}


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _token_coverage(original: str, candidate: str) -> float:
    original_tokens = set(
        re.findall(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9_-]{3,}", original.lower())
    )
    if not original_tokens:
        return 1.0
    candidate_tokens = set(
        re.findall(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9_-]{3,}", candidate.lower())
    )
    return len(original_tokens & candidate_tokens) / len(original_tokens)


def _validate_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > settings.MAX_SOURCE_ARCHIVE_FILES:
                raise SourceExtractionError(
                    "The document archive contains too many files"
                )
            total_size = sum(max(0, item.file_size) for item in members)
            if total_size > settings.MAX_SOURCE_UNCOMPRESSED_BYTES:
                raise SourceExtractionError(
                    "The document archive expands beyond the allowed size"
                )
            for item in members:
                if item.compress_size and item.file_size / item.compress_size > 10_000:
                    raise SourceExtractionError(
                        "The document archive has an unsafe compression ratio"
                    )
    except zipfile.BadZipFile as exc:
        raise SourceExtractionError(
            "The uploaded document is not a valid archive"
        ) from exc


def _validate_source_bytes(filename: str, data: bytes) -> None:
    extension = Path(filename).suffix.lower()
    if extension in {".docx", ".xlsx", ".xlsm", ".pptx"}:
        if not data.startswith(b"PK"):
            raise SourceExtractionError(
                "The uploaded Office document has an invalid file signature"
            )
        _validate_archive(data)
    elif extension == ".pdf" and not data.lstrip().startswith(b"%PDF"):
        raise SourceExtractionError("The uploaded PDF has an invalid file signature")
    elif extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = settings.MAX_SOURCE_IMAGE_PIXELS
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except Exception as exc:
            raise SourceExtractionError(
                "The uploaded image is invalid or unsafe"
            ) from exc

    if settings.MALWARE_SCAN_ENABLED:
        if settings.MALWARE_SCANNER_HOST:
            _scan_with_clamd(data)
        else:
            try:
                result = subprocess.run(
                    [settings.MALWARE_SCANNER_COMMAND, "--stream", "--no-summary"],
                    input=data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                raise SourceExtractionError("Malware scanning is unavailable") from exc
            if result.returncode != 0:
                raise SourceExtractionError("The uploaded file failed malware scanning")


@lru_cache(maxsize=1)
def _markitdown() -> Any:
    """Load MarkItDown once, and say why when it cannot be loaded.

    Every failure here used to return None in silence, and the caller logged the same
    "markitdown_unavailable_or_failed" whether the package was missing or a particular
    PDF had defeated it. A dependency that never loaded in any image was therefore
    indistinguishable from a difficult file, and both looked survivable — so the
    structure layer being dead on every single upload went unnoticed.

    Cached, so an import failure is reported once per process rather than per document.
    """
    try:
        from markitdown import MarkItDown
    except Exception as exc:
        logger.warning(
            "MarkItDown is not importable; Markdown conversion falls back to the "
            "page extractor for every document",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    try:
        return MarkItDown(enable_plugins=False)
    except TypeError:
        # MarkItDown 0.0.x has no plugin constructor argument. Its built-in
        # converters are still sufficient here because PaddleOCR remains the
        # scanned-page fallback.
        pass
    except Exception as exc:
        # Only TypeError was caught before, so any OTHER constructor failure escaped
        # this function entirely — past _convert_with_markitdown, which does not guard
        # this call, and out through extract_source_markdown, which the upload endpoint
        # does not guard either. An optional enhancement layer could take the whole
        # upload down with a 500.
        logger.warning(
            "MarkItDown could not be constructed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    try:
        return MarkItDown()
    except Exception as exc:
        logger.warning(
            "MarkItDown could not be constructed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None


def _strip_heading_emphasis(markdown: str) -> str:
    """Unwrap ``## **Section**`` into ``## Section``.

    PyMuPDF4LLM detects a heading by its font being larger AND bolder, and reports both
    facts: the ``##`` for the size and a ``**...**`` wrapper for the weight. The bold is
    redundant inside a heading, and the asterisks are not cosmetic here —
    ``department_routing._major_sections`` reads the heading text as a section TITLE and
    shows it to the reviewer, so they would surface as ``**2. Equipment**`` in the
    review UI and in every routed candidate.

    Only the wrapper around a whole heading line is removed. Emphasis inside body text
    is content and is left exactly as found.
    """
    return re.sub(
        r"^(#{1,6}\s+)\*\*\s*(.+?)\s*\*\*\s*$",
        r"\1\2",
        markdown,
        flags=re.MULTILINE,
    )


def _convert_pdf_with_pymupdf4llm(data: bytes) -> str:
    """Convert a PDF to Markdown using PyMuPDF's layout model.

    MarkItDown reads PDFs through pdfminer, which returns a flat text stream with no
    heading information at all. PyMuPDF knows each span's font size and weight, so it
    can tell a heading from body text and emit `#`/`##` plus real Markdown tables.

    That difference decides whether the LLM pass is needed: ``split_into_chunks``
    splits on ##/### boundaries, so a PDF with no headings had no structure until a
    model generated some.

    Returns "" on any failure, like every other converter here — this is an
    enhancement layer and the page extractor remains the floor.
    """
    if not settings.PYMUPDF4LLM_ENABLED:
        return ""
    try:
        import pymupdf
        import pymupdf4llm  # type: ignore[import-untyped]  # no py.typed or stubs; trusted PDF converter
    except Exception as exc:
        logger.warning(
            "PyMuPDF4LLM is not importable; PDF Markdown structure falls back to MarkItDown",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return ""
    document = None
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
        _reject_oversized_pdf(document.page_count)
        return _strip_heading_emphasis(
            _clean(str(pymupdf4llm.to_markdown(document, show_progress=False)))
        )
    except SourceExtractionError:
        # The page ceiling is a real rejection, not a converter fault: let it out.
        raise
    except Exception as exc:
        logger.warning(
            "PyMuPDF4LLM conversion failed; trying MarkItDown",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return ""
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:
                pass


#: Formats whose own extractor is the reference reader, so MarkItDown is not asked.
#:
#: MarkItDown reads .docx through mammoth and .xlsx through pandas, which are its optional extras
#: (`markitdown[docx]`, `markitdown[xlsx]`); `pyproject.toml` declares `markitdown` without them, so
#: on every upload of these two types it raised MissingDependencyException, logged a warning and
#: handed the document to the native extractor anyway (51 of 51 in one session's logs). Installing the
#: extra would not be an improvement either: measured on three real specifications, MarkItDown's
#: output escaped identifiers (`gpio\_out`), kept the table of contents and inlined images as data
#: URIs, which is worse to search than `_extract_docx` / `_extract_xlsx`. Same result as before,
#: minus the exception.
_NATIVE_EXTRACTOR_ONLY = {".docx", ".xlsx", ".xlsm"}


def _convert_with_markitdown(filename: str, data: bytes) -> str:
    if not settings.MARKITDOWN_ENABLED:
        return ""
    # MarkItDown is a structure-enhancement layer, not the authoritative
    # reader for formats that already have a lossless native extractor. Some
    # installed MarkItDown versions reject .md/.txt/.csv outright; bypassing
    # them avoids turning a valid upload into a 500 during completion.
    if Path(filename).suffix.lower() in {".md", ".txt", ".csv"}:
        return ""
    if Path(filename).suffix.lower() in _NATIVE_EXTRACTOR_ONLY:
        return ""
    converter = _markitdown()
    if converter is None:
        return ""
    try:
        result = converter.convert_stream(
            io.BytesIO(data),
            file_extension=Path(filename).suffix.lower(),
        )
        value = (
            getattr(result, "markdown", None)
            or getattr(result, "text_content", None)
            or ""
        )
        return _clean(str(value))
    except Exception as exc:
        # MarkItDown is an enhancement layer. A single unsupported or malformed
        # file must still be handled by the existing format-specific extractor —
        # but say which file and which error, so a systematic failure is
        # distinguishable from one awkward document.
        logger.warning(
            "MarkItDown conversion failed; using the page extractor",
            filename=filename,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return ""


@lru_cache(maxsize=1)
def _paddle_ocr() -> Any:
    # PaddlePaddle's Windows oneDNN executor currently fails on OCR model
    # attributes; disable that optional accelerator and use the CPU path.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    try:
        import paddle

        paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_use_onednn": False})
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise SourceExtractionError(
            "PaddleOCR is not available. Install paddlepaddle and paddleocr to process scanned files."
        ) from exc
    try:
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=settings.PADDLEOCR_LANG,
            enable_mkldnn=False,
        )
    except TypeError:
        return PaddleOCR(
            use_angle_cls=True,
            lang=settings.PADDLEOCR_LANG,
            show_log=False,
            enable_mkldnn=False,
        )


def _ocr_image(image: Any) -> str:
    engine = _paddle_ocr()
    # PaddleOCR 3.x accepts NumPy arrays (not PIL objects) and returns a
    # Result whose JSON payload nests recognition fields under ``res``.
    try:
        import numpy as np

        image_input = (
            np.asarray(image) if not isinstance(image, (str, np.ndarray)) else image
        )
    except Exception:
        image_input = image

    def collect_texts(payload: Any) -> list[str]:
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return []
        if not isinstance(payload, dict):
            return []
        result = payload.get("res", payload)
        if not isinstance(result, dict):
            return []
        return [str(value) for value in result.get("rec_texts", []) if value]

    lines: list[str] = []
    if hasattr(engine, "predict"):
        for item in engine.predict(image_input):
            payload = item.json if hasattr(item, "json") else item
            lines.extend(collect_texts(payload))
        if lines:
            return _clean("\n".join(lines))

    try:
        legacy_result = engine.ocr(image_input, cls=True)
    except TypeError:
        # PaddleOCR 3.x keeps ``ocr`` as a compatibility alias but removed
        # the ``cls`` keyword; PaddleOCR 2.x still needs it.
        legacy_result = engine.ocr(image_input)
    for page in legacy_result or []:
        for line in page or []:
            try:
                lines.append(str(line[1][0]))
            except (IndexError, TypeError):
                continue
    return _clean("\n".join(lines))


def _reject_oversized_pdf(page_count: int) -> None:
    """Reject before extraction, not after.

    The limit used to be checked on the RESULT, so a 5,000-page scan was fully
    rasterized and OCR'd — minutes of worker time and every page of it held in memory —
    only to be refused for a page count that was knowable from the header.
    """
    if page_count > settings.MAX_SOURCE_PAGES:
        raise SourceExtractionError(
            f"Documents are limited to {settings.MAX_SOURCE_PAGES} pages"
        )


def _extract_pdf_pages(data: bytes) -> tuple[list[dict[str, Any]], list[int]]:
    """Extract page text, and report which pages could not be read.

    The second element is the page numbers that FAILED, which is not the same as the
    pages that came back empty: a legitimately blank page is readable and yields nothing,
    while an unreadable one means the caller is about to index a document that is missing
    content. Only the caller can decide what to do about that, and it cannot decide at all
    unless the failure is reported instead of swallowed.
    """
    failed_pages: list[int] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        _reject_oversized_pdf(len(reader.pages))
        pages: list[dict[str, Any]] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                pages.append(_page(index, page.extract_text() or ""))
            except Exception as exc:
                # A single corrupt content stream must not decide the fate of the other
                # 400 pages, but it must not vanish either: this page is recorded as
                # failed and reported upwards.
                logger.warning(
                    "PDF page text extraction failed", page_number=index, error=str(exc)
                )
                pages.append(_page(index, ""))
                failed_pages.append(index)
        if any(item["text"] for item in pages):
            # Mixed PDFs are common: retain embedded text and OCR only image
            # pages instead of silently dropping scanned appendices.
            if any(not item["text"] for item in pages):
                try:
                    import fitz  # type: ignore[import-untyped]  # PyMuPDF ships no py.typed; no stubs installed
                    from PIL import Image

                    document = fitz.open(stream=data, filetype="pdf")
                    for index, item in enumerate(pages):
                        if item["text"]:
                            continue
                        page = document[index]
                        if not page.get_images(full=True):
                            # No image and no text: genuinely blank, not a failure.
                            continue
                        try:
                            pixmap = page.get_pixmap(
                                matrix=fitz.Matrix(1.5, 1.5), alpha=False
                            )
                            item["text"] = _ocr_image(
                                Image.open(io.BytesIO(pixmap.tobytes("png")))
                            )
                        except Exception as exc:
                            logger.warning(
                                "PDF page OCR failed",
                                page_number=index + 1,
                                error=str(exc),
                            )
                        if not item["text"] and index + 1 not in failed_pages:
                            # An image page that produced nothing is unread content,
                            # whether OCR raised or simply returned empty.
                            failed_pages.append(index + 1)
                except Exception as exc:
                    # OCR being absent entirely is a deployment condition rather than a
                    # per-page fault, so the embedded text still stands. The image-only
                    # pages are reported as failures so the document is not treated as
                    # completely extracted.
                    logger.warning("PDF OCR pass unavailable", error=str(exc))
                    failed_pages.extend(
                        index
                        for index, item in enumerate(pages, start=1)
                        if not item["text"] and index not in failed_pages
                    )
            return pages, sorted(set(failed_pages))
    except SourceExtractionError:
        raise
    except Exception as exc:
        # No embedded text could be read at all; fall through to a full OCR pass.
        logger.info(
            "PDF embedded-text extraction unusable, falling back to OCR", error=str(exc)
        )

    try:
        import fitz
        from PIL import Image

        document = fitz.open(stream=data, filetype="pdf")
        _reject_oversized_pdf(document.page_count)
        ocr_pages: list[dict[str, Any]] = []
        ocr_failures: list[int] = []
        for index, page in enumerate(document, start=1):
            try:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                ocr_pages.append(
                    _page(
                        index, _ocr_image(Image.open(io.BytesIO(pixmap.tobytes("png"))))
                    )
                )
            except SourceExtractionError:
                # OCR is not installed. That is not a per-page fault and the whole
                # document is unreadable without it, so it stays a hard failure.
                raise
            except Exception as exc:
                logger.warning(
                    "PDF page rasterization failed", page_number=index, error=str(exc)
                )
                ocr_pages.append(_page(index, ""))
                ocr_failures.append(index)
        return ocr_pages, ocr_failures
    except SourceExtractionError:
        raise
    except Exception as exc:
        raise SourceExtractionError(f"Could not extract text from PDF: {exc}") from exc


_DOCX_SKIPPED_STYLE = re.compile(
    r"^(?:toc\b|table of (?:figures|tables|contents))", re.I
)
_DOCX_HEADING_STYLE = re.compile(r"^heading\s*([1-9])$", re.I)


def _drop_dangling_relationships(data: bytes) -> bytes:
    """Rewrite a .docx without the relationships whose target part is not in the package.

    Some producers (generators, converters) write `fontTable.xml.rels -> fonts/font1.odttf`, or a
    customXml item, and then do not ship the part. Word opens such a file; python-docx does not, it
    raises KeyError("There is no item named 'word/fonts/font1.odttf' in the archive") from
    Document() and the whole upload fails with "Could not process uploaded source". Measured on two of
    the 22 files of a real specification set. A relationship to a part that is not there carries nothing
    to read, so removing it loses nothing.
    """
    from lxml import etree  # type: ignore[import-untyped]  # lxml-stubs is not a declared dependency

    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        names = set(source.namelist())
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                blob = source.read(info.filename)
                if info.filename.endswith(".rels"):
                    # word/_rels/fontTable.xml.rels describes the part word/fontTable.xml.
                    base = posixpath.dirname(posixpath.dirname(info.filename))
                    root = etree.fromstring(blob, parser)
                    changed = False
                    for rel in list(root):
                        href = rel.get("Target") or ""
                        if rel.get("TargetMode") == "External" or not href:
                            continue
                        resolved = (
                            href.lstrip("/")
                            if href.startswith("/")
                            else posixpath.normpath(posixpath.join(base, href))
                        )
                        if resolved not in names:
                            root.remove(rel)
                            changed = True
                    if changed:
                        blob = etree.tostring(
                            root,
                            xml_declaration=True,
                            encoding="UTF-8",
                            standalone=True,
                        )
                target.writestr(info, blob)
    return out.getvalue()


def _open_docx(data: bytes):
    from docx import Document

    try:
        return Document(io.BytesIO(data))
    except KeyError:
        # A relationship points at a part the package does not contain; see above.
        logger.info("DOCX has relationships to missing parts; reading it without them")
        return Document(io.BytesIO(_drop_dangling_relationships(data)))


def _docx_cell_text(cell: Any) -> str:
    return " ".join(cell.text.split())


def _extract_docx(data: bytes) -> str:
    """The document's text in the ORDER it is written, headings as Markdown headings.

    Paragraphs and tables used to be read separately, all paragraphs first and every table after
    them. For a specification that put each register table at the very end of the text, detached from
    the section it belongs to and from the heading the chunker attaches to a chunk: "PADDIR | 0x00 |
    RW" sat after the glossary, and no chunk said which block it described. Here a table stays under
    the heading above it. The table of contents (and of figures/tables) is left out: it only repeats
    every section title, so a search for a title matched the contents page as often as the section.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = _open_docx(data)
    parts: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = " ".join(paragraph.text.split())
            if not text:
                continue
            style = (
                (paragraph.style.name if paragraph.style is not None else "") or ""
            ).strip()
            if _DOCX_SKIPPED_STYLE.match(style):
                continue
            heading = _DOCX_HEADING_STYLE.match(style)
            if heading:
                parts.append(f"\n\n{'#' * int(heading.group(1))} {text}\n")
            elif style.lower() == "title":
                parts.append(f"\n\n# {text}\n")
            else:
                parts.append(paragraph.text.strip())
        elif tag == "tbl":
            rows: list[str] = []
            for row in Table(child, document).rows:
                cells: list[str] = []
                previous = None
                for cell in row.cells:
                    # A merged cell is returned once per grid column it spans.
                    if cell._tc is previous:
                        continue
                    previous = cell._tc
                    cells.append(_docx_cell_text(cell))
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                parts.append("\n\n" + "\n".join(rows) + "\n")
    return _clean("\n".join(parts))


def _extract_xlsx(data: bytes) -> str:
    """Each sheet as a heading, each row as `a | b | c`, a blank line between tables.

    Three things the first version got wrong, measured on a real pin-list and bus-generator workbook:

    * an empty cell was dropped, so a row `Name | | Description` became `Name | Description` and the
      description read as the second column's value (11 rows between the two files);
    * a cell with a line break (69 + 6 of them) put the rest of its text on a line of its own, which no
      longer looks like a table row, so the chunker stopped treating the block as a table;
    * a blank row, which is how a sheet separates one table from the next, vanished, running tables
      into each other.

    Empty cells inside a row are kept (trailing ones are not), a cell's own whitespace and line breaks
    become single spaces, and a blank row becomes a blank line.
    """
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet in workbook.worksheets:
        parts.append(f"\n## {sheet.title}\n")
        after_blank_row = False
        for row in sheet.iter_rows(values_only=True):
            cells = [
                "" if value is None else " ".join(str(value).split()) for value in row
            ]
            while cells and not cells[-1]:
                cells.pop()
            if not any(cells):
                after_blank_row = True
                continue
            if after_blank_row:
                parts.append("")
                after_blank_row = False
            parts.append(" | ".join(cells))
    return _clean("\n".join(parts))


def _extract_pptx(data: bytes) -> str:
    from pptx import Presentation

    presentation = Presentation(io.BytesIO(data))
    parts: list[str] = []
    for index, slide in enumerate(presentation.slides, start=1):
        parts.append(f"## Slide {index}")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
    return _clean("\n".join(parts))


def extraction_failed_pages(pages: list[dict[str, Any]]) -> list[int]:
    """Page numbers this extraction could not read, for callers that must flag it."""
    return [int(item["page_number"]) for item in pages if item.get("extraction_failed")]


def extract_source_pages(filename: str, data: bytes) -> list[dict[str, Any]]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise SourceExtractionError(
            f"Unsupported file type '{extension or 'unknown'}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not data:
        raise SourceExtractionError("The uploaded file is empty.")
    _validate_source_bytes(filename, data)
    failed_pages: list[int] = []
    if extension == ".pdf":
        pages, failed_pages = _extract_pdf_pages(data)
    elif extension == ".docx":
        pages = [_page(1, _extract_docx(data))]
    elif extension in {".xlsx", ".xlsm"}:
        pages = [_page(1, _extract_xlsx(data))]
    elif extension == ".pptx":
        pages = [_page(1, _extract_pptx(data))]
    elif extension == ".csv":
        rows = csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace")))
        pages = [
            _page(
                1, "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
            )
        ]
    elif extension in {".txt", ".md"}:
        pages = [_page(1, data.decode("utf-8-sig", errors="replace"))]
    else:
        from PIL import Image
        import numpy as np

        Image.MAX_IMAGE_PIXELS = settings.MAX_SOURCE_IMAGE_PIXELS
        image = Image.open(io.BytesIO(data)).convert("RGB")
        pages = [_page(1, _ocr_image(np.asarray(image)))]
    # An unreadable page is KEPT, marked, and carried into ``page_texts``. Dropping every
    # textless page made a partially-unreadable document indistinguishable from a complete
    # one: the corpus was quietly missing content, citations pointed at pages that were
    # never read, and nothing in the record said so. Everything downstream selects pages by
    # `item["text"]` being non-empty, so a marker page is never indexed or cited — it only
    # makes the gap visible.
    failures = set(failed_pages)
    pages = [
        {**item, "extraction_failed": True} if item["page_number"] in failures else item
        for item in pages
        if item["text"] or item["page_number"] in failures
    ]
    # The page-count ceiling is enforced before extraction for PDFs (_reject_oversized_pdf);
    # this catches the other formats, whose page count is only known once parsed.
    if len(pages) > settings.MAX_SOURCE_PAGES:
        raise SourceExtractionError(
            f"Documents are limited to {settings.MAX_SOURCE_PAGES} pages"
        )
    if sum(len(str(item["text"])) for item in pages) > settings.MAX_SOURCE_TEXT_CHARS:
        raise SourceExtractionError("The extracted document text is too large")
    if not any(item["text"] for item in pages):
        raise SourceExtractionError("No readable text was found in the uploaded file.")
    if failures:
        logger.warning(
            "Source extracted with unreadable pages",
            filename=filename,
            failed_pages=sorted(failures),
            total_pages=len(pages),
        )
    return pages


def extract_source_markdown(
    filename: str,
    data: bytes,
    pages: list[dict[str, Any]] | None = None,
) -> str:
    """Convert a source to Markdown without weakening page-aware extraction.

    Two converters supply document structure (headings, lists, tables and links):
    PyMuPDF4LLM for PDFs, MarkItDown for everything else. ``pages`` remains the
    authoritative page-indexed OCR/text representation used for citations and
    original-source review. If both converters return an incomplete result, the page
    extraction is used instead.

    The log lines name the converter that produced the output, because "conversion
    used the page extractor" previously covered three different situations and a
    systematically dead converter looked the same as one awkward document.
    """
    fallback = _clean(
        "\n\n".join(
            str(item["text"])
            for item in (pages or extract_source_pages(filename, data))
        )
    )
    converted = ""
    converter_used = "markitdown"
    if Path(filename).suffix.lower() == ".pdf":
        # PDFs go to PyMuPDF4LLM first because MarkItDown's pdfminer path emits no
        # headings, and headings are what decide whether the LLM pass is needed at all.
        converted = _convert_pdf_with_pymupdf4llm(data)
        if converted:
            converter_used = "pymupdf4llm"
    if not converted:
        converted = _convert_with_markitdown(filename, data)
    if not converted:
        logger.info(
            "Source Markdown conversion used page extractor",
            filename=filename,
            reason="every_converter_unavailable_or_failed",
        )
        return fallback
    coverage = _token_coverage(fallback, converted)
    if len(converted) < max(80, int(len(fallback) * 0.30)) or coverage < 0.80:
        logger.warning(
            "Markdown conversion output rejected; using page extractor",
            filename=filename,
            converter=converter_used,
            fallback_characters=len(fallback),
            markdown_characters=len(converted),
            token_coverage=round(coverage, 3),
        )
        return fallback
    logger.info(
        "Source converted to Markdown",
        filename=filename,
        converter=converter_used,
        page_extractor_characters=len(fallback),
        markdown_characters=len(converted),
        token_coverage=round(coverage, 3),
    )
    return converted


def extract_source(filename: str, data: bytes) -> str:
    """Return the legacy flattened representation used by article bodies."""
    return _clean(
        "\n\n".join(str(item["text"]) for item in extract_source_pages(filename, data))
    )
