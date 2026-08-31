"""Extract searchable text from uploaded knowledge sources."""
from __future__ import annotations

import csv
import io
import json
import os
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
    ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx", ".txt", ".md", ".csv",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}


class SourceExtractionError(ValueError):
    pass


def _scan_with_clamd(data: bytes) -> None:
    """Scan bytes through ClamAV's TCP INSTREAM protocol without temp files."""
    host = settings.MALWARE_SCANNER_HOST
    if not host:
        raise SourceExtractionError("Malware scanning is unavailable")
    try:
        with socket.create_connection((host, settings.MALWARE_SCANNER_PORT), timeout=10) as client:
            client.settimeout(30)
            client.sendall(b"zINSTREAM\0")
            for offset in range(0, len(data), 1024 * 1024):
                chunk = data[offset:offset + 1024 * 1024]
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
    original_tokens = set(re.findall(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9_-]{3,}", original.lower()))
    if not original_tokens:
        return 1.0
    candidate_tokens = set(re.findall(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9_-]{3,}", candidate.lower()))
    return len(original_tokens & candidate_tokens) / len(original_tokens)


def _validate_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > settings.MAX_SOURCE_ARCHIVE_FILES:
                raise SourceExtractionError("The document archive contains too many files")
            total_size = sum(max(0, item.file_size) for item in members)
            if total_size > settings.MAX_SOURCE_UNCOMPRESSED_BYTES:
                raise SourceExtractionError("The document archive expands beyond the allowed size")
            for item in members:
                if item.compress_size and item.file_size / item.compress_size > 10_000:
                    raise SourceExtractionError("The document archive has an unsafe compression ratio")
    except zipfile.BadZipFile as exc:
        raise SourceExtractionError("The uploaded document is not a valid archive") from exc


def _validate_source_bytes(filename: str, data: bytes) -> None:
    extension = Path(filename).suffix.lower()
    if extension in {".docx", ".xlsx", ".xlsm", ".pptx"}:
        if not data.startswith(b"PK"):
            raise SourceExtractionError("The uploaded Office document has an invalid file signature")
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
            raise SourceExtractionError("The uploaded image is invalid or unsafe") from exc

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


def _convert_with_markitdown(filename: str, data: bytes) -> str:
    if not settings.MARKITDOWN_ENABLED:
        return ""
    # MarkItDown is a structure-enhancement layer, not the authoritative
    # reader for formats that already have a lossless native extractor. Some
    # installed MarkItDown versions reject .md/.txt/.csv outright; bypassing
    # them avoids turning a valid upload into a 500 during completion.
    if Path(filename).suffix.lower() in {".md", ".txt", ".csv"}:
        return ""
    converter = _markitdown()
    if converter is None:
        return ""
    try:
        result = converter.convert_stream(
            io.BytesIO(data),
            file_extension=Path(filename).suffix.lower(),
        )
        value = getattr(result, "markdown", None) or getattr(result, "text_content", None) or ""
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
        return PaddleOCR(use_angle_cls=True, lang=settings.PADDLEOCR_LANG, show_log=False, enable_mkldnn=False)


def _ocr_image(image: Any) -> str:
    engine = _paddle_ocr()
    # PaddleOCR 3.x accepts NumPy arrays (not PIL objects) and returns a
    # Result whose JSON payload nests recognition fields under ``res``.
    try:
        import numpy as np
        image_input = np.asarray(image) if not isinstance(image, (str, np.ndarray)) else image
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
                logger.warning("PDF page text extraction failed", page_number=index, error=str(exc))
                pages.append(_page(index, ""))
                failed_pages.append(index)
        if any(item["text"] for item in pages):
            # Mixed PDFs are common: retain embedded text and OCR only image
            # pages instead of silently dropping scanned appendices.
            if any(not item["text"] for item in pages):
                try:
                    import fitz
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
                            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                            item["text"] = _ocr_image(Image.open(io.BytesIO(pixmap.tobytes("png"))))
                        except Exception as exc:
                            logger.warning("PDF page OCR failed", page_number=index + 1, error=str(exc))
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
                        index for index, item in enumerate(pages, start=1)
                        if not item["text"] and index not in failed_pages
                    )
            return pages, sorted(set(failed_pages))
    except SourceExtractionError:
        raise
    except Exception as exc:
        # No embedded text could be read at all; fall through to a full OCR pass.
        logger.info("PDF embedded-text extraction unusable, falling back to OCR", error=str(exc))

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
                ocr_pages.append(_page(index, _ocr_image(Image.open(io.BytesIO(pixmap.tobytes("png"))))))
            except SourceExtractionError:
                # OCR is not installed. That is not a per-page fault and the whole
                # document is unreadable without it, so it stays a hard failure.
                raise
            except Exception as exc:
                logger.warning("PDF page rasterization failed", page_number=index, error=str(exc))
                ocr_pages.append(_page(index, ""))
                ocr_failures.append(index)
        return ocr_pages, ocr_failures
    except SourceExtractionError:
        raise
    except Exception as exc:
        raise SourceExtractionError(f"Could not extract text from PDF: {exc}") from exc


def _extract_docx(data: bytes) -> str:
    from docx import Document
    document = Document(io.BytesIO(data))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return _clean("\n".join(parts))


def _extract_xlsx(data: bytes) -> str:
    from openpyxl import load_workbook
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet in workbook.worksheets:
        parts.append(f"## {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            values = [str(value).strip() for value in row if value is not None and str(value).strip()]
            if values:
                parts.append(" | ".join(values))
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
    return [
        int(item["page_number"])
        for item in pages
        if item.get("extraction_failed")
    ]


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
        pages = [_page(1, "\n".join(" | ".join(cell.strip() for cell in row) for row in rows))]
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
        raise SourceExtractionError(f"Documents are limited to {settings.MAX_SOURCE_PAGES} pages")
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

    MarkItDown supplies document structure (headings, lists, tables and links).
    ``pages`` remains the authoritative page-indexed OCR/text representation
    used for citations and original-source review. If MarkItDown returns an
    incomplete result, the page extraction is used instead.
    """
    fallback = _clean("\n\n".join(str(item["text"]) for item in (pages or extract_source_pages(filename, data))))
    converted = _convert_with_markitdown(filename, data)
    if not converted:
        logger.info("Source Markdown conversion used page extractor", filename=filename, reason="markitdown_unavailable_or_failed")
        return fallback
    coverage = _token_coverage(fallback, converted)
    if len(converted) < max(80, int(len(fallback) * 0.30)) or coverage < 0.80:
        logger.warning(
            "MarkItDown output rejected; using page extractor",
            filename=filename,
            fallback_characters=len(fallback),
            markdown_characters=len(converted),
            token_coverage=round(coverage, 3),
        )
        return fallback
    logger.info(
        "Source converted to Markdown with MarkItDown",
        filename=filename,
        page_extractor_characters=len(fallback),
        markdown_characters=len(converted),
        token_coverage=round(coverage, 3),
    )
    return converted


def extract_source(filename: str, data: bytes) -> str:
    """Return the legacy flattened representation used by article bodies."""
    return _clean("\n\n".join(str(item["text"]) for item in extract_source_pages(filename, data)))
