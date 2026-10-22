"""Correct text inside a stored original file, leaving everything else in it alone.

A reviewer who spots a wrong figure (``1234`` where the document should say ``1244``) should
be able to fix it in the document people actually read, not only in the extracted Markdown
beside it. This module takes the bytes of the stored file and a list of find/replace pairs and
returns the bytes of a corrected copy. It never touches the original: the caller stores the
result as a NEW revision, so the file as first uploaded stays available for comparison.

What each format supports, and the limits, because the limits are what a reader of the result
needs to know:

* Word, PowerPoint, Excel (``.docx`` ``.pptx`` ``.xlsx`` ``.xlsm``): the XML inside the package
  is edited directly, so styles, images, charts, formulas and macros are carried over exactly.
  A match may span several runs (Word splits text wherever formatting changes); the
  replacement takes the formatting of the run the match started in.
  In a workbook a *shared* string is one entry used by every cell that holds it, so editing it
  changes all of them; a number is changed only when the cell holds exactly that number and is
  not a formula.
* PDF with a text layer: the old text is removed (redacted) and the new text is drawn in its
  place with Noto Sans, at the same size, colour and position. The original font is usually a
  subset that lacks glyphs for whatever is typed, so the typeface can differ, and a match that
  wraps across two lines is not found. Images, vector graphics and everything else on the page
  are kept.
* ``.txt`` ``.md`` ``.csv``: plain text replacement.
* Scanned PDFs and pictures (``.png`` ``.jpg`` ``.tif`` ``.bmp`` ``.webp``) have no text to edit,
  only pixels, and are refused with a reason rather than faked.

Matching is exact (case and accents significant) and replacements apply in order, each to the
result of the one before.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

MAX_REPLACEMENTS = 20
MAX_TEXT_LENGTH = 500

_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_TEXT_EXTENSIONS = {".txt", ".md", ".csv"}


class SourceEditError(ValueError):
    """The edit cannot be made. The message is written for the person who asked for it."""


@dataclass(frozen=True)
class Replacement:
    find: str
    replace: str


@dataclass
class EditOutcome:
    #: How many places each replacement matched, in the order given.
    counts: list[int]
    #: The corrected file, or ``None`` for a dry run.
    data: bytes | None
    #: Plain-language caveats about this particular result (a font change, a scanned page).
    notes: list[str] = field(default_factory=list)


def validate_replacements(replacements: Iterable[Replacement]) -> list[Replacement]:
    items = list(replacements)
    if not items:
        raise SourceEditError("Give at least one piece of text to change.")
    if len(items) > MAX_REPLACEMENTS:
        raise SourceEditError(
            f"At most {MAX_REPLACEMENTS} changes can be made at once."
        )
    for item in items:
        if not item.find:
            raise SourceEditError("The text to find cannot be empty.")
        if len(item.find) > MAX_TEXT_LENGTH or len(item.replace) > MAX_TEXT_LENGTH:
            raise SourceEditError(f"Text is limited to {MAX_TEXT_LENGTH} characters.")
        if item.find == item.replace:
            raise SourceEditError("The new text is the same as the old text.")
    return items


def supports_editing(filename: str) -> bool:
    return Path(filename).suffix.lower() not in _IMAGE_EXTENSIONS


def apply_source_edits(
    filename: str,
    data: bytes,
    replacements: Iterable[Replacement],
    *,
    write: bool = True,
) -> EditOutcome:
    """Apply the replacements to a file. ``write=False`` only counts what would change."""
    items = validate_replacements(replacements)
    extension = Path(filename).suffix.lower()
    if extension in _IMAGE_EXTENSIONS:
        raise SourceEditError(
            "This file is a picture or a scan: it has no text to edit, only pixels. "
            "Edit the extracted text instead."
        )
    if extension in {".docx", ".pptx", ".xlsx", ".xlsm"}:
        return _edit_ooxml(extension, data, items, write)
    if extension == ".pdf":
        return _edit_pdf(data, items, write)
    if extension in _TEXT_EXTENSIONS:
        return _edit_text(data, items, write)
    raise SourceEditError(
        f"Editing {extension or 'this kind of file'} in place is not supported."
    )


# ── plain text ─────────────────────────────────────────────────────────────────────────────


def _edit_text(data: bytes, items: list[Replacement], write: bool) -> EditOutcome:
    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceEditError(
            "This text file is not UTF-8, so it cannot be edited safely."
        ) from exc
    counts: list[int] = []
    for item in items:
        counts.append(text.count(item.find))
        text = text.replace(item.find, item.replace)
    if not write:
        return EditOutcome(counts, None)
    encoded = text.encode("utf-8")
    return EditOutcome(counts, (b"\xef\xbb\xbf" + encoded) if bom else encoded)


# ── Office Open XML ────────────────────────────────────────────────────────────────────────


def _part_names(extension: str, names: list[str]) -> list[str]:
    if extension == ".docx":
        pattern = r"word/(document|header\d*|footer\d*|footnotes|endnotes)\.xml"
    elif extension == ".pptx":
        pattern = r"ppt/(slides/slide|notesSlides/notesSlide)\d+\.xml"
    else:
        return [
            n
            for n in names
            if n == "xl/sharedStrings.xml"
            or re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)
        ]
    return [n for n in names if re.fullmatch(pattern, n)]


def _paragraph_tag(extension: str, part: str) -> tuple[str, str]:
    """The element that holds one run of text, and the one inside it that carries the text."""
    if extension == ".docx":
        return _W + "p", _W + "t"
    if extension == ".pptx":
        return _A + "p", _A + "t"
    if part.endswith("sharedStrings.xml"):
        return _S + "si", _S + "t"
    # A worksheet holds strings inline when the writer did not use the shared table.
    return _S + "is", _S + "t"


def _texts_of(paragraph: Any, paragraph_tag: str, text_tag: str) -> list[Any]:
    """Text elements belonging to this paragraph itself.

    Not those of a paragraph nested inside it (a text box inside a Word paragraph), which is
    visited in its own right, and not a spreadsheet's phonetic guide (``rPh``), which is
    pronunciation rather than content.
    """
    found: list[Any] = []
    stack = list(reversed(list(paragraph)))
    while stack:
        node = stack.pop()
        tag = node.tag
        if tag == paragraph_tag or tag == _S + "rPh":
            continue
        if tag == text_tag:
            found.append(node)
            continue
        stack.extend(reversed(list(node)))
    return found


def _replace_in_paragraph(
    texts: list[Any], items: list[Replacement], counts: list[int]
) -> bool:
    changed = False
    for index, item in enumerate(items):
        pieces = [node.text or "" for node in texts]
        joined = "".join(pieces)
        if item.find not in joined:
            continue
        starts = []
        position = joined.find(item.find)
        while position != -1:
            starts.append(position)
            position = joined.find(item.find, position + len(item.find))
        offsets = []
        running = 0
        for piece in pieces:
            offsets.append(running)
            running += len(piece)
        # Right to left so an earlier edit does not move the offsets of a later one.
        for start in reversed(starts):
            end = start + len(item.find)
            first = max(i for i, offset in enumerate(offsets) if offset <= start)
            last = max(i for i, offset in enumerate(offsets) if offset < end)
            head = pieces[first][: start - offsets[first]]
            tail = pieces[last][end - offsets[last] :]
            pieces[first] = head + item.replace + (tail if first == last else "")
            for middle in range(first + 1, last):
                pieces[middle] = ""
            if last != first:
                pieces[last] = tail
        for node, piece in zip(texts, pieces):
            if (node.text or "") != piece:
                node.text = piece
                if piece != piece.strip():
                    node.set(_XML_SPACE, "preserve")
        counts[index] += len(starts)
        changed = True
    return changed


def _edit_numeric_cells(root: Any, items: list[Replacement], counts: list[int]) -> bool:
    """A number typed into a spreadsheet is stored as ``<v>1234</v>``, not as text."""
    changed = False
    for cell in root.iter(_S + "c"):
        if cell.get("t") not in (None, "n") or cell.find(_S + "f") is not None:
            continue
        value = cell.find(_S + "v")
        if value is None or value.text is None:
            continue
        for index, item in enumerate(items):
            if value.text.strip() != item.find.strip():
                continue
            try:
                float(item.replace)
            except ValueError:
                continue
            value.text = item.replace.strip()
            counts[index] += 1
            changed = True
            break
    return changed


def _edit_ooxml(
    extension: str, data: bytes, items: list[Replacement], write: bool
) -> EditOutcome:
    from lxml import etree  # type: ignore[import-untyped]  # lxml-stubs is not a declared dependency

    counts = [0] * len(items)
    try:
        source = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise SourceEditError(
            "This file is damaged or is not an Office document."
        ) from exc
    with source:
        parts = _part_names(extension, source.namelist())
        replaced: dict[str, bytes] = {}
        # Never resolve entities or fetch anything: the file is untrusted.
        parser = etree.XMLParser(
            resolve_entities=False, no_network=True, huge_tree=False
        )
        for part in parts:
            root = etree.fromstring(source.read(part), parser)
            changed = False
            tags = _paragraph_tag(extension, part)
            for paragraph in root.iter(tags[0]):
                texts = _texts_of(paragraph, *tags)
                if texts and _replace_in_paragraph(texts, items, counts):
                    changed = True
            if extension in {".xlsx", ".xlsm"} and not part.endswith(
                "sharedStrings.xml"
            ):
                changed = _edit_numeric_cells(root, items, counts) or changed
            if changed:
                replaced[part] = etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8", standalone=True
                )
        if not write:
            return EditOutcome(counts, None)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as target:
            for info in source.infolist():
                target.writestr(
                    info, replaced.get(info.filename, source.read(info.filename))
                )
    return EditOutcome(counts, output.getvalue())


# ── PDF ────────────────────────────────────────────────────────────────────────────────────

# Redaction modes, spelled as the numbers pymupdf defines (PDF_REDACT_IMAGE_NONE and
# PDF_REDACT_LINE_ART_NONE are both 0): keep images and vector art under a redacted rectangle.
_REDACT_IMAGE_NONE = 0
_REDACT_LINE_ART_NONE = 0

_PDF_FONT_REGULAR, _PDF_FONT_BOLD, _PDF_FONT_ITALIC, _PDF_FONT_BOLD_ITALIC = (
    "notos",
    "notosbo",
    "notosit",
    "notosbi",
)


@dataclass
class _Hit:
    rect: Any
    text: str
    size: float
    color: tuple[float, float, float]
    origin: tuple[float, float]
    bold: bool
    italic: bool
    #: The text to the end of the line was redrawn too, in one style.
    reflowed: bool = False


def _line_chars(line: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    return [
        (char["c"], char, span)
        for span in line.get("spans", [])
        for char in span.get("chars", [])
    ]


def _style(
    span: dict[str, Any],
) -> tuple[float, tuple[float, float, float], bool, bool]:
    flags = int(span.get("flags", 0))
    name = str(span.get("font", "")).lower()
    color = int(span.get("color", 0))
    return (
        float(span.get("size", 11)),
        (((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255),
        bool(flags & 16) or "bold" in name,
        bool(flags & 2) or "italic" in name or "oblique" in name,
    )


def _font_name(bold: bool, italic: bool) -> str:
    if bold and italic:
        return _PDF_FONT_BOLD_ITALIC
    return _PDF_FONT_BOLD if bold else _PDF_FONT_ITALIC if italic else _PDF_FONT_REGULAR


def _union(pymupdf: Any, chars: list[tuple[str, Any, dict[str, Any]]]) -> Any:
    rect = pymupdf.Rect(chars[0][1]["bbox"])
    for _, char, _ in chars[1:]:
        rect |= pymupdf.Rect(char["bbox"])
    return rect


def _line_occurrences(
    joined: str, items: list[Replacement], counts: list[int]
) -> list[tuple[int, int, str]]:
    """Non-overlapping matches in a line as (start, end, replacement), left to right."""
    found: list[tuple[int, int, str, int]] = []
    for index, item in enumerate(items):
        position = joined.find(item.find)
        while position != -1:
            end = position + len(item.find)
            found.append((position, end, item.replace, index))
            position = joined.find(item.find, end)
    found.sort(key=lambda entry: (entry[0], entry[3]))
    kept: list[tuple[int, int, str]] = []
    last_end = 0
    for start, end, replacement, index in found:
        if start < last_end:
            continue
        kept.append((start, end, replacement))
        counts[index] += 1
        last_end = end
    return kept


def _find_hits(
    page: Any, items: list[Replacement], counts: list[int], fonts: dict[str, Any]
) -> tuple[list[_Hit], bool]:
    """Every exact match on a page, as a place to redact and the text to put there.

    Replacement text that is wider than what it replaces would run into the words after it, so
    from the first such match to the end of the line the whole tail is redrawn, shifted, in
    one style (that of the first character), as a word processor would re-flow it.
    """
    import pymupdf

    hits: list[_Hit] = []
    has_text = False
    for block in page.get_text("rawdict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            chars = _line_chars(line)
            if not chars:
                continue
            has_text = True
            joined = "".join(c for c, _, _ in chars)
            occurrences = _line_occurrences(joined, items, counts)
            cursor = 0
            for number, (start, end, replacement) in enumerate(occurrences):
                size, color, bold, italic = _style(chars[start][2])
                rect = _union(pymupdf, chars[start:end])
                width = fonts[_font_name(bold, italic)].text_length(
                    replacement, fontsize=size
                )
                if (
                    width <= rect.width + 0.5
                    or end == len(chars)
                    and width <= page.rect.x1 - 36 - rect.x0
                ):
                    hits.append(
                        _Hit(
                            rect,
                            replacement,
                            size,
                            color,
                            tuple(chars[start][1]["origin"]),
                            bold,
                            italic,
                        )
                    )
                    cursor = end
                    continue
                # Wider than the original and something follows it: redraw to the line's end.
                pieces: list[str] = []
                position = start
                for later_start, later_end, later_replacement in occurrences[number:]:
                    pieces.append(joined[position:later_start])
                    pieces.append(later_replacement)
                    position = later_end
                pieces.append(joined[position:])
                tail_rect = _union(pymupdf, chars[start:])
                hits.append(
                    _Hit(
                        tail_rect,
                        "".join(pieces),
                        size,
                        color,
                        tuple(chars[start][1]["origin"]),
                        bold,
                        italic,
                        reflowed=True,
                    )
                )
                break
    return hits, has_text


def _background(page: Any, rect: Any) -> tuple[float, float, float]:
    """The most common colour under a rectangle, so the patch blends into a tinted cell."""
    try:
        pixmap = page.get_pixmap(clip=rect, dpi=72, alpha=False)
        samples = pixmap.samples
        step = pixmap.n
        colour, _ = Counter(
            tuple(samples[i : i + 3]) for i in range(0, len(samples) - 2, step)
        ).most_common(1)[0]
        return tuple(channel / 255 for channel in colour)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - a blank patch beats a failed edit
        return (1.0, 1.0, 1.0)


def _edit_pdf(data: bytes, items: list[Replacement], write: bool) -> EditOutcome:
    import pymupdf

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise SourceEditError("This PDF could not be opened.") from exc
    with document:
        if document.needs_pass or document.is_encrypted:
            raise SourceEditError(
                "This PDF is password-protected, so it cannot be edited."
            )
        counts = [0] * len(items)
        notes: set[str] = set()
        scanned_pages = 0
        fonts = _pdf_fonts()
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            hits, has_text = _find_hits(page, items, counts, fonts)
            if not has_text:
                scanned_pages += 1
            if not (hits and write):
                continue
            for hit in hits:
                page.add_redact_annot(hit.rect, fill=_background(page, hit.rect))
            page.apply_redactions(
                images=_REDACT_IMAGE_NONE,
                graphics=_REDACT_LINE_ART_NONE,
            )
            registered: set[str] = set()
            for hit in hits:
                name = _font_name(hit.bold, hit.italic)
                if name not in registered:
                    page.insert_font(fontname=name, fontbuffer=fonts[name].buffer)
                    registered.add(name)
                size = hit.size
                width = fonts[name].text_length(hit.text, fontsize=size)
                room = (
                    max(hit.rect.width, page.rect.x1 - 36 - hit.rect.x0)
                    if hit.reflowed
                    else hit.rect.width
                )
                if width > room > 0:
                    # Still too wide: shrink a little rather than run off the page or over the
                    # next word, but not so far that it becomes unreadable.
                    size = max(size * 0.6, size * room / width)
                page.insert_text(
                    hit.origin,
                    hit.text,
                    fontname=name,
                    fontsize=size,
                    color=hit.color,
                    overlay=True,
                )
            if any(hit.reflowed for hit in hits):
                notes.add(
                    "Text after a longer change was moved along the line and drawn in one style."
                )
        if scanned_pages == document.page_count:
            raise SourceEditError(
                "This PDF is a scan: it has no text layer to edit, only page images. "
                "Edit the extracted text instead."
            )
        if scanned_pages:
            notes.add(f"{scanned_pages} page(s) are scans and were left unchanged.")
        if any(counts):
            notes.add(
                "New text is drawn in Noto Sans; the typeface may differ from the surrounding text."
            )
        if not write:
            return EditOutcome(counts, None, sorted(notes))
        return EditOutcome(
            counts, document.tobytes(garbage=3, deflate=True), sorted(notes)
        )


def _pdf_fonts() -> dict[str, Any]:
    """Noto Sans (regular, bold, italic, bold italic): the only fonts here with Vietnamese."""
    try:
        import pymupdf
        import pymupdf_fonts  # type: ignore[import-untyped,import-not-found]  # PyMuPDF's font pack ships no stubs
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise SourceEditError("Editing PDFs needs the pymupdf-fonts package.") from exc
    return {
        name: pymupdf.Font(fontbuffer=pymupdf_fonts.myfont(name))
        for name in (
            _PDF_FONT_REGULAR,
            _PDF_FONT_BOLD,
            _PDF_FONT_ITALIC,
            _PDF_FONT_BOLD_ITALIC,
        )
    }
