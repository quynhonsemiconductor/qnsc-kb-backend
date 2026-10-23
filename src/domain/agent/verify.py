"""Step 6: check the numbers in an answer against the material it was written from.

A language model can state a figure that appears in none of the sources it was given, or
quietly transpose one (``1244`` for ``1234``), and a fluent sentence hides it. This is a plain
code check, not a second opinion from a model: every number that looks like a *quantity* in
the answer must also occur in the retrieved passages, in a verified calculation, or in a chart
built from a table. One that occurs nowhere is returned.

Separators are ambiguous (``1.234`` is 1234 in Vietnamese and 1.234 in English), so a token is
read every way it could be read and counts as found if ANY reading is in the sources. That
errs toward silence, on purpose: a false alarm teaches readers to ignore the warning, and a
transposed digit is still caught because no reading of it matches.

Citation markers (``[C1]``), list numbering, years, and small whole numbers (under 100, not
percentages or decimals) are not quantities and are not checked.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

_CITATION = re.compile(
    r"\[(?:C\d{1,3}|Source ID:\s*\d+|\d{1,2})(?:[^\]]{0,200})\]", re.IGNORECASE
)
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*]|\d{1,2}[.)])\s+")
# A number with optional separators and an optional percent sign. The lookarounds keep it
# from starting or ending inside a longer token (an id, a code, a date).
_NUMBER = re.compile(
    r"(?<![\w.,/-])-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?%?(?![\w/-])"
    r"|(?<![\w.,/-])-?\d+(?:[.,]\d+)?%?(?![\w/-])"
)
_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_GROUPED = re.compile(r"^-?\d{1,3}(?:[.,]\d{3})+$")


def _decimal(text: str) -> Decimal | None:
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def interpretations(token: str) -> set[Decimal]:
    """Every value this token could mean, normalised (``Decimal.normalize``)."""
    percent = token.endswith("%")
    text = token.rstrip("%").strip()
    values: set[Decimal] = set()
    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        # The later separator is the decimal point; the other groups thousands.
        decimal_sep = "." if text.rfind(".") > text.rfind(",") else ","
        group_sep = "," if decimal_sep == "." else "."
        candidate = _decimal(text.replace(group_sep, "").replace(decimal_sep, "."))
        if candidate is not None:
            values.add(candidate)
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        if text.count(sep) == 1:
            candidate = _decimal(text.replace(sep, "."))  # as a decimal point
            if candidate is not None:
                values.add(candidate)
        if _GROUPED.match(text):
            candidate = _decimal(text.replace(".", "").replace(",", ""))  # as thousands
            if candidate is not None:
                values.add(candidate)
    else:
        candidate = _decimal(text)
        if candidate is not None:
            values.add(candidate)
    if percent:
        values |= {value / 100 for value in values}
    return {value.normalize() for value in values}


def numbers_in(text: str) -> set[Decimal]:
    """Every reading of every number in free text."""
    found: set[Decimal] = set()
    for match in _NUMBER.finditer(text or ""):
        found |= interpretations(match.group(0))
    return found


def _is_quantity(token: str) -> bool:
    if token.endswith("%") or "." in token or "," in token:
        return True
    plain = token.lstrip("-")
    if _YEAR.match(plain):
        return False
    value = _decimal(plain)
    return value is not None and abs(value) >= 100


def source_numbers(
    context_results: list[dict[str, Any]],
    analytics_results: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> set[Decimal]:
    """Everything the answer is allowed to quote."""
    allowed: set[Decimal] = set()
    for result in context_results:
        for key in ("context_text", "parent_text", "chunk_text", "title"):
            allowed |= numbers_in(str(result.get(key) or ""))
        data = (result.get("evidence") or {}).get("structured_data")
        if isinstance(data, dict):
            rows = data.get("rows")
            for row in rows if isinstance(rows, list) else []:
                for cell in row if isinstance(row, list) else []:
                    allowed |= numbers_in(str(cell))
    for item in analytics_results:
        allowed |= numbers_in(f"{item.get('result', '')} {item.get('formula', '')}")
        operands = item.get("operands")
        for operand in operands if isinstance(operands, list) else []:
            allowed |= numbers_in(
                str(operand.get("value", "")) if isinstance(operand, dict) else ""
            )
    for artifact in artifacts:
        series_list = artifact.get("series")
        for series in series_list if isinstance(series_list, list) else []:
            for value in series.get("values", []):
                if isinstance(value, (int, float)):
                    allowed.add(Decimal(str(value)).normalize())
    return allowed


def find_unverified_numbers(
    answer: str,
    context_results: list[dict[str, Any]],
    analytics_results: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Quantities stated in the answer that no source contains, in order of appearance."""
    cleaned = _LIST_MARKER.sub("", _CITATION.sub(" ", answer or ""))
    allowed = source_numbers(context_results, analytics_results or [], artifacts or [])
    unverified: list[str] = []
    for match in _NUMBER.finditer(cleaned):
        token = match.group(0)
        if not _is_quantity(token):
            continue
        readings = interpretations(token)
        if (
            not readings
            or readings & allowed
            or {abs(value) for value in readings} & allowed
        ):
            continue
        if token not in unverified:
            unverified.append(token)
    return unverified
