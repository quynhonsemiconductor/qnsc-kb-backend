"""Citation parsing compatible with DocNexus and legacy QNSC answers."""
import re


def extract_citation_indices(answer: str) -> list[int]:
    matches = re.findall(r"\[Source ID:\s*(\d+)\]|\[(\d+)\]", answer or "")
    return sorted({int(first or second) for first, second in matches})


# Backend-issued markers are always C-prefixed (`[C1]`, `[Source ID: C1]`).
# Bare numeric markers are accepted for legacy/model-lenient output, but only
# as 1-2 digit indices: a bare `[2024]` or `[12345]` is a year, footnote, or
# Markdown reference in the answer text, not a citation, and treating it as
# one fails the whole grounded answer closed.
_CITATION_MARKER = re.compile(
    r"\[Source ID:\s*(C\d+)\]|\[(C\d+)\]"
    r"|\[Source ID:\s*([0-9]{1,2})\]|\[([0-9]{1,2})\]",
    re.IGNORECASE,
)


def extract_citation_ids(answer: str) -> list[str]:
    """Extract backend-issued citation IDs, accepting legacy numeric markers."""
    ids: set[str] = set()
    for groups in _CITATION_MARKER.findall(answer or ""):
        value = next(part for part in groups if part)
        value = value.upper()
        ids.add(value if value.startswith("C") else f"C{value}")
    return sorted(ids, key=lambda value: int(value[1:]))
