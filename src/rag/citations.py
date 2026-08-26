"""Citation parsing compatible with DocNexus and legacy QNSC answers."""
import re


def extract_citation_indices(answer: str) -> list[int]:
    matches = re.findall(r"\[Source ID:\s*(\d+)\]|\[(\d+)\]", answer or "")
    return sorted({int(first or second) for first, second in matches})


# One bracketed run, capped so a stray `[` cannot swallow the rest of the answer.
_BRACKET = re.compile(r"\[([^\[\]]{1,240})\]")
# Backend-issued markers are C-prefixed. The prompt asks for a bare `[C1]`, but a model
# that is also told to surface review dates and owners writes what it considers one
# citation: `[C4: 2026-08-08T16:34:46, admin@example.com]` or `[C1, C2]`. Requiring the
# bracket to close immediately after the id read those as NO citation at all, and an
# uncited grounded answer is refused wholesale — the user lost a correct, sourced answer
# over punctuation. So the id is matched inside the bracket instead of against it.
_MARKER_IN_BRACKET = re.compile(r"\bC(\d{1,3})\b", re.IGNORECASE)
# Bare numeric markers stay legacy-only and 1-2 digits: a lone `[2024]` or `[12345]` is a
# year, footnote, or Markdown reference, not a citation, and promoting one to a citation
# fails the whole grounded answer closed.
_LEGACY_BARE = re.compile(r"^\s*(?:Source ID:\s*)?([0-9]{1,2})\s*$", re.IGNORECASE)


def extract_citation_ids(answer: str) -> list[str]:
    """Extract backend-issued citation IDs, accepting legacy numeric markers."""
    ids: set[str] = set()
    for inner in _BRACKET.findall(answer or ""):
        found = _MARKER_IN_BRACKET.findall(inner)
        if found:
            ids.update(f"C{int(value)}" for value in found)
            continue
        legacy = _LEGACY_BARE.match(inner)
        if legacy:
            ids.add(f"C{int(legacy.group(1))}")
    return sorted(ids, key=lambda value: int(value[1:]))
