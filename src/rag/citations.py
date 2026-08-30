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


def strip_unknown_markers(answer: str, known: set[str]) -> str:
    """Remove citation markers that name a passage which was never retrieved.

    A model that invents `[C3]` when only two passages were retrieved used to cost the
    user the WHOLE answer: the guard failed closed and replaced a correct, cited response
    with "no grounded answer could be produced". The valid `[C1]`/`[C2]` citations went
    with it.

    Dropping the dangling marker is both safer and less destructive. Every marker left in
    the text still resolves to a retrieved passage, so nothing is attributed to a source
    that was not consulted -- which is the property the guard exists to protect -- while
    the sourced answer survives.

    Markers are removed with any immediately adjacent whitespace collapsed, so stripping
    `[C3]` from "routing [C1] [C3]." does not leave a double space before the full stop.
    """
    if not answer:
        return answer

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        found = _MARKER_IN_BRACKET.findall(inner)
        if not found:
            legacy = _LEGACY_BARE.match(inner)
            found = [legacy.group(1)] if legacy else []
        if not found:
            # Not a citation at all -- a footnote, a year, a Markdown link. Leave it.
            return match.group(0)
        kept = [value for value in found if f"C{int(value)}" in known]
        if not kept:
            return ""
        if len(kept) == len(found):
            return match.group(0)
        # A grouped marker such as `[C1, C3]` keeps only the retrieved half.
        return "[" + ", ".join(f"C{int(value)}" for value in kept) + "]"

    stripped = _BRACKET.sub(replace, answer)
    # Collapse the gap a removed marker leaves behind, without touching newlines.
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", stripped).strip()
