"""Parsing and rendering for grounded and general-knowledge answer sections."""
from __future__ import annotations

import re

GROUNDED_SENTINEL = "<<<GROUNDED>>>"
EXTENDED_SENTINEL = "<<<EXTENDED>>>"
GROUNDED_HEADING = "## Answer from the Knowledge Base"
EXTENDED_HEADING = "## Additional context (general knowledge — not from the Knowledge Base, not cited)"

# Tolerant on the bracket count by necessity. The sentinels are written by a language
# model, not by a protocol: glm-4.5-flash emits `<<<GROUNDED>>` — two closing angles —
# and an exact match then failed to find any section boundary at all. The consequences
# were not cosmetic. The raw `<<<GROUNDED>>` showed up in the answer, and everything
# after `<<<EXTENDED>>` — uncited general knowledge, by definition — was folded into the
# grounded section instead of being separated, stripped of markers, and put under its
# own "not from the Knowledge Base" heading.
_SENTINEL_LINE_RE = re.compile(r"^\s*<{2,4}\s*(GROUNDED|EXTENDED)\s*>{2,4}\s*$", re.IGNORECASE)
_SENTINEL_WORDS = ("GROUNDED", "EXTENDED")


def could_begin_a_sentinel(probe: str) -> bool:
    """Whether `probe` is still a possible start of a sentinel line.

    Used by the incremental stream to decide what to withhold. It must accept the same
    malformed shapes `_SENTINEL_LINE_RE` does, or a `<<GROUNDED>>` would stream out one
    character at a time before the completed line was recognised as a boundary.
    """

    if not probe.startswith("<"):
        return False
    body = probe.lstrip("<")
    if len(probe) - len(body) > 4:
        return False
    name = body.rstrip(">").upper()
    return any(word.startswith(name) for word in _SENTINEL_WORDS)
# Matches the bare `[C1]` the prompt asks for AND the enriched bracket a model writes
# when it is also told to surface review dates and owners: `[C4: 2026-08-08, a@b.c]`.
# The extended section must carry no marker in either shape — nothing there is
# attributable to the knowledge base.
_CITATION_MARKER_RE = re.compile(
    r"\[[^\[\]]{0,240}?\bC\d{1,3}\b[^\[\]]{0,240}?\]|\[(?:Source ID:\s*)?\d{1,2}\]",
    re.IGNORECASE,
)


def _sentinel_lines(text: str) -> list[tuple[str, int, int]]:
    lines = text.splitlines(keepends=True)
    found: list[tuple[str, int, int]] = []
    offset = 0
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            match = _SENTINEL_LINE_RE.match(line.rstrip("\r\n"))
            if match:
                found.append((match.group(1).upper(), offset, offset + len(line)))
        offset += len(line)
    return found


def _strip_sentinel_lines(text: str) -> str:
    kept: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            kept.append(line)
        elif in_fence or not _SENTINEL_LINE_RE.match(line):
            kept.append(line)
    return "\n".join(kept)


def normalize_answer_markdown(answer: str) -> str:
    """Remove malformed fence/citation fragments without changing content."""
    value = (answer or "").strip()
    citation_markers = r"((?:\[(?:Source ID:\s*)?C?\d+\]\s*)+)"
    value = re.sub(
        rf"(?m)^([ \t]*```)[ \t]+{citation_markers}$",
        r"\1\n\n\2",
        value,
    )
    value = re.sub(r"```[ \t]*\[[ \t]*$", "```", value)
    value = re.sub(r"(?m)^[ \t]*\[[ \t]*$", "", value)
    return value.strip()


def split_answer_sections(raw: str) -> tuple[str, str]:
    """Return normalized grounded and extended sections.

    Unmarked output is deliberately treated as grounded. Sentinel-like text in
    fenced code blocks is content, not a section boundary.
    """
    text = raw or ""
    markers = _sentinel_lines(text)
    grounded_marker = next((item for item in markers if item[0] == "GROUNDED"), None)
    if grounded_marker is None:
        return normalize_answer_markdown(_strip_sentinel_lines(text)), ""

    extended_marker = next(
        (item for item in markers if item[0] == "EXTENDED" and item[1] >= grounded_marker[2]),
        None,
    )
    prefix = text[:grounded_marker[1]]
    if extended_marker:
        grounded_raw = prefix + text[grounded_marker[2]:extended_marker[1]]
        extended_raw = text[extended_marker[2]:]
    else:
        grounded_raw = prefix + text[grounded_marker[2]:]
        extended_raw = ""
    return (
        normalize_answer_markdown(_strip_sentinel_lines(grounded_raw)),
        normalize_answer_markdown(_strip_sentinel_lines(extended_raw)),
    )


#: A line is only ever a fence or a sentinel. While the text received so far could still
#: grow into one of these, it is held back; the moment it cannot, it is safe to release.
_FENCE = "```"


class IncrementalAnswerStream:
    """Release provider tokens the instant they cannot belong to a sentinel line.

    The provider streams, but the raw stream cannot be forwarded as-is: it carries
    ``<<<GROUNDED>>>`` / ``<<<EXTENDED>>>`` section markers, and half of one on screen is
    worse than no streaming at all. The previous answer to that was to forward NOTHING and
    replace the message atomically at the end — correct, but it turned a streaming
    provider into a spinner followed by a wall of text.

    Only two constructs matter, and both are whole-line: a fence toggle and a sentinel. So
    the rule is simply that a line is withheld only while what has arrived of it is still a
    prefix of one of those. "The" is released immediately; "<" waits one character. In
    practice that is token-level streaming for everything except the sentinels themselves.

    Text after ``<<<EXTENDED>>>`` is never released: the extended section is rendered under
    its own heading behind a divider, and arrives with the final replace event.
    """

    def __init__(self) -> None:
        self._line = ""
        # How much of the current line has already gone out. Once any of a line has been
        # released, that line is committed: it cannot turn out to be a sentinel.
        self._released = 0
        self._in_fence = False
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _holding(self) -> bool:
        if self._released:
            return False
        probe = self._line.lstrip()
        if not probe:
            # Leading whitespace only: "   <<<GROUNDED>>>" is still reachable.
            return True
        return _FENCE.startswith(probe) or could_begin_a_sentinel(probe)

    def _finish_line(self, out: list[str]) -> None:
        line = self._line
        if line.strip().startswith("```"):
            self._in_fence = not self._in_fence
        elif not self._in_fence:
            match = _SENTINEL_LINE_RE.match(line)
            if match:
                # Guaranteed unreleased: a sentinel is always held by _holding.
                if match.group(1).upper() == "EXTENDED":
                    self._stopped = True
                self._line = ""
                self._released = 0
                return
        out.append(line[self._released:] + "\n")
        self._line = ""
        self._released = 0

    def feed(self, chunk: str) -> str:
        """Consume raw provider text; return only what is safe to show now."""
        if self._stopped or not chunk:
            return ""
        out: list[str] = []
        for character in chunk:
            if character == "\n":
                self._finish_line(out)
                if self._stopped:
                    break
            else:
                self._line += character
                if not self._holding():
                    out.append(self._line[self._released:])
                    self._released = len(self._line)
        return "".join(out)

    def finish(self) -> str:
        """Flush a trailing line that never got its newline."""
        if self._stopped:
            return ""
        line = self._line
        self._line = ""
        released, self._released = self._released, 0
        if not self._in_fence and _SENTINEL_LINE_RE.match(line):
            return ""
        return line[released:]


def strip_citation_markers(text: str) -> str:
    """Remove source markers from the non-grounded section."""
    cleaned = _CITATION_MARKER_RE.sub("", text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def render_answer_sections(grounded: str, extended: str, enabled: bool) -> str:
    """Build the copy-safe combined answer returned by the API/UI."""
    grounded = normalize_answer_markdown(grounded)
    extended = normalize_answer_markdown(extended)
    if not enabled:
        return grounded
    rendered = f"{GROUNDED_HEADING}\n\n{grounded}" if grounded else ""
    if extended:
        rendered += f"\n\n---\n\n{EXTENDED_HEADING}\n\n{extended}"
    return rendered.strip()
