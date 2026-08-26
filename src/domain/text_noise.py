"""Noise removal for the RETRIEVAL representation of a document.

WHERE THIS APPLIES. Only to the text on its way into chunks and embeddings. It never
touches ``DocumentSource.page_texts`` (the audit copy behind the PDF review view) or
``Article.body_md`` (the approved reading representation, protected by the token and
numeric coverage guards in content_restructure). Those stay byte-identical; this is a
projection of them for search.

WHY IT IS NEEDED. When a source has no restructured body, indexing.py indexes ONE
SECTION PER PAGE. A running header such as "Cong ty ABC - Quy dinh noi bo | Trang 3/45"
therefore lands in every chunk of the document. That pulls every embedding toward the
boilerplate rather than the content, and makes every chunk a lexical match for the
company name — so a query mentioning it retrieves the whole corpus, ranked by accident.

WHY IT IS CONSERVATIVE. Deleting real content is worse than keeping noise: a passage
that never gets indexed cannot be found, and nobody sees an error. So every rule here
needs positive evidence that a line is furniture rather than content:

* a repeated line must recur across a MAJORITY of at least three pages, and be short
  enough to be a header or footer;
* a page label must be the ENTIRE line, which means it carries no retrieval signal to
  lose in the first place;
* a dot-leader row must end in the page number that makes it a table-of-contents entry.

Anything that only looks suspicious is kept.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

# Running headers and footers usually carry the page number, which is exactly what makes
# them all textually distinct. Digit runs are masked before lines are compared, so
# "Trang 3/45" and "Trang 4/45" are recognised as the same piece of furniture.
_DIGIT_RUN_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# A line that is ONLY a page label: "7", "- 7 -", "[7]", "Trang 7", "Trang 7/45",
# "Page 7 of 45", "7/45". Anchored at both ends on purpose — "1. Gioi thieu" and
# "Dieu 7 quy dinh..." must survive, and they do, because they carry trailing words.
_PAGE_LABEL_RE = re.compile(
    r"^[-–—|(\[]*\s*"
    r"(?:(?:trang|page|pg|p)\s*\.?\s*)?"
    r"\d+"
    r"(?:\s*(?:/|\\|of|tren|trên|-|–|—)\s*\d+)?"
    r"\s*[-–—|)\]]*$",
    re.IGNORECASE,
)

# A table-of-contents row: "Chuong 1 .......... 12".
_DOT_LEADER_RE = re.compile(r"[.·•…]{4,}\s*\d+\s*$")

# Three pages is the minimum at which "this line repeats" means anything. Below it, a
# heading legitimately reused on both pages of a two-page document would be deleted.
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MIN_RATIO = 0.6
# Headers and footers are short. A long line recurring across pages is more likely to be
# a real repeated clause, and a clause is content.
BOILERPLATE_MAX_LINE_CHARS = 120


def _fingerprint(line: str) -> str:
    """Identity of a line for repetition purposes: spacing, case and digits ignored."""
    return _DIGIT_RUN_RE.sub("#", _WHITESPACE_RE.sub(" ", line).strip().lower())


def detect_boilerplate(
    pages: Sequence[tuple[int, str]] | Iterable[tuple[int, str]] | None,
    *,
    min_pages: int = BOILERPLATE_MIN_PAGES,
    min_ratio: float = BOILERPLATE_MIN_RATIO,
) -> frozenset[str]:
    """Return the fingerprints of lines that recur across most pages.

    Counted once per page, so a line repeated ten times on one page is not mistaken for
    a header — that is a formatting artefact of a single page, not document furniture.
    """
    texts = [text for _page_number, text in (pages or []) if text and text.strip()]
    if len(texts) < min_pages:
        return frozenset()

    pages_containing: Counter[str] = Counter()
    for text in texts:
        fingerprints = {
            _fingerprint(line)
            for raw_line in text.splitlines()
            for line in (raw_line.strip(),)
            if line and len(line) <= BOILERPLATE_MAX_LINE_CHARS
        }
        pages_containing.update(fingerprints - {""})

    threshold = max(min_pages, math.ceil(len(texts) * min_ratio))
    return frozenset(
        fingerprint
        for fingerprint, count in pages_containing.items()
        if count >= threshold
    )


def is_navigation_line(line: str) -> bool:
    """Whether a line is a page label or a table-of-contents row.

    Both are navigation furniture for a paper reader. Neither carries retrieval signal:
    a bare page number matches nothing anyone would search for.
    """
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_PAGE_LABEL_RE.match(stripped) or _DOT_LEADER_RE.search(stripped))


def strip_noise(text: str, boilerplate: frozenset[str] = frozenset()) -> str:
    """Drop navigation lines and known boilerplate, preserving everything else.

    Blank lines are kept so paragraph structure survives for the chunker; runs left
    behind by removed lines are collapsed the same way extraction collapses them.
    """
    if not text:
        return text
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            if is_navigation_line(line):
                continue
            if boilerplate and _fingerprint(line) in boilerplate:
                continue
        kept.append(raw_line)
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(kept)).strip()
