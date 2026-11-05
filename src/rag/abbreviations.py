"""Find the abbreviated name a question spells out: "clock divider" -> CLKDIV, "interrupt map" -> INTMAP.

Why: specifications name things by abbreviation (`CLKDIV`, `CLK_EN`, `INTMAP`, `WSTRB`, `PSLVERR`) and people ask in words
("what is the clock divider field of the SPI host?"). The passage that answers holds the abbreviation and none of
the question's words, so the keyword leg ranks it with the common words and the vector leg does not tie the two
together reliably. Measured on 12 questions that spell out names present in the 22 chip specifications, 8 reached
the first 20 results; when the question also carried the names found here, 12 did, and 11 were in the first 5 (8).

How: the names are taken from the corpus itself (capitalised tokens of 4 to 16 characters in the published
chunks of the company, with how many articles use each), so nothing has to be written by an administrator. A name
is proposed for two or three consecutive words of the question when it reads as their abbreviation: each word
contributes its beginning ("int" of interrupt, "div" of divider), or its first letter followed by some of its
consonants ("clk" of clock), or only its first letter when there are three words ("p" "r" "data" for PRDATA).
The words of the question are left alone; the names are added to it, as identifiers, and the usual retrieval and
ranking do the rest.

Pure functions for the matching; `VocabularyCache` keeps one vocabulary per company and loads it in the
background, so the first questions after a start are searched as before and none waits for it.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping

import structlog

logger = structlog.get_logger()

#: Words that never start or sit inside a spelled-out name.
_STOP_WORDS = frozenset(
    "the a an of for is are was were what which how do does to in on and or i me my that this with by at from be can "
    "it its as if not no".split()
)
_VOWELS = frozenset("aeiou")
_WORD = re.compile(r"[a-zA-Z]+")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_TRAILING_DIGITS = re.compile(r"\d+$")
#: A name as it is written in a specification: capitals, digits and underscores, starting with a capital.
NAME_PATTERN = r"\m[A-Z][A-Z0-9_]{3,15}\M"
#: Shorter names (`WRAP`, `PROT`, `TRST`) read as an abbreviation of too many word pairs by chance.
MIN_CORE_LENGTH = 5
MAX_ALIASES = 3
#: Windows of this many consecutive words are tried.
WINDOW_SIZES = (2, 3)


def _is_subsequence(piece: str, word: str) -> bool:
    letters = iter(word)
    return all(char in letters for char in piece)


def _abbreviates(piece: str, word: str) -> bool:
    """`piece` is the beginning of `word`, or its first letter and then some of its consonants in order."""
    if word.startswith(piece):
        return True
    return all(char not in _VOWELS for char in piece[1:]) and _is_subsequence(
        piece, word
    )


def _can_form(rest: str, words: list[str], index: int = 0) -> bool:
    if index == len(words):
        return rest == ""
    word = words[index]
    for length in range(1, len(rest) + 1):
        piece = rest[:length]
        if piece[0] != word[0]:
            return False
        if len(rest) - length < len(words) - index - 1:
            return False  # every word still to come needs at least a letter
        if not _abbreviates(piece, word):
            if length >= len(word):
                return False
            continue
        # With two words an initial alone is only enough beside a longer part (p + addr, w + strb); three may be an
        # acronym with a word at the end.
        if length == 1 and len(words) == 2 and len(rest) - 1 < 3:
            continue
        if _can_form(rest[length:], words, index + 1):
            return True
    return False


def _names_something(token: str) -> bool:
    """A token written as the name of a thing asked about: with an underscore, or in capitals of five or more.

    `SPI`, `UART0` and `I2C` are not that: they name the block a question is about, and the register it asks for is
    often still spelled out ("I2C clock divider").
    """
    if "_" in token.strip("_") and any(char.isalpha() for char in token):
        return True
    letters = [char for char in token if char.isalpha()]
    return len(letters) >= 5 and token.isupper()


def name_core(name: str) -> str:
    """The letters of a name without underscores or a trailing number: `CLK_EN` -> `clken`, `UART0` -> `uart`."""
    return _TRAILING_DIGITS.sub("", name.lower().replace("_", ""))


def find_aliases(
    text: str, vocabulary: Mapping[str, int], limit: int = MAX_ALIASES
) -> list[str]:
    """Names from `vocabulary` (name -> number of articles using it) that read as abbreviations of the words of `text`.

    Several spellings with the same letters (`INTMAP`, `INTMAP_001`) give the most used one. The result is ordered by
    how many articles use the name, so the names the corpus leans on come first, and cut to `limit`.
    """
    text = text or ""
    if any(_names_something(token) for token in _TOKEN.findall(text)):
        # The question already names what it asks about (`INTR_ENABLE offset`): the identifier leg has it, and
        # a guess beside a name that was given only adds noise.
        return []
    words = [word.lower() for word in _WORD.findall(text)]
    found: dict[str, tuple[int, str]] = {}
    for size in WINDOW_SIZES:
        for start in range(len(words) - size + 1):
            window = words[start : start + size]
            if any(len(word) < 2 or word in _STOP_WORDS for word in window):
                continue
            letters = sum(len(word) for word in window)
            for name, uses in vocabulary.items():
                core = name_core(name)
                if (
                    len(core) < MIN_CORE_LENGTH
                    or len(core) > letters
                    or core in window
                    or core in words
                ):
                    continue
                if _can_form(core, window):
                    best = found.get(core)
                    if best is None or uses > best[0]:
                        found[core] = (uses, name)
    ranked = sorted(found.values(), key=lambda item: (-item[0], item[1]))
    return [name for _uses, name in ranked[:limit]]


#: Results of the ordinary search that keep their place, and how many the extended search may put right after them.
KEEP_ORDINARY = 2
INSERT_EXTENDED = 3


def merge_alias_results(
    ordinary: list[dict],
    extended: list[dict],
    keep: int = KEEP_ORDINARY,
    insert: int = INSERT_EXTENDED,
) -> list[dict]:
    """The ordinary search's first `keep` results, then up to `insert` the extended search found and it did not,
    then the rest of the ordinary results, then the rest of the extended ones. Each passage once.

    A name added to a question that was about something else can only move the answer down by `insert` places.
    """
    merged: list[dict] = []
    seen: set[str] = set()

    def add(item: dict) -> None:
        key = str(item.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            merged.append(item)

    ordinary_ids = {str(item.get("chunk_id")) for item in ordinary}
    for item in ordinary[:keep]:
        add(item)
    new = [item for item in extended if str(item.get("chunk_id")) not in ordinary_ids]
    for item in new[:insert]:
        add(item)
    for item in ordinary[keep:]:
        add(item)
    for item in extended:
        add(item)
    return merged


class VocabularyCache:
    """One vocabulary per company, loaded in the background and refreshed after `ttl_seconds`.

    `get` never waits: it returns what is loaded (possibly stale) or None, and starts a load when there is nothing
    or it is old and none is running. A failed load is not retried for `retry_seconds`, so a database that cannot
    answer costs one attempt, not one per search.
    """

    def __init__(self, ttl_seconds: float = 6 * 3600, retry_seconds: float = 300):
        self.ttl_seconds = ttl_seconds
        self.retry_seconds = retry_seconds
        self._loaded: dict[str, tuple[float, dict[str, int]]] = {}
        self._running: set[str] = set()
        self._failed_at: dict[str, float] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def get(
        self,
        company: str,
        loader: Callable[[str], Awaitable[dict[str, int]]],
    ) -> dict[str, int] | None:
        entry = self._loaded.get(company)
        now = time.monotonic()
        stale = entry is None or now - entry[0] > self.ttl_seconds
        recently_failed = now - self._failed_at.get(company, -1e9) < self.retry_seconds
        if stale and company not in self._running and not recently_failed:
            self._start(company, loader)
        return entry[1] if entry else None

    def _start(
        self, company: str, loader: Callable[[str], Awaitable[dict[str, int]]]
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._running.add(company)
        task = loop.create_task(self._load(company, loader))
        self._tasks.add(
            task
        )  # a reference, or the loop may drop the task before it ends
        task.add_done_callback(self._tasks.discard)

    async def _load(
        self, company: str, loader: Callable[[str], Awaitable[dict[str, int]]]
    ) -> None:
        try:
            vocabulary = await loader(company)
            self._loaded[company] = (time.monotonic(), vocabulary)
            self._failed_at.pop(company, None)
            logger.info(
                "Abbreviation vocabulary loaded",
                company_domain=company,
                names=len(vocabulary),
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a missing vocabulary only means no expansion
            self._failed_at[company] = time.monotonic()
            logger.warning(
                "Abbreviation vocabulary could not be loaded; questions are searched as asked",
                company_domain=company,
                error=str(exc),
            )
        finally:
            self._running.discard(company)

    def forget(self, company: str | None = None) -> None:
        """Drop one company's vocabulary (or all), so the next question reloads it."""
        if company is None:
            self._loaded.clear()
        else:
            self._loaded.pop(company, None)


vocabulary_cache = VocabularyCache()


async def load_company_vocabulary(company_domain: str) -> dict[str, int]:
    """Every capitalised name in the company's published chunks, with the number of articles that use it.

    Read in a session of its own bound to the company, so row-level security shows the same rows the company's
    searches see. The vocabulary holds names only: what a question finds with them is still decided by the
    asker's own permissions.
    """
    from sqlalchemy import text

    from src.api.deps import SessionLocal, set_database_context

    # The pattern is a bind parameter, not text spliced into the statement.
    statement = text("""
        SELECT name, count(DISTINCT article_id) AS uses
        FROM (
            SELECT c.article_id, (regexp_matches(c.chunk_text, :pattern, 'g'))[1] AS name
            FROM article_chunks c
            JOIN articles a ON a.id = c.article_id
            WHERE a.company_domain = :company
              AND a.status = 'published'
              AND a.lifecycle_status = 'active'
        ) names
        GROUP BY name
        """)
    async with SessionLocal() as db:
        await set_database_context(db, company_domain)
        rows = (
            await db.execute(
                statement, {"company": company_domain, "pattern": NAME_PATTERN}
            )
        ).all()
    return {str(name): int(uses) for name, uses in rows}
