"""Tokens in a question that name one specific thing: a register, a signal, a part, an error code.

`PADOUTSET`, `CONFIGOPTS`, `CFG_REG_LO`, `I2C`, `UART0`, `WDOG_BARK_THOLD`, `APB_M12`, `0x10`.

The keyword leg ORs every word of a question and ranks with `ts_rank_cd`, which counts occurrences and knows
nothing about how rare a word is. In a natural-language question the identifier is one word among many, and
the common ones ("thanh ghi", "offset", "địa chỉ") match many passages many times over, so the passage that
holds the identifier ranked 128th to 493rd of 600 matches and never reached the answer.

Weighting by inverse document frequency does not repair that, measured on 17 chip-specification questions:
in a corpus that is mostly English technical text the Vietnamese question words ("nhiều", "nào", "gì")
occur in 2 to 10 chunks, so they score as rare and win. What does separate the useful words is their shape,
which does not depend on the language of the question: identifiers are written in capitals, with an
underscore, or with a digit next to letters.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[A-Za-z0-9_]+")

#: A question in which EVERY word is identifier-shaped and there are more than this many of them is typed in
#: capitals ("WHAT IS THE RESET VALUE"), not a list of names, so none of its words is evidence of anything.
#: Short keyword queries stay: "GPIO PADOUTSET" and "GPIO PADOUTSET CLR" are all names and are wanted.
_SHOUTING_MIN_WORDS = 4
#: Enough for a question that names several registers; bounds the size of the tsquery.
MAX_IDENTIFIER_TERMS = 8


def _fold(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    ).lower()


def _looks_like_an_identifier(word: str) -> bool:
    if len(word) < 2:
        return False
    has_letter = any(char.isalpha() for char in word)
    if not has_letter:
        return False  # a number on its own ("12", "2026") is not a name
    return (
        (word.isupper() and any(char.isalpha() for char in word))
        or "_" in word.strip("_")
        or any(char.isdigit() for char in word)
    )


def identifier_terms(query: str, limit: int = MAX_IDENTIFIER_TERMS) -> list[str]:
    """The identifier-shaped words of `query`, folded the way the keyword leg folds, in order, without repeats."""

    words = _TOKEN.findall(query or "")
    letters = [word for word in words if any(char.isalpha() for char in word)]
    if not letters:
        return []
    shaped = [word for word in letters if _looks_like_an_identifier(word)]
    if len(shaped) == len(letters) and len(letters) >= _SHOUTING_MIN_WORDS:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for word in shaped:
        key = _fold(word.strip("_"))
        if len(key) > 1 and key not in seen:
            seen.add(key)
            found.append(key)
    return found[:limit]


def usable_identifiers(query: str, normalized_query: str) -> list[str]:
    """The names of `query` that survive the normalisation retrieval and scoring both work from.

    `normalize_query` drops English stopwords, and several are real chip names in capitals: `CAN`, `ON`, `IN`, `DO`,
    `AT`. The reranker can only weigh a term it still has, so a name it dropped must not get a retrieval query of
    its own either ("CAN bus baud rate?" would otherwise pull in the passages that repeat the verb "can").
    """
    kept = set(normalized_query.split())
    return [name for name in identifier_terms(query) if name in kept]
