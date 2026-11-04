"""Search a question in the language the documents are written in, as well as the one it was asked in.

Why: a Vietnamese question over English specifications matches the wrong documents. Its filler words
("thanh ghi", "địa chỉ", "bao nhiêu") share no text with the English passage that answers it, but they
do match the Vietnamese-language documents in the same knowledge base, so those win whatever the topic is.
Measured on 85 generated Vietnamese questions whose answers are in the knowledge base: 60 reached the
prompt as asked, 77 when searched in English translation, against 79 of 85 for questions written in English
(the answer-bearing chunk in the first 40 results and above the context gate).

Used by `SearchService.search` when `QUERY_TRANSLATION_ENABLED`: the question and its translation are both
retrieved and each passage keeps its better score, so a passage that is only in Vietnamese is still found
by the question as asked.

Never raises: a failed, slow or empty translation is no translation, and the search is exactly the one that
would have run without this feature. One short, non-streaming model call, remembered per question.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict

import structlog

from src.core.config import settings
from src.rag.reranker import fold_diacritics

logger = structlog.get_logger()

# Letters that exist in Vietnamese and in no language this knowledge base is written in. A question
# without any is left alone: it is almost always already English, and a call per English question
# would be a cost with nothing to gain.
_VIETNAMESE_LETTERS = re.compile(
    "[ăâđêôơưàáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "Translate the user's question into {language} for a search engine over engineering "
    "specifications. Keep every identifier, register name, signal name, number and code token exactly as "
    "written. Output only the translated question, nothing else."
)

_CACHE_MAX = 512
_cache: OrderedDict[str, str | None] = OrderedDict()
#: After a failed call, questions are searched as asked for this long, so a provider outage costs one
#: timeout, not one per search.
_COOLDOWN_SECONDS = 60.0
_cooldown_until = 0.0


def needs_translation(question: str) -> bool:
    """True when the question looks Vietnamese (carries a letter only Vietnamese has)."""
    return bool(_VIETNAMESE_LETTERS.search(question or ""))


def _remember(key: str, value: str | None) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


async def translate_query(question: str) -> str | None:
    """The question in `QUERY_TRANSLATION_TARGET_LANGUAGE`, or None when there is none to use."""
    if not needs_translation(question):
        return None
    key = question.strip()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    # Imported here, not at load, so tests can monkeypatch the client like every other best-effort
    # model call in this codebase.
    from src.domain.llm_client import complete, resolve_provider

    global _cooldown_until
    if time.monotonic() < _cooldown_until or not resolve_provider():
        return None
    try:
        reply, _tokens, _model, _provider = await complete(
            [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(
                        language=settings.QUERY_TRANSLATION_TARGET_LANGUAGE
                    ),
                },
                {"role": "user", "content": key},
            ],
            timeout=settings.QUERY_TRANSLATION_TIMEOUT_SECONDS,
            thinking=False,
            # A reasoning model spends part of this on thinking; too small a budget comes back empty.
            max_tokens=settings.QUERY_TRANSLATION_MAX_TOKENS,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a missing translation must never fail a search
        _cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
        logger.warning(
            "Query translation failed; searching the question as asked", error=str(exc)
        )
        return None
    lines = (reply or "").strip().splitlines()
    translated = lines[0].strip().strip("\"'").strip() if lines else ""
    # Nothing new (the model echoed the question, or it was already close to English).
    if not translated or fold_diacritics(translated) == fold_diacritics(key):
        _remember(key, None)
        return None
    _remember(key, translated)
    return translated
