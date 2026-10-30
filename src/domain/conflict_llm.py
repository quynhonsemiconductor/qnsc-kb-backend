"""Model-based contradiction detection between a newly published article and similar ones.

The labelled-fact pattern check (`ai_service._detect_explicit_conflicts`) only sees lines that
look like "Deadline: ...". Two procedures that disagree in prose -- one says approval takes
three working days, the other five -- never match a pattern. This asks a model instead, with
the same non-blocking posture as the pattern check: a failure, an empty reply or a reply that
does not parse yields no conflicts and never touches the publish.

The model is told to report only statements that cannot both be true. A newer version of the
same rule, or a rule for a different scope, is not a conflict, and a false positive costs a
reviewer's time, so the bar is high and the number reported per article is capped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()

MAX_CONFLICTS = 5
#: Characters of each article the model sees. Enough for a policy page, bounded for cost of
#: a call nobody is waiting on.
EXCERPT_CHARS = 5000
MAX_STATEMENT_CHARS = 400
_KINDS = {"date", "numerical", "status", "ownership", "procedure", "other"}
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)

_SYSTEM_PROMPT = """You compare a NEW knowledge-base article with OTHER articles and report
statements that contradict each other.

Return ONLY a JSON object, no explanation and no markdown fence:

{"conflicts": [{"other": <number of the OTHER article>, "topic": "<what the disagreement is
about, a short noun phrase>", "new_statement": "<the claim in the NEW article, quoted or
closely paraphrased>", "other_statement": "<the incompatible claim in that OTHER article>",
"kind": "date|numerical|status|ownership|procedure|other"}]}

Report a pair only when both statements are about the same subject and cannot both be true at
once: a different deadline, limit, duration, owner or required step for the same thing.
Do NOT report: a rule for a different department, product, country or situation; a newer
revision that clearly replaces an older one and says so; a difference in wording or level of
detail; anything you are unsure about. When nothing qualifies return {"conflicts": []}.
Write topic and statements in the language of the articles.
"""


@dataclass(frozen=True)
class LlmConflict:
    other_index: int
    topic: str
    new_statement: str
    other_statement: str
    kind: str


def _clean(raw: object, limit: int) -> str:
    return re.sub(r"\s+", " ", str(raw or "").strip())[:limit]


def parse_conflicts(reply: str, other_count: int) -> list[LlmConflict]:
    """Parse the model's reply defensively; anything malformed is dropped, never raised."""
    try:
        payload = json.loads(_CODE_FENCE_RE.sub("", reply.strip()).strip())
    except (json.JSONDecodeError, TypeError):
        return []
    items = payload.get("conflicts") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    found: list[LlmConflict] = []
    seen: set[tuple[int, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(str(item.get("other")))
        except (TypeError, ValueError):
            continue
        topic = _clean(item.get("topic"), 200)
        new_statement = _clean(item.get("new_statement"), MAX_STATEMENT_CHARS)
        other_statement = _clean(item.get("other_statement"), MAX_STATEMENT_CHARS)
        if not (1 <= index <= other_count) or not topic:
            continue
        if not new_statement or not other_statement:
            continue
        # Two statements that normalise to the same text are the same claim, not a conflict.
        if new_statement.casefold() == other_statement.casefold():
            continue
        key = (index, topic.casefold())
        if key in seen:
            continue
        seen.add(key)
        kind = str(item.get("kind") or "other").strip().lower()
        found.append(
            LlmConflict(
                other_index=index,
                topic=topic,
                new_statement=new_statement,
                other_statement=other_statement,
                kind=kind if kind in _KINDS else "other",
            )
        )
        if len(found) >= MAX_CONFLICTS:
            break
    return found


async def detect_llm_conflicts(
    new_title: str, new_body: str, others: list[dict[str, Any]]
) -> list[LlmConflict]:
    """Ask the model; `others` items carry `title` and `body_md`. Never raises."""
    from src.core.config import settings

    if not settings.CONFLICT_LLM_DETECTION_ENABLED or not others:
        return []
    # Imported here so a test can monkeypatch `src.domain.llm_client` like every other
    # LLM-calling module.
    from src.domain.llm_client import complete, resolve_provider

    if not resolve_provider():
        return []
    sections = [f"NEW ARTICLE: {new_title}\n{new_body[:EXCERPT_CHARS]}"]
    for number, other in enumerate(others, start=1):
        sections.append(
            f"OTHER ARTICLE {number}: {other.get('title', '')}\n"
            f"{str(other.get('body_md') or '')[:EXCERPT_CHARS]}"
        )
    try:
        reply, _tokens, _model, _provider = await complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n---\n\n".join(sections)},
            ],
            timeout=settings.CONFLICT_LLM_TIMEOUT_SECONDS,
            thinking=False,
            max_tokens=1200,
            background=True,
        )
    except Exception as exc:
        logger.warning("LLM conflict detection call failed", error=str(exc))
        return []
    return parse_conflicts(reply, len(others))
