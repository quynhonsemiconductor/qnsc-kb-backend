"""Best-effort tag suggestion for one document, run automatically during ingestion.

`api/routers/articles.py::auto_tag_articles` already does this for a human-selected
batch of already-PUBLISHED articles, as an explicit, opt-in action a reviewer confirms
afterwards (`confirm_article_tags`). This module is the single-document version of the
same prompt, called automatically while a draft is still being restructured -- the same
shape department routing already has (`domain/department_routing.py`, invoked
automatically from `workers/tasks.py::run_restructure_pending_draft`), so a freshly
uploaded document arrives at review with both a suggested department AND suggested tags
instead of only the former.

Deliberately never raises and never blocks ingestion: a tag suggestion is a convenience a
reviewer can accept, edit, or ignore entirely (PendingDraft.tags stays a plain editable
field), never a requirement for a draft to reach the review queue.
"""
from __future__ import annotations

import json
import re
import unicodedata

import structlog

logger = structlog.get_logger()

MIN_TAGS = 3
MAX_TAGS = 8
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
_VALID_TAG_RE = re.compile(r"[\w -]+", re.UNICODE)

_SYSTEM_PROMPT = """You generate concise search tags for one knowledge-base document.

Return ONLY a JSON object in this exact shape, nothing else -- no explanation, no
markdown code fence: {"tags": ["tag1", "tag2"]}

Use 3 to 8 specific tags, lowercase, using only letters, numbers, spaces, or hyphens. Do
not invent tags unrelated to the document's actual content.

If a list of EXISTING TAGS is provided, prefer reusing one of them whenever it genuinely
fits the document -- only suggest a new tag when nothing in that list applies. This keeps
the tenant's tag vocabulary from fragmenting into near-duplicates.
"""

# Cap on how many catalogue tags get listed in the prompt. This is steering, not
# enforcement -- `catalogue` (the normalized set) still filters the result afterward --
# so an oversized tenant vocabulary degrades to "the model sees a partial list" rather
# than a token-budget failure.
MAX_CATALOGUE_EXAMPLES = 200


def _clean_tag(raw: object) -> str | None:
    """Same normalization confirm_article_tags applies to a human-reviewed tag list,
    so an automatic suggestion and a manually confirmed one are indistinguishable once
    stored."""
    value = re.sub(r"\s+", " ", str(raw).strip().lower())
    if not value or len(value) > 50 or not _VALID_TAG_RE.fullmatch(value):
        return None
    return value


def _normalize_for_catalogue(value: str) -> str:
    """Same accent-insensitive normalization `articles.py`'s tag endpoints use to
    compare a tag against the catalogue, so "an toan" and "an toàn" match one entry."""
    return re.sub(
        r"\s+",
        " ",
        "".join(
            ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)
        ).strip().casefold(),
    )


async def suggest_tags_for_document(
    title: str,
    body_md: str,
    doc_type: str = "",
    *,
    catalogue: set[str] | None = None,
    catalogue_examples: list[str] | None = None,
) -> list[str]:
    """Best-effort tag suggestions for one document. Returns [] rather than raising.

    Every failure mode (no provider configured, the call fails, the reply is not
    parseable JSON) is swallowed and logged: this runs inline in the same background
    pass that formats the document, and a broken tag suggestion must not turn into a
    broken upload.

    `catalogue` mirrors `auto_tag_articles`'s own governance rule (articles.py): when the
    tenant has an approved tag vocabulary, only suggest tags already in it -- and when the
    vocabulary is empty (`set()`, not `None`), suggest nothing at all rather than seeding
    the tenant's tags from unreviewed AI output. `None` means "no catalogue enforcement",
    used by callers (or tests) that have not loaded one; a real caller in production
    always has a set, even if it is empty.

    `catalogue_examples` is the display-cased vocabulary (`TagCatalog.tag`, not the
    accent-folded `normalized_tag` `catalogue` filters against) shown to the model so it
    is steered toward reusing existing tags instead of inventing near-duplicates that
    `catalogue` then silently drops. Purely a prompt hint -- `catalogue` remains the sole
    enforcement mechanism, so a stale or truncated example list only costs suggestion
    quality, never correctness.
    """
    # Imported here, not at module load, so a test can monkeypatch
    # `src.domain.llm_client.complete`/`resolve_provider` the same way every other
    # LLM-calling module in this codebase is tested (see department_routing.py /
    # test_llm_department_routing.py).
    from src.domain.llm_client import complete, resolve_provider

    provider = resolve_provider()
    if not provider:
        return []

    examples_block = ""
    if catalogue_examples:
        shown = sorted(catalogue_examples)[:MAX_CATALOGUE_EXAMPLES]
        examples_block = "EXISTING TAGS (prefer reusing these when they fit):\n" + ", ".join(shown) + "\n\n"

    try:
        answer, _tokens, _model, _provider_name = await complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"TITLE: {title}\nTYPE: {doc_type or '(unspecified)'}\n\n"
                        f"{examples_block}"
                        f"CONTENT:\n{body_md[:5000]}"
                    ),
                },
            ],
            max_tokens=150,
        )
    except Exception as exc:
        logger.warning("Automatic tag suggestion call failed", error=str(exc))
        return []

    cleaned = _CODE_FENCE_RE.sub("", answer.strip()).strip()
    try:
        payload = json.loads(cleaned)
        raw_tags = payload.get("tags") if isinstance(payload, dict) else None
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning(
            "Automatic tag suggestion returned unparseable JSON",
            error=str(exc),
            reply=cleaned[:200],
        )
        return []
    if not isinstance(raw_tags, list):
        # A string value here is the trap: `for raw in "not-a-list"` iterates its
        # characters rather than raising, which used to turn a malformed reply into a
        # list of single-letter "tags" instead of the empty list this should return.
        if raw_tags is not None:
            logger.warning(
                "Automatic tag suggestion's 'tags' field was not a list",
                reply=cleaned[:200],
            )
        return []

    tags: list[str] = []
    for raw in raw_tags:
        cleaned_tag = _clean_tag(raw)
        if cleaned_tag and cleaned_tag not in tags:
            tags.append(cleaned_tag)

    if catalogue is not None:
        tags = [tag for tag in tags if _normalize_for_catalogue(tag) in catalogue]
    return tags[:MAX_TAGS]
