"""Best-effort extraction of document identity fields (number, dates, signer).

A separate LLM call from `content_restructure.py` on purpose, not an addition to its
prompt: that prompt's contract is "return only the reformatted Markdown", tuned and
covered by its own numeric-coverage checks, and asking the same call to also return a
JSON metadata block would mean parsing two different things out of one response with no
way to fail one half without risking the other. This module can fail, return nothing, or
be wrong without touching the reformatted document at all -- the one thing it writes is
`Article.structured_metadata`, never `body_md`.

Called best-effort, after a draft publishes (see governance.py), the same way
`domain/contradiction_check.py` is: a bad extraction is a missing convenience, not a
reason to fail a publish that would otherwise have succeeded.
"""
from __future__ import annotations

import json
import re

import structlog

from src.core.config import settings

logger = structlog.get_logger()

#: Every key is optional in the response; a document with no visible issue date, say,
#: should answer null for it rather than the model inventing one to fill the shape.
_FIELDS = ("document_number", "issue_date", "expiry_date", "signed_by")

_SYSTEM_PROMPT = """You extract document identity fields from a knowledge-base article,
if and only if they are explicitly present in the text.

Return ONLY a JSON object with exactly these keys, and nothing else -- no explanation,
no markdown code fence:

{"document_number": string or null, "issue_date": "YYYY-MM-DD" or null,
 "expiry_date": "YYYY-MM-DD" or null, "signed_by": string or null}

document_number is a reference/control number the document itself states (e.g. "SOP-114",
"QD-2026-08"), not a page number or section number. issue_date is when the document itself
says it was issued or took effect. expiry_date is an explicit review-by or valid-until
date, not a date mentioned in passing. signed_by is the name or title of whoever the
document names as approver/signer. Leave a field null rather than guessing when the text
does not state it directly.
"""

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class StructuredMetadataUnavailable(RuntimeError):
    """The extraction call failed or returned something unusable. Never fatal to a caller."""


async def extract_structured_metadata(title: str, body_md: str) -> dict[str, str | None]:
    """Best-effort document identity fields, or raise `StructuredMetadataUnavailable`.

    Every value in the returned dict is either a clean string or explicitly `None` --
    never missing, so a caller can always merge the full four keys into storage without
    checking for KeyError first. Truncates `body_md` the same way auto-tagging does
    (articles.py's `AutoTagRequest` prompt): the document's identity fields are almost
    always in the first page, and sending the whole body wastes tokens on content that
    cannot change the answer.
    """
    from src.domain.llm_client import complete, resolve_provider

    provider = resolve_provider(settings.RESTRUCTURE_MODEL or None)
    if not provider:
        raise StructuredMetadataUnavailable("No LLM provider is configured")

    try:
        answer, _tokens, _model, _provider_name = await complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"TITLE: {title}\n\nDOCUMENT:\n{body_md[:6000]}",
                },
            ],
            model_override=settings.RESTRUCTURE_MODEL or None,
            max_tokens=200,
        )
    except Exception as exc:
        raise StructuredMetadataUnavailable(f"extraction call failed: {exc}") from exc

    cleaned = _CODE_FENCE_RE.sub("", answer.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StructuredMetadataUnavailable(
            f"extraction response was not valid JSON: {cleaned[:200]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredMetadataUnavailable(
            f"extraction response was not a JSON object: {cleaned[:200]!r}"
        )

    result: dict[str, str | None] = {}
    for field in _FIELDS:
        value = payload.get(field)
        if value is None:
            result[field] = None
            continue
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "unknown"}:
            result[field] = None
            continue
        if field in {"issue_date", "expiry_date"} and not _DATE_RE.match(text):
            # A date the model could not put in YYYY-MM-DD is not usable as a date field
            # (it cannot be sorted or compared), so it is dropped rather than stored as
            # a string that looks like a date but is not one.
            logger.warning(
                "Structured metadata extraction returned an unparseable date",
                field=field,
                value=text,
            )
            result[field] = None
            continue
        result[field] = text[:255]
    return result
