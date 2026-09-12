"""Contextual chunk headers: situate a chunk within its document before it is embedded.

A retrieval child chunk is ~250 characters (see chunker.py) -- precise, but often
ambiguous on its own: "Employees must submit the form at least three days in advance"
could be about leave, expenses, or a dozen other processes. Anthropic's Contextual
Retrieval measured prepending a short, chunk-specific context blurb before embedding cut
top-20 retrieval failures 49% (67% combined with reranking) -- this is the same
technique, scoped to what this codebase already computes per SECTION during indexing.

ONE HEADER PER SECTION, not per child and not even per parent: a section is the
indexing loop's natural unit (see indexing.py), and the header's job is disambiguation --
"which document and section is this snippet even from" -- not per-sentence nuance. Doing
this per child would multiply LLM calls by the child count (often dozens per document)
for no proportional gain.

EMBEDDING INPUT ONLY, never stored. `chunk_text` (used for BM25 tsvector, citations, and
UI highlight snippets — see ai_service.py's `highlight_text`/`excerpt` fields) is left
completely unchanged; only the string handed to the embedder gets the header prepended.
Storing a synthetic header inside `chunk_text` would leak it into citation excerpts and
highlighted passages shown to end users, which is a worse outcome than the retrieval gain
is worth. This scopes the win to the dense leg, not the sparse (BM25) leg.

Deliberately never raises, mirroring auto_tagging.py/entity_extraction.py: an indexing
pass without a header embeds exactly what today's pipeline already embeds, never worse.
"""
from __future__ import annotations

import structlog

logger = structlog.get_logger()

# ~100 tokens at a pessimistic 4 chars/token, matching the "50-100 token" target loosely
# in characters -- consistent with how chunker.py already bounds everything in characters
# rather than doing a second tokenisation pass just for this.
MAX_HEADER_CHARS = 400
SECTION_EXCERPT_CHARS = 3000

_SYSTEM_PROMPT = """You write a short context header for one section of a knowledge-base document, so a snippet pulled from it later can be understood on its own.

Return ONLY the header text: one or two sentences, no more than 100 words, no markdown, no quotes, no explanation. State what document this is from and what this section covers. Do not restate the section's specific details -- the header situates the snippet, it does not replace it.
"""


async def generate_section_context(title: str, heading: str | None, section_text: str) -> str:
    """Best-effort context header for one section. Returns "" rather than raising.

    Called once per section during indexing; the result is prepended to every child
    chunk's embedding input within that section (see indexing.py), never stored.
    """
    # Imported here, not at module load, so a test can monkeypatch
    # `src.domain.llm_client.complete`/`resolve_provider`, the same convention every
    # other best-effort LLM-calling module in this codebase follows.
    from src.domain.llm_client import complete, resolve_provider

    provider = resolve_provider()
    if not provider:
        return ""

    try:
        answer, _tokens, _model, _provider_name = await complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT TITLE: {title}\n"
                        f"SECTION HEADING: {heading or '(untitled section)'}\n\n"
                        f"SECTION CONTENT (excerpt):\n{section_text[:SECTION_EXCERPT_CHARS]}"
                    ),
                },
            ],
            max_tokens=120,
        )
    except Exception as exc:
        logger.warning("Contextual chunk header generation failed", error=str(exc))
        return ""

    return answer.strip()[:MAX_HEADER_CHARS]


def apply_header(header: str, chunk_text: str) -> str:
    """Combine a section header with one child's text for embedding, if there is one.

    A named function rather than an inline f-string so the join format has one
    definition, and so a test can assert the format without duplicating it.
    """
    if not header:
        return chunk_text
    return f"{header}\n\n{chunk_text}"
