"""Agentic multi-query retrieval: expand one question into several targeted search
queries, and adapt with a follow-up query when the first round of evidence is weak.

Off by default (MULTI_QUERY_RETRIEVAL_ENABLED, core/config.py), same posture as every
other retrieval-side addition in this codebase (CROSS_ENCODER_RERANKER_ENABLED,
GRAPH_RETRIEVAL_ENABLED): a new code path with no production traffic behind it yet.

Only used for the ordinary, non-comparison case. A comparison query
(query_router.py::is_comparison_query) already gets its own free, deterministic
decomposition in search_service.py -- this is not layered on top of that one, to avoid
paying for both an LLM call and a second retrieval pass on the same query.

Deliberately never raises, mirroring contextual_header.py/auto_tagging.py: a caller that
gets [] (or None from the follow-up) back falls through to exactly the single-query
search it would have run before this feature existed.
"""
from __future__ import annotations

import structlog

from src.core.config import settings
from src.rag.reranker import fold_diacritics

logger = structlog.get_logger()

_SUBQUERY_SYSTEM_PROMPT = """You expand one search query into several distinct, targeted search queries for a knowledge-base retrieval system.

Given the user's question, propose up to {max_queries} alternative search queries that together cover different phrasings, synonyms, and sub-topics of the question, so a search engine that only matches close wording still finds the relevant passages. If the question mixes languages, or its topic has a common technical-term equivalent in another language, include a variant using that term.

Each query must be short (a few words to one short sentence) and stay strictly on the same subject as the question -- never broaden it into a different topic, and never assume or state an answer.

Return ONLY the queries, one per line, no numbering, no bullet points, no explanation. Return fewer lines if the question does not need more than one search angle, and return nothing if you have no genuinely distinct angle to add."""

_FOLLOWUP_SYSTEM_PROMPT = """A knowledge-base search for the user's question returned only weakly related passages. Propose ONE different, more targeted search query that might reach a better-matching passage -- a synonym, a narrower sub-topic, or a related term the earlier searches likely missed.

Do not repeat the question or any of the queries already tried. Do not invent a topic the question does not ask about. If you have no genuinely different angle to try, reply with exactly: NONE

Return ONLY the new query text, or NONE -- nothing else."""


def _dedup_queries(candidates: list[str], *, original: str, limit: int) -> list[str]:
    """First `limit` candidates, stripped, non-empty, and distinct from `original` and
    each other. Comparison is diacritic/case-folded so a near-identical rephrasing of the
    same query doesn't count as a second search angle."""
    seen = {fold_diacritics(original)}
    deduped: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip(" \t-*•")
        if not candidate:
            continue
        key = fold_diacritics(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return deduped


async def generate_subqueries(question: str) -> list[str]:
    """Up to MULTI_QUERY_MAX_SUBQUERIES alternate search queries for `question`.

    Best-effort: returns [] on any failure (no provider configured, timeout, malformed
    response) -- same posture as every other auxiliary LLM call in this codebase.
    """
    # Imported here, not at module load, so a test can monkeypatch
    # `src.domain.llm_client.complete`/`resolve_provider`, the same convention every
    # other best-effort LLM-calling module in this codebase follows.
    from src.domain.llm_client import complete, resolve_provider

    if not question.strip():
        return []

    max_queries = settings.MULTI_QUERY_MAX_SUBQUERIES
    if max_queries <= 0:
        return []

    provider = resolve_provider()
    if not provider:
        return []

    try:
        answer, _tokens, _model, _provider_name = await complete(
            [
                {
                    "role": "system",
                    "content": _SUBQUERY_SYSTEM_PROMPT.format(max_queries=max_queries),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=200,
            thinking=False,
        )
    except Exception as exc:
        logger.warning("Multi-query sub-query generation failed", error=str(exc))
        return []

    lines = [line.strip() for line in answer.splitlines()]
    return _dedup_queries(lines, original=question, limit=max_queries)


async def generate_followup_query(
    question: str, tried_queries: list[str], weak_titles: list[str]
) -> str | None:
    """One additional search query to try when the first retrieval round was weak.

    Returns None on failure, on an explicit "no better angle" response, or when the
    model repeats a query already tried -- the caller falls back to the results it
    already has, exactly as it would without this feature.
    """
    from src.domain.llm_client import complete, resolve_provider

    if not question.strip():
        return None

    provider = resolve_provider()
    if not provider:
        return None

    tried_text = "\n".join(f"- {q}" for q in tried_queries if q) or "(none)"
    weak_text = "\n".join(f"- {t}" for t in weak_titles if t) or "(none)"

    try:
        answer, _tokens, _model, _provider_name = await complete(
            [
                {"role": "system", "content": _FOLLOWUP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION: {question}\n\n"
                        f"QUERIES ALREADY TRIED:\n{tried_text}\n\n"
                        f"TITLES OF THE WEAK RESULTS FOUND SO FAR:\n{weak_text}"
                    ),
                },
            ],
            max_tokens=60,
            thinking=False,
        )
    except Exception as exc:
        logger.warning("Multi-query follow-up generation failed", error=str(exc))
        return None

    candidate = answer.strip().strip('"')
    if not candidate or candidate.upper() == "NONE":
        return None
    tried_keys = {fold_diacritics(q) for q in [question, *tried_queries] if q}
    if fold_diacritics(candidate) in tried_keys:
        return None
    return candidate
