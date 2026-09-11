"""LLM-judged RAG quality metrics: online, non-deterministic, opt-in.

Companion to `rag/evaluator.py`, not a replacement or an upgrade to it. That module's
docstring is a promise -- "offline-friendly metrics used by the evaluation dashboard and
CI" -- and every function in it is a pure, deterministic string comparison so CI never
depends on network access, an API key, or token spend. The functions here break every one
of those properties: they call whichever provider `domain/llm_client.py` is configured
with, so they cost tokens and latency and can disagree with themselves between two calls
on the same input.

Call these from a scheduled job or an on-demand dashboard action. Never from the CI path
`evaluator.py` serves, and never as a silent fallback when a caller "just" wants a score --
an LLM outage should surface as `JudgeUnavailable`, the same way `VectorSearchUnavailable`
surfaces an embedding outage in `domain/search_service.py`, rather than as a fabricated
number that makes a real quality regression invisible on the dashboard.
"""
from __future__ import annotations

import re

_FAITHFULNESS_SYSTEM_PROMPT = """You are a strict fact-checker for a retrieval-augmented \
answer. You are given an ANSWER and the CONTEXT it was supposed to be grounded in.

Judge what fraction of the factual claims in the ANSWER are directly supported by the \
CONTEXT. A claim is supported only if the CONTEXT states it or something that entails it \
-- not merely related, not "consistent with a plausible reading". An answer that says it \
found no information should score 1.0 when the CONTEXT genuinely lacks the information.

Respond with exactly one line, a single number between 0 and 100, and nothing else: no \
explanation, no punctuation, no words.
"""

_CONTEXT_PRECISION_SYSTEM_PROMPT = """You are grading search results for a knowledge-base \
question. You are given a QUESTION and a numbered list of PASSAGES retrieved for it.

Decide which passage numbers are actually relevant to answering the QUESTION -- a passage \
that is on-topic but does not help answer this specific question is NOT relevant.

Respond with exactly one line: the relevant passage numbers separated by commas (for \
example "1,3,4"), or the single word "none" if no passage is relevant. Nothing else.
"""

_ENTAILMENT_SYSTEM_PROMPT = """You check whether ONE sentence is actually supported by \
ONE source passage. This is a narrower question than "is this related to the passage" --
a passage can mention the same topic without stating the claim at all.

Answer "yes" only if the passage states the claim in the sentence, or states something \
that directly entails it. Answer "no" if the passage is silent on it, contradicts it, or \
only shares a topic with it.

Respond with exactly one word, "yes" or "no", and nothing else.
"""

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_INDEX_RE = re.compile(r"\d+")


class JudgeUnavailable(RuntimeError):
    """The LLM judge could not produce a usable score.

    Raised on a provider failure and on a response this module cannot parse -- the two
    are kept as one exception because a caller building a metrics dashboard needs to
    treat both the same way: skip this data point, do not plot a fabricated one.
    """


async def judge_faithfulness(answer: str, context: str) -> float:
    """What fraction of ANSWER's claims are supported by CONTEXT, per an LLM judge.

    Returns a score in [0, 1]. Raises `JudgeUnavailable` rather than guessing when the
    provider fails or its response contains no parseable number -- see the module
    docstring for why a fabricated score is worse than a missing data point.
    """
    # Imported here, not at module load, so a test can monkeypatch
    # `src.domain.llm_client.complete` the same way every other LLM-calling module in
    # this codebase is tested (see department_routing.py / test_llm_department_routing.py).
    from src.domain.llm_client import complete

    try:
        text, _tokens, _model, _provider = await complete(
            [
                {"role": "system", "content": _FAITHFULNESS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
                },
            ],
            max_tokens=16,
        )
    except Exception as exc:
        raise JudgeUnavailable(f"faithfulness judge call failed: {exc}") from exc

    match = _NUMBER_RE.search(text)
    if not match:
        raise JudgeUnavailable(
            f"faithfulness judge returned no parseable number: {text!r}"
        )
    score = float(match.group())
    # A judge that answers "0.8" meant a fraction; one that answers "80" meant a
    # percentage. Both are legitimate readings of "a number between 0 and 100" once a
    # model treats the instruction loosely, so values already in [0, 1] are left alone.
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


async def judge_context_precision(question: str, retrieved_passages: list[str]) -> float:
    """What fraction of `retrieved_passages` an LLM judge calls relevant to `question`.

    The counterpart to `evaluator.context_recall`: recall asks whether the passages the
    answer NEEDED were retrieved, this asks whether the passages that WERE retrieved were
    worth retrieving. Raises `JudgeUnavailable` on a provider failure or an unparseable
    response, same reasoning as `judge_faithfulness`.
    """
    if not retrieved_passages:
        return 1.0
    numbered = "\n\n".join(
        f"[{index}] {passage}" for index, passage in enumerate(retrieved_passages, start=1)
    )
    from src.domain.llm_client import complete

    try:
        text, _tokens, _model, _provider = await complete(
            [
                {"role": "system", "content": _CONTEXT_PRECISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nPASSAGES:\n{numbered}",
                },
            ],
            max_tokens=64,
        )
    except Exception as exc:
        raise JudgeUnavailable(f"context precision judge call failed: {exc}") from exc

    normalized = text.strip().lower()
    if normalized.startswith("none"):
        return 0.0
    indices = {int(value) for value in _INDEX_RE.findall(text)}
    if not indices:
        raise JudgeUnavailable(
            f"context precision judge returned no parseable indices: {text!r}"
        )
    valid = {index for index in indices if 1 <= index <= len(retrieved_passages)}
    return len(valid) / len(retrieved_passages)


async def judge_entailment(claim: str, passage: str) -> bool:
    """Whether `passage` actually supports `claim` -- not merely shares its topic.

    This is the AIS ("Attributable to Identified Sources") check: a citation marker
    asserts the passage next to it supports the specific sentence it is attached to, and
    `rag/citations.py` today only confirms the marker names a passage that was retrieved
    at all, not that the sentence is true of that passage. This function is the missing
    half, meant to run per flagged sentence rather than per answer -- callers should
    reserve it for sentences carrying a specific, checkable claim (a number, a date, a
    name) rather than every sentence, both for cost and because vague sentences rarely
    have a crisp yes/no answer.

    Raises `JudgeUnavailable` on a provider failure or a reply that is neither yes nor no
    -- guessing "supported" on an unparseable reply would silently defeat the check.
    """
    from src.domain.llm_client import complete

    try:
        text, _tokens, _model, _provider = await complete(
            [
                {"role": "system", "content": _ENTAILMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"PASSAGE:\n{passage}\n\nSENTENCE:\n{claim}",
                },
            ],
            max_tokens=4,
        )
    except Exception as exc:
        raise JudgeUnavailable(f"entailment judge call failed: {exc}") from exc

    normalized = text.strip().lower()
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    raise JudgeUnavailable(f"entailment judge returned neither yes nor no: {text!r}")
