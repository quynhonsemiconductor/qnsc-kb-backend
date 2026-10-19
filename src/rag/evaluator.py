"""Offline-friendly metrics used by the evaluation dashboard and CI."""

from __future__ import annotations

import re
from collections.abc import Iterable


def context_recall(
    retrieved_chunk_ids: Iterable[str], expected_chunk_ids: Iterable[str]
) -> float:
    expected = {str(item) for item in expected_chunk_ids}
    if not expected:
        return 1.0
    return len(expected.intersection(str(item) for item in retrieved_chunk_ids)) / len(
        expected
    )


def lexical_faithfulness(answer: str, context: str) -> float:
    """Conservative proxy: proportion of answer content words in retrieved context."""
    if not context.strip() and re.search(
        r"not found|no information|không tìm thấy|không có thông tin",
        answer,
        re.IGNORECASE,
    ):
        return 1.0
    # Citation identifiers are provenance metadata, not answer claims.
    answer = re.sub(r"\[[^\]]+\]", " ", answer)
    answer_terms = {
        item for item in re.findall(r"[\w'-]+", answer.lower()) if len(item) > 3
    }
    if not answer_terms:
        return 1.0
    context_terms = set(re.findall(r"[\w'-]+", context.lower()))
    return len(answer_terms.intersection(context_terms)) / len(answer_terms)


def answer_correctness(answer: str, expected_answer: str) -> float:
    expected_terms = {
        item
        for item in re.findall(r"[\w'-]+", expected_answer.lower())
        if len(item) > 3
    }
    if not expected_terms:
        return 1.0
    answer_terms = set(re.findall(r"[\w'-]+", answer.lower()))
    return len(expected_terms.intersection(answer_terms)) / len(expected_terms)
