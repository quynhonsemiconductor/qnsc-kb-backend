"""Parsing discipline for the LLM-judged metrics: never fabricate a score.

`rag/llm_judge.py` exists precisely because `rag/evaluator.py` cannot call an LLM and stay
offline-friendly. These functions are the risky half of that split, so what has to be
proven here is not "the happy path returns a number" -- it is that every way a model can
answer badly (silence, prose instead of a number, an out-of-range passage index) raises
`JudgeUnavailable` instead of quietly returning a plausible-looking score.
"""
from __future__ import annotations

import asyncio

import pytest

from src.rag.llm_judge import (
    JudgeUnavailable,
    find_unverified_claims,
    judge_context_precision,
    judge_entailment,
    judge_faithfulness,
)


@pytest.fixture
def llm(monkeypatch):
    """Install a fake provider. Returns a setter for what the model replies."""
    from src.domain import llm_client

    state: dict = {"reply": "100", "raises": None}

    async def fake_complete(messages, **kwargs):
        state["messages"] = messages
        if state["raises"] is not None:
            raise state["raises"]
        return state["reply"], 4, "fake-model", "fake"

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    return state


def _faithfulness(llm, answer="The deadline is Friday.", context="Deadline: Friday."):
    return asyncio.run(judge_faithfulness(answer, context))


def _precision(llm, question="What is the deadline?", passages=("Deadline: Friday.", "Unrelated text.")):
    return asyncio.run(judge_context_precision(question, list(passages)))


@pytest.mark.parametrize("reply,expected", [("100", 1.0), ("0", 0.0), ("62", 0.62), ("1", 1.0), ("0.42", 0.42)])
def test_faithfulness_accepts_percentage_or_fraction(llm, reply, expected):
    llm["reply"] = reply
    assert _faithfulness(llm) == pytest.approx(expected)


def test_faithfulness_tolerates_surrounding_prose():
    async def fake_complete(messages, **kwargs):
        return "The score is 75 out of 100.", 8, "fake-model", "fake"

    import src.domain.llm_client as llm_client_module

    original = llm_client_module.complete
    llm_client_module.complete = fake_complete
    try:
        assert asyncio.run(judge_faithfulness("answer", "context")) == pytest.approx(0.75)
    finally:
        llm_client_module.complete = original


@pytest.mark.parametrize("reply", ["", "not a number", "no idea", "n/a"])
def test_faithfulness_raises_on_unparseable_reply(llm, reply):
    llm["reply"] = reply
    with pytest.raises(JudgeUnavailable):
        _faithfulness(llm)


def test_faithfulness_raises_on_provider_failure(llm):
    llm["raises"] = RuntimeError("provider is down")
    with pytest.raises(JudgeUnavailable):
        _faithfulness(llm)


def test_faithfulness_clamps_out_of_range_scores(llm):
    # Not a reply a well-behaved judge should send, but a parser that trusts its input
    # completely is exactly how a dashboard ends up plotting a score above 100%.
    llm["reply"] = "500"
    assert _faithfulness(llm) == 1.0


def test_precision_counts_only_relevant_indices(llm):
    llm["reply"] = "1"
    assert _precision(llm) == pytest.approx(0.5)


def test_precision_all_relevant(llm):
    llm["reply"] = "1, 2"
    assert _precision(llm) == pytest.approx(1.0)


def test_precision_none_relevant(llm):
    llm["reply"] = "none"
    assert _precision(llm) == 0.0


def test_precision_ignores_out_of_range_indices(llm):
    # A judge that hallucinates a fourth passage must not inflate the denominator or
    # silently pass -- only indices that name a real passage count.
    llm["reply"] = "1, 99"
    assert _precision(llm) == pytest.approx(0.5)


def test_precision_empty_passage_list_short_circuits(llm):
    assert asyncio.run(judge_context_precision("question", [])) == 1.0


def test_precision_raises_when_no_index_and_not_none(llm):
    llm["reply"] = "all of them look fine"
    with pytest.raises(JudgeUnavailable):
        _precision(llm)


def test_precision_raises_on_provider_failure(llm):
    llm["raises"] = RuntimeError("provider is down")
    with pytest.raises(JudgeUnavailable):
        _precision(llm)


@pytest.mark.parametrize("reply,expected", [("yes", True), ("Yes.", True), ("no", False), ("No, it does not.", False)])
def test_entailment_reads_yes_or_no(llm, reply, expected):
    llm["reply"] = reply
    assert asyncio.run(judge_entailment("The deadline is Friday.", "Deadline: Friday.")) is expected


@pytest.mark.parametrize("reply", ["", "maybe", "it depends on context"])
def test_entailment_raises_on_anything_else(llm, reply):
    llm["reply"] = reply
    with pytest.raises(JudgeUnavailable):
        asyncio.run(judge_entailment("claim", "passage"))


def test_entailment_raises_on_provider_failure(llm):
    llm["raises"] = RuntimeError("provider is down")
    with pytest.raises(JudgeUnavailable):
        asyncio.run(judge_entailment("claim", "passage"))


# --- find_unverified_claims: the live-path composition -------------------------------


def test_flags_a_checkable_sentence_the_judge_rejects(llm):
    llm["reply"] = "no"
    result = asyncio.run(
        find_unverified_claims(
            "The deadline is 5 business days [C1].",
            {"C1": "This document does not mention any deadline."},
        )
    )
    assert result == [{"sentence": "The deadline is 5 business days [C1].", "source_id": "C1"}]


def test_does_not_flag_a_sentence_the_judge_confirms(llm):
    llm["reply"] = "yes"
    result = asyncio.run(
        find_unverified_claims("The deadline is 5 business days [C1].", {"C1": "Deadline: 5 business days."})
    )
    assert result == []


def test_skips_a_sentence_with_no_checkable_claim(llm):
    result = asyncio.run(
        find_unverified_claims("This policy covers the finance department [C1].", {"C1": "Some passage."})
    )
    assert result == []
    assert "messages" not in llm, "no judge call should have been made"


def test_skips_a_sentence_citing_more_than_one_source(llm):
    result = asyncio.run(
        find_unverified_claims("The deadline is 5 business days [C1][C2].", {"C1": "a", "C2": "b"})
    )
    assert result == []
    assert "messages" not in llm


def test_skips_a_sentence_whose_cited_source_was_not_retrieved(llm):
    result = asyncio.run(find_unverified_claims("The deadline is 5 business days [C9].", {"C1": "a"}))
    assert result == []
    assert "messages" not in llm


def test_a_judge_failure_is_not_reported_as_unverified(llm):
    """JudgeUnavailable means 'never checked', not 'wrong' -- the whole point of never
    running this from the live path unguarded."""
    llm["raises"] = RuntimeError("provider is down")
    result = asyncio.run(
        find_unverified_claims("The deadline is 5 business days [C1].", {"C1": "Deadline: 5 business days."})
    )
    assert result == []


def test_stops_at_max_sentences(llm):
    llm["reply"] = "no"
    answer = " ".join(f"Section {index} lasts {index} days [C1]." for index in range(1, 8))
    result = asyncio.run(find_unverified_claims(answer, {"C1": "irrelevant"}, max_sentences=3))
    assert len(result) == 3
