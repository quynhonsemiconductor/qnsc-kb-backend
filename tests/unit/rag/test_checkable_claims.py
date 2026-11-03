"""What is worth an entailment check, and which sentence a citation marker belongs to.

Both functions exist to make `rag/llm_judge.py::judge_entailment` affordable: checking
every sentence in every answer would multiply the LLM cost of a question by its sentence
count, so `has_checkable_claim` narrows the set to sentences with something concrete to
verify, and `split_cited_sentences` is what tells a caller which passage marker to check
each one against.
"""
from __future__ import annotations

from src.rag.citations import has_checkable_claim, split_cited_sentences


def test_a_number_makes_a_claim_checkable():
    assert has_checkable_claim("The retention period is 90 days.")


def test_a_percentage_makes_a_claim_checkable():
    assert has_checkable_claim("Coverage reached 96.7%.")


def test_a_vague_sentence_is_not_checkable():
    assert not has_checkable_claim("This policy covers the finance department.")


def test_an_empty_sentence_is_not_checkable():
    assert not has_checkable_claim("")
    assert not has_checkable_claim(None)  # type: ignore[arg-type]


def test_split_pairs_each_sentence_with_its_own_markers():
    answer = "The deadline is Friday [C1]. Approval needs two signatures [C2][C3]."
    result = split_cited_sentences(answer)
    assert result == [
        ("The deadline is Friday [C1].", ["C1"]),
        ("Approval needs two signatures [C2][C3].", ["C2", "C3"]),
    ]


def test_split_keeps_a_sentence_with_no_marker_and_an_empty_id_list():
    answer = "This is background. The limit is 5 [C1]."
    result = split_cited_sentences(answer)
    assert result[0] == ("This is background.", [])
    assert result[1] == ("The limit is 5 [C1].", ["C1"])


def test_split_of_empty_answer_returns_nothing():
    assert split_cited_sentences("") == []
    assert split_cited_sentences(None) == []  # type: ignore[arg-type]


def test_checkable_sentences_are_the_ones_worth_checking():
    """The intended combination: filter to checkable sentences that actually cite something."""
    answer = "This section is general guidance. The fee is 500,000 VND [C1]. See the owner for details [C2]."
    checkable_cited = [
        (sentence, ids)
        for sentence, ids in split_cited_sentences(answer)
        if ids and has_checkable_claim(sentence)
    ]
    assert checkable_cited == [("The fee is 500,000 VND [C1].", ["C1"])]
