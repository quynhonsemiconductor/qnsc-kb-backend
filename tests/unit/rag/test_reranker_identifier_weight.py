"""The reranker counts a named thing (a register, a signal) for more than the words around it.

Measured on 17 chip-specification questions (real e5-large embeddings, DeepSeek answers, 3 runs each):
questions answered correctly 9.0 -> 11.3 of 16. Before this, `thanh ghi` + `offset` (3 of 6 terms, in another
module's prose) outscored `PADOUTSET` + `GPIO` (2 of 6, the register table).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.rag.identifier_terms import identifier_terms
from src.rag.reranker import (
    identifier_weights,
    normalize_query,
    prepare_query,
    rerank_chunks_with_scores,
    retrieval_score,
)

QUESTION = "Thanh ghi PADOUTSET của GPIO nằm ở offset nào?"


def _chunk(text: str, title: str = "doc"):
    return SimpleNamespace(
        chunk_text=text,
        article=SimpleNamespace(title=title),
        parent_chunk=None,
        id=text[:12],
    )


GPIO = _chunk(
    "PADOUTSET | 0x10 | Write SET | Write one to set the PADOUT bit on this GPIO."
)
PROSE = _chunk(
    "6.1 Thanh ghi kênh CH0 (base + offset). Thanh ghi nào cũng có offset, ghi theo word."
)


def test_the_register_table_outranks_another_modules_prose():
    ranked = rerank_chunks_with_scores(
        normalize_query(QUESTION),
        [PROSE, GPIO],
        identifiers=identifier_terms(QUESTION),
    )
    assert ranked[0][0] is GPIO


def test_unweighted_the_prose_won(monkeypatch):
    """The behaviour this replaces, kept as a test of the setting that restores it."""
    monkeypatch.setattr(settings, "RAG_IDENTIFIER_TERM_WEIGHT", 1.0)
    ranked = rerank_chunks_with_scores(
        normalize_query(QUESTION),
        [GPIO, PROSE],
        identifiers=identifier_terms(QUESTION),
    )
    assert ranked[0][0] is PROSE


def test_a_normalised_question_has_lost_the_capitals_that_mark_a_name():
    """Why the caller must pass `identifiers` from the ORIGINAL question: normalize_query lower-cases, and
    only names with a digit or an underscore survive that."""
    assert identifier_terms(QUESTION) == ["padoutset", "gpio"]
    assert identifier_terms(normalize_query(QUESTION)) == []
    assert identifier_terms(normalize_query("Địa chỉ của I2C và UART0?")) == [
        "i2c",
        "uart0",
    ]


def test_without_the_original_question_a_plain_name_is_not_weighted():
    prepared = prepare_query(normalize_query(QUESTION))
    assert prepared.identifier_weights == {}


def test_with_it_the_names_are_weighted():
    prepared = prepare_query(
        normalize_query(QUESTION), identifiers=identifier_terms(QUESTION)
    )
    assert set(prepared.identifier_weights) == {"padoutset", "gpio"}
    # No pool statistics were given, so each is worth the maximum.
    assert set(prepared.identifier_weights.values()) == {
        settings.RAG_IDENTIFIER_TERM_WEIGHT
    }


def test_a_name_that_no_candidate_contains_is_not_counted_against_everything():
    """The existing rule for terms nobody can match, kept: only names that are findable are weighted."""
    prepared = prepare_query(
        "thanh ghi zzzname",
        findable=frozenset({"thanh", "ghi"}),
        identifiers=["zzzname"],
    )
    assert prepared.identifier_weights == {}


@pytest.mark.parametrize(
    "question",
    [
        "Nhân viên được nghỉ phép bao nhiêu ngày một năm?",
        "Chính sách bảo hiểm sức khỏe của công ty như thế nào?",
    ],
)
def test_a_question_that_names_nothing_scores_exactly_as_before(monkeypatch, question):
    chunk = _chunk("Mỗi nhân viên chính thức được hưởng 12 ngày nghỉ phép năm.")
    prepared = prepare_query(normalize_query(question), identifiers=[])
    weighted = retrieval_score(normalize_query(question), chunk, prepared)
    monkeypatch.setattr(settings, "RAG_IDENTIFIER_TERM_WEIGHT", 1.0)
    unweighted = retrieval_score(
        normalize_query(question),
        chunk,
        prepare_query(normalize_query(question), identifiers=[]),
    )
    assert weighted == unweighted


def test_a_passage_holding_every_term_still_scores_full_marks():
    chunk = _chunk("PADOUTSET GPIO thanh ghi offset nam")
    prepared = prepare_query(
        normalize_query(QUESTION), identifiers=identifier_terms(QUESTION)
    )
    assert retrieval_score("", chunk, prepared) == pytest.approx(1.0, abs=0.36)
    assert retrieval_score("", chunk, prepared) <= 1.35


# --- how much a name is worth depends on how common it is among the candidates --------


def test_a_name_in_a_few_candidates_keeps_nearly_the_whole_weight():
    weights = identifier_weights(["padoutset"], {"padoutset": 8}, 400, 6.0)
    assert weights["padoutset"] > 4.0


def test_a_name_in_a_third_of_the_candidates_is_worth_little_more_than_a_word():
    weights = identifier_weights(["gpio"], {"gpio": 150}, 400, 6.0)
    assert 1.0 < weights["gpio"] < 2.0


def test_a_name_in_every_candidate_is_worth_an_ordinary_word():
    assert identifier_weights(["apb"], {"apb": 400}, 400, 6.0)["apb"] == pytest.approx(
        1.0
    )


def test_the_rarer_the_name_the_more_it_weighs():
    weights = identifier_weights(["a", "b", "c"], {"a": 2, "b": 20, "c": 200}, 400, 6.0)
    assert weights["a"] > weights["b"] > weights["c"] > 1.0


def test_the_weight_never_exceeds_the_configured_maximum():
    weights = identifier_weights(["a"], {"a": 1}, 400, 6.0)
    assert weights["a"] <= 6.0


def test_a_pool_too_small_to_say_gives_the_maximum():
    assert identifier_weights(["a"], {"a": 1}, 1, 6.0) == {"a": 6.0}


def _pool():
    """What the fix is for: hundreds of chunks that merely mention the module, one that holds the register."""
    mentions = [
        _chunk(f"Each GPIO pin number {n} can be muxed to another function.", "pins")
        for n in range(60)
    ]
    return mentions, GPIO


def test_chunks_that_only_mention_the_module_no_longer_tie_with_the_register_table():
    """A fixed weight scored every chunk containing GPIO the same as the one containing PADOUTSET, and with
    enough of them tied, the register table fell outside the top results."""
    mentions, table = _pool()
    ranked = rerank_chunks_with_scores(
        normalize_query(QUESTION),
        [*mentions, table],
        limit=5,
        identifiers=identifier_terms(QUESTION),
    )
    assert ranked[0][0] is table
    assert ranked[0][1] > ranked[1][1]


def test_the_score_of_a_named_register_chunk_is_high_enough_to_be_used_as_context():
    mentions, table = _pool()
    ranked = rerank_chunks_with_scores(
        normalize_query(QUESTION),
        [*mentions, table],
        limit=5,
        identifiers=identifier_terms(QUESTION),
    )
    assert ranked[0][1] >= settings.RAG_MIN_CONTEXT_SCORE
