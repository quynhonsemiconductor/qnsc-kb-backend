"""Which chunks are worth a cross-encoder pair, and how the result is folded back into the lexical ranking."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from src.rag.cross_encoder import cross_passage, promote, select_candidates


def _chunk(label: str):
    return SimpleNamespace(id=uuid.uuid4(), label=label)


def _ranked(n: int):
    """n chunks in lexical order with descending lexical scores 0.9, 0.8, ..."""
    return [(_chunk(f"c{i}"), round(0.9 - 0.1 * i, 2)) for i in range(n)]


# --- select_candidates --------------------------------------------------------------------------------------------


def test_the_first_lexical_candidates_are_chosen():
    assert select_candidates(_ranked(20), {}, 5, 0) == [0, 1, 2, 3, 4]


def test_the_best_vector_hits_are_added_wherever_they_sit_in_the_lexical_order():
    ranked = _ranked(20)
    ranks = {ranked[17][0].id: 1, ranked[9][0].id: 2, ranked[3][0].id: 3}
    assert select_candidates(ranked, ranks, 3, 2) == [0, 1, 2, 9, 17]


def test_a_vector_hit_already_among_the_lexical_ones_is_not_counted_twice():
    ranked = _ranked(10)
    ranks = {ranked[1][0].id: 1, ranked[8][0].id: 2}
    assert select_candidates(ranked, ranks, 3, 1) == [0, 1, 2]


def test_fewer_chunks_than_asked_for_is_not_an_error():
    assert select_candidates(_ranked(3), {}, 10, 5) == [0, 1, 2]
    assert select_candidates([], {}, 10, 5) == []


def test_zero_vector_candidates_means_lexical_only():
    ranked = _ranked(10)
    assert select_candidates(ranked, {ranked[9][0].id: 1}, 2, 0) == [0, 1]


def test_the_result_is_in_lexical_order():
    ranked = _ranked(10)
    ranks = {ranked[8][0].id: 1, ranked[5][0].id: 2}
    chosen = select_candidates(ranked, ranks, 2, 2)
    assert chosen == sorted(chosen)


# --- promote ------------------------------------------------------------------------------------------------------


def _promote(ranked, candidates, scores, top_k=0, floor=0.4, minimum=0.05):
    return promote(ranked, candidates, scores, top_k, floor, minimum)


def test_scored_candidates_come_first_best_score_first_then_the_rest_in_lexical_order():
    ranked = _ranked(6)
    out = _promote(ranked, [0, 1, 2], [0.1, 0.9, 0.5])
    assert [c.label for c, _ in out] == ["c1", "c2", "c0", "c3", "c4", "c5"]


def test_without_assist_each_item_keeps_its_own_lexical_score():
    ranked = _ranked(4)
    out = _promote(ranked, [0, 1], [0.1, 0.9])
    assert dict((c.label, s) for c, s in out) == {c.label: s for c, s in ranked}


def test_the_top_k_the_model_is_sure_of_are_lifted_to_the_floor():
    ranked = [(_chunk("buried"), 0.05), (_chunk("fine"), 0.8), (_chunk("weak"), 0.05)]
    out = _promote(ranked, [0, 1, 2], [0.95, 0.5, 0.01], top_k=2, floor=0.4)
    scores = {c.label: s for c, s in out}
    assert scores["buried"] == 0.4  # lifted
    assert scores["fine"] == 0.8  # never lowered
    assert scores["weak"] == 0.05  # outside the top 2


def test_a_model_that_is_not_sure_lifts_nothing():
    ranked = [(_chunk("a"), 0.05)]
    out = _promote(ranked, [0], [0.02], top_k=3, floor=0.4, minimum=0.05)
    assert out[0][1] == 0.05


def test_ties_keep_the_lexical_order_and_never_compare_the_chunks():
    class Unorderable(SimpleNamespace):
        def __lt__(self, other):  # pragma: no cover
            raise TypeError

    ranked = [(Unorderable(id=1), 0.5), (Unorderable(id=2), 0.4)]
    out = _promote(ranked, [0, 1], [0.7, 0.7])
    assert [c.id for c, _ in out] == [1, 2]


def test_a_length_mismatch_is_a_caller_bug():
    with pytest.raises(ValueError):
        _promote(_ranked(3), [0, 1], [0.5])


def test_nothing_to_promote_returns_the_ranking_unchanged():
    ranked = _ranked(3)
    assert _promote(ranked, [], []) == ranked


# --- cross_passage ------------------------------------------------------------------------------------------------


def test_the_passage_reads_where_it_is_from_then_what_it_says():
    chunk = SimpleNamespace(
        chunk_text="PADOUTSET | 0x10 | Write SET",
        heading="Register Map",
        article=SimpleNamespace(title="QNSC_MAS_GPIO"),
        parent_chunk=SimpleNamespace(section_ref="Section 6", text="parent"),
    )
    assert (
        cross_passage(chunk)
        == "QNSC_MAS_GPIO - Register Map\nPADOUTSET | 0x10 | Write SET"
    )


def test_the_section_is_used_when_the_chunk_has_no_heading():
    chunk = SimpleNamespace(
        chunk_text="text",
        heading=None,
        article=SimpleNamespace(title="T"),
        parent_chunk=SimpleNamespace(section_ref="Section 4", text="parent"),
    )
    assert cross_passage(chunk).startswith("T - Section 4\n")


def test_a_very_long_chunk_is_cut():
    chunk = SimpleNamespace(
        chunk_text="x" * 10_000,
        heading="h",
        article=SimpleNamespace(title="t"),
        parent_chunk=None,
    )
    assert len(cross_passage(chunk)) == 1800
