from datetime import datetime
from types import SimpleNamespace

from src.rag.dedupe import collapse_duplicates, passage_key

CLAUSE = (
    "Nhân viên được nghỉ phép năm 12 ngày làm việc mỗi năm theo quy định của công ty."
)


def _chunk(
    cid,
    article,
    text=CLAUSE,
    embedding=None,
    reviewed=None,
    updated=None,
    evidence=None,
):
    return SimpleNamespace(
        id=cid,
        chunk_text=text,
        embedding=embedding,
        evidence_id=evidence,
        article=SimpleNamespace(
            id=article,
            title=f"Doc {article}",
            dept="HR",
            last_reviewed=reviewed,
            updated_at=updated,
        ),
    )


def _run(ranked, similarity=0.97, min_chars=20):
    return collapse_duplicates(ranked, similarity=similarity, min_chars=min_chars)


def test_the_key_ignores_case_punctuation_and_spacing():
    assert passage_key("Nghỉ  phép, 12 ngày!") == passage_key("nghỉ phép 12 ngày")


def test_the_same_passage_in_two_documents_becomes_one_result_that_names_the_other():
    kept, also_in = _run([(_chunk("c1", "a"), 0.9), (_chunk("c2", "b"), 0.8)])
    assert [c.id for c, _ in kept] == ["c1"]
    assert also_in == {
        "c1": [{"chunk_id": "c2", "article_id": "b", "title": "Doc b", "dept": "HR"}]
    }


def test_wording_that_differs_only_in_spacing_and_punctuation_is_still_a_copy():
    other = CLAUSE.replace(" năm 12", "  năm, 12").upper()
    kept, _ = _run([(_chunk("c1", "a"), 0.9), (_chunk("c2", "b", text=other), 0.9)])
    assert len(kept) == 1


def test_a_repeat_inside_one_article_is_kept():
    kept, also_in = _run([(_chunk("c1", "a"), 0.9), (_chunk("c2", "a"), 0.8)])
    assert len(kept) == 2 and also_in == {}


def test_the_better_scoring_copy_stays_even_when_it_arrives_second():
    kept, also_in = _run([(_chunk("c1", "a"), 0.5), (_chunk("c2", "b"), 0.9)])
    assert [c.id for c, _ in kept] == ["c2"]
    assert [s["article_id"] for s in also_in["c2"]] == ["a"]


def test_a_tie_goes_to_the_more_recently_reviewed_copy():
    old = _chunk("c1", "a", reviewed=datetime(2024, 1, 1))
    new = _chunk("c2", "b", reviewed=datetime(2026, 1, 1))
    kept, _ = _run([(old, 0.900), (new, 0.903)])
    assert [c.id for c, _ in kept] == ["c2"]


def test_the_winner_keeps_the_place_of_the_copy_it_replaces():
    other = _chunk(
        "x", "z", text="Một đoạn hoàn toàn khác về quy trình mua sắm vật tư."
    )
    kept, _ = _run([(_chunk("c1", "a"), 0.5), (other, 0.7), (_chunk("c2", "b"), 0.9)])
    assert [c.id for c, _ in kept] == ["c2", "x"]


def test_three_copies_list_every_other_document_once():
    kept, also_in = _run(
        [
            (_chunk("c1", "a"), 0.9),
            (_chunk("c2", "b"), 0.8),
            (_chunk("c3", "c"), 0.7),
            (_chunk("c4", "b"), 0.6),
        ]
    )
    assert [c.id for c, _ in kept] == ["c1"]
    assert sorted(s["article_id"] for s in also_in["c1"]) == ["b", "c"]


def test_short_passages_are_never_merged():
    kept, _ = _run(
        [
            (_chunk("c1", "a", text="Mục đích"), 0.9),
            (_chunk("c2", "b", text="Mục đích"), 0.8),
        ],
        min_chars=40,
    )
    assert len(kept) == 2


def test_figures_and_tables_are_not_compared_as_text():
    kept, _ = _run([(_chunk("c1", "a"), 0.9), (_chunk("c2", "b", evidence="e1"), 0.8)])
    assert len(kept) == 2


def test_near_identical_embeddings_merge_and_distant_ones_do_not():
    base = [1.0, 0.0, 0.0, 0.0]
    close = [0.999, 0.02, 0.0, 0.0]
    far = [0.2, 1.0, 0.0, 0.0]
    kept, also_in = _run(
        [
            (_chunk("c1", "a", text=CLAUSE, embedding=base), 0.9),
            (
                _chunk("c2", "b", text=CLAUSE + " Áp dụng từ 2026.", embedding=close),
                0.8,
            ),
            (
                _chunk(
                    "c3",
                    "c",
                    text="Quy trình mua sắm vật tư văn phòng phẩm hằng quý.",
                    embedding=far,
                ),
                0.7,
            ),
        ]
    )
    assert [c.id for c, _ in kept] == ["c1", "c3"]
    assert [s["article_id"] for s in also_in["c1"]] == ["b"]


def test_a_similarity_above_one_leaves_only_the_exact_match():
    base = [1.0, 0.0]
    kept, _ = _run(
        [
            (_chunk("c1", "a", embedding=base), 0.9),
            (_chunk("c2", "b", text=CLAUSE + " Thêm một câu.", embedding=base), 0.8),
        ],
        similarity=1.1,
    )
    assert len(kept) == 2


def test_missing_or_mismatched_embeddings_do_not_break_it():
    kept, _ = _run(
        [
            (_chunk("c1", "a", embedding=[1.0, 0.0]), 0.9),
            (_chunk("c2", "b", text=CLAUSE + " Khác.", embedding=None), 0.8),
            (
                _chunk(
                    "c3", "c", text=CLAUSE + " Khác nữa.", embedding=[1.0, 0.0, 0.0]
                ),
                0.7,
            ),
        ]
    )
    assert len(kept) == 3


def test_nothing_to_collapse_returns_the_input_in_order():
    ranked = [
        (
            _chunk(f"c{i}", f"a{i}", text=f"Đoạn số {i} nói về chủ đề riêng của nó."),
            1 - i / 10,
        )
        for i in range(4)
    ]
    kept, also_in = _run(ranked)
    assert [c.id for c, _ in kept] == ["c0", "c1", "c2", "c3"] and also_in == {}
