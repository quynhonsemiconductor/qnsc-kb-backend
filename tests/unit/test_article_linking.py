"""The pure ranking core of topical cross-linking: collapse chunk rows to articles.

`collapse_to_top_articles` is what decides which articles get suggested as related, so
its correctness is what this file checks -- the pgvector query around it (`find_topical_matches`)
cannot be exercised without a real Postgres+pgvector connection, but everything it does
AFTER getting rows back from the database is ordinary Python and belongs here instead.
"""
from __future__ import annotations

from src.domain.article_linking import collapse_to_top_articles

A, B, C = "article-a", "article-b", "article-c"


def test_ranks_closest_first():
    rows = [(A, 0.30), (B, 0.10), (C, 0.20)]
    assert collapse_to_top_articles(rows, threshold=0.45, limit=5) == [B, C, A]


def test_an_article_is_scored_by_its_best_chunk_not_its_worst():
    # Article A has one great chunk and one poor one; article B has one mediocre chunk.
    # A's best (0.05) beats B's only option (0.20), so A must win even though it also
    # contributed a worse row that would rank behind B on its own.
    rows = [(A, 0.05), (A, 0.40), (B, 0.20)]
    assert collapse_to_top_articles(rows, threshold=0.45, limit=5) == [A, B]


def test_distance_above_threshold_is_excluded():
    rows = [(A, 0.10), (B, 0.50)]
    assert collapse_to_top_articles(rows, threshold=0.45, limit=5) == [A]


def test_distance_exactly_at_threshold_is_included():
    rows = [(A, 0.45)]
    assert collapse_to_top_articles(rows, threshold=0.45, limit=5) == [A]


def test_limit_caps_the_result_even_with_many_good_matches():
    rows = [(f"article-{i}", 0.01 * i) for i in range(10)]
    result = collapse_to_top_articles(rows, threshold=0.45, limit=3)
    assert result == [f"article-{i}" for i in range(3)]


def test_no_rows_returns_no_matches():
    assert collapse_to_top_articles([], threshold=0.45, limit=5) == []


def test_no_row_within_threshold_returns_no_matches():
    rows = [(A, 0.9), (B, 0.8)]
    assert collapse_to_top_articles(rows, threshold=0.45, limit=5) == []


def test_ids_are_stringified_for_a_stable_key():
    """Real rows carry UUID objects, not strings; the function must key on str(id)."""
    class _FakeUUID:
        def __init__(self, label):
            self._label = label

        def __str__(self):
            return self._label

    rows = [(_FakeUUID(A), 0.1), (_FakeUUID(A), 0.05)]
    result = collapse_to_top_articles(rows, threshold=0.45, limit=5)
    assert result == [A]
