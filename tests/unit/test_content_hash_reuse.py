"""Skip-if-unchanged re-embedding: a child whose own text has not changed since the last
successful index reuses its existing vector instead of paying for another embed call.

`_partition_for_reuse` is the pure decision at the center of this -- `reusable_by_hash`
is already scoped to the current EMBEDDING_VERSION by its caller in indexing.py (a hash
match under a different encoder is a vector in an unrelated space), so this only has to
do the lookup and preserve order.
"""
from __future__ import annotations

from src.domain.indexing import _partition_for_reuse


def test_a_hash_present_in_the_cache_is_reused():
    embeddings, to_embed = _partition_for_reuse(["abc"], {"abc": [1.0, 2.0]})
    assert embeddings == [[1.0, 2.0]]
    assert to_embed == []


def test_a_hash_absent_from_the_cache_needs_embedding():
    embeddings, to_embed = _partition_for_reuse(["abc"], {})
    assert embeddings == [None]
    assert to_embed == [0]


def test_order_is_preserved_across_a_mixed_batch():
    embeddings, to_embed = _partition_for_reuse(
        ["new-1", "cached", "new-2"], {"cached": [9.0]}
    )
    assert embeddings == [None, [9.0], None]
    assert to_embed == [0, 2]


def test_an_empty_batch_needs_nothing():
    embeddings, to_embed = _partition_for_reuse([], {"anything": [1.0]})
    assert embeddings == []
    assert to_embed == []


def test_a_duplicate_hash_within_the_batch_reuses_the_same_vector_for_both():
    """Two children that happen to have identical text (a repeated boilerplate line, for
    instance) are not a bug -- both legitimately reuse the one cached vector."""
    embeddings, to_embed = _partition_for_reuse(["dup", "dup"], {"dup": [5.0]})
    assert embeddings == [[5.0], [5.0]]
    assert to_embed == []
