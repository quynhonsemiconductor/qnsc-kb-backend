"""e5 needs `query: ` / `passage: `; getting it wrong costs recall silently.

Nothing raises when the prefix is missing or when a query is embedded as a
passage: the vectors are still unit-norm and still retrieve something, just
worse. So these assertions are the only thing standing between a model swap and
a quiet regression.
"""
from __future__ import annotations

import pytest

from src.core.config import Settings, settings
from src.lib.embeddings import _decorate, _needs_instruction_prefix


@pytest.fixture
def as_model(monkeypatch):
    def _set(name: str) -> None:
        monkeypatch.setattr(settings, "EMBEDDING_MODEL", name)

    return _set


def test_e5_query_and_passage_get_different_prefixes(as_model) -> None:
    as_model("intfloat/multilingual-e5-small")
    assert _decorate(["thủ đô"], "RETRIEVAL_QUERY") == ["query: thủ đô"]
    assert _decorate(["thủ đô"], "RETRIEVAL_DOCUMENT") == ["passage: thủ đô"]


def test_minilm_is_left_alone(as_model) -> None:
    # The previous model was trained without instructions; adding one would put
    # every vector somewhere the corpus was not embedded.
    as_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert _decorate(["thủ đô"], "RETRIEVAL_QUERY") == ["thủ đô"]
    assert not _needs_instruction_prefix()


def test_bge_m3_is_left_alone(as_model) -> None:
    # bge-m3's own FAQ: it no longer requires instructions.
    as_model("BAAI/bge-m3")
    assert _decorate(["thủ đô"], "RETRIEVAL_DOCUMENT") == ["thủ đô"]
    assert not _needs_instruction_prefix()


def test_every_text_in_a_batch_is_prefixed(as_model) -> None:
    as_model("intfloat/multilingual-e5-base")
    assert _decorate(["a", "b"], "RETRIEVAL_DOCUMENT") == ["passage: a", "passage: b"]


def test_e5_small_keeps_the_384_wide_column() -> None:
    # Same width as MiniLM, which is why this swap needs no pgvector migration.
    assert Settings(EMBEDDING_MODEL="intfloat/multilingual-e5-small").EMBEDDING_DIMENSION == 384


def test_e5_base_is_768_wide() -> None:
    assert Settings(EMBEDDING_MODEL="intfloat/multilingual-e5-base").EMBEDDING_DIMENSION == 768
