"""What one search costs the database and the API process.

Measured on a 25k-chunk corpus, before these changes:

* the keyword leg was ONE query whose WHERE joined the full-text match and the title match with an OR.
  The title match (`ILIKE '%term%'`) cannot use an index, which makes the whole OR unindexable, so every
  search scanned article_chunks, computed to_tsvector() for every row and ranked every row that passed:
  197 ms for a term that matches nothing at all, 393 ms for a term matching a fifth of the corpus, and
  growing with the corpus. Split into a full-text leg and a title leg, each uses its own existing index:
  2 ms and 41 ms for the same two queries.
* every candidate (a pool of 240 per leg) and every sibling child chunk was loaded WITH its 1024-float
  embedding, ~480 vectors parsed from text per request, to feed one consumer that only needs the final
  ranked list.

These tests pin the shape of the SQL and of the loading, not their speed.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import attributes

from src.models.chunk import ArticleChunk
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User
from src.repositories.chunk import ChunkRepository, attach_embeddings


def _user() -> User:
    user = User(
        id=uuid.uuid4(),
        email="a@acme.test",
        name="Admin",
        company_domain="acme.test",
        role="Admin",
        active=True,
    )
    role = Role(name="Admin", company_domain="acme.test", active=True)
    for key in ("article.read", "article.review", "article.publish"):
        role.permissions.append(
            RolePermission(permission=Permission(key=key, name=key), scope="global")
        )
    user.roles.append(role)
    return user


class _Db:
    """Records every statement as Postgres SQL and returns nothing."""

    def __init__(self):
        self.sql: list[str] = []

    async def execute(self, statement, *_args, **_kwargs):
        self.sql.append(str(statement.compile(dialect=postgresql.dialect())))
        return SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: []), all=lambda: []
        )


def _search(embedding):
    db = _Db()
    asyncio.run(
        ChunkRepository(db).hybrid_search(
            query="bao tri may khac",
            query_embedding=embedding,
            user=_user(),
            limit=10,
            filters={},
        )
    )
    return db.sql


def _chunk_selects(statements: list[str]) -> list[str]:
    return [sql for sql in statements if "FROM article_chunks" in sql]


# ── the keyword leg is two indexable queries, not one unindexable one ─────────────────────


def test_the_full_text_match_and_the_title_match_are_never_in_one_or():
    selects = _chunk_selects(_search(None))
    full_text = [sql for sql in selects if "@@" in sql]
    title = [sql for sql in selects if "ILIKE" in sql]
    assert len(full_text) == 1 and len(title) == 1, selects
    assert full_text[0] != title[0]
    # The whole point: an unindexable predicate must not share an OR with the indexable one.
    assert (
        "ILIKE" not in full_text[0]
    ), "the title match came back into the full-text query"
    assert "@@" not in title[0], "the full-text match came back into the title query"


def test_a_query_with_a_vector_adds_one_leg_and_keeps_the_other_two():
    with_vector = _chunk_selects(_search([0.1] * 8))
    without = _chunk_selects(_search(None))
    assert len(with_vector) == len(without) + 1
    assert sum("<=>" in sql.split("ORDER BY")[-1] for sql in with_vector) == 1


def test_the_full_text_leg_ranks_and_is_bounded():
    [full_text] = [sql for sql in _chunk_selects(_search(None)) if "@@" in sql]
    assert "ts_rank_cd" in full_text.split("ORDER BY")[-1]
    assert "LIMIT" in full_text


def test_the_title_leg_is_bounded_so_a_common_title_word_cannot_return_the_corpus():
    [title] = [sql for sql in _chunk_selects(_search(None)) if "ILIKE" in sql]
    assert "LIMIT" in title
    assert (
        "ts_rank_cd" not in title
    ), "the title leg must not rank every chunk it matches"


# ── candidates are loaded without their vectors ───────────────────────────────────────────


def _deferred_embedding_paths() -> list[list[str]]:
    """The attribute path of every option in RETRIEVAL_LOAD_OPTIONS that defers `embedding`.

    Read from the options themselves. Compiling a statement on its own does not apply loader options, so
    the SQL text proves nothing about deferral; the behaviour itself is checked against a real database in
    tests/integration/test_retrieval_legs.py.
    """
    from src.repositories.chunk import RETRIEVAL_LOAD_OPTIONS

    paths: list[list[str]] = []
    for option in RETRIEVAL_LOAD_OPTIONS:
        for element in option.context:
            deferred = dict(getattr(element, "strategy", None) or ()).get("deferred")
            keys = [
                getattr(entry, "key", None)
                for entry in (getattr(element, "path", None) or ())
            ]
            if deferred and keys and keys[-1] == "embedding":
                paths.append([key for key in keys if isinstance(key, str)])
    return paths


def test_every_candidate_is_loaded_without_its_vector():
    assert ["embedding"] in _deferred_embedding_paths(), (
        "a candidate would be loaded with its 1024-float vector; rag/dedupe.py gets vectors for the "
        "ranked list through attach_embeddings"
    )


def test_sibling_child_chunks_are_also_loaded_without_their_vector():
    # selectinload of ParentChunk.child_chunks is a separate statement run by the ORM; its options must
    # carry the defer too, or every parent drags all of its children's vectors along.
    assert any(
        "parent_chunk" in path and "child_chunks" in path
        for path in _deferred_embedding_paths()
    ), _deferred_embedding_paths()


def test_each_chunk_statement_applies_the_shared_options():
    # The three statements in hybrid_search all pass RETRIEVAL_LOAD_OPTIONS (also pinned in
    # test_retrieval_eager_loads); here, that the vector leg is among them when a vector is given.
    import inspect

    source = inspect.getsource(ChunkRepository.hybrid_search)
    assert source.count(".options(*RETRIEVAL_LOAD_OPTIONS)") >= 3


# ── attach_embeddings ─────────────────────────────────────────────────────────────────────


class _VectorDb:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    async def execute(self, statement):
        self.calls += 1
        return SimpleNamespace(all=lambda: list(self.vectors.items()))


def _unloaded_chunk() -> ArticleChunk:
    # Constructed without `embedding`, so the attribute is unloaded, as for a row fetched with defer().
    return ArticleChunk(id=uuid.uuid4(), chunk_text="x")


def test_vectors_for_the_ranked_list_come_back_in_one_query():
    chunks = [_unloaded_chunk() for _ in range(5)]
    db = _VectorDb({chunk.id: [float(index)] for index, chunk in enumerate(chunks)})
    asyncio.run(attach_embeddings(db, chunks))
    assert db.calls == 1
    for index, chunk in enumerate(chunks):
        assert "embedding" not in attributes.instance_state(chunk).unloaded
        assert chunk.embedding == [float(index)]


def test_a_chunk_that_already_has_its_vector_costs_nothing():
    chunk = _unloaded_chunk()
    attributes.set_committed_value(chunk, "embedding", [1.0])
    db = _VectorDb({})
    asyncio.run(attach_embeddings(db, [chunk]))
    assert db.calls == 0 and chunk.embedding == [1.0]


def test_anything_that_is_not_a_mapped_chunk_is_left_alone():
    # Test doubles and plain objects flow through the search path; they have no vector to load.
    db = _VectorDb({})
    asyncio.run(attach_embeddings(db, [SimpleNamespace(id=1), object(), None]))
    assert db.calls == 0


def test_a_chunk_without_a_stored_vector_ends_up_with_none_rather_than_unloaded():
    chunk = _unloaded_chunk()
    asyncio.run(attach_embeddings(_VectorDb({}), [chunk]))
    # Reading it must not raise MissingGreenlet later.
    assert chunk.embedding is None
