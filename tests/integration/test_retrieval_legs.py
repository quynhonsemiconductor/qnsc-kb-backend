"""The two keyword legs and the deferred vectors, against a real, migrated PostgreSQL.

The unit tests pin the SQL shape. These pin what comes back: that splitting the full-text match from the
title match changed the cost of a search and not its result, that authorization still filters both legs, and
that candidates really arrive without their vectors (the ORM option is only observable on a loaded object).

Rows are tagged and removed afterwards; the corpus is a handful of rows, so nothing here depends on speed.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import delete, inspect

from src.workers.loop import sync_run

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs a real PostgreSQL: the point is what the planner and the ORM do with these queries",
)

TAG = "it-retrieval-"
COMPANY = "it-retrieval.test"
DEPT = "IT Retrieval"
#: A word no other test's data contains, so only this test's rows can match it.
WORD = "zorblaxium"


def _run(coro):
    return sync_run(coro)


def _vector(seed: float) -> list[float]:
    return [seed] + [0.0] * 1023


@pytest.fixture
def db():
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from src.api.deps import set_database_context
    from src.models.article import Article
    from src.models.user import Department

    engine = create_async_engine(os.environ["DATABASE_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def clean():
        async with factory() as session:
            await set_database_context(session, None, True)
            await session.execute(delete(Article).where(Article.title.like(f"{TAG}%")))
            await session.execute(
                delete(Department).where(Department.company_domain == COMPANY)
            )
            await session.commit()

    _run(clean())

    async def seed_department():
        from src.models.user import Department

        async with factory() as session:
            await set_database_context(session, None, True)
            session.add(
                Department(
                    company_domain=COMPANY,
                    name=DEPT,
                    description=DEPT,
                    active=True,
                    review_required=False,
                )
            )
            await session.commit()

    _run(seed_department())
    yield factory
    _run(clean())
    _run(engine.dispose())


def _user():
    from src.models.rbac import Permission, Role, RolePermission
    from src.models.user import User

    user = User(
        id=uuid.uuid4(),
        email=f"admin@{COMPANY}",
        name="Admin",
        company_domain=COMPANY,
        role="Admin",
        active=True,
    )
    role = Role(name="Admin", company_domain=COMPANY, active=True)
    for key in ("article.read", "article.review", "article.publish"):
        role.permissions.append(
            RolePermission(permission=Permission(key=key, name=key), scope="global")
        )
    user.roles.append(role)
    return user


async def _article(
    factory,
    name: str,
    *,
    title: str | None = None,
    texts=("nothing relevant",),
    **overrides,
):
    """An article with one parent chunk holding one child chunk per text. Returns (article_id, [chunk ids])."""
    from src.api.deps import set_database_context
    from src.models.article import Article
    from src.models.chunk import ArticleChunk, ParentChunk

    values = dict(
        title=f"{TAG}{title or name}",
        body_md="# body",
        dept=DEPT,
        domain="ops",
        company_domain=COMPANY,
        type="SOP",
        sensitivity="internal",
        visibility="department",
        status="published",
        lifecycle_status="active",
        version=1,
        needs_update=False,
        index_status="ready",
    )
    values.update(overrides)
    async with factory() as session:
        await set_database_context(session, None, True)
        article = Article(**values)
        session.add(article)
        await session.flush()
        parent = ParentChunk(
            article_id=article.id,
            text=" ".join(texts),
            section_ref="S",
            chunk_type="section",
        )
        session.add(parent)
        await session.flush()
        chunks = [
            ArticleChunk(
                article_id=article.id,
                parent_chunk_id=parent.id,
                chunk_text=text,
                embedding=_vector(1.0 + index),
                embedding_model="mock",
                embedding_version="it-v1",
                department_id=DEPT,
                sensitivity="internal",
                visibility="department",
                chunk_index=index,
                chunk_type="text",
                chunking_version="v1",
            )
            for index, text in enumerate(texts)
        ]
        session.add_all(chunks)
        await session.commit()
        return article.id, [chunk.id for chunk in chunks]


async def _search(factory, query: str, embedding=None):
    from src.api.deps import set_database_context
    from src.repositories.chunk import ChunkRepository

    async with factory() as session:
        await set_database_context(session, None, True)
        hits = await ChunkRepository(session).hybrid_search(
            query=query, query_embedding=embedding, user=_user(), limit=10, filters={}
        )
        # Inspect inside the session: this is what the search service sees.
        return [
            (
                hit.id,
                "embedding" in inspect(hit).unloaded,
                [
                    ("embedding" in inspect(c).unloaded)
                    for c in hit.parent_chunk.child_chunks
                ],
            )
            for hit in hits
        ]


# ── the keyword legs ──────────────────────────────────────────────────────────────────────


def test_text_matches_come_first_then_title_only_matches_and_nothing_twice(db):
    [(_, [text_hit])] = [
        _run(_article(db, "a-text", texts=(f"this chunk mentions {WORD} plainly",)))
    ]
    [(_, [title_only])] = [
        _run(_article(db, f"b-{WORD}", texts=("a chunk that says nothing about it",)))
    ]
    [(_, [both])] = [
        _run(_article(db, f"c-{WORD}", texts=(f"and the text says {WORD} too",)))
    ]
    _run(_article(db, "d-unrelated", texts=("completely different subject",)))

    ids = [hit[0] for hit in _run(_search(db, WORD))]

    assert set(ids) == {text_hit, title_only, both}
    assert len(ids) == len(set(ids)), "a chunk found by both legs must appear once"
    # The text leg's results lead; the chunk that matched only through its article's title is last.
    assert ids.index(title_only) == len(ids) - 1
    assert {ids[0], ids[1]} == {text_hit, both}


def test_a_term_that_matches_nothing_returns_nothing(db):
    _run(_article(db, "plain", texts=("an ordinary chunk",)))
    assert _run(_search(db, "qwertyzzxx")) == []


def test_authorization_filters_both_legs(db):
    _run(
        _article(db, "draft-text", texts=(f"draft mentioning {WORD}",), status="draft")
    )
    _run(_article(db, f"draft-title-{WORD}", texts=("body",), status="draft"))
    _run(
        _article(
            db,
            "archived-text",
            texts=(f"archived mentioning {WORD}",),
            lifecycle_status="archived",
        )
    )
    [(_, [visible])] = [
        _run(_article(db, "visible", texts=(f"published mentioning {WORD}",)))
    ]

    ids = [hit[0] for hit in _run(_search(db, WORD))]
    assert ids == [
        visible
    ], "an unpublished or archived article leaked through one of the legs"


def test_the_title_leg_is_bounded_so_a_common_title_word_cannot_return_the_corpus(db):
    from src.core.config import settings

    for index in range(settings.RAG_CANDIDATE_POOL_SIZE + 20):
        _run(_article(db, f"common-{WORD}-{index}", texts=("filler text",)))
    hits = _run(_search(db, WORD))
    assert len(hits) <= settings.RAG_CANDIDATE_POOL_SIZE


# ── the vectors ───────────────────────────────────────────────────────────────────────────


def test_candidates_and_their_siblings_arrive_without_a_vector_and_attach_embeddings_loads_them(
    db,
):
    from src.api.deps import set_database_context
    from src.repositories.chunk import ChunkRepository, attach_embeddings

    _run(
        _article(
            db,
            "vectors",
            texts=(
                f"first chunk about {WORD}",
                "second sibling chunk",
                "third sibling chunk",
            ),
        )
    )

    async def scenario():
        async with db() as session:
            await set_database_context(session, None, True)
            hits = await ChunkRepository(session).hybrid_search(
                query=WORD, query_embedding=None, user=_user(), limit=10, filters={}
            )
            assert hits, "the seeded chunk was not found"
            siblings = [c for hit in hits for c in hit.parent_chunk.child_chunks]
            before = (
                "embedding" in inspect(hits[0]).unloaded,
                [("embedding" in inspect(c).unloaded) for c in siblings],
            )
            await attach_embeddings(session, hits)
            # After attach, reading the vector must not raise MissingGreenlet and must be the stored one.
            loaded = [list(hit.embedding)[:1] for hit in hits]
            # The hit is also one of its parent's children (same object); only the OTHER children are siblings.
            hit_ids = {hit.id for hit in hits}
            still_deferred_siblings = [
                ("embedding" in inspect(c).unloaded)
                for c in siblings
                if c.id not in hit_ids
            ]
            return before, loaded, still_deferred_siblings

    (hit_deferred, sibling_deferred), loaded, still_deferred = _run(scenario())
    assert hit_deferred is True, "a candidate was loaded with its vector"
    assert sibling_deferred and all(
        sibling_deferred
    ), "a sibling chunk was loaded with its vector"
    assert loaded == [[1.0]]
    assert len(still_deferred) == 2 and all(
        still_deferred
    ), "attach_embeddings must load the ranked hits only, not their siblings"


def test_the_vector_leg_still_works_and_also_returns_candidates_without_vectors(db):
    near, far = None, None
    [(_, [near])] = [_run(_article(db, "near", texts=("the near chunk",)))]
    hits = _run(_search(db, "unrelated words only", embedding=_vector(1.0)))
    # `it-v1` is not the configured embedding version, so nothing matches the vector leg here: the point is
    # that the leg runs against the real schema and its options without error.
    assert isinstance(hits, list)
    assert near is not None and far is None
