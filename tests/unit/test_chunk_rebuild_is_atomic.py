"""A reindex must never commit an article into a state where it has no chunks.

`delete_by_article_id` used to commit the wipe on its own, and `create_parent_chunk`
committed once per parent on top of that. So a reindex was a long sequence of independent
commits starting with "this article now has zero chunks" -- and the article stays
`published` throughout. Any failure between the wipe and the last parent left the document
searchable with nothing behind it, permanently, while `index_status` still read `ready`
from the previous successful run. A concurrent search during a healthy reindex saw the same
empty window.

These assert the commit BOUNDARY, not the SQL: that the repository stages its writes, that
every standalone caller owns a commit, and that a mid-rebuild embedding failure rolls the
wipe back instead of publishing a partial index.
"""
from __future__ import annotations

import asyncio
import inspect
import uuid

from src.domain import indexing
from src.repositories.chunk import ChunkRepository


class _Session:
    """Records commit/flush/rollback order against the statements that preceded them."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def execute(self, statement, params=None, *args, **kwargs):
        rendered = " ".join(str(statement).lower().split())
        verb = rendered.split(" ", 1)[0]
        target = rendered.split()[2] if verb == "delete" else ""
        self.events.append(f"{verb} {target}".strip())
        return None

    def add(self, _obj) -> None:
        self.events.append("add")

    def add_all(self, objs) -> None:
        for _obj in objs:
            self.events.append("add")

    async def flush(self) -> None:
        self.events.append("flush")

    async def commit(self) -> None:
        self.events.append("commit")

    async def rollback(self) -> None:
        self.events.append("rollback")

    async def refresh(self, _obj) -> None:
        return None


def test_the_chunk_wipe_is_staged_and_not_committed() -> None:
    """The wipe alone is the dangerous commit: it publishes an empty article."""
    session = _Session()
    asyncio.run(ChunkRepository(session).delete_by_article_id(uuid.uuid4()))

    assert session.events == ["delete parent_chunks", "delete article_chunks"], session.events
    assert "commit" not in session.events


def test_writing_chunks_flushes_for_ids_rather_than_committing() -> None:
    """The parent's id is needed by its children; a flush produces it without ending
    the transaction the rebuild depends on."""
    from src.models.chunk import ParentChunk

    session = _Session()
    repo = ChunkRepository(session)
    asyncio.run(repo.create_parent_chunk(ParentChunk(article_id=uuid.uuid4(), text="x")))

    assert session.events == ["add", "flush"]


def test_a_standalone_chunk_deletion_still_commits() -> None:
    """Staging without a commit would make plain deletion a no-op, which is the failure
    mode moving the boundary invites."""
    source = inspect.getsource(indexing.delete_article_chunks)
    delete_at = source.index("delete_by_article_id")
    assert "await db.commit()" in source[delete_at:]


def test_an_embedding_failure_rolls_the_wipe_back_before_recording_the_dlq() -> None:
    """The DLQ write commits. Without the rollback ahead of it, that commit would publish
    the half-built chunk set it is reporting as broken."""
    source = inspect.getsource(indexing._index_article)
    failure_at = source.index("if embedding_failures:")
    tail = source[failure_at:]

    assert tail.index("await db.rollback()") < tail.index("DeadLetterJob")


def test_the_success_path_has_exactly_one_commit() -> None:
    """One commit for the whole rebuild: the wipe, every chunk, and index_status='ready'
    land together or not at all."""
    source = inspect.getsource(indexing._index_article)
    # The only other commit in the function belongs to the failure branch, and it commits a
    # DLQ row on a session the rollback above it already emptied. Everything the rebuild
    # itself writes is bounded below.
    build_phase = source[: source.index("if embedding_failures:")]
    ready_at = source.index('article.index_status = "ready"')

    assert build_phase.count("await db.commit()") == 0
    assert source[ready_at:].count("await db.commit()") == 1
