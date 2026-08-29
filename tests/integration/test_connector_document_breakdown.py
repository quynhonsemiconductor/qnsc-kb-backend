"""A source has to be able to say where its documents got to.

A sync reported "241 checked, 70 imported" and the Articles list was empty. Nothing was
wrong -- connector documents become PendingDrafts under governed publication and only
become Articles once somebody approves them -- but nothing said so either. The sync
summary describes what one RUN did and is then over; it cannot answer "how much of this
drive is live and how much is waiting for a reviewer", which is the question an
administrator actually has.

This is tested against a real database on purpose. The whole thing is SQL: joins,
grouping, and which rows are excluded. A test with a faked session would assert the
shape of the code rather than the truth of the numbers, and every mistake worth catching
here -- counting a tombstone, joining the wrong way, missing a status -- would survive
it.

Skips without PostgreSQL, matching test_tenant_context_survives_commit.py; CI applies
migrations before pytest, so it runs there.
"""
from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.domain.connectors import document_breakdown
from src.models.article import Article
from src.models.connectors import ExternalDocument
from src.models.governance import PendingDraft
from src.models.ops import Connector

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="counts a real join across four tables; a fake session would prove nothing",
)

DOMAIN = "breakdown.test"


@pytest_asyncio.fixture
async def session():
    """This test's own engine, disposed with it -- see the note in
    test_tenant_context_survives_commit.py about SessionLocal and event loops."""
    engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=None)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db:
        # Provider callbacks and background work use internal context; RLS would
        # otherwise hide the rows this test just inserted.
        from src.api.deps import set_database_context

        await set_database_context(db, None, True)
        yield db
    await engine.dispose()


async def _document(db, connector, *, state="active", held=False):
    document = ExternalDocument(
        connector_id=connector.id,
        corpus_id="corpus-1",
        external_id=str(uuid.uuid4()),
        name="doc.pdf",
        state=state,
        metadata_json={"ingest_failure": {"attempts": 3}} if held else {},
    )
    db.add(document)
    await db.flush()
    return document


async def _draft(db, document, status):
    db.add(
        PendingDraft(
            title="A draft",
            source_ref=f"test://{document.external_id}",
            source_hash=str(uuid.uuid4()),
            company_domain=DOMAIN,
            status=status,
            external_document_id=document.id,
        )
    )
    await db.flush()


async def _published_article(db, document):
    article = Article(
        title="Published",
        body_md="body",
        dept="Engineering",
        domain="general",
        type="reference",
        company_domain=DOMAIN,
        status="published",
        lifecycle_status="active",
    )
    db.add(article)
    await db.flush()
    document.article_id = article.id
    await db.flush()
    return article


@pytest_asyncio.fixture
async def connector(session):
    row = Connector(name="Breakdown source", system="sharepoint", company_domain=DOMAIN)
    session.add(row)
    await session.flush()
    yield row
    await session.rollback()


@pytest.mark.asyncio
async def test_an_untouched_source_counts_nothing(session, connector):
    assert await document_breakdown(session, connector.id) == {
        "total": 0,
        "published": 0,
        "pending_review": 0,
        "draft": 0,
        "rejected": 0,
        "approved": 0,
        "held": 0,
    }


@pytest.mark.asyncio
async def test_each_draft_status_is_counted_separately(session, connector):
    for status in ("pending", "pending", "draft", "rejected", "approved"):
        await _draft(session, await _document(session, connector), status)

    counts = await document_breakdown(session, connector.id)

    assert counts["pending_review"] == 2
    assert counts["draft"] == 1
    assert counts["rejected"] == 1
    assert counts["approved"] == 1
    assert counts["total"] == 5


@pytest.mark.asyncio
async def test_a_published_document_is_counted_as_published(session, connector):
    await _published_article(session, await _document(session, connector))

    counts = await document_breakdown(session, connector.id)

    assert counts["published"] == 1
    assert counts["total"] == 1


@pytest.mark.asyncio
async def test_an_unapproved_article_is_not_counted_as_published(session, connector):
    """An Article row exists well before it is published; joining on article_id alone
    would report drafts as live."""
    document = await _document(session, connector)
    article = await _published_article(session, document)
    article.status = "draft"
    await session.flush()

    assert (await document_breakdown(session, connector.id))["published"] == 0


@pytest.mark.asyncio
async def test_an_archived_article_is_not_counted_as_published(session, connector):
    document = await _document(session, connector)
    article = await _published_article(session, document)
    article.lifecycle_status = "archived"
    await session.flush()

    assert (await document_breakdown(session, connector.id))["published"] == 0


@pytest.mark.asyncio
async def test_a_deleted_document_is_not_part_of_the_corpus(session, connector):
    """A tombstone would make a drive look fuller than it is, and would keep counting a
    draft for something the provider no longer has."""
    deleted = await _document(session, connector, state="deleted")
    await _draft(session, deleted, "pending")
    await _document(session, connector)

    counts = await document_breakdown(session, connector.id)

    assert counts["total"] == 1
    assert counts["pending_review"] == 0


@pytest.mark.asyncio
async def test_held_documents_are_counted(session, connector):
    await _document(session, connector, held=True)
    await _document(session, connector)

    assert (await document_breakdown(session, connector.id))["held"] == 1


@pytest.mark.asyncio
async def test_another_source_is_not_counted(session, connector):
    """The whole point is per-source figures; a company-wide count would be useless."""
    other = Connector(name="Other", system="google_drive", company_domain=DOMAIN)
    session.add(other)
    await session.flush()
    await _draft(session, await _document(session, other), "pending")
    await _document(session, connector)

    counts = await document_breakdown(session, connector.id)

    assert counts["total"] == 1
    assert counts["pending_review"] == 0
