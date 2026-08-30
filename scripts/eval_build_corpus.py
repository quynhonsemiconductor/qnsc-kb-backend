"""Index benchmark contexts into the eval database as real articles and chunks.

Each unique benchmark context becomes one published Article whose ``body_md`` is
that context. Chunking, embedding, and persistence then go through the SAME code
the product uses -- ``create_parent_child_chunks`` and the configured embedding
backend -- so a retrieval number measured afterwards describes the real pipeline
rather than a reimplementation of it.

WHY NOT the full ``index_article`` domain function: it needs the outbox/event
machinery, a Celery or inline job runner, and a governance actor. This writes the
same rows that function writes, calling the same chunker and embedder, minus the
orchestration -- which is why the chunk/parent row shapes below are kept
deliberately identical to src/domain/indexing.py:213-264.

Deduplication is by ``doc_id`` (a hash of the context), so questions sharing a
paragraph share a document, and retrieval has to discriminate between thousands
of near-neighbour Wikipedia paragraphs. That is the point: a corpus of one
document per question would make Recall@k meaningless.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import settings
from src.lib.embeddings import get_bge_embeddings
from src.models.article import Article
from src.models.chunk import ArticleChunk, ParentChunk
from src.models.user import Department
from src.rag.chunker import create_parent_child_chunks

# One department, one bitmap, one visibility for every eval article: the
# permission surface is not what is being measured here, and a uniform ACL keeps
# hybrid_search's authorization predicates satisfied without modelling an org.
EVAL_DEPT = "eval"
EVAL_COMPANY = "eval.local"
EVAL_BITMAP = 1


def _load_records(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique_documents(records: list[dict]) -> dict[str, dict]:
    documents: dict[str, dict] = {}
    for record in records:
        doc_id = record["doc_id"]
        if doc_id not in documents:
            documents[doc_id] = {
                "doc_id": doc_id,
                "title": record.get("title") or doc_id,
                "context": record["context"],
                "lang": record.get("lang_c") or "vi",
            }
    return documents


async def _ensure_department(session: AsyncSession) -> None:
    existing = await session.execute(
        select(Department).where(
            Department.name == EVAL_DEPT, Department.company_domain == EVAL_COMPANY
        )
    )
    if existing.scalars().first():
        return
    session.add(
        Department(
            name=EVAL_DEPT,
            company_domain=EVAL_COMPANY,
            kind="org",
            description="Benchmark corpus for MLQA / UIT-ViQuAD evaluation",
            active=True,
        )
    )
    await session.commit()

async def _purge(session: AsyncSession) -> int:
    """Drop every eval article (chunks cascade) so a rebuild is reproducible."""
    result = await session.execute(
        select(Article.id).where(Article.company_domain == EVAL_COMPANY)
    )
    ids = [row[0] for row in result.all()]
    if ids:
        await session.execute(delete(ArticleChunk).where(ArticleChunk.article_id.in_(ids)))
        await session.execute(delete(ParentChunk).where(ParentChunk.article_id.in_(ids)))
        await session.execute(delete(Article).where(Article.id.in_(ids)))
        await session.commit()
    return len(ids)


async def index_documents(
    session: AsyncSession, documents: dict[str, dict], batch_size: int
) -> tuple[int, int]:
    """Write Article + ParentChunk + ArticleChunk rows for every document."""
    article_count = 0
    chunk_count = 0
    started = time.time()
    pending: list[tuple[ArticleChunk, str]] = []

    async def flush(rows: list[tuple[ArticleChunk, str]]) -> int:
        if not rows:
            return 0
        vectors = get_bge_embeddings([text_value for _chunk, text_value in rows])
        if vectors is None or len(vectors) != len(rows):
            raise RuntimeError(
                f"embedding backend returned {0 if vectors is None else len(vectors)} "
                f"vectors for {len(rows)} chunks"
            )
        for (chunk, _text_value), vector in zip(rows, vectors):
            chunk.embedding = vector
            session.add(chunk)
        await session.commit()
        return len(rows)

    for index, document in enumerate(documents.values(), start=1):
        article = Article(
            id=uuid.uuid4(),
            title=(document["title"] or document["doc_id"])[:255],
            body_md=document["context"],
            external_id=document["doc_id"],
            dept=EVAL_DEPT,
            domain=EVAL_DEPT,
            company_domain=EVAL_COMPANY,
            type="REFERENCE",
            sensitivity="public",
            visibility="public",
            language=document["lang"],
            status="published",
            lifecycle_status="active",
            index_status="ready",
        )
        session.add(article)
        await session.flush()
        article_count += 1

        for parent_spec in create_parent_child_chunks(
            document["context"], heading=document["title"] or None
        ):
            parent = ParentChunk(
                id=uuid.uuid4(),
                article_id=article.id,
                text=str(parent_spec["parent_text"]),
                section_ref=document["doc_id"],
                chunk_type=str(parent_spec["chunk_type"]),
                heading=(parent_spec.get("heading") or None),
            )
            session.add(parent)
            await session.flush()
            for child_index, child_text in enumerate(parent_spec["children"]):
                clean = str(child_text).strip()
                if not clean:
                    continue
                pending.append(
                    (
                        ArticleChunk(
                            article_id=article.id,
                            parent_chunk_id=parent.id,
                            chunk_text=clean,
                            embedding_model=settings.EMBEDDING_MODEL,
                            embedding_version=settings.EMBEDDING_VERSION,
                            chunk_type=str(parent_spec["chunk_type"]),
                            heading=(parent_spec.get("heading") or None),
                            chunking_version=settings.CHUNKING_VERSION,
                            access_group_bitmap=EVAL_BITMAP,
                            department_id=EVAL_DEPT,
                            sensitivity="public",
                            visibility="public",
                            chunk_index=child_index,
                        ),
                        clean,
                    )
                )

        if len(pending) >= batch_size:
            chunk_count += await flush(pending)
            pending = []

        if index % 200 == 0:
            elapsed = time.time() - started
            rate = index / elapsed if elapsed else 0.0
            print(
                f"  {index}/{len(documents)} docs, {chunk_count} chunks, "
                f"{rate:.1f} docs/s",
                flush=True,
            )

    chunk_count += await flush(pending)
    return article_count, chunk_count


async def main_async(args: argparse.Namespace) -> int:
    records: list[dict] = []
    for path in args.inputs:
        found = _load_records(path)
        records.extend(found)
        print(f"loaded {len(found)} questions from {path.name}")

    documents = _unique_documents(records)
    print(f"{len(documents)} unique contexts to index")

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await _ensure_department(session)
        if args.purge:
            removed = await _purge(session)
            print(f"purged {removed} existing eval articles")
        started = time.time()
        articles, chunks = await index_documents(session, documents, args.batch_size)
        elapsed = time.time() - started
        total = await session.execute(
            text(
                "select count(*) from article_chunks c join articles a on a.id=c.article_id "
                "where a.company_domain=:domain"
            ),
            {"domain": EVAL_COMPANY},
        )
        stored = total.scalar_one()
    await engine.dispose()

    print(
        f"indexed {articles} articles, {chunks} chunks in {elapsed:.0f}s "
        f"({chunks/elapsed if elapsed else 0:.1f} chunks/s); {stored} chunks in database"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--purge", action="store_true", help="delete existing eval articles first")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
