import uuid
import re

import structlog

from src.api.deps import SessionLocal, set_database_context
from src.core.config import settings
from src.domain.search_service import get_text_embeddings
from src.domain.text_noise import detect_boilerplate, strip_noise
from src.models.chunk import ArticleChunk, ParentChunk
from src.models.ops import DeadLetterJob
from src.models.article import DocumentSource
from sqlalchemy import select
from src.repositories.article import ArticleRepository
from src.repositories.chunk import ChunkRepository
from src.rag.chunker import create_parent_child_chunks
from src.lib.locking import article_lock

logger = structlog.get_logger()


def _normalized_tokens(value: str) -> list[str]:
    return re.findall(r"[\w'-]+", (value or "").lower())


def _prepare_source_pages(
    source_pages: list[tuple[int, str]] | None,
) -> list[tuple[int, set[str], str]]:
    """Tokenise each page ONCE per document, not once per chunk.

    _match_source_page is called for every parent and every child chunk, and it used to
    re-tokenise, re-set and re-join every page on each of those calls — O(chunks x pages)
    passes over the whole extracted document. Measured at ~104 ms per call across 120
    pages, which is ~80 seconds of pure re-tokenisation for an 800-chunk file, on the
    worker's own thread. Pages do not change while a document is being indexed, so the
    work belongs here.
    """
    prepared: list[tuple[int, set[str], str]] = []
    for page_number, page_text in source_pages or []:
        page_tokens = _normalized_tokens(page_text)
        if not page_tokens:
            continue
        prepared.append((page_number, set(page_tokens), " ".join(page_tokens)))
    return prepared


def _match_source_page(
    text: str, prepared_pages: list[tuple[int, set[str], str]]
) -> int | None:
    """Map structured article text back to the original extracted page."""
    query_tokens = _normalized_tokens(text)
    if not query_tokens or not prepared_pages:
        return None
    query_set = set(query_tokens)
    # Neither of these depends on the page, so they are computed before the loop.
    denominator = max(min(len(query_set), 80), 1)
    # Longer exact runs are more reliable than common-token overlap.
    query_preview = " ".join(query_tokens[:24])
    compare_exact = len(query_preview) >= 24
    best_page: int | None = None
    best_score = 0.0
    for page_number, page_set, page_normalized in prepared_pages:
        overlap = len(query_set & page_set) / denominator
        exact_bonus = 0.75 if compare_exact and query_preview in page_normalized else 0.0
        score = overlap + exact_bonus
        if score > best_score:
            best_score = score
            best_page = page_number
    return best_page if best_score >= 0.12 else None


def _section_ref_and_text(
    section_idx: int, section_heading: str, section_body: str, page_number: int | None
) -> tuple[str, str]:
    """Derive a section's citation label and the text that will be indexed.

    Extracted so noise removal can be decided for the WHOLE document before the first
    chunk is written — the emptiness guard has to see every section at once.
    """
    lines = section_body.strip().split("\n")
    first_line = lines[0].strip() if lines else ""
    if page_number is not None:
        return (section_heading[:255] or f"Page {page_number}", section_body.strip())
    if first_line.startswith("#"):
        return (
            first_line.lstrip("#").strip()[:255] or f"Section {section_idx + 1}",
            "\n".join(lines[1:]),
        )
    return (f"Section {section_idx + 1}", section_body.strip())


def _child_chunk_metadata(parent_spec: dict, section_heading: str) -> tuple[str, str | None]:
    """Capture parent metadata while the parent spec is still in scope."""
    chunk_type = str(parent_spec.get("chunk_type") or "section")
    heading = str(parent_spec.get("heading") or section_heading or "")[:255] or None
    return chunk_type, heading


async def set_index_status(article_id: uuid.UUID, status: str, error: str | None = None) -> None:
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        article_repo = ArticleRepository(db)
        article = await article_repo.get_by_id(article_id)
        if article:
            article.index_status = status
            article.index_error = error
            await article_repo.update(article)


async def index_article(article_id: uuid.UUID) -> None:
    """Index an Article and persist a terminal failure state on exceptions."""
    await set_index_status(article_id, "processing")
    try:
        indexed = await _index_article(article_id)
        if not indexed:
            await set_index_status(
                article_id,
                "pending",
                "Article is not active and published; indexing was skipped",
            )
    except Exception as exc:
        logger.exception("Article indexing failed", article_id=str(article_id))
        try:
            await set_index_status(article_id, "failed", str(exc)[:2000])
        except Exception:
            # Preserve the original indexing exception if the status update
            # cannot be persisted, while leaving a diagnostic trail.
            logger.exception(
                "Unable to persist failed Article index status",
                article_id=str(article_id),
            )
        raise


async def _index_article(article_id: uuid.UUID) -> bool:
    """Create searchable chunks in-process while Celery is disabled."""
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        async with article_lock(db, str(article_id)):
            article_repo = ArticleRepository(db)
            chunk_repo = ChunkRepository(db)
            article = await article_repo.get_by_id(article_id)

            if not article or article.status != "published" or article.lifecycle_status != "active":
                logger.warning(
                    "Skipping article indexing",
                    article_id=str(article_id),
                    reason="article missing or not published",
                    status=article.status if article else None,
                )
                return False

            await chunk_repo.delete_by_article_id(article_id)
            source_result = await db.execute(
                select(DocumentSource)
                .where(DocumentSource.article_id == article_id)
                .order_by(DocumentSource.ingested_at.desc())
            )
            source = source_result.scalars().first()
            source_pages = []
            if source and source.page_texts:
                source_pages = [
                    (int(item.get("page_number", index)), str(item.get("text", "")))
                    for index, item in enumerate(source.page_texts, start=1)
                    if item.get("text")
                ]
            # Tokenised once here, then reused by every _match_source_page call below.
            prepared_pages = _prepare_source_pages(source_pages)
            # Uploaded sources keep their original page text on DocumentSource
            # for audit/PDF review, while the article body is the approved,
            # losslessly restructured reading representation used for indexing.
            if article.body_md and article.body_md.strip():
                sections = [
                    (section.split("\n", 1)[0].strip() if section else "", section, None)
                    for section in article.body_md.split("\n## ")
                ]
            elif source and source.page_texts:
                sections = [
                    (f"Page {item.get('page_number', index)}", str(item.get("text", "")), item.get("page_number", index))
                    for index, item in enumerate(source.page_texts, start=1)
                    if item.get("text")
                ]
            else:
                sections = [(section.split("\n", 1)[0].strip() if section else "", section, None) for section in article.body_md.split("\n## ")]
            chunk_count = 0
            embedding_failures: list[dict[str, object]] = []

            indexable = [
                _section_ref_and_text(section_idx, section_heading, section_body, page_number)
                for section_idx, (section_heading, section_body, page_number) in enumerate(sections)
            ]
            # Retrieval-only noise removal. The stored page text and the article body are
            # left untouched; this is the projection of them that becomes chunks and
            # vectors, and running headers in it poison every chunk of the document.
            boilerplate = detect_boilerplate(source_pages)
            if boilerplate:
                denoised = [
                    (section_ref, strip_noise(section_text, boilerplate))
                    for section_ref, section_text in indexable
                ]
                if any(section_text.strip() for _section_ref, section_text in denoised):
                    indexable = denoised
                else:
                    # An article with no chunks is silently unfindable: index_status still
                    # becomes "ready". Keeping the noise beats losing the document.
                    logger.warning(
                        "Noise removal would have emptied the document; indexing raw text",
                        article_id=str(article_id),
                        boilerplate_lines=len(boilerplate),
                    )

            for (section_ref, section_text), (section_heading, _section_body, page_number) in zip(
                indexable, sections
            ):
                child_chunks = []
                pending_children: list[tuple[str, int | None, uuid.UUID, str, str | None]] = []
                for parent_spec in create_parent_child_chunks(section_text, heading=section_heading):
                    parent_text = str(parent_spec["parent_text"])
                    parent_page_number = page_number or _match_source_page(parent_text, prepared_pages)
                    parent_chunk_type, parent_heading = _child_chunk_metadata(parent_spec, section_heading)
                    parent = await chunk_repo.create_parent_chunk(
                        ParentChunk(
                            article_id=article_id,
                            text=parent_text,
                            section_ref=section_ref,
                            chunk_type=parent_chunk_type,
                            heading=parent_heading,
                            page_number=parent_page_number,
                        )
                    )
                    for child_text in parent_spec["children"]:
                        clean_text = str(child_text).strip()
                        child_page_number = page_number or _match_source_page(clean_text, prepared_pages) or parent_page_number
                        pending_children.append((clean_text, child_page_number, parent.id, parent_chunk_type, parent_heading))

                embeddings = await get_text_embeddings([item[0] for item in pending_children])
                if embeddings is None or len(embeddings) != len(pending_children):
                    embedding_failures.extend({"section": section_ref, "chunk_index": index} for index in range(len(pending_children)))
                else:
                    for child_index, ((clean_text, child_page_number, parent_id, chunk_type, heading), embedding) in enumerate(zip(pending_children, embeddings)):
                        child_chunks.append(
                            ArticleChunk(
                                article_id=article_id,
                                parent_chunk_id=parent_id,
                                chunk_text=clean_text,
                                embedding=embedding,
                                embedding_model=settings.EMBEDDING_MODEL,
                                embedding_version=settings.EMBEDDING_VERSION,
                                chunk_type=chunk_type,
                                heading=heading,
                                chunking_version=settings.CHUNKING_VERSION,
                                department_id=article.dept,
                                sensitivity=article.sensitivity,
                                visibility=article.visibility,
                                chunk_index=child_index,
                                page_number=child_page_number,
                            )
                        )

                if child_chunks:
                    await chunk_repo.create_child_chunks(child_chunks)
                    chunk_count += len(child_chunks)

            logger.info("Article indexing completed", article_id=str(article_id), chunk_count=chunk_count)
            if embedding_failures:
                # Discard the rebuild BEFORE recording the failure. The wipe and every chunk
                # written above are still uncommitted, so this restores the previous chunk
                # set wholesale: the article keeps serving its old passages instead of
                # becoming a published document with a partial index. Committing the DLQ row
                # first would have committed those partial chunks with it.
                await db.rollback()
                db.add(DeadLetterJob(
                    source_queue="embedding",
                    payload={"article_id": str(article_id), "failures": embedding_failures},
                    error=f"Embedding unavailable for {len(embedding_failures)} chunk(s)",
                ))
                await db.commit()
                logger.warning("Embedding failures recorded in DLQ", article_id=str(article_id), failure_count=len(embedding_failures))
                raise RuntimeError(f"Embedding unavailable for {len(embedding_failures)} chunk(s)")
            article.index_status = "ready"
            article.index_error = None
            await db.commit()
            return True


async def recompute_article_permissions(article_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        async with article_lock(db, str(article_id)):
            article_repo = ArticleRepository(db)
            chunk_repo = ChunkRepository(db)
            article = await article_repo.get_by_id(article_id)
            if not article:
                logger.warning("Skipping permission refresh; article not found", article_id=str(article_id))
                return
            await chunk_repo.update_permissions(
                article_id=article_id,
                sensitivity=article.sensitivity,
                visibility=article.visibility,
                dept=article.dept,
            )
            logger.info("Article permissions refreshed", article_id=str(article_id))


async def delete_article_chunks(article_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        # The repository stages the delete and leaves the commit to whoever owns the
        # transaction, because a reindex has to wipe and rebuild atomically. Here the wipe
        # IS the whole operation, so this is the owner.
        await ChunkRepository(db).delete_by_article_id(article_id)
        await db.commit()
        logger.info("Article chunks deleted", article_id=str(article_id))
