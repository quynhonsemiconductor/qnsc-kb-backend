"""End-to-end MLQA / UIT-ViQuAD 2.0 evaluation: retrieval + extractive reading.

The measured path is: question -> SearchService.search (normalization, hybrid
dense+FTS SQL, RRF, lexical reranker, relevance floor) -> parent-context
selection -> extractive reader -> SQuAD-style EM/F1.

NO GOLD CONTEXT IS EVER INJECTED. The reader only sees passages the retriever
actually returned, which is the whole point: an answer score here is a statement
about the system, not about the reader.

Context assembly mirrors src/domain/ai_service.py `_select_context`: child hits
are collapsed to their parents, ordered by score, capped by
RAG_MAX_CONTEXT_PARENTS, and each parent trimmed to RAG_PARENT_CONTEXT_CHARS.
Reproducing that here (rather than calling `AIService.ask`) is deliberate --
`ask` also builds prompts, calls a hosted LLM, writes usage logs and caches
answers, none of which belongs in a CPU-only extractive measurement. The
selection parameters come from the same settings the product uses, so a knob
change moves both.

Per-question rows are written for failure analysis, carrying enough state to
classify a failure without re-running: retrieval ranks, whether the gold document
was retrieved, whether the answer string was present in the context, the reader's
span, its null-relative score, and the abstention decision.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import statistics
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from src.core.config import settings
from src.domain.search_service import SearchService
from src.models.article import Article
from src.models.user import User
from src.rag.compressor import compress_context
from src.rag.squad_metrics import aggregate, score_answer
from src.repositories.chunk import ChunkRepository
from src.repositories.governance import GovernanceRepository

EVAL_COMPANY = "eval.local"


async def _eval_user(session) -> User:
    """A persisted global-read user with relationships eagerly loaded.

    Persisted because SearchService writes a search_logs row per query (FK to
    users); eager because the permission helpers read `groups`/`roles`
    synchronously and lazy loading raises MissingGreenlet under asyncio.
    """
    query = (
        select(User)
        .where(User.email == "eval@eval.local")
        .options(
            selectinload(User.groups),
            selectinload(User.roles),
            selectinload(User.departments),
            selectinload(User.department_ownerships),
        )
    )
    user = (await session.execute(query)).scalars().first()
    if user is None:
        session.add(
            User(
                id=uuid.uuid4(),
                email="eval@eval.local",
                name="Evaluation Harness",
                password_hash="unused-by-retrieval",
                role="Admin",
                dept="eval",
                company_domain=EVAL_COMPANY,
                active=True,
            )
        )
        await session.commit()
        user = (await session.execute(query)).scalars().first()
    return user


def _select_context(results: list[dict]) -> list[dict]:
    """Collapse child hits to parents under the product's context budgets."""
    by_parent: dict[str, dict] = {}
    for result in results:
        key = result.get("parent_chunk_id") or result.get("chunk_id")
        existing = by_parent.get(key)
        if existing is None or (result.get("score") or 0) > (existing.get("score") or 0):
            by_parent[key] = result
    ordered = sorted(by_parent.values(), key=lambda row: row.get("score") or 0, reverse=True)

    selected: list[dict] = []
    total_chars = 0
    per_article: dict[str, int] = {}
    for result in ordered:
        if len(selected) >= settings.RAG_MAX_CONTEXT_PARENTS:
            break
        article_id = result.get("article_id") or ""
        if per_article.get(article_id, 0) >= settings.RAG_MAX_PARENTS_PER_ARTICLE:
            continue
        passage = compress_context(
            result.get("parent_text") or result.get("chunk_text") or "",
            max_characters=settings.RAG_PARENT_CONTEXT_CHARS,
        )
        if not passage.strip():
            continue
        if total_chars + len(passage) > settings.RAG_CONTEXT_MAX_CHARS:
            continue
        total_chars += len(passage)
        per_article[article_id] = per_article.get(article_id, 0) + 1
        selected.append({**result, "passage": passage})
    return selected


def _load(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(records):
        return records
    return random.Random(seed).sample(records, limit)


async def evaluate(
    records: list[dict],
    limit: int,
    progress_every: int,
    apply_confidence_gate: bool,
) -> tuple[dict, list[dict]]:
    from src.lib.reader import resolve_reader

    reader = resolve_reader()
    reader.warm_up()

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    rows: list[dict] = []
    retrieval_latency: list[float] = []
    reader_latency: list[float] = []
    started = time.time()

    async with factory() as session:
        # A real GovernanceRepository, not None: SearchService records a "gap" row
        # whenever a search returns nothing, and `gov_repo` is a required
        # collaborator. Passing None made every zero-result question raise twice
        # inside _record_gap and log two tracebacks per question.
        service = SearchService(ChunkRepository(session), GovernanceRepository(session), None)
        user = await _eval_user(session)
        mapping = await session.execute(
            select(Article.external_id, Article.id).where(
                Article.company_domain == EVAL_COMPANY
            )
        )
        article_by_doc = {external: str(ident) for external, ident in mapping.all()}

        for index, record in enumerate(records, start=1):
            question = record["question"]
            answers = record.get("answers") or []
            lang = record.get("lang_q") or "vi"
            gold_article = article_by_doc.get(record["doc_id"])

            search_started = time.perf_counter()
            try:
                results = await service.search(user, question, limit=limit)
                search_error = None
            except Exception as exc:  # noqa: BLE001 - recorded per question
                results, search_error = [], f"{type(exc).__name__}: {exc}"
            retrieval_latency.append(time.perf_counter() - search_started)

            context = _select_context(results)
            passages = [item["passage"] for item in context]
            top_score = results[0]["score"] if results else None

            # The product refuses before generating when the top lexical score is
            # below RAG_MIN_CONTEXT_SCORE. Measuring with the gate ON shows what a
            # user gets; with it OFF shows what the reader could have done. Both
            # are recorded so the gate's cost is attributable.
            gated_out = bool(
                apply_confidence_gate
                and (top_score is None or top_score < settings.RAG_MIN_CONTEXT_SCORE)
            )

            reader_started = time.perf_counter()
            if gated_out or not passages:
                prediction, span_score, null_score, passage_index = "", 0.0, 0.0, None
            else:
                span = reader.read(question, passages)
                threshold = settings.READER_NULL_THRESHOLD
                prediction = span.text if span.score > threshold else ""
                span_score = span.score
                null_score = span.null_score
                passage_index = span.passage_index if span.text else None
            reader_latency.append(time.perf_counter() - reader_started)

            exact, f1 = score_answer(prediction, answers, lang)

            retrieved_articles = [item.get("article_id") for item in results]
            first_hit = next(
                (
                    position
                    for position, article in enumerate(retrieved_articles)
                    if article == gold_article
                ),
                None,
            )
            context_text = " ".join(passages).lower()
            answer_in_context = any(answer.lower() in context_text for answer in answers)

            rows.append(
                {
                    "qid": record["qid"],
                    "dataset": record["dataset"],
                    "split": record["split"],
                    "lang_q": lang,
                    "lang_c": record.get("lang_c"),
                    "question": question,
                    "answers": answers,
                    "is_impossible": record.get("is_impossible", False),
                    "prediction": prediction,
                    "exact_match": exact,
                    "f1": f1,
                    "gold_doc": record["doc_id"],
                    "gold_indexed": gold_article is not None,
                    "gold_retrieved": first_hit is not None,
                    "first_hit_rank": first_hit,
                    "result_count": len(results),
                    "context_parents": len(context),
                    "answer_in_context": answer_in_context,
                    "top_score": top_score,
                    "gated_out": gated_out,
                    "span_score": span_score,
                    "null_score": null_score,
                    "answer_passage_index": passage_index,
                    "search_error": search_error,
                }
            )

            if progress_every and index % progress_every == 0:
                elapsed = time.time() - started
                current = aggregate([(row["exact_match"], row["f1"]) for row in rows])
                print(
                    f"  {index}/{len(records)} q, {index/elapsed:.2f} q/s, "
                    f"EM {current['exact_match']:.1f} F1 {current['f1']:.1f}",
                    flush=True,
                )

    await engine.dispose()
    return _summarize(rows, retrieval_latency, reader_latency), rows


def _summarize(
    rows: list[dict], retrieval_latency: list[float], reader_latency: list[float]
) -> dict:
    scores = [(row["exact_match"], row["f1"]) for row in rows]
    summary = aggregate(scores)

    answerable = [row for row in rows if not row["is_impossible"]]
    unanswerable = [row for row in rows if row["is_impossible"]]
    summary["answerable"] = aggregate(
        [(row["exact_match"], row["f1"]) for row in answerable]
    )
    summary["unanswerable"] = aggregate(
        [(row["exact_match"], row["f1"]) for row in unanswerable]
    )

    scored = [row for row in answerable if row["gold_indexed"]]
    summary["retrieval"] = {
        "gold_retrieved_rate": _rate(scored, lambda row: row["gold_retrieved"]),
        "recall@1": _rate(scored, lambda row: row["first_hit_rank"] == 0),
        "answer_in_context_rate": _rate(scored, lambda row: row["answer_in_context"]),
        "zero_result_rate": _rate(rows, lambda row: row["result_count"] == 0),
        "gated_out_rate": _rate(rows, lambda row: row["gated_out"]),
        "abstention_rate": _rate(rows, lambda row: not row["prediction"]),
    }
    # The ceiling the reader was actually given: when the answer string never
    # reaches the context, no reader can score. This is the retrieval-imposed cap.
    summary["reader_ceiling_f1"] = summary["retrieval"]["answer_in_context_rate"]
    summary["errors"] = sum(1 for row in rows if row["search_error"])

    for label, samples in (("retrieval", retrieval_latency), ("reader", reader_latency)):
        if samples:
            ordered = sorted(samples)
            summary[f"{label}_latency_p50_ms"] = 1000 * statistics.median(ordered)
            summary[f"{label}_latency_p95_ms"] = (
                1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
            )
            summary[f"{label}_latency_mean_ms"] = 1000 * statistics.fmean(ordered)
    return summary


def _rate(rows: list[dict], predicate) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(1 for row in rows if predicate(row)) / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=settings.RAG_RERANK_LIMIT)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--lang-q", default=None)
    parser.add_argument("--lang-c", default=None)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--no-confidence-gate",
        action="store_true",
        help="skip the RAG_MIN_CONTEXT_SCORE pre-generation refusal",
    )
    parser.add_argument("--tag", default="baseline", help="label recorded with the run")
    args = parser.parse_args()

    records = _load(args.input)
    if args.lang_q:
        records = [row for row in records if row["lang_q"] == args.lang_q]
    if args.lang_c:
        records = [row for row in records if row["lang_c"] == args.lang_c]
    records = _sample(records, args.sample, args.seed)
    print(f"evaluating {len(records)} questions from {args.input.name}")

    summary, rows = asyncio.run(
        evaluate(records, args.limit, args.progress_every, not args.no_confidence_gate)
    )
    summary["tag"] = args.tag
    summary["confidence_gate"] = not args.no_confidence_gate
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote per-question detail to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
