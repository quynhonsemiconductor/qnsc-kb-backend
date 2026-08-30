"""Measure retrieval quality on MLQA / UIT-ViQuAD 2.0 through the real pipeline.

Every question goes through ``SearchService.search`` -- the same entry point
``AIService.ask`` uses -- so normalization, the query-embedding cache, hybrid
SQL (dense HNSW + Postgres FTS), RRF fusion, the lexical reranker, and the
relevance floor are all exercised. Nothing here reimplements retrieval.

WHAT COUNTS AS A HIT. A benchmark question is answerable from one specific
context paragraph, which this corpus stored as one Article whose ``external_id``
is the ``doc_id``. A retrieved chunk is relevant when its article's external_id
equals the question's doc_id. That is document-level truth and it is the honest
ceiling on this pipeline: if the right document never arrives, no reader can
answer.

A second, stricter signal is also recorded: whether the retrieved chunk text
actually CONTAINS a gold answer string. A document hit whose chunk lost the
answer span is the "correct document but wrong chunk" failure mode, and telling
those apart is the whole point of measuring both.

Metrics: Recall@1/5/10, MRR@10, nDCG@10, plus answer-bearing recall at the same
depths. Unanswerable questions (ViQuAD 2.0) are excluded from retrieval scoring
-- there is no gold document to find -- but counted so the split is auditable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
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
from src.repositories.chunk import ChunkRepository
from src.repositories.governance import GovernanceRepository

EVAL_COMPANY = "eval.local"
DEPTHS = (1, 5, 10)


async def _eval_user(session) -> User:
    """A persisted global-read user with every relationship eagerly loaded.

    Persisted rather than transient because SearchService writes a `search_logs`
    row per query, and that table has an FK to `users`. A fake in-memory user
    makes the first insert fail and poisons the session for every question after
    it, which reads as a retrieval failure and is not one.

    Eagerly loaded because PermissionService reads `groups`/`roles` synchronously
    while building the access bitmask. Left lazy, the first attribute access
    raises MissingGreenlet under asyncio and every question records an error
    instead of a result.
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
    existing = await session.execute(query)
    user = existing.scalars().first()
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


def _load(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(records):
        return records
    return random.Random(seed).sample(records, limit)


def _ndcg(relevances: list[int], depth: int) -> float:
    gains = [
        relevance / math.log2(position + 2)
        for position, relevance in enumerate(relevances[:depth])
    ]
    dcg = sum(gains)
    # One relevant document per question, so the ideal ranking puts it first.
    ideal = 1.0
    return dcg / ideal if ideal else 0.0


async def evaluate(
    records: list[dict], limit: int, progress_every: int
) -> tuple[dict, list[dict]]:
    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    per_question: list[dict] = []
    latencies: list[float] = []
    started = time.time()

    async with factory() as session:
        # A real GovernanceRepository, not None: a zero-result search records a
        # gap row, and `gov_repo` is a required collaborator — None raises twice
        # inside _record_gap for every question that finds nothing.
        service = SearchService(ChunkRepository(session), GovernanceRepository(session), None)
        user = await _eval_user(session)
        # Map doc_id -> article id once; the harness needs it to decide hits.
        rows = await session.execute(
            select(Article.external_id, Article.id).where(
                Article.company_domain == EVAL_COMPANY
            )
        )
        article_by_doc = {external: str(identifier) for external, identifier in rows.all()}

        for index, record in enumerate(records, start=1):
            question = record["question"]
            gold_doc = record["doc_id"]
            gold_article = article_by_doc.get(gold_doc)
            answers = record.get("answers") or []

            call_started = time.perf_counter()
            try:
                results = await service.search(user, question, limit=limit)
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                results, error = [], f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - call_started
            latencies.append(latency)

            ranks = [
                1 if result.get("article_id") == gold_article else 0
                for result in results
            ]
            answer_ranks = []
            for result in results:
                haystack = " ".join(
                    str(result.get(field) or "")
                    for field in ("chunk_text", "parent_text")
                )
                lowered = haystack.lower()
                answer_ranks.append(
                    1 if any(answer.lower() in lowered for answer in answers) else 0
                )

            first_hit = next((position for position, hit in enumerate(ranks) if hit), None)
            first_answer = next(
                (position for position, hit in enumerate(answer_ranks) if hit), None
            )
            per_question.append(
                {
                    "qid": record["qid"],
                    "dataset": record["dataset"],
                    "split": record["split"],
                    "lang_q": record["lang_q"],
                    "lang_c": record["lang_c"],
                    "question": question,
                    "answers": answers,
                    "is_impossible": record.get("is_impossible", False),
                    "gold_doc": gold_doc,
                    "gold_article_id": gold_article,
                    "gold_indexed": gold_article is not None,
                    "result_count": len(results),
                    "first_hit_rank": first_hit,
                    "first_answer_rank": first_answer,
                    "top_score": results[0]["score"] if results else None,
                    "scores": [result.get("score") for result in results[:10]],
                    "retrieved_article_ids": [result.get("article_id") for result in results[:10]],
                    "retrieved_chunk_ids": [result.get("chunk_id") for result in results[:10]],
                    "latency_seconds": latency,
                    "error": error,
                }
            )

            if progress_every and index % progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"  {index}/{len(records)} questions, {index/elapsed:.1f} q/s",
                    flush=True,
                )

    await engine.dispose()
    return _summarize(per_question, latencies), per_question


def _summarize(per_question: list[dict], latencies: list[float]) -> dict:
    scored = [
        row for row in per_question if not row["is_impossible"] and row["gold_indexed"]
    ]
    summary: dict = {
        "questions_total": len(per_question),
        "questions_scored": len(scored),
        "questions_unanswerable": sum(1 for row in per_question if row["is_impossible"]),
        "questions_gold_missing": sum(1 for row in per_question if not row["gold_indexed"]),
        "errors": sum(1 for row in per_question if row["error"]),
        "zero_result_rate": (
            100.0 * sum(1 for row in per_question if row["result_count"] == 0) / len(per_question)
            if per_question
            else 0.0
        ),
    }
    for depth in DEPTHS:
        summary[f"recall@{depth}"] = (
            100.0
            * sum(
                1
                for row in scored
                if row["first_hit_rank"] is not None and row["first_hit_rank"] < depth
            )
            / len(scored)
            if scored
            else 0.0
        )
        summary[f"answer_recall@{depth}"] = (
            100.0
            * sum(
                1
                for row in scored
                if row["first_answer_rank"] is not None and row["first_answer_rank"] < depth
            )
            / len(scored)
            if scored
            else 0.0
        )
    summary["mrr@10"] = (
        100.0
        * sum(
            1.0 / (row["first_hit_rank"] + 1)
            for row in scored
            if row["first_hit_rank"] is not None and row["first_hit_rank"] < 10
        )
        / len(scored)
        if scored
        else 0.0
    )
    summary["ndcg@10"] = (
        100.0
        * sum(
            _ndcg(
                [
                    1 if position == row["first_hit_rank"] else 0
                    for position in range(10)
                ],
                10,
            )
            for row in scored
            if row["first_hit_rank"] is not None
        )
        / len(scored)
        if scored
        else 0.0
    )
    if latencies:
        ordered = sorted(latencies)
        summary["latency_p50_ms"] = 1000 * statistics.median(ordered)
        summary["latency_p95_ms"] = 1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        summary["latency_mean_ms"] = 1000 * statistics.fmean(ordered)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=settings.RAG_RERANK_LIMIT)
    parser.add_argument("--sample", type=int, default=None, help="evaluate N sampled questions")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--lang-q", default=None, help="filter by question language")
    parser.add_argument("--lang-c", default=None, help="filter by context language")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    records = _load(args.input)
    if args.lang_q:
        records = [row for row in records if row["lang_q"] == args.lang_q]
    if args.lang_c:
        records = [row for row in records if row["lang_c"] == args.lang_c]
    records = _sample(records, args.sample, args.seed)
    print(f"evaluating {len(records)} questions from {args.input.name}")

    summary, per_question = asyncio.run(evaluate(records, args.limit, args.progress_every))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for row in per_question:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote per-question detail to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
