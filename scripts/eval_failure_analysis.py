"""Classify every failed question into one cause, and quantify where F1 is lost.

An aggregate score says the system is wrong; it does not say which stage was
wrong. This reads the per-question rows written by `eval_rag.py` and assigns each
failure to exactly ONE category, chosen by walking the pipeline in order and
stopping at the first stage that broke. Attributing a failure to every plausible
cause would double-count and make the percentages meaningless.

The decision order matters and is the argument of this script:

1. The corpus never held the gold document      -> harness/corpus fault, not RAG
2. Search returned nothing at all               -> retrieval failure (empty)
3. Gold document absent from top-k              -> retrieval failure (ranking)
4. Gold document retrieved, answer text absent  -> chunking / context assembly
5. Confidence gate refused before reading       -> gate (lexical score too low)
6. Answer was in context, reader abstained      -> reader abstention
7. Answer was in context, reader picked wrong   -> reader extraction
8. Partial overlap only                         -> span boundary
9. Unanswerable but an answer was produced      -> false answer (abstention miss)

Categories map onto the 16 requested failure types; the mapping is printed with
the counts so nothing is silently renamed.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

# Requested taxonomy -> the condition this script tests for it.
TAXONOMY = {
    "retrieval_empty": "1. Retrieval failure (no results at all)",
    "retrieval_ranking": "2/4. Wrong document / evidence outside top-k",
    "chunk_lost_answer": "3/5/6. Correct document, answer not in selected chunk",
    "confidence_gate": "9/10. Reranking/gate refused a retrievable answer",
    "reader_abstained": "12. Answer extraction failure (abstained with evidence)",
    "reader_wrong_span": "12. Answer extraction failure (wrong span)",
    "span_boundary": "12. Answer extraction failure (partial span)",
    "false_answer": "14. Unanswerable-question failure (answered anyway)",
    "corpus_missing": "16. Other (gold context absent from corpus)",
    "correct": "-- scored 1.0",
    "partial_credit": "-- scored between 0 and 1",
}


def classify(row: dict) -> str:
    exact = row.get("exact_match", 0.0)
    f1 = row.get("f1", 0.0)
    impossible = row.get("is_impossible", False)
    prediction = (row.get("prediction") or "").strip()

    if impossible:
        return "correct" if not prediction else "false_answer"
    if exact == 1.0:
        return "correct"
    if not row.get("gold_indexed", True):
        return "corpus_missing"
    if row.get("result_count", 0) == 0:
        return "retrieval_empty"
    if not row.get("gold_retrieved", False):
        return "retrieval_ranking"
    if not row.get("answer_in_context", False):
        return "chunk_lost_answer"
    if row.get("gated_out", False):
        return "confidence_gate"
    if not prediction:
        return "reader_abstained"
    if f1 == 0.0:
        return "reader_wrong_span"
    if f1 < 1.0:
        return "span_boundary"
    return "partial_credit"


def analyse(rows: list[dict]) -> dict:
    counts = collections.Counter(classify(row) for row in rows)
    total = len(rows)
    # F1 actually lost to each category: how much the aggregate would rise if
    # every question in that bucket scored perfectly. This is what ranks the
    # bottlenecks, rather than raw counts.
    recoverable: dict[str, float] = collections.defaultdict(float)
    for row in rows:
        recoverable[classify(row)] += (1.0 - row.get("f1", 0.0)) / total * 100.0
    return {
        "total": total,
        "counts": dict(counts),
        "f1_points_recoverable": {k: round(v, 2) for k, v in recoverable.items()},
    }


def _print_report(name: str, rows: list[dict]) -> None:
    report = analyse(rows)
    total = report["total"]
    print(f"\n=== {name}  ({total} questions)")
    current = 100.0 * sum(row.get("f1", 0.0) for row in rows) / total if total else 0.0
    print(f"    current F1: {current:.2f}")
    print(f"    {'category':22} {'count':>6} {'share':>8} {'F1 pts':>8}   taxonomy")
    ordered = sorted(
        report["counts"].items(),
        key=lambda item: -report["f1_points_recoverable"].get(item[0], 0.0),
    )
    for category, count in ordered:
        share = 100.0 * count / total if total else 0.0
        points = report["f1_points_recoverable"].get(category, 0.0)
        print(
            f"    {category:22} {count:6d} {share:7.1f}% {points:7.2f}   "
            f"{TAXONOMY.get(category, '?')}"
        )


def _print_examples(rows: list[dict], category: str, limit: int) -> None:
    matching = [row for row in rows if classify(row) == category][:limit]
    if not matching:
        return
    print(f"\n--- examples: {category}")
    for row in matching:
        print(f"    Q ({row['lang_q']}): {row['question'][:110]}")
        print(f"      gold: {row['answers']}")
        print(f"      pred: {row['prediction']!r}  f1={row['f1']:.2f}")
        print(
            f"      retrieved={row['gold_retrieved']} rank={row['first_hit_rank']} "
            f"answer_in_context={row['answer_in_context']} top_score={row['top_score']} "
            f"span_score={row.get('span_score')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--examples", default=None, help="print examples for a category")
    parser.add_argument("--example-count", type=int, default=5)
    args = parser.parse_args()

    everything: list[dict] = []
    for path in args.inputs:
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        everything.extend(rows)
        _print_report(path.stem, rows)
        if args.examples:
            _print_examples(rows, args.examples, args.example_count)

    if len(args.inputs) > 1:
        _print_report("ALL SPLITS COMBINED", everything)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
