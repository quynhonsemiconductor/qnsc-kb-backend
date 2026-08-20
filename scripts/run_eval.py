"""Run the QNSC RAG regression gate.

Two modes, and the distinction matters:

* Live mode (``--api-base``) is the real regression gate: every case's
  ``question`` is sent to a running stack's ``POST /ai/ask`` and the returned
  answer/citations are scored against ``expected_answer`` /
  ``expected_chunk_ids``. This exercises retrieval, reranking, prompt
  assembly, the LLM call, citation mapping, and guardrails.

* Offline mode (no ``--api-base``) only sanity-checks the *scoring
  functions* against explicit ``answer``/``context`` fixtures. It never runs
  the pipeline and certifies nothing about RAG quality. Cases whose
  ``answer`` is byte-identical to ``expected_answer`` are rejected: that
  pattern produced a permanently green, self-comparing "gate" historically.

Both modes exit non-zero when any configured score falls below its
threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the script runnable directly from the repository root as well as via
# an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.evaluator import answer_correctness, context_recall, lexical_faithfulness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QNSC RAG regression gate")
    parser.add_argument("--input", type=Path, required=True, help="JSON file containing evaluation cases")
    parser.add_argument("--api-base", type=str, default=None, help="Base URL of a live API (e.g. http://localhost:8000/api/v1) to run the real pipeline gate")
    parser.add_argument("--email", type=str, default=None, help="Login email for live mode")
    parser.add_argument("--password", type=str, default=None, help="Login password for live mode")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout for live mode")
    parser.add_argument("--min-context-recall", type=float, default=0.80)
    parser.add_argument("--min-faithfulness", type=float, default=0.80)
    parser.add_argument("--min-correctness", type=float, default=0.80)
    return parser.parse_args()


def score_case(case: dict, answer: str, context: str, retrieved_chunk_ids: list) -> dict:
    return {
        "case": case.get("id", "?"),
        "context_recall": context_recall(retrieved_chunk_ids, case.get("expected_chunk_ids", [])),
        "faithfulness": lexical_faithfulness(answer, context),
        "answer_correctness": answer_correctness(answer, str(case.get("expected_answer", ""))),
    }


def run_offline(cases: list[dict]) -> dict:
    """Score explicit fixtures. Tests the scorers, not the pipeline."""
    results = []
    for index, case in enumerate(cases, start=1):
        answer = str(case.get("answer", ""))
        expected = str(case.get("expected_answer", ""))
        if answer == expected:
            raise ValueError(
                f"Offline case {case.get('id', index)} has answer identical to "
                "expected_answer; a self-comparing fixture cannot fail and "
                "certifies nothing. Use --api-base for the real gate."
            )
        results.append(
            score_case(
                case,
                answer,
                str(case.get("context", "")),
                case.get("retrieved_chunk_ids", []),
            )
        )
    return _summarize(results)


def run_live(cases: list[dict], args: argparse.Namespace) -> dict:
    """Run each case's question through the live pipeline and score it."""
    import httpx

    if not args.email or not args.password:
        raise SystemExit("Live mode requires --email and --password")
    cases = [case for case in cases if case.get("question")]
    if not cases:
        raise SystemExit("Live mode requires cases with a 'question' field")

    with httpx.Client(timeout=args.timeout) as client:
        login = client.post(
            f"{args.api_base}/auth/login",
            json={"email": args.email, "password": args.password},
        )
        login.raise_for_status()
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        results = []
        for case in cases:
            response = client.post(
                f"{args.api_base}/ai/ask",
                json={"question": case["question"]},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
            answer = str(payload.get("answer", ""))
            citations = payload.get("citations") or []
            retrieved_chunk_ids = [
                str(item.get("chunk_id")) for item in citations if item.get("chunk_id")
            ]
            context = "\n".join(
                str(item.get("excerpt", "")) for item in citations
            )
            results.append(
                score_case(case, answer, context, retrieved_chunk_ids)
            )
    return _summarize(results)


def _summarize(results: list[dict]) -> dict:
    if not results:
        raise ValueError("Evaluation produced no scored cases")
    return {
        "case_count": len(results),
        "averages": {
            metric: sum(item[metric] for item in results) / len(results)
            for metric in ("context_recall", "faithfulness", "answer_correctness")
        },
        "results": results,
    }


def main() -> int:
    args = parse_args()
    cases = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Evaluation input must be a JSON array")

    if args.api_base:
        print("Running LIVE pipeline gate against", args.api_base)
        report = run_live(cases, args)
    else:
        print(
            "Running OFFLINE scorer sanity check only — the RAG pipeline is "
            "NOT exercised. Use --api-base for a real regression gate."
        )
        report = run_offline(cases)
    print(json.dumps(report, indent=2, sort_keys=True))

    averages = report["averages"]
    failures = {
        "context_recall": averages["context_recall"] < args.min_context_recall,
        "faithfulness": averages["faithfulness"] < args.min_faithfulness,
        "answer_correctness": averages["answer_correctness"] < args.min_correctness,
    }
    if any(failures.values()):
        print(f"Regression gate failed: {failures}", file=sys.stderr)
        return 1
    print("Regression gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
