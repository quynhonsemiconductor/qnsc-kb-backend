"""Run a supplied 15-question live UAT corpus against the authenticated AI API.

The runner deliberately refuses to invent questions or expected answers. The
input must contain exactly 15 business questions and authorized Article IDs;
the shared test password is read from ``UAT_TEST_PASSWORD`` and is never
printed or accepted in the fixture file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Make the script runnable directly from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.evaluator import answer_correctness

DEFAULT_FIXTURES = (
    ("staff", "kb-staff"),
    ("reviewer", "kb-reviewer"),
    ("ceo", "kb-ceo"),
    ("admin", "kb-admin"),
)
REFUSAL_MARKER = "not found in the knowledge base"


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(not str(item).strip() for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return [str(item).strip() for item in value]


def validate_cases(cases: Any) -> list[dict[str, Any]]:
    """Validate the business corpus before any live request is made."""
    if not isinstance(cases, list) or len(cases) != 15:
        raise ValueError("UAT input must contain exactly 15 question objects")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, raw_case in enumerate(cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"UAT question {position} must be an object")
        case_id = str(raw_case.get("id", "")).strip()
        question = str(raw_case.get("question", "")).strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"UAT question {position} has a missing or duplicate id")
        if not question:
            raise ValueError(f"UAT question {case_id} has no question text")
        seen_ids.add(case_id)

        expected_ids = _string_list(
            raw_case.get("expected_article_ids", []),
            field=f"{case_id}.expected_article_ids",
        )
        expected_answer = str(raw_case.get("expected_answer", "")).strip()
        expect_refusal = bool(raw_case.get("expect_refusal", False))
        if expect_refusal:
            if expected_ids:
                raise ValueError(
                    f"{case_id} refusal cases cannot list expected Article IDs"
                )
        elif not expected_answer:
            raise ValueError(
                f"{case_id} requires expected_answer unless expect_refusal is true"
            )
        elif not expected_ids:
            raise ValueError(
                f"{case_id} requires expected_article_ids for a grounded answer"
            )
        if "forbidden_markers" in raw_case:
            _string_list(
                raw_case["forbidden_markers"], field=f"{case_id}.forbidden_markers"
            )

        expectations = raw_case.get("fixture_expectations", {})
        if not isinstance(expectations, dict):
            raise ValueError(f"{case_id}.fixture_expectations must be an object")
        for fixture_name, raw_expectation in expectations.items():
            if not isinstance(raw_expectation, dict):
                raise ValueError(
                    f"{case_id}.fixture_expectations.{fixture_name} must be an object"
                )
            override_refusal = bool(
                raw_expectation.get("expect_refusal", expect_refusal)
            )
            if "expected_article_ids" in raw_expectation:
                override_ids = _string_list(
                    raw_expectation["expected_article_ids"],
                    field=f"{case_id}.fixture_expectations.{fixture_name}.expected_article_ids",
                )
            else:
                override_ids = [] if override_refusal else expected_ids
            override_answer = str(
                raw_expectation.get("expected_answer", expected_answer)
            ).strip()
            if override_refusal and override_ids:
                raise ValueError(
                    f"{case_id}/{fixture_name} refusal cases cannot list expected Article IDs"
                )
            if not override_refusal and (not override_answer or not override_ids):
                raise ValueError(
                    f"{case_id}/{fixture_name} needs an expected answer and Article IDs"
                )
            if "forbidden_markers" in raw_expectation:
                _string_list(
                    raw_expectation["forbidden_markers"],
                    field=f"{case_id}/{fixture_name}.forbidden_markers",
                )

        if "password" in raw_case or any(
            isinstance(item, dict) and ("password" in item or "secret" in item)
            for item in expectations.values()
        ):
            raise ValueError(
                "UAT passwords must come from UAT_TEST_PASSWORD, never the input file"
            )
        normalized.append({**raw_case, "id": case_id, "question": question})
    return normalized


def load_fixtures(path: Path | None, company_domain: str) -> list[dict[str, str]]:
    if path is None:
        return [
            {"name": name, "email": f"{local_part}@{company_domain}"}
            for name, local_part in DEFAULT_FIXTURES
        ]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Fixture input must be a non-empty JSON array")
    fixtures: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each fixture must be an object")
        if "password" in item or "secret" in item:
            raise ValueError("Fixture files cannot contain passwords or secrets")
        name = str(item.get("name", "")).strip()
        email = str(item.get("email", "")).strip()
        if not name or name in seen_names or "@" not in email:
            raise ValueError("Each fixture needs a unique name and an email address")
        seen_names.add(name)
        fixtures.append({"name": name, "email": email})
    return fixtures


def validate_fixture_coverage(
    cases: list[dict[str, Any]], fixtures: list[dict[str, str]]
) -> None:
    configured = {fixture["name"] for fixture in fixtures}
    unknown = sorted(
        {
            name
            for case in cases
            for name in (case.get("fixture_expectations") or {})
            if name not in configured
        }
    )
    if unknown:
        raise ValueError(
            f"UAT corpus references fixtures not configured for this run: {', '.join(unknown)}"
        )


def _expectation(case: dict[str, Any], fixture_name: str) -> dict[str, Any]:
    base = {
        "expected_answer": str(case.get("expected_answer", "")).strip(),
        "expected_article_ids": [
            str(item) for item in case.get("expected_article_ids", [])
        ],
        "expect_refusal": bool(case.get("expect_refusal", False)),
        "forbidden_markers": [str(item) for item in case.get("forbidden_markers", [])],
    }
    override = (case.get("fixture_expectations") or {}).get(fixture_name, {})
    if not isinstance(override, dict):
        return base
    override_refusal = bool(override.get("expect_refusal", base["expect_refusal"]))
    override_ids = override.get("expected_article_ids")
    if override_ids is None:
        override_ids = [] if override_refusal else base["expected_article_ids"]
    return {
        **base,
        "expected_answer": str(
            override.get("expected_answer", base["expected_answer"])
        ).strip(),
        "expected_article_ids": [str(item) for item in override_ids],
        "expect_refusal": override_refusal,
        "forbidden_markers": [
            str(item)
            for item in override.get("forbidden_markers", base["forbidden_markers"])
        ],
    }


def evaluate_response(
    *,
    case: dict[str, Any],
    fixture_name: str,
    status_code: int,
    payload: dict[str, Any],
    min_answer_score: float,
) -> dict[str, Any]:
    expectation = _expectation(case, fixture_name)
    answer = str(payload.get("answer", ""))
    citations = (
        payload.get("citations") if isinstance(payload.get("citations"), list) else []
    )
    expected_ids = set(expectation["expected_article_ids"])
    actual_ids = sorted(
        {
            str(item.get("article_id"))
            for item in citations
            if isinstance(item, dict) and item.get("article_id")
        }
    )
    actual_id_set = set(actual_ids)
    allowed_ids = expected_ids.intersection(actual_id_set)
    unexpected_ids = sorted(actual_id_set - expected_ids)
    refusal = REFUSAL_MARKER in answer.lower()
    source_urls_missing = [
        str(item.get("article_id"))
        for item in citations
        if isinstance(item, dict)
        and item.get("article_id")
        and not str(item.get("source_url", "")).strip()
    ]
    serialized_payload = json.dumps(payload, ensure_ascii=False).lower()
    forbidden_exposed = [
        marker
        for marker in expectation["forbidden_markers"]
        if marker.lower() in serialized_payload
    ]

    if expectation["expect_refusal"]:
        answer_score = 1.0 if refusal and not citations else 0.0
    else:
        answer_score = answer_correctness(answer, expectation["expected_answer"])
    citation_precision = (
        len(allowed_ids) / len(actual_id_set)
        if actual_id_set
        else (1.0 if not expected_ids else 0.0)
    )
    citation_recall = (
        len(allowed_ids) / len(expected_ids)
        if expected_ids
        else (1.0 if not actual_id_set else 0.0)
    )
    passed = (
        status_code == 200
        and answer_score >= min_answer_score
        and not unexpected_ids
        and not source_urls_missing
        and not forbidden_exposed
        and (
            (expectation["expect_refusal"] and refusal and not citations)
            or (
                not expectation["expect_refusal"]
                and bool(allowed_ids)
                and citation_recall >= 1.0
            )
        )
    )
    return {
        "case": case["id"],
        "fixture": fixture_name,
        "question": case["question"],
        "status": status_code,
        "answer": answer,
        "citations": citations,
        "expected_article_ids": sorted(expected_ids),
        "actual_article_ids": actual_ids,
        "missing_article_ids": sorted(expected_ids - actual_id_set),
        "unexpected_article_ids": unexpected_ids,
        "missing_source_urls": sorted(set(source_urls_missing)),
        "forbidden_markers_exposed": forbidden_exposed,
        "expected_refusal": expectation["expect_refusal"],
        "refusal_detected": refusal,
        "answer_score": round(answer_score, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "passed": passed,
    }


async def run_uat(
    *,
    base_url: str,
    password: str,
    cases: list[dict[str, Any]],
    fixtures: list[dict[str, str]],
    min_answer_score: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=base_url, timeout=120.0, trust_env=False
    ) as client:
        tokens: dict[str, str] = {}
        for fixture in fixtures:
            login = await client.post(
                "/api/v1/auth/login",
                data={"username": fixture["email"], "password": password},
            )
            if login.status_code != 200:
                raise RuntimeError(
                    f"login failed for fixture {fixture['name']} with HTTP {login.status_code}"
                )
            tokens[fixture["name"]] = str(login.json()["access_token"])

        for case in cases:
            for fixture in fixtures:
                response = await client.post(
                    "/api/v1/ai/ask",
                    headers={"Authorization": f"Bearer {tokens[fixture['name']]}"},
                    json={"question": case["question"]},
                )
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"answer": response.text[:500], "citations": []}
                results.append(
                    evaluate_response(
                        case=case,
                        fixture_name=fixture["name"],
                        status_code=response.status_code,
                        payload=payload if isinstance(payload, dict) else {},
                        min_answer_score=min_answer_score,
                    )
                )

    scores = [float(item["answer_score"]) for item in results]
    passed = [item for item in results if item["passed"]]
    return {
        "case_count": len(cases),
        "fixture_count": len(fixtures),
        "measurement_count": len(results),
        "min_answer_score": min_answer_score,
        "answer_accuracy": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "passed_measurements": len(passed),
        "pass_rate": round(len(passed) / len(results), 4) if results else 0.0,
        "gate_passed": bool(results)
        and len(passed) == len(results)
        and (sum(scores) / len(scores)) >= min_answer_score,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the supplied 15-question QNSC live UAT corpus"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing exactly 15 business questions",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Optional JSON fixture list; defaults to seeded staff/reviewer/CEO/admin accounts",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--company-domain", default=os.getenv("SEED_COMPANY_DOMAIN", "acme.test")
    )
    parser.add_argument(
        "--min-answer-score",
        type=float,
        default=0.80,
        help="Provisional per-answer lexical score threshold; replace with the approved UAT KPI",
    )
    args = parser.parse_args()
    password = os.getenv("UAT_TEST_PASSWORD", "").strip()
    if not password:
        parser.error(
            "UAT_TEST_PASSWORD must be set; no password is stored in this script"
        )
    if not 0.0 <= args.min_answer_score <= 1.0:
        parser.error("--min-answer-score must be between 0 and 1")

    try:
        cases = validate_cases(json.loads(args.input.read_text(encoding="utf-8")))
        fixtures = load_fixtures(args.fixtures, args.company_domain)
        validate_fixture_coverage(cases, fixtures)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        report = asyncio.run(
            run_uat(
                base_url=args.base_url,
                password=password,
                cases=cases,
                fixtures=fixtures,
                min_answer_score=args.min_answer_score,
            )
        )
    except (httpx.HTTPError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
