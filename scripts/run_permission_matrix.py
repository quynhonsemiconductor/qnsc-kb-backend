"""Run the authenticated Role x Resource permission matrix against a live stack.

The runner uses only the deterministic seeded identities. The shared password
is read from ``MATRIX_TEST_PASSWORD`` and is never accepted as a command-line
argument, from a JSON input file, or written to the output report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROLES = ("Admin", "CEO", "Reviewer", "Staff")
ROLE_EMAIL_PREFIX = {
    "Admin": "kb-admin",
    "CEO": "kb-ceo",
    "Reviewer": "kb-reviewer",
    "Staff": "kb-staff",
}


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    resource: str
    method: str
    path: str
    allowed_roles: frozenset[str]
    body: dict[str, Any] | None = None


ALL_ROLES = frozenset(ROLES)
MANAGEMENT_ROLES = frozenset({"Admin", "CEO"})
REVIEW_ROLES = frozenset({"Admin", "CEO", "Reviewer"})
ADMIN_ONLY = frozenset({"Admin"})


CASES = (
    MatrixCase("identity.me", "identity", "GET", "/api/v1/auth/me", ALL_ROLES),
    MatrixCase(
        "identity.departments",
        "departments",
        "GET",
        "/api/v1/auth/departments",
        ALL_ROLES,
    ),
    MatrixCase(
        "identity.roles", "roles", "GET", "/api/v1/auth/roles", MANAGEMENT_ROLES
    ),
    MatrixCase(
        "identity.users", "users", "GET", "/api/v1/auth/users", MANAGEMENT_ROLES
    ),
    MatrixCase(
        "identity.groups",
        "access groups",
        "GET",
        "/api/v1/auth/groups",
        MANAGEMENT_ROLES,
    ),
    MatrixCase(
        "articles.list",
        "articles",
        "GET",
        "/api/v1/articles/?status=published",
        ALL_ROLES,
    ),
    MatrixCase(
        "articles.browse",
        "article browse",
        "GET",
        "/api/v1/knowledge/browse",
        ALL_ROLES,
    ),
    MatrixCase(
        "search.hybrid",
        "hybrid search",
        "GET",
        "/api/v1/search?q=procedure&limit=10",
        ALL_ROLES,
    ),
    MatrixCase(
        "ai.conversations",
        "AI conversations",
        "GET",
        "/api/v1/ai/conversations",
        ALL_ROLES,
    ),
    MatrixCase(
        "ai.answer",
        "AI answer",
        "POST",
        "/api/v1/ai/ask",
        ALL_ROLES,
        {"question": "What is the retention period?"},
    ),
    MatrixCase(
        "sources.catalog",
        "source catalog",
        "GET",
        "/api/v1/knowledge/sources",
        ALL_ROLES,
    ),
    MatrixCase(
        "home.summary",
        "permission-aware home",
        "GET",
        "/api/v1/knowledge/home",
        ALL_ROLES,
    ),
    MatrixCase("metadata.tags", "metadata", "GET", "/api/v1/meta/tags", ALL_ROLES),
    MatrixCase(
        "notifications.list", "notifications", "GET", "/api/v1/notifications", ALL_ROLES
    ),
    MatrixCase(
        "interactions.bookmarks",
        "bookmarks",
        "GET",
        "/api/v1/interactions/bookmarks",
        ALL_ROLES,
    ),
    MatrixCase(
        "connectors.list",
        "SharePoint connectors",
        "GET",
        "/api/v1/connectors",
        MANAGEMENT_ROLES,
    ),
    MatrixCase(
        "review.pending",
        "draft review queue",
        "GET",
        "/api/v1/governance/pending-drafts",
        REVIEW_ROLES,
    ),
    MatrixCase(
        "governance.audit",
        "audit log",
        "GET",
        "/api/v1/governance/audit-log",
        ADMIN_ONLY,
    ),
    MatrixCase(
        "governance.health",
        "dependency health",
        "GET",
        "/api/v1/governance/health-metrics",
        ADMIN_ONLY,
    ),
    MatrixCase(
        "governance.llm",
        "LLM configuration",
        "GET",
        "/api/v1/admin/llm/config",
        ADMIN_ONLY,
    ),
)


def _payload_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"kind": "list", "count": len(payload)}
    if isinstance(payload, dict):
        summary: dict[str, Any] = {"kind": "object", "keys": sorted(payload)[:30]}
        if "citations" in payload and isinstance(payload["citations"], list):
            summary["citation_count"] = len(payload["citations"])
        if "answer" in payload and isinstance(payload["answer"], str):
            summary["answer_length"] = len(payload["answer"])
        return summary
    return {"kind": type(payload).__name__}


async def _login(
    client: httpx.AsyncClient, role: str, company_domain: str, password: str
) -> str:
    email = f"{ROLE_EMAIL_PREFIX[role]}@{company_domain}"
    response = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": password}
    )
    if response.status_code != 200:
        raise RuntimeError(f"login failed for {role} with HTTP {response.status_code}")
    return str(response.json()["access_token"])


async def run(
    *, base_url: str, company_domain: str, password: str, include_ai: bool
) -> dict[str, Any]:
    cases = tuple(case for case in CASES if include_ai or case.case_id != "ai.answer")
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=base_url, timeout=90.0, trust_env=False
    ) as client:
        for role in ROLES:
            token = await _login(client, role, company_domain, password)
            headers = {"Authorization": f"Bearer {token}"}
            for case in cases:
                response = await client.request(
                    case.method, case.path, headers=headers, json=case.body
                )
                expected_allowed = role in case.allowed_roles
                expected_status = 200 if expected_allowed else 403
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                results.append(
                    {
                        "role": role,
                        "resource": case.resource,
                        "case_id": case.case_id,
                        "method": case.method,
                        "path": case.path,
                        "expected": "allow/200" if expected_allowed else "deny/403",
                        "actual_status": response.status_code,
                        "passed": response.status_code == expected_status,
                        "response_summary": _payload_summary(payload),
                    }
                )

    passed = sum(bool(item["passed"]) for item in results)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "company_domain": company_domain,
        "roles": list(ROLES),
        "case_count": len(cases),
        "result_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.getenv("MATRIX_BASE_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument(
        "--company-domain", default=os.getenv("MATRIX_COMPANY_DOMAIN", "acme.test")
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip the provider-backed AI answer case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON evidence artifact to this path.",
    )
    args = parser.parse_args()
    password = os.getenv("MATRIX_TEST_PASSWORD", "").strip()
    if not password:
        parser.error(
            "MATRIX_TEST_PASSWORD must be set; no password is stored in this script"
        )

    report = asyncio.run(
        run(
            base_url=args.base_url,
            company_domain=args.company_domain,
            password=password,
            include_ai=not args.skip_ai,
        )
    )
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
