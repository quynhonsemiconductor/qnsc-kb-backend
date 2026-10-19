"""Exercise the MVP-1 upload-to-AI chain against a running real stack.

Required environment variables:
  ACCEPTANCE_TEST_PASSWORD  Password shared by the seeded test accounts.

Optional environment variables:
  ACCEPTANCE_BASE_URL          Defaults to http://127.0.0.1:8000
  ACCEPTANCE_COMPANY_DOMAIN   Defaults to acme.test
  ACCEPTANCE_OTHER_DOMAIN     Defaults to other.test
  ACCEPTANCE_MARKER           Defaults to a generated marker

Seed accounts first, for example:
  $env:SEED_TEST_PASSWORD = '...'; $env:SEED_COMPANY_DOMAIN = 'acme.test';
  poetry run python scripts/seed_data.py
  $env:SEED_COMPANY_DOMAIN = 'other.test'; poetry run python scripts/seed_data.py
  $env:ACCEPTANCE_TEST_PASSWORD = '...'; poetry run python scripts/real_stack_acceptance.py

The test intentionally retains the approved Article as an auditable acceptance
record. It never prints bearer tokens or presigned URLs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import uuid
from urllib.parse import urlsplit

import httpx


def _safe_response(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError:
        return {"status": response.status_code, "text": response.text[:240]}
    if isinstance(payload, dict):
        return {
            "status": response.status_code,
            "body": {
                key: (
                    "<redacted>"
                    if key
                    in {
                        "access_token",
                        "refresh_token",
                        "upload_url",
                        "authorization_url",
                        "state",
                    }
                    else value
                )
                for key, value in payload.items()
            },
        }
    return {"status": response.status_code, "body": payload}


async def run(
    base_url: str, company_domain: str, other_domain: str, password: str, marker: str
) -> dict[str, object]:
    body = (
        f"# {marker} access procedure\n\n"
        f"## Authorized handling\n\nThe {marker} procedure requires the Finance department and reviewer approval.\n\n"
        f"## Retention\n\nThe {marker} source must be retained for 30 days after approval.\n"
    ).encode("utf-8")
    source_hash = hashlib.sha256(body).hexdigest()

    async with httpx.AsyncClient(
        base_url=base_url, timeout=90, follow_redirects=False, trust_env=False
    ) as client:

        async def login(local_part: str, domain: str) -> tuple[str, dict[str, object]]:
            email = f"{local_part}@{domain}"
            response = await client.post(
                "/api/v1/auth/login", data={"username": email, "password": password}
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"login failed for {email}: {_safe_response(response)}"
                )
            payload = response.json()
            return str(payload["access_token"]), payload["user"]

        staff_token, _ = await login("kb-staff", company_domain)
        admin_token, _ = await login("kb-admin", company_domain)
        reviewer_token, reviewer = await login("kb-reviewer", company_domain)
        ceo_token, _ = await login("kb-ceo", company_domain)
        other_token, _ = await login("kb-staff", other_domain)
        staff_headers = {"Authorization": f"Bearer {staff_token}"}
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        reviewer_headers = {"Authorization": f"Bearer {reviewer_token}"}
        ceo_headers = {"Authorization": f"Bearer {ceo_token}"}
        other_headers = {"Authorization": f"Bearer {other_token}"}

        intent = await client.post(
            "/api/v1/articles/source-uploads",
            headers=staff_headers,
            json={
                "filename": f"{marker}.md",
                "source_hash": source_hash,
                "content_length": len(body),
                "dept": "Finance",
                "tags": ["real-stack", "acceptance"],
            },
        )
        if intent.status_code != 201:
            raise RuntimeError(f"upload intent failed: {_safe_response(intent)}")
        intent_payload = intent.json()
        draft_id = str(intent_payload["draft_id"])
        upload_url = str(intent_payload["upload_url"])

        put = await client.put(upload_url, content=body)
        if put.status_code not in {200, 201, 204}:
            raise RuntimeError(f"signed R2 PUT failed: status={put.status_code}")
        signed = urlsplit(upload_url)
        unsigned_url = f"{signed.scheme}://{signed.netloc}{signed.path}"
        unsigned = await client.get(unsigned_url)
        if 200 <= unsigned.status_code < 300:
            raise RuntimeError(
                f"unsigned R2 object was reachable: status={unsigned.status_code}"
            )

        complete = await client.post(
            f"/api/v1/articles/source-uploads/{draft_id}/complete",
            headers=staff_headers,
            json={"content_length": len(body)},
        )
        if complete.status_code != 201:
            raise RuntimeError(f"upload completion failed: {_safe_response(complete)}")
        complete_payload = complete.json()
        if complete_payload.get("status") != "pending":
            raise RuntimeError(
                f"completed upload was not pending: {complete_payload.get('status')}"
            )

        duplicate = await client.post(
            "/api/v1/articles/source-uploads",
            headers=staff_headers,
            json={
                "filename": f"{marker}.md",
                "source_hash": source_hash,
                "content_length": len(body),
                "dept": "Finance",
            },
        )
        if duplicate.status_code != 409:
            raise RuntimeError(
                f"duplicate upload was not rejected: {_safe_response(duplicate)}"
            )

        assign = await client.post(
            f"/api/v1/governance/pending-drafts/{draft_id}/assign-approver",
            headers=admin_headers,
            json={"approver_id": reviewer["id"], "use_rule": False},
        )
        if assign.status_code != 200:
            raise RuntimeError(f"approver assignment failed: {_safe_response(assign)}")

        candidates_response = await client.get(
            f"/api/v1/governance/pending-drafts/{draft_id}/candidates",
            headers=reviewer_headers,
        )
        candidates = (
            candidates_response.json() if candidates_response.status_code == 200 else []
        )
        if candidates_response.status_code != 200 or not candidates:
            raise RuntimeError(
                f"split candidates were not available after upload completion: {_safe_response(candidates_response)}"
            )

        approve = await client.post(
            f"/api/v1/governance/pending-drafts/{draft_id}/approve",
            headers=reviewer_headers,
            # The probe creates a new auditable Article on every run. The
            # backend still performs duplicate/similarity detection; this
            # explicit flag records the human decision to retain a new copy.
            json={
                "dept": "Finance",
                "treat_as_new": True,
                "review_note": "Real-stack acceptance approval",
            },
        )
        if approve.status_code != 200:
            raise RuntimeError(f"approval failed: {_safe_response(approve)}")
        article_id = str(approve.json()["id"])

        article = None
        for _ in range(30):
            response = await client.get(
                f"/api/v1/articles/{article_id}", headers=ceo_headers
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"approved Article could not be read by CEO: {_safe_response(response)}"
                )
            article = response.json()
            if article.get("index_status") == "ready":
                break
            await asyncio.sleep(1)
        if not article or article.get("index_status") != "ready":
            raise RuntimeError(
                f"Article index did not become ready: {article.get('index_status') if article else None}"
            )

        search = await client.get(
            "/api/v1/search",
            headers=ceo_headers,
            params={"q": marker, "status": "published", "limit": 10},
        )
        search_results = search.json() if search.status_code == 200 else []
        if search.status_code != 200 or not any(
            str(item.get("article_id")) == article_id for item in search_results
        ):
            raise RuntimeError(f"authorized search failed: {_safe_response(search)}")

        source = await client.get(
            f"/api/v1/articles/{article_id}/source", headers=ceo_headers
        )
        if source.status_code != 200 or source.content != body:
            raise RuntimeError(
                f"authorized source opening failed: status={source.status_code}, bytes={len(source.content)}"
            )

        ai = await client.post(
            "/api/v1/ai/ask",
            headers=ceo_headers,
            json={"question": f"What is the retention period for {marker}?"},
        )
        if ai.status_code != 200:
            raise RuntimeError(f"AI answer failed: {_safe_response(ai)}")
        ai_payload = ai.json()
        citations = ai_payload.get("citations") or []
        if not citations or not any(
            str(item.get("article_id")) == article_id for item in citations
        ):
            raise RuntimeError(
                f"AI answer did not return an authorized citation: {_safe_response(ai)}"
            )
        if not any(
            f"/articles/{article_id}/source" in str(item.get("source_url", ""))
            for item in citations
        ):
            raise RuntimeError(
                f"AI citation did not expose a source-opening URL: {_safe_response(ai)}"
            )

        # Exercise the same question through users who cannot read the
        # Article. The API must not leak the marker in answer text, excerpts,
        # or citations even when the question itself names the source.
        security_question = f"What is the retention period for {marker}?"
        reviewer_ai = await client.post(
            "/api/v1/ai/ask",
            headers=reviewer_headers,
            json={"question": security_question},
        )
        cross_ai = await client.post(
            "/api/v1/ai/ask",
            headers=other_headers,
            json={"question": security_question},
        )
        reviewer_ai_payload = (
            reviewer_ai.json() if reviewer_ai.status_code == 200 else {}
        )
        cross_ai_payload = cross_ai.json() if cross_ai.status_code == 200 else {}
        marker_lower = marker.lower()
        if (
            reviewer_ai.status_code != 200
            or marker_lower in json.dumps(reviewer_ai_payload).lower()
            or reviewer_ai_payload.get("citations")
        ):
            raise RuntimeError(
                f"same-tenant AI isolation failed: {_safe_response(reviewer_ai)}"
            )
        if (
            cross_ai.status_code != 200
            or marker_lower in json.dumps(cross_ai_payload).lower()
            or cross_ai_payload.get("citations")
        ):
            raise RuntimeError(
                f"cross-tenant AI isolation failed: {_safe_response(cross_ai)}"
            )

        reviewer_article = await client.get(
            f"/api/v1/articles/{article_id}", headers=reviewer_headers
        )
        reviewer_list = await client.get(
            "/api/v1/articles/",
            headers=reviewer_headers,
            params={"q": marker, "status": "published"},
        )
        reviewer_search = await client.get(
            "/api/v1/search",
            headers=reviewer_headers,
            params={"q": marker, "status": "published", "limit": 10},
        )
        cross_article = await client.get(
            f"/api/v1/articles/{article_id}", headers=other_headers
        )
        cross_list = await client.get(
            "/api/v1/articles/",
            headers=other_headers,
            params={"q": marker, "status": "published"},
        )
        cross_source = await client.get(
            f"/api/v1/articles/{article_id}/source", headers=other_headers
        )
        cross_search = await client.get(
            "/api/v1/search",
            headers=other_headers,
            params={"q": marker, "status": "published", "limit": 10},
        )
        reviewer_list_results = (
            reviewer_list.json() if reviewer_list.status_code == 200 else []
        )
        cross_list_results = cross_list.json() if cross_list.status_code == 200 else []
        reviewer_results = (
            reviewer_search.json() if reviewer_search.status_code == 200 else []
        )
        cross_results = cross_search.json() if cross_search.status_code == 200 else []
        if (
            reviewer_article.status_code not in {403, 404}
            or reviewer_list.status_code != 200
            or reviewer_list_results
            or reviewer_search.status_code != 200
            or reviewer_results
        ):
            raise RuntimeError(
                f"same-tenant department isolation failed: article={_safe_response(reviewer_article)}, list={_safe_response(reviewer_list)}, search={_safe_response(reviewer_search)}"
            )
        if (
            cross_article.status_code not in {403, 404}
            or cross_list.status_code != 200
            or cross_list_results
            or cross_source.status_code not in {403, 404}
            or cross_search.status_code != 200
            or cross_results
        ):
            raise RuntimeError(
                f"cross-tenant authorization failed: article={_safe_response(cross_article)}, list={_safe_response(cross_list)}, source={_safe_response(cross_source)}, search={_safe_response(cross_search)}"
            )

        return {
            "marker": marker,
            "draft_id": draft_id,
            "article_id": article_id,
            "article_status": article.get("status"),
            "index_status": article.get("index_status"),
            "candidates_after_completion": len(candidates),
            "authorized_search_hits": len(search_results),
            "citation_count": len(citations),
            "citation_source_url_present": True,
            "unsigned_r2_status": unsigned.status_code,
            "duplicate_intent_status": duplicate.status_code,
            "assigned_approver_approval_status": approve.status_code,
            "same_tenant_reviewer_article_status": reviewer_article.status_code,
            "same_tenant_reviewer_list_hits": len(reviewer_list_results),
            "same_tenant_reviewer_search_hits": len(reviewer_results),
            "same_tenant_reviewer_ai_citations": len(
                reviewer_ai_payload.get("citations") or []
            ),
            "same_tenant_reviewer_ai_marker_exposed": False,
            "cross_tenant_article_status": cross_article.status_code,
            "cross_tenant_list_hits": len(cross_list_results),
            "cross_tenant_source_status": cross_source.status_code,
            "cross_tenant_search_hits": len(cross_results),
            "cross_tenant_ai_citations": len(cross_ai_payload.get("citations") or []),
            "cross_tenant_ai_marker_exposed": False,
            "ai_answer_prefix": str(ai_payload.get("answer", ""))[:80],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.getenv("ACCEPTANCE_BASE_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument(
        "--company-domain", default=os.getenv("ACCEPTANCE_COMPANY_DOMAIN", "acme.test")
    )
    parser.add_argument(
        "--other-domain", default=os.getenv("ACCEPTANCE_OTHER_DOMAIN", "other.test")
    )
    parser.add_argument(
        "--marker",
        default=os.getenv(
            "ACCEPTANCE_MARKER", f"QNSC-RSTACK-{uuid.uuid4().hex[:12].upper()}"
        ),
    )
    args = parser.parse_args()
    password = os.getenv("ACCEPTANCE_TEST_PASSWORD", "").strip()
    if not password:
        parser.error(
            "ACCEPTANCE_TEST_PASSWORD must be set; no password is stored in this script"
        )
    print(
        json.dumps(
            asyncio.run(
                run(
                    args.base_url,
                    args.company_domain,
                    args.other_domain,
                    password,
                    args.marker,
                )
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
