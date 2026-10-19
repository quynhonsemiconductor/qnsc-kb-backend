"""Prove, with the application's own sender, that Graph mail works and is scoped to ONE mailbox.

Run by the owner from a workstation after the setup in docs/runbooks/email-graph.md (and by CI-free
local tests against a fake Microsoft). It uses src/services/email.py unchanged, so a pass here is a pass
for the code that runs in the worker.

    # develop's app: its own mailbox must work, the PRODUCTION mailbox must be refused
    python scripts/graph_mail_check.py \\
        --tenant <tenant-guid> --client-id <develop-mail-app-client-id> \\
        --cert mail.crt --key mail.key \\
        --sender noreply-kb-dev@qnsc.vn --to you@qnsc.vn \\
        --denied-mailbox noreply-kb@qnsc.vn --denied-mailbox someone-else@qnsc.vn

Checks, each printed as PASS or FAIL (exit status 1 on any FAIL):
  1. TOKEN   the certificate is accepted and Entra issues a Graph token;
  2. SEND    sendMail AS the sender mailbox is accepted (HTTP 202);
  3. DENIED  sendMail AS each --denied-mailbox is REFUSED with 403 (repeat the flag). This is the negative test: it proves the
             Exchange RBAC scope is what limits the app. If it is accepted, the app can send as anyone and
             the setup is wrong (usually an unscoped Mail.Send consent is still granted in Entra).

Nothing here prints the key, the certificate, or the token. The mail it sends is a plain test message.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import Settings  # noqa: E402
from src.services import email  # noqa: E402


def _claims(token: str) -> dict:
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:
        return {}


async def run(args: argparse.Namespace) -> int:
    if args.authority:
        email.LOGIN_AUTHORITY = args.authority.rstrip("/")
    if args.graph_api:
        email.GRAPH_API = args.graph_api.rstrip("/")
    email.settings = Settings(
        _env_file=None,
        ENVIRONMENT="production",
        MICROSOFT_TENANT_ID=args.tenant,
        MAIL_GRAPH_CLIENT_ID=args.client_id,
        MAIL_GRAPH_CERT_PEM=Path(args.cert).read_text(encoding="utf-8"),
        MAIL_GRAPH_PRIVATE_KEY_PEM=Path(args.key).read_text(encoding="utf-8"),
        MICROSOFT_GRAPH_SENDER=args.sender,
    )
    failures = 0

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:7s} {detail}")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            token = await email._access_token(
                client
            )  # noqa: SLF001 - the point is to exercise the real path
        claims = _claims(token)
        roles = claims.get("roles") or []
        report(
            "TOKEN",
            True,
            f"issued for appid={claims.get('appid') or claims.get('azp')}; "
            f"roles in the token: {roles or 'none (expected when authority comes from Exchange RBAC only)'}",
        )
        if roles:
            # ANY application role means something was consented in Entra, and Entra grants are tenant-wide, so
            # they would make the Exchange scope meaningless. The app should hold no role at all.
            report(
                "SCOPE",
                False,
                f"the token carries application role(s) {roles}: remove every application permission "
                "consent from this app in Entra (runbook step 7), or the Exchange scope is meaningless",
            )
            return 1
    except email.MailSendError as exc:
        report("TOKEN", False, str(exc))
        return 1

    sender = email.MicrosoftGraphEmailSender()
    try:
        await sender.send(
            to=args.to,
            subject="QNSC KB mail check",
            text="Test message from scripts/graph_mail_check.py.",
        )
        report("SEND", True, f"sent as {args.sender} to {args.to} (HTTP 202)")
    except email.MailSendError as exc:
        report("SEND", False, f"{exc} (request-id {exc.request_id})")

    if args.denied_mailbox:
        for denied in args.denied_mailbox:
            try:
                await sender.send(
                    to=args.to,
                    subject="QNSC KB mail check (must be refused)",
                    text="This must not arrive.",
                    mailbox=denied,
                )
                report(
                    "DENIED",
                    False,
                    f"sendMail AS {denied} was ACCEPTED: the app can send as other mailboxes",
                )
            except email.MailSendError as exc:
                report(
                    "DENIED",
                    exc.status == 403,
                    f"as {denied}: HTTP {exc.status} {exc.code} (request-id {exc.request_id})"
                    + ("" if exc.status == 403 else "; expected 403"),
                )
    else:
        print("SKIP  DENIED  no --denied-mailbox given; the scope has NOT been proven")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--cert", required=True, help="PEM file with the certificate")
    parser.add_argument("--key", required=True, help="PEM file with the private key")
    parser.add_argument(
        "--sender", required=True, help="the mailbox the app is scoped to"
    )
    parser.add_argument("--to", required=True, help="recipient of the test message")
    parser.add_argument(
        "--denied-mailbox",
        action="append",
        help="a mailbox OUTSIDE the scope (repeatable); sending as it must be refused with 403. Always include the OTHER environment's mailbox",
    )
    parser.add_argument(
        "--authority",
        help="override https://login.microsoftonline.com (local fake Microsoft)",
    )
    parser.add_argument(
        "--graph-api",
        help="override https://graph.microsoft.com/v1.0 (local fake Microsoft)",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
