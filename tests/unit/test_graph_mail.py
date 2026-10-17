"""Application email through Microsoft Graph with a certificate (no client secret).

Everything Microsoft is replaced by an httpx MockTransport that behaves the way the real services do where it
matters: the token endpoint VERIFIES the client assertion's signature against the certificate it was given, and
Graph answers 202 / 403 / 401 the way Exchange does. Credentials are generated at runtime; no key material is
committed (and a scanner would rightly object).
"""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
import structlog
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from src.core.config import Settings
from src.services import email

TENANT = "11111111-2222-4333-8444-555555555555"
CLIENT = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
SENDER = "noreply-kb-dev@qnsc.vn"  # develop's own mailbox
PROD_MAILBOX = "noreply-kb@qnsc.vn"


def make_credential(*, days=365, started=None, key=None):
    key = key or rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = dt.datetime.now(dt.timezone.utc)
    start = started or now - dt.timedelta(days=1)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "qnsc-kb-mail-test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start).not_valid_after(start + dt.timedelta(days=days))
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    key_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    return key, cert, cert_pem, key_pem


class FakeMicrosoft:
    """The token endpoint and Graph, as one MockTransport."""

    def __init__(self, cert, *, graph_status=202, graph_error=None, token_error=None, expires_in=3600, allowed_mailbox=None):
        self.cert = cert
        self.allowed_mailbox = allowed_mailbox  # the one mailbox this app's Exchange RBAC scope covers
        self.graph_status, self.graph_error, self.token_error, self.expires_in = graph_status, graph_error, token_error, expires_in
        self.token_calls: list[dict] = []
        self.graph_calls: list[httpx.Request] = []
        self.issued = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_calls.append({"url": str(request.url), "form": form})
            if self.token_error:
                return httpx.Response(400, json=self.token_error)
            # What Entra does: verify the assertion against the REGISTERED certificate's public key.
            claims = jwt.decode(
                form["client_assertion"], self.cert.public_key(), algorithms=["RS256"],
                audience=f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token",
            )
            assert claims["iss"] == claims["sub"] == CLIENT == form["client_id"]
            self.issued += 1
            return httpx.Response(200, json={"access_token": f"graph-token-{self.issued}", "expires_in": self.expires_in})
        self.graph_calls.append(request)
        headers = {"request-id": "req-123"}
        mailbox = request.url.path.split("/")[3]
        if self.allowed_mailbox and mailbox != self.allowed_mailbox:  # what Exchange RBAC for Applications does
            return httpx.Response(403, headers=headers, json={"error": {"code": "ErrorAccessDenied", "message": "Access is denied. Check credentials and try again."}})
        if self.graph_status == 202:
            return httpx.Response(202, headers=headers)
        return httpx.Response(self.graph_status, headers=headers, json=self.graph_error or {"error": {"code": "ErrorAccessDenied", "message": "Access is denied. Check credentials and try again."}})


@pytest.fixture
def world(monkeypatch):
    key, cert, cert_pem, key_pem = make_credential()
    settings = Settings(
        _env_file=None, ENVIRONMENT="production", MICROSOFT_TENANT_ID=TENANT, MAIL_GRAPH_CLIENT_ID=CLIENT,
        MAIL_GRAPH_CERT_PEM=cert_pem, MAIL_GRAPH_PRIVATE_KEY_PEM=key_pem, MICROSOFT_GRAPH_SENDER=SENDER,
    )
    monkeypatch.setattr(email, "settings", settings)
    email._credential_cache.clear()
    email.reset_token_cache()
    yield {"key": key, "cert": cert, "cert_pem": cert_pem, "key_pem": key_pem, "settings": settings}
    email._credential_cache.clear()
    email.reset_token_cache()


def send(world_or_fake, **kw):
    sender = email.MicrosoftGraphEmailSender(transport=httpx.MockTransport(world_or_fake))
    args = dict(to="user@qnsc.vn", subject="Welcome", text="plain", html="<p>hi</p>")
    args.update(kw)
    return asyncio.run(sender.send(**args))


# ── the client assertion ─────────────────────────────────────────────────────────────────────
def test_the_client_assertion_is_signed_with_the_certificate_key_and_names_its_thumbprint(world):
    assertion = email.build_client_assertion(tenant=TENANT, client_id=CLIENT, now=1_700_000_000)

    header = jwt.get_unverified_header(assertion)
    der = world["cert"].public_bytes(serialization.Encoding.DER)
    b64 = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    assert header["alg"] == "RS256"
    assert header["x5t#S256"] == b64(hashlib.sha256(der).digest())
    assert header["x5t"] == b64(hashlib.sha1(der).digest())

    claims = jwt.decode(assertion, world["cert"].public_key(), algorithms=["RS256"],
                        audience=f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token",
                        options={"verify_exp": False, "verify_nbf": False})
    assert claims["iss"] == claims["sub"] == CLIENT
    assert claims["exp"] == 1_700_000_600 and claims["nbf"] == 1_700_000_000 - 30
    assert claims["jti"], "every assertion needs a unique id, or Entra rejects a replay"


def test_two_assertions_never_share_an_id(world):
    a = jwt.decode(email.build_client_assertion(tenant=TENANT, client_id=CLIENT), options={"verify_signature": False})
    b = jwt.decode(email.build_client_assertion(tenant=TENANT, client_id=CLIENT), options={"verify_signature": False})
    assert a["jti"] != b["jti"]


def test_an_assertion_signed_with_some_other_key_is_rejected_by_the_token_endpoint(world, monkeypatch):
    other_key, _, other_cert_pem, other_key_pem = make_credential()
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={
        "MAIL_GRAPH_CERT_PEM": other_cert_pem, "MAIL_GRAPH_PRIVATE_KEY_PEM": other_key_pem}))
    email._credential_cache.clear()
    fake = FakeMicrosoft(world["cert"])  # Entra knows the ORIGINAL certificate
    with pytest.raises(jwt.InvalidSignatureError):
        send(fake)


# ── sending ──────────────────────────────────────────────────────────────────────────────────
def test_a_send_posts_sendmail_as_the_configured_mailbox_with_a_bearer_token(world):
    fake = FakeMicrosoft(world["cert"])
    send(fake)

    request = fake.graph_calls[0]
    assert request.method == "POST"
    assert str(request.url) == f"https://graph.microsoft.com/v1.0/users/{SENDER}/sendMail"
    assert request.headers["authorization"] == "Bearer graph-token-1"
    body = json.loads(request.content)
    assert body["saveToSentItems"] is False
    assert body["message"]["subject"] == "Welcome"
    assert body["message"]["body"] == {"contentType": "HTML", "content": "<p>hi</p>"}
    assert body["message"]["toRecipients"] == [{"emailAddress": {"address": "user@qnsc.vn"}}]


def test_plain_text_is_sent_as_text(world):
    fake = FakeMicrosoft(world["cert"])
    send(fake, html=None)
    assert json.loads(fake.graph_calls[0].content)["message"]["body"] == {"contentType": "Text", "content": "plain"}


def test_the_token_request_is_a_certificate_client_credentials_grant_with_no_secret(world):
    fake = FakeMicrosoft(world["cert"])
    send(fake)

    call = fake.token_calls[0]
    assert call["url"] == f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
    form = call["form"]
    assert form["grant_type"] == "client_credentials"
    assert form["scope"] == "https://graph.microsoft.com/.default"
    assert form["client_assertion_type"] == "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
    assert "client_secret" not in form, "the mail app has no client secret"


def test_the_token_is_reused_across_sends(world):
    fake = FakeMicrosoft(world["cert"])
    for _ in range(3):
        send(fake)
    assert len(fake.token_calls) == 1 and len(fake.graph_calls) == 3


def test_a_token_about_to_expire_is_replaced(world):
    fake = FakeMicrosoft(world["cert"], expires_in=60)  # inside the 120 s refresh margin
    send(fake)
    send(fake)
    assert len(fake.token_calls) == 2


def test_a_401_drops_the_cached_token_so_the_next_send_gets_a_fresh_one(world):
    fake = FakeMicrosoft(world["cert"], graph_status=401, graph_error={"error": {"code": "InvalidAuthenticationToken", "message": "expired"}})
    with pytest.raises(email.MailSendError) as caught:
        send(fake)
    assert caught.value.status == 401
    fake.graph_status = 202
    send(fake)
    assert len(fake.token_calls) == 2


# ── failure: the 403 the RBAC scope exists to produce ────────────────────────────────────────
def test_a_403_from_graph_is_reported_with_its_code_and_request_id(world):
    fake = FakeMicrosoft(world["cert"], graph_status=403)
    with pytest.raises(email.MailSendError) as caught:
        send(fake, mailbox="ceo@qnsc.vn")  # a mailbox outside the RBAC scope

    error = caught.value
    assert error.status == 403 and error.code == "ErrorAccessDenied" and error.request_id == "req-123"
    assert "ceo@qnsc.vn" in str(error) and "403" in str(error)
    assert fake.graph_calls[0].url.path == "/v1.0/users/ceo@qnsc.vn/sendMail"


def test_no_error_text_ever_contains_credential_material(world):
    fake = FakeMicrosoft(world["cert"], graph_status=403)
    with pytest.raises(email.MailSendError) as caught:
        send(fake)
    text = str(caught.value)
    assert "BEGIN" not in text and "graph-token" not in text and "Bearer" not in text
    assert world["key_pem"].splitlines()[1] not in text


def test_a_rejected_assertion_surfaces_the_token_endpoint_error(world):
    fake = FakeMicrosoft(world["cert"], token_error={"error": "invalid_client", "error_description": "AADSTS700027: Client assertion contains an invalid signature."})
    with pytest.raises(email.MailSendError) as caught:
        send(fake)
    assert caught.value.status == 400 and caught.value.code == "invalid_client"
    assert "AADSTS700027" in str(caught.value) and "Requesting the Graph token" in str(caught.value)
    assert fake.graph_calls == [], "no mail request may follow a failed token request"


# ── misconfiguration is named, early, and without secrets ────────────────────────────────────
@pytest.mark.parametrize("override,needle", [
    ({"MICROSOFT_GRAPH_SENDER": ""}, "MICROSOFT_GRAPH_SENDER"),
    ({"MAIL_GRAPH_CLIENT_ID": ""}, "MAIL_GRAPH_CLIENT_ID"),
    ({"MICROSOFT_TENANT_ID": "common"}, "tenant GUID"),
    ({"MAIL_GRAPH_CERT_PEM": ""}, "MAIL_GRAPH_CERT_PEM"),
    ({"MAIL_GRAPH_PRIVATE_KEY_PEM": ""}, "MAIL_GRAPH_PRIVATE_KEY_PEM"),
    ({"MAIL_GRAPH_CERT_PEM": "not a pem"}, "could not be parsed"),
])
def test_misconfiguration_fails_with_the_name_of_the_setting(world, monkeypatch, override, needle):
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update=override))
    email._credential_cache.clear()
    fake = FakeMicrosoft(world["cert"])
    with pytest.raises(email.MailSendError, match=needle):
        send(fake)
    assert fake.graph_calls == []


def test_a_certificate_that_does_not_belong_to_the_key_is_refused(world, monkeypatch):
    _, _, other_cert_pem, _ = make_credential()
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MAIL_GRAPH_CERT_PEM": other_cert_pem}))
    email._credential_cache.clear()
    with pytest.raises(email.MailSendError, match="does not belong"):
        send(FakeMicrosoft(world["cert"]))


def test_an_expired_certificate_is_refused_with_the_date(world, monkeypatch):
    _, _, cert_pem, key_pem = make_credential(days=10, started=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=100))
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MAIL_GRAPH_CERT_PEM": cert_pem, "MAIL_GRAPH_PRIVATE_KEY_PEM": key_pem}))
    email._credential_cache.clear()
    with pytest.raises(email.MailSendError, match="expired on"):
        send(FakeMicrosoft(world["cert"]))


def test_a_certificate_close_to_expiry_still_sends_and_the_daily_check_warns(world, monkeypatch):
    key, cert, cert_pem, key_pem = make_credential(days=10)
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MAIL_GRAPH_CERT_PEM": cert_pem, "MAIL_GRAPH_PRIVATE_KEY_PEM": key_pem}))
    email._credential_cache.clear()
    send(FakeMicrosoft(cert))  # a send is not blocked by a certificate that is merely close to expiry
    from src.workers import tasks

    with structlog.testing.capture_logs() as logs:
        tasks.check_mail_certificate()
    assert any(e["event"] == "Mail certificate expires soon" and e["log_level"] == "warning" for e in logs)


def test_the_daily_check_errors_on_an_expired_certificate_and_is_quiet_when_healthy(world, monkeypatch):
    from src.workers import tasks

    with structlog.testing.capture_logs() as logs:
        tasks.check_mail_certificate()
    assert [e["log_level"] for e in logs] == ["info"]

    _, _, cert_pem, _ = make_credential(days=10, started=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=100))
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MAIL_GRAPH_CERT_PEM": cert_pem}))
    with structlog.testing.capture_logs() as logs:
        tasks.check_mail_certificate()
    assert any(e["event"] == "Mail certificate has expired" and e["log_level"] == "error" for e in logs)


def test_an_elliptic_curve_key_is_refused_because_the_assertion_is_rs256(world, monkeypatch):
    ec_key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ec")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ec_key.public_key())
            .serial_number(1).not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=30))
            .sign(ec_key, hashes.SHA256()))
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={
        "MAIL_GRAPH_CERT_PEM": cert.public_bytes(serialization.Encoding.PEM).decode(),
        "MAIL_GRAPH_PRIVATE_KEY_PEM": ec_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()}))
    email._credential_cache.clear()
    with pytest.raises(email.MailSendError, match="RSA"):
        send(FakeMicrosoft(world["cert"]))


def test_pems_stored_with_literal_backslash_n_still_load(world, monkeypatch):
    escaped = {"MAIL_GRAPH_CERT_PEM": world["cert_pem"].strip().replace("\n", "\\n"),
               "MAIL_GRAPH_PRIVATE_KEY_PEM": world["key_pem"].strip().replace("\n", "\\n")}
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update=escaped))
    email._credential_cache.clear()
    fake = FakeMicrosoft(world["cert"])
    send(fake)
    assert len(fake.graph_calls) == 1


# ── provider selection ───────────────────────────────────────────────────────────────────────
def settings_for(**kw):
    return Settings(_env_file=None, **kw)


def test_graph_is_the_provider_when_the_mail_app_is_configured(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production", MAIL_GRAPH_CLIENT_ID=CLIENT))
    assert isinstance(email.get_email_sender(), email.MicrosoftGraphEmailSender)


def test_local_development_without_a_mail_app_prints_instead_of_failing(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="development"))
    assert isinstance(email.get_email_sender(), email.ConsoleEmailSender)


def test_local_development_with_a_mail_app_still_uses_graph(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="development", MAIL_GRAPH_CLIENT_ID=CLIENT))
    assert isinstance(email.get_email_sender(), email.MicrosoftGraphEmailSender)


def test_the_console_provider_is_refused_outside_development(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production", EMAIL_PROVIDER="console"))
    with pytest.raises(RuntimeError, match="local development only"):
        email.get_email_sender()


def test_the_console_provider_logs_the_message(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="development", EMAIL_PROVIDER="console"))
    sender = email.get_email_sender()
    with structlog.testing.capture_logs() as logs:
        asyncio.run(sender.send(to="a@qnsc.vn", subject="Reset", text="https://kb.example/reset?token=abc"))
    entry = next(e for e in logs if "console provider" in e["event"])
    assert entry["to"] == "a@qnsc.vn" and "token=abc" in entry["text"]


def test_a_misconfigured_deployed_environment_does_not_fall_back_to_printing(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production"))  # graph, nothing configured
    assert isinstance(email.get_email_sender(), email.MicrosoftGraphEmailSender)  # which then fails loudly on send


def test_the_test_environment_records_instead_of_sending(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="test"))
    assert isinstance(email.get_email_sender(), email.FakeEmailSender)


# ── the worker's start-up guard (the API never refuses to boot over mail) ────────────────────
def test_the_api_boots_with_no_mail_configuration_at_all():
    """validate_production() used to refuse; the worker now owns that check, so the knowledge base keeps serving."""
    from tests.unit.test_admin_bootstrap import _production_settings  # a settings object valid apart from mail

    _production_settings(MICROSOFT_GRAPH_SENDER=None, MAIL_GRAPH_CLIENT_ID=None).validate_production()


def test_no_mail_app_configured_is_reported_but_not_fatal(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production"))
    assert email.configuration_problems() == ([], True)


@pytest.mark.parametrize("missing", ["MICROSOFT_GRAPH_SENDER", "MAIL_GRAPH_CERT_PEM", "MAIL_GRAPH_PRIVATE_KEY_PEM"])
def test_a_client_id_without_the_rest_is_fatal(monkeypatch, missing):
    values = dict(ENVIRONMENT="production", MAIL_GRAPH_CLIENT_ID=CLIENT, MICROSOFT_GRAPH_SENDER=SENDER,
                  MAIL_GRAPH_CERT_PEM="cert", MAIL_GRAPH_PRIVATE_KEY_PEM="key")
    values[missing] = ""
    monkeypatch.setattr(email, "settings", settings_for(**values))
    fatal, unconfigured = email.configuration_problems()
    assert unconfigured is False and len(fatal) == 1 and missing in fatal[0]


def test_a_complete_configuration_has_no_problems(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production", MAIL_GRAPH_CLIENT_ID=CLIENT,
                                                        MICROSOFT_GRAPH_SENDER=SENDER, MAIL_GRAPH_CERT_PEM="c", MAIL_GRAPH_PRIVATE_KEY_PEM="k"))
    assert email.configuration_problems() == ([], False)


def test_console_is_fatal_outside_development_and_fine_inside(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production", EMAIL_PROVIDER="console"))
    assert email.configuration_problems()[0]
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="development", EMAIL_PROVIDER="console"))
    assert email.configuration_problems() == ([], False)


def test_other_providers_have_nothing_to_check(monkeypatch):
    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production", EMAIL_PROVIDER="fake"))
    assert email.configuration_problems() == ([], False)


def test_the_worker_refuses_to_start_on_a_fatal_problem_and_logs_an_error_for_the_unconfigured_case(monkeypatch):
    from src.workers import tasks

    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production", MAIL_GRAPH_CLIENT_ID=CLIENT))
    with pytest.raises(RuntimeError, match="Mail configuration"):
        tasks.worker_init_mail_guard()

    monkeypatch.setattr(email, "settings", settings_for(ENVIRONMENT="production"))
    with structlog.testing.capture_logs() as logs:
        tasks.worker_init_mail_guard()  # does not raise
    assert any(e["log_level"] == "error" and e["event"].startswith("Mail is not configured") for e in logs)


def test_the_secrets_are_secretstr_and_never_appear_in_a_repr():
    s = settings_for(MAIL_GRAPH_CERT_PEM="CERTBODY", MAIL_GRAPH_PRIVATE_KEY_PEM="KEYBODY")
    assert "KEYBODY" not in repr(s) and "CERTBODY" not in repr(s) and "KEYBODY" not in str(s.model_dump())


def test_a_sender_with_odd_characters_is_quoted_into_the_url(world, monkeypatch):
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MICROSOFT_GRAPH_SENDER": "a b/c?d@qnsc.vn"}))
    fake = FakeMicrosoft(world["cert"])
    send(fake)
    assert "/users/a%20b%2Fc%3Fd@qnsc.vn/sendMail" in str(fake.graph_calls[0].url)


def test_retry_after_is_parsed_capped_and_ignored_when_not_a_number(world):
    for header, expected in (("30", 30), ("999999", 3600), ("Wed, 21 Oct 2026 07:28:00 GMT", None)):
        fake = FakeMicrosoft(world["cert"], graph_status=429, graph_error={"error": {"code": "TooManyRequests", "message": "slow"}})
        original = fake.__call__

        def with_header(request, original=original, header=header):
            response = original(request)
            if request.url.host != "login.microsoftonline.com":
                response.headers["retry-after"] = header
            return response

        with pytest.raises(email.MailSendError) as caught:
            send(with_header)
        assert caught.value.retry_after == expected and caught.value.stops_the_batch is True


def test_error_classification(world):
    fake = FakeMicrosoft(world["cert"], graph_status=403)
    with pytest.raises(email.MailSendError) as caught:
        send(fake)
    assert caught.value.permanent is True and caught.value.stops_the_batch is False

    email.reset_token_cache()
    token_fail = FakeMicrosoft(world["cert"], token_error={"error": "invalid_client", "error_description": "bad"})
    with pytest.raises(email.MailSendError) as caught:
        send(token_fail)
    assert caught.value.stage == "token" and caught.value.stops_the_batch is True and caught.value.permanent is False

    transient = FakeMicrosoft(world["cert"], graph_status=500, graph_error={"error": {"code": "ErrorInternalServerError", "message": "x"}})
    with pytest.raises(email.MailSendError) as caught:
        send(transient)
    assert caught.value.permanent is False and caught.value.stops_the_batch is False


def test_the_certificate_status_reports_days_left_and_none_without_a_certificate(world, monkeypatch):
    status = email.certificate_status()
    assert status["valid"] and 360 <= status["days_left"] <= 365 and status["expired"] is False
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MAIL_GRAPH_CERT_PEM": None}))
    assert email.certificate_status() is None
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MAIL_GRAPH_CERT_PEM": "garbage"}))
    assert email.certificate_status()["valid"] is False


# ── one mailbox per environment ──────────────────────────────────────────────────────────────
def test_the_develop_app_can_send_as_its_own_mailbox_but_not_as_the_production_one(world):
    """Plan 7.4: a non-production environment never sends from the production mailbox. The scope is Exchange's
    to enforce; what the application owes is to surface the refusal as a 403, per mailbox, unmodified."""
    fake = FakeMicrosoft(world["cert"], allowed_mailbox=SENDER)

    send(fake)  # its own mailbox
    with pytest.raises(email.MailSendError) as caught:
        send(fake, mailbox=PROD_MAILBOX)

    assert caught.value.status == 403 and caught.value.code == "ErrorAccessDenied"
    assert PROD_MAILBOX in str(caught.value)
    assert [r.url.path for r in fake.graph_calls] == [f"/v1.0/users/{SENDER}/sendMail", f"/v1.0/users/{PROD_MAILBOX}/sendMail"]


def test_the_mirror_image_the_production_app_cannot_send_as_the_dev_mailbox(world, monkeypatch):
    monkeypatch.setattr(email, "settings", world["settings"].model_copy(update={"MICROSOFT_GRAPH_SENDER": PROD_MAILBOX}))
    fake = FakeMicrosoft(world["cert"], allowed_mailbox=PROD_MAILBOX)

    send(fake)
    with pytest.raises(email.MailSendError) as caught:
        send(fake, mailbox=SENDER)
    assert caught.value.status == 403


# ── scripts/graph_mail_check.py: any application role in the token fails the check ──────────────────────
def _check_script():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "graph_mail_check.py"
    spec = importlib.util.spec_from_file_location("graph_mail_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _token_with(claims: dict) -> str:
    import base64
    import json

    return "h." + base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode() + ".s"


@pytest.mark.parametrize("roles,expected_exit", [(["Mail.Send"], 1), (["Sites.Read.All"], 1), (["User.Read.All", "Mail.Read"], 1), ([], 0)])
def test_the_check_script_fails_on_any_role_claim(tmp_path, monkeypatch, capsys, roles, expected_exit):
    script = _check_script()
    (tmp_path / "c.pem").write_text("not-a-real-certificate")
    (tmp_path / "k.pem").write_text("not-a-real-key")
    monkeypatch.setattr(email, "settings", email.settings)  # the script reassigns it; monkeypatch restores it

    async def fake_token(client):
        return _token_with({"appid": CLIENT, "roles": roles} if roles else {"appid": CLIENT})

    async def fake_send(self, **kwargs):
        if kwargs.get("mailbox"):
            raise email.MailSendError("denied", status=403, code="ErrorAccessDenied")

    monkeypatch.setattr(email, "_access_token", fake_token)
    monkeypatch.setattr(email.MicrosoftGraphEmailSender, "send", fake_send)
    args = SimpleNamespace(authority=None, graph_api=None, tenant=TENANT, client_id=CLIENT, cert=str(tmp_path / "c.pem"),
                           key=str(tmp_path / "k.pem"), sender=SENDER, to="me@qnsc.vn", denied_mailbox=[PROD_MAILBOX])
    monkeypatch.setattr(email, "LOGIN_AUTHORITY", email.LOGIN_AUTHORITY)
    monkeypatch.setattr(email, "GRAPH_API", email.GRAPH_API)

    assert asyncio.run(script.run(args)) == expected_exit
    out = capsys.readouterr().out
    assert ("FAIL  SCOPE" in out) == bool(roles)
