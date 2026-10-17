"""The mail wiring in Terraform, asserted on the text (no AWS or OpenTofu needed to run these).

What they pin, each one a review finding on kb#215:
  * the mail secrets are gated on mail_graph_client_id, so merging before the owner's steps references nothing
    that does not exist yet;
  * the certificate and key reach the WORKER container only;
  * (this branch is the CONTRACT half) develop has no provider override and no SES send grant remains;
  * the three mail alarms exist.
"""

from __future__ import annotations

import re
from pathlib import Path

INFRA = Path(__file__).resolve().parents[2] / "infra"
STACK = (INFRA / "modules" / "stack" / "main.tf").read_text()
DEVELOP = (INFRA / "live" / "develop" / "main.tf").read_text()
DEV_VARS = (INFRA / "live" / "develop" / "variables.tf").read_text()
PROD_VARS = (INFRA / "live" / "prod" / "variables.tf").read_text()


def block(text: str, start: str) -> str:
    """The brace-balanced block that begins at ``start``."""
    i = text.index(start)
    depth, j = 0, text.index("{", i)
    for k in range(j, len(text)):
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if depth == 0:
            return text[i : k + 1]
    raise AssertionError(start)


def test_the_switch_is_the_client_id():
    assert re.search(r'mail_graph_enabled\s*=\s*var\.email_provider == "graph" && var\.mail_graph_client_id != ""', STACK)


def test_the_bundle_keys_are_created_only_when_the_switch_is_on():
    assert re.search(r'local\.mail_graph_enabled \? \{\s*"mail-graph-certificate"', STACK)
    assert 'var.email_provider == "graph" ? {' not in STACK


def test_mail_secrets_are_not_in_the_shared_secret_list():
    common = block(STACK, "common_secrets = concat(")
    assert "MAIL_GRAPH" not in common and "mail-graph" not in common


def test_mail_secrets_are_gated_and_contain_exactly_the_two_pems():
    start = STACK.index("  mail_secrets = ")
    mail = STACK[start : STACK.index("] : []", start)]
    assert "local.mail_graph_enabled" in mail
    assert re.findall(r'name = "(\w+)"', mail) == ["MAIL_GRAPH_CERT_PEM", "MAIL_GRAPH_PRIVATE_KEY_PEM"]


def test_only_the_worker_container_receives_them():
    assert STACK.count("local.mail_secrets") == 1  # the worker module and nowhere else
    worker = block(STACK, 'module "worker"')
    assert "secrets          = concat(local.common_secrets, local.mail_secrets)" in worker
    api = block(STACK, 'module "api"')
    assert "mail_secrets" not in api and "secrets = local.common_secrets" in api
    beat = STACK[STACK.index('name      = "beat"') : STACK.index('name      = "beat"') + 900]
    assert "mail_secrets" not in beat and "local.common_secrets" in beat


def test_develop_has_no_provider_override_and_there_is_no_ses_grant_any_more():
    """The contract half of expand/contract (kb#216): develop is plain `graph`, and nothing grants ses:SendEmail."""
    assert "email_provider" not in DEVELOP
    assert 'resource "aws_iam_role_policy" "worker_ses_send"' not in STACK
    assert '"ses:SendEmail"' not in STACK and "var.mail_from" not in STACK and "local.mail_from" not in STACK


def test_the_three_mail_alarms_watch_the_worker_log_group():
    for line in ("Mail is not configured", "Mail certificate expires soon", "Mail certificate has expired", "Notification marked dead"):
        assert line in STACK
    filt = block(STACK, 'resource "aws_cloudwatch_log_metric_filter" "mail"')
    assert "log_group_name = local.worker_log_group" in filt
    assert "alarm_actions = [module.observability.alarm_topic_arn]" in block(STACK, 'resource "aws_cloudwatch_metric_alarm" "mail"')


def test_each_environment_keeps_its_own_mailbox_rule():
    assert 'can(regex("-dev@", var.microsoft_graph_sender))' in DEV_VARS
    assert '!can(regex("-dev@", var.microsoft_graph_sender))' in PROD_VARS
