"""The mail wiring in Terraform, asserted on the text (no AWS or OpenTofu needed to run these).

What they pin, each one a review finding on kb#215:
  * the mail credential is a STANDALONE secret (one PEM), injected only once mail_graph_client_id is set, so
    merging before the owner's steps references nothing that has no value yet;
  * it reaches the WORKER container only, is granted to the worker's execution role only, no task role can read
    any secret, and a resource policy denies everyone else;
  * develop stays on SES until the client id is set;
  * the SES send grant is NOT keyed on the provider (expand/contract: it outlives the switch to Graph);
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


def test_the_mail_credential_is_not_a_key_of_the_shared_bundle():
    """The bundle's ARN is on the API's execution role too, so a key inside it is readable by the API's role."""
    names = STACK[STACK.index("secret_names ="):STACK.index('// NOTE: "tunnel-token"')]
    assert "mail-graph" not in names and "MAIL_GRAPH" not in names
    assert 'module.secrets.secret_arns["mail-graph' not in STACK


def test_the_mail_credential_is_a_standalone_secret_created_before_the_client_id_is_set():
    secret = block(STACK, 'resource "aws_secretsmanager_secret" "mail_graph"')
    assert "count = local.mail_secret_exists ? 1 : 0" in secret
    assert 'mail_secret_exists = var.email_provider != "fake"' in STACK  # develop is "ses" until its client id is set
    assert "count = local.mail_secret_exists ? 1 : 0" in block(STACK, 'resource "aws_secretsmanager_secret_policy" "mail_graph"')
    assert 'name                    = "${var.product}/${var.env}/mail-graph-certificate"' in secret
    assert "kms_key_id" in secret
    # the container is not gated on the client id (the owner must be able to put the value first); injection is
    assert "mail_graph_client_id" not in secret
    assert "aws_secretsmanager_secret_version\" \"mail" not in STACK  # the value never enters state


def test_nobody_but_the_worker_execution_role_can_read_it():
    policy = block(STACK, 'resource "aws_secretsmanager_secret_policy" "mail_graph"')
    assert 'Effect    = "Deny"' in policy and 'Principal = "*"' in policy
    assert 'Action    = "secretsmanager:GetSecretValue"' in policy
    assert 'ArnNotEquals = { "aws:PrincipalArn" = module.worker.execution_role_arn }' in policy
    assert "module.api" not in policy  # not the api's roles


def test_no_task_role_in_the_stack_can_read_any_secret():
    """ecs-service grants a task role secret access only through task_secret_arns; this stack must never pass it."""
    for tf in INFRA.rglob("*.tf"):
        text = re.sub(r"(//|#)[^\n]*", "", tf.read_text())  # comments may name it
        assert "task_secret_arns" not in text, tf
        assert "task_ssm_parameter_arns" not in text, tf


def test_mail_secrets_are_gated_and_are_exactly_the_one_pem():
    start = STACK.index("  mail_secrets = ")
    mail = STACK[start : STACK.index("] : []", start)]
    assert "local.mail_graph_enabled" in mail
    assert re.findall(r'name = "(\w+)"', mail) == ["MAIL_GRAPH_PEM"]
    assert "aws_secretsmanager_secret.mail_graph[0].arn" in mail


def test_only_the_worker_container_receives_it_and_only_the_worker_execution_role_is_granted_it():
    assert STACK.count("local.mail_secrets") == 1  # the worker module and nowhere else
    worker = block(STACK, 'module "worker"')
    assert "secrets          = concat(local.common_secrets, local.mail_secrets)" in worker
    assert "aws_secretsmanager_secret.mail_graph[0].arn" in worker  # its execution role's secret_arns
    api = block(STACK, 'module "api"')
    assert "mail_graph" not in api and "mail_secrets" not in api and "secrets = local.common_secrets" in api
    beat = STACK[STACK.index('name      = "beat"') : STACK.index('name      = "beat"') + 900]
    assert "mail_secrets" not in beat and "local.common_secrets" in beat
    assert "MAIL_GRAPH" not in block(STACK, "common_secrets = concat(")


def test_develop_stays_on_ses_until_the_client_id_is_set():
    assert 'email_provider        = var.mail_graph_client_id != "" ? "graph" : "ses"' in DEVELOP


def test_the_ses_grant_is_not_keyed_on_the_provider():
    grant = block(STACK, 'resource "aws_iam_role_policy" "worker_ses_send"')
    assert 'count = var.mail_from_email != "" ? 1 : 0' in grant
    assert "email_provider" not in grant
    assert 'mail_from_domain = var.mail_from_email != "" ?' in STACK


def test_the_three_mail_alarms_watch_the_worker_log_group():
    for line in ("Mail is not configured", "Mail certificate expires soon", "Mail certificate has expired",
                 "Notification marked dead", "Mail sender refused"):
        assert line in STACK
    filt = block(STACK, 'resource "aws_cloudwatch_log_metric_filter" "mail"')
    assert "log_group_name = local.worker_log_group" in filt
    assert "alarm_actions = [module.observability.alarm_topic_arn]" in block(STACK, 'resource "aws_cloudwatch_metric_alarm" "mail"')


def test_each_environment_keeps_its_own_mailbox_rule():
    # lower(): noreply-kb-DEV@ must not slip past either rule
    assert 'can(regex("-dev@", lower(var.microsoft_graph_sender)))' in DEV_VARS
    assert '!can(regex("-dev@", lower(var.microsoft_graph_sender)))' in PROD_VARS
