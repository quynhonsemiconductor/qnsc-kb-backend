# qnsc-kb · shared
#
# Everything that is per-PRODUCT rather than per-environment: the ECR repositories the
# deploy pipeline pushes to, and every GitHub OIDC role that pipeline assumes. Applied
# on both a main push and a release tag (see .github/workflows/infra-apply.yml), and
# always BEFORE the environment stacks — nothing else can assume a role until the roles
# exist.
#
# The frontend (qnsc-kb-frontend) deploys to Cloudflare Pages and needs no AWS role, so
# it appears nowhere in this file. Only qnsc-kb-backend touches AWS.
terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }

  backend "s3" {
    bucket         = "qnsc-tofu-state"
    key            = "qnsc-kb/shared/terraform.tfstate"
    region         = "ap-southeast-1"
    encrypt        = true
    dynamodb_table = "qnsc-tofu-locks"
  }
}

provider "aws" {
  region = "ap-southeast-1"
  default_tags {
    tags = {
      Project   = "qnsc-kb"
      ManagedBy = "opentofu"
      Layer     = "shared"
    }
  }
}

locals {
  github_org = var.github_org
}

data "aws_caller_identity" "current" {}

# ── Read shared platform outputs from qnsc-infra bootstrap ───────────────────
# Gives us: kms_key_arn, artifacts_bucket_name, oidc_provider_arn.
# Dependency: qnsc-infra/live/bootstrap must be applied before this stack. The OIDC
# provider in particular is an ACCOUNT SINGLETON — AWS permits one per issuer URL —
# so this product must consume it and must never create its own.
data "terraform_remote_state" "platform" {
  backend = "s3"
  config = {
    bucket = "qnsc-tofu-state"
    key    = "platform/bootstrap/terraform.tfstate"
    region = "ap-southeast-1"
  }
}

# ── ECR repositories ──────────────────────────────────────────────────────────
# Three, matching the three targets of the repo-root Dockerfile: api, worker,
# migrator. `beat` is deliberately absent — Celery beat runs as a second container
# off the WORKER image with its own command, so it needs no image of its own.
module "ecr" {
  source = "git::https://github.com/QNSC-VN/qnsc-tf-modules.git//modules/ecr?ref=ecr-v2.0.0"

  # Lower than the module defaults (30 releases / 20 builds), but for a narrower reason
  # than "the images are big".
  #
  # These images ARE big — the api and worker carry ONNX Runtime, paddle and the baked bge-m3
  # weights — but ECR bills unique LAYERS, and the `deps` and `model-cache` stages are
  # shared across all three images and unchanged between builds unless poetry.lock or the
  # model changes. The per-build delta is the application COPY layer, which is small. So
  # steady-state storage is roughly one ~4-5 GB layer set plus deltas, and multiplying
  # image size by keep-count overstates it by an order of magnitude.
  #
  # What these counts actually insure against is the case where that assumption breaks: a
  # dependency bump invalidates the shared layers, and every build after it carries its
  # own multi-GB copy until the old ones expire. 10 and 5 bound that without losing
  # anything anyone reads — a build older than the last ten is not something we roll back
  # to, we rebuild. (The one period that assumption DID break was the ONNX migration
  # itself: three distinct layer generations in two days — torch-era, transition with
  # both runtimes, onnx-only — and the torch-era tags age out of these counts on their
  # own within a few more builds.)
  #
  # Re-run `aws ecr start-lifecycle-policy-preview` (a dry run) before changing these; it
  # is the only way to see what a policy will delete.
  keep_release_count = 10
  keep_build_count   = 5

  repository_names     = ["qnsc-kb-api", "qnsc-kb-worker", "qnsc-kb-migrator"]
  image_tag_mutability = "MUTABLE" # allows re-tagging :latest
  kms_key_arn          = data.terraform_remote_state.platform.outputs.kms_key_arn
  tags                 = { Layer = "shared" }
}

# ── GitHub OIDC ───────────────────────────────────────────────────────────────
# Owns every qnsc-kb AWS role: per-environment deploy, ECR push, infra plan/apply.
# Keyless — no access keys exist for CI to leak.
#
# v3.0.1, and both parts of that version matter.
#
# v3 adds ssm:DescribeParameters to the deploy role, which the shared deploy workflow's
# secret preflight needs to verify that SSM SecureString parameters were populated —
# metadata only (names and version numbers, never values). rally still pins v2.1.0.
#
# v3.0.1 trusts BOTH shapes of the GitHub OIDC subject, and THIS REPOSITORY REQUIRES IT.
# GitHub issues newer repositories an ID-augmented subject:
#
#   repo:QNSC-VN@297362956/qnsc-kb-backend@1312007613:pull_request
#
# while older ones like rally emit `repo:QNSC-VN/rally:...`. Nobody configures this — it
# is fixed by GitHub at repository creation, and both repos report identical, default
# settings. On v3.0.0 every job here failed with "Not authorized to perform
# sts:AssumeRoleWithWebIdentity", which names neither the subject nor the mismatch; the
# presented claim is only visible in CloudTrail's userIdentity.userName.
module "iam_oidc" {
  source = "git::https://github.com/QNSC-VN/qnsc-tf-modules.git//modules/iam-oidc?ref=iam-oidc-v3.0.1"

  product           = "qnsc-kb"
  github_org        = local.github_org
  oidc_provider_arn = data.terraform_remote_state.platform.outputs.oidc_provider_arn

  environments = {
    develop = {
      allowed_subjects = [
        "repo:${local.github_org}/qnsc-kb-backend:ref:refs/heads/main",
        "repo:${local.github_org}/qnsc-kb-backend:environment:develop"
      ]
    }
    production = {
      allowed_subjects = [
        "repo:${local.github_org}/qnsc-kb-backend:ref:refs/heads/main",
        "repo:${local.github_org}/qnsc-kb-backend:ref:refs/tags/v*",
        "repo:${local.github_org}/qnsc-kb-backend:environment:production"
      ]
    }
  }

  # Both are the BACKEND repo: qnsc-kb splits frontend and backend across two repos,
  # but infra lives beside the backend (infra/ in this repo), exactly as rally's lives
  # in its monorepo. The frontend repo is absent on purpose — it deploys to Cloudflare
  # Pages and assumes no AWS role, so listing it would grant access nothing uses.
  app_repo_names         = ["qnsc-kb-backend"]
  infra_repo_name        = "qnsc-kb-backend"
  ecr_repository_pattern = "qnsc-kb-*"
  ecs_passrole_pattern   = "qnsc-kb-*" # shared ecs-service names roles <cluster>-<service>-task
  tags                   = { Layer = "shared" }

  # infra_plan_subjects / infra_apply_subjects: the infra-apply jobs run in the
  # shared/develop/production GitHub Environments, which match the module defaults —
  # no override needed.

  # Blast-radius guardrail: an explicit Deny on this product's infra-apply role, so a
  # buggy qnsc-kb apply cannot destroy the platform's foundations (state bucket, lock
  # table, OIDC provider, CMK, artifacts bucket) or mint IAM users. Those are owned by
  # qnsc-infra bootstrap and shared with every other product — the one place where a
  # mistake here would take rally and opshub down with it.
  infra_apply_guardrail = {
    state_bucket_arn     = "arn:aws:s3:::qnsc-tofu-state"
    lock_table_arn       = "arn:aws:dynamodb:ap-southeast-1:${data.aws_caller_identity.current.account_id}:table/qnsc-tofu-locks"
    oidc_provider_arn    = data.terraform_remote_state.platform.outputs.oidc_provider_arn
    kms_key_arn          = data.terraform_remote_state.platform.outputs.kms_key_arn
    artifacts_bucket_arn = data.terraform_remote_state.platform.outputs.artifacts_bucket_arn
  }
}

# ── RDS wake guard — develop deploy role only ────────────────────────────────
# develop runs at min_count = 0 with its database STOPPED off-hours (idle_schedule in
# live/develop), so a deploy landing on a sleeping environment must be able to start it.
# The shared deploy reusable's `ensure_rds` step does exactly that, and this grant is
# what lets it.
#
# Start and Describe only — never Stop. Stopping is the scheduler's job, under its own
# narrowly-scoped role; a deploy role that can stop a database is a deploy that can
# cause an outage.
#
# Scoped to develop and deliberately ABSENT from the production role: production is
# meant to be running, so a deploy needing to start it is an anomaly that should be
# loud rather than silently handled.
#
# The ARN is built from account + region + a fixed identifier rather than looked up
# with a data source. A lookup fails hard whenever the instance does not exist yet —
# a first apply, or a full teardown/redeploy — which would make this stack unable to
# apply independently of develop's RDS lifecycle. A string needs no resource.
locals {
  qnsc_kb_develop_rds_arn = "arn:aws:rds:ap-southeast-1:${data.aws_caller_identity.current.account_id}:db:qnsc-kb-develop"
}

resource "aws_iam_role_policy" "deploy_rds_dev_guard" {
  name = "qnsc-kb-deploy-develop-rds-guard"
  role = split("/", module.iam_oidc.deploy_role_arns["develop"])[1]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "RDSDevGuard"
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:StartDBInstance",
        ]
        Resource = local.qnsc_kb_develop_rds_arn
      }
    ]
  })
}
