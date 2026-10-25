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

// ============================================================================
// AWS EXIT (E1, 2026-10-31): the shared stack is being destroyed.
//
// ECR (image fallback is GHCR), the GitHub OIDC provider + deploy roles go.
// kb has no production AWS state (never applied), so nothing depends on this
// stack after the develop destroy lands.
//
// MERGE ORDER: strictly AFTER the develop-stack destroy PR. CI deploys to
// kb develop stop with this merge. Draft until the owner's go.
// ============================================================================
