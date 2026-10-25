// qnsc-kb · develop
//
// Values only. The whole stack lives in ../../modules/stack, so develop and production
// cannot drift structurally — only the numbers below differ. Develop leans on cheap,
// interruptible infrastructure (Fargate Spot, a small database, short log retention);
// production takes the durable settings.
terraform {
  required_version = ">= 1.9"
  required_providers {
    aws        = { source = "hashicorp/aws", version = "~> 5.0" }
    cloudflare = { source = "cloudflare/cloudflare", version = "~> 4.0" }
  }

  backend "s3" {
    bucket         = "qnsc-tofu-state"
    key            = "qnsc-kb/develop/terraform.tfstate"
    region         = "ap-southeast-1"
    encrypt        = true
    dynamodb_table = "qnsc-tofu-locks"
  }
}

provider "aws" {
  region = "ap-southeast-1"
  default_tags {
    tags = {
      Project     = "qnsc-kb"
      Environment = "develop"
      ManagedBy   = "opentofu"
    }
  }
}

// Reads TF_VAR_cloudflare_api_token. Pages and DNS resources are count-gated on the
// account id, so this stack applies before Cloudflare exists.
provider "cloudflare" {
  api_token = var.cloudflare_api_token != "" ? var.cloudflare_api_token : null
}

// ============================================================================
// AWS EXIT (E1, 2026-10-31): this develop stack is being destroyed.
//
// Provider blocks + backend stay byte-identical; the stack module is removed
// so `tofu plan` shows the full destroy set — a deleted directory plans
// nothing and proves nothing.
//
// Order of operations (product-docs/platform/runbooks/aws-exit-teardown.md):
//   1. copy-out checklist done (secrets -> kv-qnsc-dev incl. the two
//      graph-mail certificate secrets)
//   2. final manual RDS snapshot taken and verified `available`
//      (kb dev is NOT in the disposable set)
//   3. merge with the `allow-dns-destroy` label (dev records + tunnel)
//   4. `infra/live/_shared` is destroyed by the follow-up PR, AFTER this one
// ============================================================================
