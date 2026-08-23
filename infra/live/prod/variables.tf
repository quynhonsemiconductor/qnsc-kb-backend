// PUBLIC identifiers live here, in git, for the same reason as in live/develop: the
// infra-plan job has no `environment:` context, so an environment-scoped Actions
// variable resolves to "" during plan and the plan lies about count-gated resources.
//
// Only the Cloudflare API token and the release image tag come from CI.

variable "cloudflare_api_token" {
  type        = string
  sensitive   = true
  default     = ""
  description = "Cloudflare API token with Pages + DNS edit scope, from TF_VAR_cloudflare_api_token."
}

variable "image_tag" {
  type    = string
  default = "latest"

  description = <<-EOT
    Container image tag deployed to production.

    infra-apply.yml passes TF_VAR_image_tag = the release tag that triggered the apply,
    so the value below is only a fallback for a local plan. Production should never
    actually run "latest": that tag follows develop, so an apply that used it would
    quietly move production onto an untested build.
  EOT
}

variable "cloudflare_account_id" {
  type    = string
  default = ""

  description = <<-EOT
    Cloudflare account that owns the Pages project. NOT a secret.

    Supplied by CI as TF_VAR_cloudflare_account_id from the ORG-level
    CLOUDFLARE_ACCOUNT_ID variable, in BOTH the plan and the apply. That parity is the
    point: the Pages and DNS modules are count-gated on this value, so a plan that
    cannot see it reports a phantom create or destroy of both — and a real destroy is
    easy to miss among phantoms.

    Org-level rather than environment-scoped for the same reason: an environment-scoped
    variable resolves to "" in a plan job, which has no `environment:` context.

    The empty default is only for a local plan without it, which is a valid
    intermediate state — the AWS half of the stack applies on its own and Cloudflare
    arrives in a later apply.
  EOT
}


variable "microsoft_client_id" {
  type        = string
  default     = ""
  description = "Entra application (client) ID for the Microsoft connector. A public identifier."
}

variable "google_client_id" {
  type        = string
  default     = ""
  description = "Google OAuth client ID for the Google connector. A public identifier."
}

variable "allowed_email_domains" {
  type        = list(string)
  default     = ["qnsc.vn"]
  description = <<-EOT
    Restricts which email domains may be registered. Empty accepts ANY, which under
    company-scoped RLS means an admin-created account on another domain quietly becomes a
    second tenant whose rows nobody else can see.
  EOT
}

variable "entra_admin_emails" {
  type    = list(string)
  default = ["nghiavt@qnsc.vn", "sinhhpt@qnsc.vn", "quangld@qnsc.vn", "hieuvbm@qnsc.vn"]

  description = <<-EOT
    Provisioned as GLOBAL administrators on their FIRST Entra sign-in, rather than Staff.

    Read only at account creation, so it never overrides a role changed later in the admin
    UI, and removing an address does not demote anyone — promote and demote in the UI once
    an account exists.
  EOT
}

variable "alarm_emails" {
  type    = list(string)
  default = []

  description = <<-EOT
    Addresses subscribed to the alarm SNS topic. Each subscription must be confirmed
    from the email itself — Terraform creates it as `pending confirmation` and cannot
    complete it, so an unconfirmed address is silently no alerting at all.
  EOT
}
