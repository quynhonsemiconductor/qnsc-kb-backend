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
  type = string
  # A placeholder that can never be a real tag, NOT "latest". `latest` does not exist in the immutable ECR
  # repositories (only <sha> and release tags are pushed), so a baseline task definition carrying it leaves any
  # container the deploy does not rewrite unable to start; that took the develop worker down. The module
  # rejects "latest" outright. This default only ever applies to a plan with no TF_VAR_image_tag (a PR plan, a
  # local plan), which never deploys anything.
  default = "set-by-the-release-never-a-real-tag"

  description = <<-EOT
    Container image tag deployed to production.

    infra-apply.yml passes TF_VAR_image_tag = the release tag that triggered the apply,
    so the value above is only a fallback for a plan with none (a pull-request plan or a
    local one). It is deliberately not a real tag, and never "latest": that tag does not
    exist in the immutable repositories, and the module refuses it.
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

variable "microsoft_graph_sender" {
  type        = string
  default     = ""
  description = <<-EOT
    Mailbox that invitations and password resets are sent FROM, e.g. "no-reply@qnsc.vn".

    Empty means outbound mail is DEAD, silently: ENVIRONMENT is pinned to "production", so
    the development FakeEmailSender is never selected and the Graph sender raises before any
    HTTP call. Invitations and password resets queue and retry every 30 seconds forever
    without arriving, and validate_production() does not catch it.

    Requires the mailbox to exist in the tenant and the Entra app to hold Mail.Send
    application permission with admin consent.
  EOT
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

variable "alarm_emails" {
  type = list(string)
  # devops@qnsc.vn is an M365 SHARED MAILBOX, deliberately, not an alias on a person.
  # Recipients are managed in the admin centre rather than here, and the address survives
  # any individual leaving.
  #
  # ARMED 2026-09-12, ahead of this environment's first apply, so it is never live without
  # a notification path — the failure mode found across every other stack on that date.
  default = ["devops@qnsc.vn"]

  description = <<-EOT
    Addresses subscribed to the alarm SNS topic. Each subscription must be confirmed
    from the email itself — Terraform creates it as `pending confirmation` and cannot
    complete it, so an unconfirmed address is silently no alerting at all.
  EOT
}
