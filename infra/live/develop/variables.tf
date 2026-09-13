// PUBLIC identifiers live here, in git, deliberately — not in Actions variables.
//
// The infra-plan job has no `environment:` context (adding one would gate every PR
// behind the production reviewer), so an environment-scoped Actions variable resolves
// to "" during plan. Values that reach apply but not plan make the plan LIE: resources
// gated on them appear as phantom creates or destroys, and a real destroy hidden among
// them is easy to miss. Held here, plan and apply see the same value.
//
// Only the Cloudflare API token — an actual credential — is passed in from CI.

variable "cloudflare_api_token" {
  type      = string
  sensitive = true
  default   = ""

  description = <<-EOT
    Cloudflare API token with Pages + DNS edit scope, from TF_VAR_cloudflare_api_token.
    Empty skips provider auth so the stack can be planned before Cloudflare is set up.
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
  default     = "dbd99dbb-d20e-4076-8f8b-75c15e733414"
  description = "Entra application (client) ID for the Microsoft connector. A public identifier. Empty leaves that connector dormant."
}

variable "microsoft_graph_sender" {
  type        = string
  default     = ""
  description = <<-EOT
    Mailbox that invitations and password resets are sent FROM, e.g. "no-reply@qnsc.vn".

    Left EMPTY deliberately until the mailbox exists and the Entra app has Mail.Send
    application permission with admin consent. While it is empty, outbound mail is dead:
    ENVIRONMENT is pinned to "production", so the development FakeEmailSender is never
    selected, and the Graph sender raises before any HTTP call. Invitations and resets will
    queue and retry every 30 seconds without ever arriving.

    Set this before relying on the invite or forgot-password flows.
  EOT
}

variable "microsoft_tenant_id" {
  type        = string
  default     = "dc0f2078-ac28-4ff2-b21a-d4b28df32361"
  description = <<-EOT
    The QNSC Entra tenant, NOT "common".

    The connector requests DELEGATED Graph scopes, so whoever completes the OAuth flow
    connects THEIR SharePoint. Under "common" that is any user in any Microsoft tenant on
    earth, and their documents would be ingested into this knowledge base. Pinning the
    tenant is what limits the flow to qnsc.vn accounts.
  EOT
}

variable "email_provider" {
  type    = string
  default = "ses"

  description = <<-EOT
    Overrides the module default ("graph") for this environment: develop sends through
    AWS SES rather than Microsoft Graph, so outbound mail does not wait on Entra admin
    consent for Mail.Send. See infra/modules/stack/variables.tf for what switching this
    requires (mail_from_email below, a verified SES identity, and sandbox/production
    access for the recipients being invited).
  EOT
}

variable "mail_from_email" {
  type    = string
  default = "no-reply@qnsc.vn"

  description = <<-EOT
    Mailbox that invitations and password resets are sent FROM under EMAIL_PROVIDER=ses.

    The domain is verified automatically (aws_sesv2_email_identity + module.dns_ses_dkim
    in modules/stack/main.tf, gated on email_provider == "ses"), so this only needs to
    change if the local part should differ. Emptying it reintroduces the dead-mail shape
    microsoft_graph_sender has above: SesEmailSender raises "MAIL_FROM_EMAIL is not
    configured" on every attempt, and deliver_notification_queue retries every 30 seconds
    without ever arriving.
  EOT
}

variable "google_client_id" {
  type        = string
  default     = ""
  description = "Google OAuth client ID for the Google connector. A public identifier. Empty leaves that connector dormant."
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
    an account exists. Add the rest of the admin team here.
  EOT
}

variable "alarm_emails" {
  type = list(string)
  # devops@qnsc.vn is an M365 SHARED MAILBOX, deliberately, not an alias on a person.
  # Recipients are managed in the admin centre rather than here, and the address survives
  # any individual leaving.
  #
  # ARMED 2026-09-12. This was `[]`, and `qnsc-kb-develop-alarms` existed in AWS with ZERO
  # subscriptions.
  default = ["devops@qnsc.vn"]

  description = <<-EOT
    Addresses subscribed to the alarm SNS topic. Each subscription must be confirmed
    from the email itself — Terraform creates it as `pending confirmation` and cannot
    complete it, so an unconfirmed address is silently no alerting at all.
  EOT
}
