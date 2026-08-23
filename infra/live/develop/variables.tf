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
  type    = list(string)
  default = []

  description = <<-EOT
    Addresses subscribed to the alarm SNS topic. Each subscription must be confirmed
    from the email itself — Terraform creates it as `pending confirmation` and cannot
    complete it, so an unconfirmed address is silently no alerting at all.
  EOT
}
