# Proposal: grant KB Admin from an Entra app role

Status: **proposal — no code in this PR**
Date: 2026-11-05

## Problem

The first global Admin is a database bootstrap (`scripts/provision_admin.py`, now fixed to
set its own RLS context). Every new environment needs that one-off run before anybody can
sign in, and the account it creates is outside the IdP's lifecycle: removing a person from
the admin list does not remove KB Admin from them.

OpsHub already models admin as an Entra app role on its sign-in app and reads the ID
token's `roles` claim. **OpsHub assigns that role to the GROUP `grp-qnsc-platform-admins`,
which this tenant cannot support: group-based assignment to an enterprise application
requires Entra ID P1 or P2, and this tenant has no P1** (Microsoft Learn, "Manage users and
groups assignment to an application": "Group-based assignment requires Microsoft Entra ID
P1 or P2 edition."
<https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/assign-user-or-group-access-portal>).
KB must therefore use **direct user assignments**, the pattern already used by
`infra/live/identity/proxmox.tf`.

## Proposal

1. **Identity (one owner apply).** `infra/live/identity/kb_prod_apps.tf`:
   - Add one app role to `azuread_application.kb_prod`: `value = "kb.admin"`,
     `display_name = "KB Administrator"`, `allowed_member_types = ["User"]`, `enabled = true`.
     The role **id is a fixed GUID committed in the Terraform** (an `azuread_application_app_role`
     resource, not a generated id) so assignments can reference it deterministically.
   - Assign the role to a **USER object-id set**, not a group — the Proxmox pattern, because
     group-to-app assignment needs P1 (above):

     ```hcl
     variable "kb_admin_object_ids" {
       type        = set(string)
       description = "Entra USER object ids granted the KB Admin app role. No P1: groups cannot be assigned."
     }

     resource "azuread_app_role_assignment" "kb_prod_admin" {
       for_each            = var.kb_admin_object_ids
       app_role_id         = azuread_application_app_role.kb_prod_admin.id
       principal_object_id = each.value
       resource_object_id  = azuread_service_principal.kb_prod.object_id
     }
     ```

   - **Adding or removing an admin is one IaC edit:** add/remove the person's user object id
     in `kb_admin_object_ids` and apply. The same people must be kept in the Proxmox list
     (`proxmox_admin_object_ids`) and the Entra group `grp-qnsc-platform-admins` (Cloudflare
     Access and Proxmox key on them); the identity owner should move both apps to ONE shared
     local list before this lands, so onboarding/offboarding is one edit rather than three.
     Direct assignment also removes Entra's group-replication delay: the assignment is on the
     user when the next token is issued.
   - Leave `app_role_assignment_required = false`: requiring an assignment would lock every
     non-admin employee out of KB. The claim is present for assigned users and absent for
     everyone else, which is exactly the discriminator.
   - No optional id-token claim is needed: Entra emits `roles` in the ID token for a user
     assigned to an app role. This reverses the comment in that file ("NO app roles") — the
     proposal is the reversal.

2. **KB auth.**
   - `src/domain/entra_auth.py`: after signature/tenant/audience/nonce verification, read
     `claims["roles"]` as a `list[str]`; ignore non-string entries. Expose a boolean
     `entra_admin = "kb.admin" in roles`. Nothing else about the token is trusted before the
     signature check.
   - **Guest exclusion (claim path only):** require the verified `tid` to equal the pinned
     tenant (already enforced) **and** the email/UPN to be an internal member of the allowed
     domain: `preferred_username`/`email` must not contain `#EXT#` and its domain must equal
     the KB admin domain (`qnsc.vn`). A B2B guest object can be assigned an app role in some
     tenants; it must never become a KB Admin through this path. Guests keep the existing
     invitation flow.
   - `src/api/routers/auth.py` (`/entra/callback`), **after** the account lookup and before
     the session is issued:
     - claim present → ensure the user holds the global Admin role (`Role.name == "Admin"`,
       `company_domain IS NULL`), set `users.admin_source = 'entra'`, and audit
       `entra_admin_grant`. A user who already is a DB bootstrap Admin keeps
       `admin_source = 'db'` — the claim does not silently re-label the break-glass path.
     - claim absent **and `users.admin_source = 'entra'`** → remove the global Admin role,
       clear `admin_source`, and revoke the user's refresh sessions and bump `auth_version`
       (see the correction below). This is the revocation the Problem section promises; it
       must ship in step 1, not later, or the ADR promises a property it does not have.
     - claim absent and `admin_source` is `db`/NULL → leave the account alone. The
       break-glass Admin is not demoted by signing in.
   - **Schema, in step 1:** migration adds `users.admin_source` (`'db' | 'entra'`, nullable;
     NULL means legacy/bootstrap). `provision_admin.py` writes `'db'`.
   - First sign-in by an assigned admin with no account: this is the part that needs a
     product decision. Either (a) keep KB invite-only and require an Invitation as today
     (then no DB bootstrap is needed only for an account an invite already created), or
     (b) let a verified `kb.admin` claim from the pinned tenant, with an internal UPN, create
     the account without an Invitation. **Recommendation: (b), restricted to @qnsc.vn**,
     audited as `entra_admin_provision`; it removes the bootstrap without widening access
     beyond the assigned user list.
   - **`auth_version` correction:** it is NOT bumped on a role change today — only on a
     password/account change (`auth.py`). Role grants are read from the database on the next
     request, but an existing session stays alive. The revocation branch above therefore
     explicitly revokes refresh sessions and increments `auth_version`; without that, removing
     an admin from the assigned list would leave their live session an Admin until expiry.

3. **Rollout.**
   - Apply the identity change (app role + direct assignments) and the `admin_source`
     migration first, then the auth code.
   - Sign in once with an assigned user; confirm the claim and the grant.
   - Keep `scripts/provision_admin.py --no-password` as the break-glass path (it sets the
     Admin RLS context itself, so it still works with no application path) until a
     claim-backed login is verified.
   - Only after that, remove the DB bootstrap step from the go-live runbook.

## Risks

- **The app role becomes the root of admin trust.** Anyone who can edit
  `kb_admin_object_ids` (or assign the app role in Entra) can mint a global KB Admin. That is
  the intended model, but the IaC list's reviewers and the Entra audit trail must be treated
  as production admin access.
- **Token latency.** An Entra ID token lives up to ~1 hour. A removed admin keeps admin until
  their next sign-in presents a token without the claim; the revocation branch then ends the
  session. If removal must take effect immediately regardless of sign-in, an operator must
  also revoke the user's KB sessions (or a scheduled reconciliation job is needed).
- **Guests and B2B users** can hold app-role assignments in some tenant configurations. The
  pinned-tenant + internal-UPN + `#EXT#` checks above are the boundary; they must be tested.
- **Audience/tenant pinning** already exists (`verify_id_token`); reading `roles` adds no new
  token trust, but a regression there would be a privilege-escalation bug. Tests must assert
  a token with an unverified/foreign tenant never yields admin.
- **Irreversible step:** dropping the DB bootstrap before a successful Entra sign-in leaves
  an environment unadministrable. Keep the script and the runbook path until the first
  claim-backed login is verified.
