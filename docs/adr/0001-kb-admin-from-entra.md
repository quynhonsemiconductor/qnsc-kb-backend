# Proposal: grant KB Admin from an Entra app role

Status: **proposal — no code in this PR**
Date: 2026-11-05

## Problem

The first global Admin is a database bootstrap (`scripts/provision_admin.py`, now fixed to
set its own RLS context). Every new environment needs that one-off run before anybody can
sign in, and the account it creates is outside the IdP's lifecycle: removing someone from
`grp-qnsc-platform-admins` does not remove KB Admin from them.

OpsHub already models admin as an Entra app role on its sign-in app, assigned to
`grp-qnsc-platform-admins` in `infra/live/identity` and read from the ID token's `roles`
claim. This proposal applies that shape to KB.

## Proposal

1. **Identity (one owner apply).** `infra/live/identity/kb_prod_apps.tf`:
   - Add one app role to `azuread_application.kb_prod`: `value = "kb.admin"`,
     `display_name = "KB Administrator"`, `allowed_member_types = ["User"]`, `enabled = true`.
     The role **id is a fixed GUID committed in the Terraform** (an `azuread_application_app_role`
     resource, not a generated id) so the group assignment can reference it deterministically.
   - Assign it to the existing `grp-qnsc-platform-admins` (`var.platform_admins_object_id`,
     already used by the OpsHub grant):

     ```hcl
     resource "azuread_app_role_assignment" "kb_prod_admin" {
       app_role_id         = azuread_application_app_role.kb_prod_admin.id
       principal_object_id = var.platform_admins_object_id
       resource_object_id  = azuread_service_principal.kb_prod.object_id
     }
     ```

   - Leave `app_role_assignment_required = false`: requiring an assignment would lock every
     non-admin employee out of KB. The claim is present for admin-group members and absent for
     everyone else, which is exactly the discriminator.
   - No optional id-token claim is needed: Entra emits `roles` in the ID token for a user
     assigned to an app role. This reverses the comment in that file ("NO app roles") — the
     proposal is the reversal.

2. **KB auth.**
   - `src/domain/entra_auth.py`: after signature/tenant/audience/nonce verification, read
     `claims["roles"]` as a `list[str]`; ignore non-string entries. Expose a boolean
     `entra_admin = "kb.admin" in roles`. Nothing else about the token is trusted before the
     signature check.
   - `src/api/routers/auth.py` (`/entra/callback`), **after** the account lookup and before
     the session is issued:
     - claim present → ensure the user holds the global Admin role (`Role.name == "Admin"`,
       `company_domain IS NULL`) through the same `bootstrap_rbac` + role-assignment path the
       script uses; audit `entra_admin_grant`.
     - claim absent → do **not** revoke automatically in this first step. Revocation needs a
       source marker so a break-glass DB Admin is not demoted by signing in; add
       `users.admin_source` (`db` | `entra`, nullable) in a later migration and revoke only
       `entra` rows. Until then, removal from the group takes effect for new sessions only if
       the operator also removes the role (documented, not silent).
   - First sign-in by an admin-group member with no account: this is the part that needs a
     product decision. Either (a) keep KB invite-only and require an Invitation as today
     (then no DB bootstrap is needed only for an account that an invite already created), or
     (b) let a verified `kb.admin` claim from the pinned tenant create the account without an
     Invitation. **Recommendation: (b), restricted to @qnsc.vn**, audited as
     `entra_admin_provision`; it removes the bootstrap without widening access beyond the
     admin group.
   - `auth_version` is bumped on any role change, so existing sessions end on the next
     request; a group removal should therefore be paired with the revocation work above.

3. **Rollout.**
   - Apply the identity change and the group assignment first.
   - Sign in once with a `grp-qnsc-platform-admins` member; confirm the claim and the grant.
   - Keep `scripts/provision_admin.py --no-password` as the break-glass path (it sets the
     Admin RLS context itself, so it still works with no application path).
   - Only after that, remove the DB bootstrap step from the go-live runbook.

## Risks

- **The app role becomes the root of admin trust.** Anyone who can add a member to
  `grp-qnsc-platform-admins` can mint a global KB Admin. That is the intended model, but the
  group's owners and audit trail must be treated as production admin access.
- **Token latency.** An Entra ID token lives up to ~1 hour and sessions are longer. Removing
  a person from the group does not end existing KB sessions. Mitigations: `auth_version` bump
  on the next sign-in grant change, plus explicit session revocation on offboarding; a later
  revocation sync is required for "group removal ends access".
- **Guests and B2B users** do not receive app-role claims the same way; they keep the
  existing invitation/DB path. The claim path must be additive, never the only way in.
- **Audience/tenant pinning** already exists (`verify_id_token`); reading `roles` adds no new
  token trust, but a regression there would be a privilege-escalation bug. Tests must assert
  a token with an unverified/foreign tenant never yields admin.
- **Irreversible step:** dropping the DB bootstrap before a successful Entra sign-in leaves
  an environment unadministrable. Keep the script and the runbook path until the first
  claim-backed login is verified.
