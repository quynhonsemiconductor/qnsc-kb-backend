# Entra apps for KB: one single-tenant app per environment

Microsoft sign-in and the SharePoint/OneDrive connector use an Entra app registration. This is the standard
(plan §7.4) and the owner's steps. It follows an Entra audit that found the single app `QNSC Knowledge Base`
(`dbd99dbb-d20e-4076-8f8b-75c15e733414`) is **multi-tenant** (`signInAudience = AzureADMultipleOrgs`), is shared by
every environment, and has only `localhost` redirect URIs.

## What the code does about it (kb-backend, this change)

The registration decides who Entra will *let reach us*; the code decides who we *accept*. Both are now closed:

| | Before | Now |
|---|---|---|
| `MICROSOFT_TENANT_ID` default | `common` | empty |
| `common`, `organizations`, `consumers`, empty | accepted as the authority; with `common`, **a validly signed token from any tenant verified** (the tenant comparison was skipped) | **not a pin**: sign-in is "not configured" (503), the connector card is unavailable, token verification rejects everything |
| ID token `tid` / `iss` | compared with each other under `common` | both compared with OUR configured tenant; no escape hatch |
| Connector consent / token / refresh URLs | `/{MICROSOFT_TENANT_ID}`, i.e. `/common` by default | `/{tenant GUID}` only; refuses to build a URL otherwise |
| Terraform | module default `common`; prod did not pass a tenant | module requires a GUID (`validation`); prod passes the QNSC tenant |

What an **external-tenant user** got, measured on `main` (`d7bc045`) by calling the real functions with a token signed
for a foreign tenant:

* authority = our tenant GUID (what every **deployed** environment uses; production refuses to boot otherwise):
  **rejected** (`Microsoft ID token tenant is invalid`).
* authority = `common` (the code default, and `docker-compose.yml`'s `:-common`, so a local or ad-hoc run): the token
  **verified**. The callback then looks the user up by email: with **no invitation** it answers **403 "no active QNSC
  invitation"** and creates nothing (verified: zero `db.add`, zero commits). **With an invitation** for that email it
  would have created the user with the invitation's role (read from the code, not run), and a role is the only thing
  between that user and the data. So there was no path to data without an admin inviting that address, but the tenant
  check was not what stopped it.
* Not tested live: whether Entra itself turns a foreign user away at `/{our GUID}/oauth2/v2.0/authorize` even though the
  app is multi-tenant (it does for an unknown user; B2B guests of our tenant carry our `tid`).

Production note: `infra/live/prod` had `microsoft_client_id = ""`, so production could not boot (production requires the
client id). Nothing is exposed there; it must be set before the first prod deploy.

## The two apps

| | **KB Dev** | **KB Prod** |
|---|---|---|
| Supported account types | **Accounts in this organizational directory only (single tenant)** | same |
| Platform | **Web** (server-side code flow with a client secret). Not SPA, not mobile | same |
| Redirect URIs | `http://localhost:8000/api/v1/auth/entra/callback`, `http://localhost:8000/api/v1/connectors/oauth/callback`, `https://kb-api-dev.qnsc.vn/api/v1/auth/entra/callback`, `https://kb-api-dev.qnsc.vn/api/v1/connectors/oauth/callback` | `https://kb-api.qnsc.vn/api/v1/auth/entra/callback`, `https://kb-api.qnsc.vn/api/v1/connectors/oauth/callback` **only** |
| Used by | local dev **and** the develop environment | production only |
| Secrets bundle | `qnsc-kb/develop/app`, key `microsoft-client-secret` | `qnsc-kb/prod/app`, key `microsoft-client-secret` |
| Terraform | `microsoft_client_id` in `infra/live/develop/variables.tf` | `microsoft_client_id` in `infra/live/prod/variables.tf` |

Production gets **no `localhost` URI**: a production client secret must not be usable from a developer's machine.

### Permissions: delegated only, no application permissions

**API permissions > Microsoft Graph > Delegated permissions** (and nothing under *Application permissions*):

| Permission | For |
|---|---|
| `openid`, `profile`, `email` | Sign-in (the ID token). These need no admin consent |
| `offline_access`, `User.Read` | Connector refresh token and the signed-in identity |
| `Files.Read.All`, `Sites.Read.All` | The connector reads SharePoint and OneDrive **as the user who consented** |

**The connector's delegated scopes stay on the sign-in app.** The code uses one `MICROSOFT_CLIENT_ID` for both sign-in
(`/auth/entra/callback`) and the connector (`/connectors/oauth/callback`), and the deployed connector mode is
`delegated` (the default; Terraform does not set `MICROSOFT_CONNECTOR_AUTH_MODE`). Delegated means the files an admin
can read through the connector are the files *that admin* can read, which is the property to keep. Whether your tenant
lets users consent to these two themselves, or needs an administrator to **Grant admin consent**, depends on its
user-consent policy: check the *Status* column after adding them and grant if it says "Not granted".

**If the connector is ever switched to `application` mode** (unattended, `MICROSOFT_CONNECTOR_AUTH_MODE=application`),
it needs *application* permissions, and those do not belong on the sign-in app. That would be a third app (for
example `KB Prod Connector`) with `Sites.Selected` and its own client id: a design change, not this runbook.

### Other settings to set on each app

* **Authentication**: no implicit grant or hybrid flow (leave both unchecked); **Allow public client flows = No**.
* **Certificates & secrets**: one client secret, shortest lifetime your policy allows, calendar the expiry. (A
  certificate is preferable and a follow-up; this code path uses a secret today.)
* **Enterprise applications > the app > Properties**: **Assignment required = Yes**, then assign the group(s) that may
  sign in. Without it every member of the tenant can reach the sign-in; the invitation is the only gate after that.
* **Token configuration**: nothing to add. The code reads `oid`, `tid`, `iss`, `aud`, `nonce` and
  `preferred_username`, all present in a v2.0 ID token by default.
* Record the **Application (client) ID** and **Directory (tenant) ID** (`dc0f2078-ac28-4ff2-b21a-d4b28df32361`).

## Owner steps, in order

1. Create **KB Dev** and **KB Prod** as above (Entra admin center > App registrations > New registration).
2. For each, create the client secret and put it in that environment's bundle key `microsoft-client-secret`.
   **Merge, do not replace** the bundle, and keep the value off the command line (a pipe, not an argument; the secret
   comes from the clipboard or a file, never from the shell history):

   ```bash
   ID=qnsc-kb/<env>/app
   aws secretsmanager get-secret-value --secret-id "$ID" --query SecretString --output text \
     | jq --rawfile s kb-<env>.secret '. + {"microsoft-client-secret": ($s | rtrimstr("\n"))}' \
     | aws secretsmanager put-secret-value --secret-id "$ID" --secret-string file:///dev/stdin
   aws secretsmanager get-secret-value --secret-id "$ID" --query SecretString --output text \
     | jq -e 'has("secret-key") and has("data-encryption-key") and has("microsoft-client-secret")'   # prints true
   ```
3. Set `microsoft_client_id` to that app's Application (client) ID in `infra/live/<env>/variables.tf` and open a pull
   request. Develop's default is still the old shared app; this replaces it. `microsoft_tenant_id` already defaults to
   the QNSC tenant in both environments, and the module now rejects anything that is not a GUID.
4. Merge; apply; deploy. Order matters only for the brief window between the secret changing and the client id
   changing; sign-in on the old app cannot work on develop today anyway (its only redirect URIs are `localhost`).
5. Local development: in `.env` set `MICROSOFT_CLIENT_ID` (KB Dev), `MICROSOFT_CLIENT_SECRET`,
   `MICROSOFT_TENANT_ID=<the GUID>`, `MICROSOFT_LOGIN_REDIRECT_URI=http://localhost:8000/api/v1/auth/entra/callback`
   and `MICROSOFT_REDIRECT_URI=http://localhost:8000/api/v1/connectors/oauth/callback`. Without the tenant GUID the
   Microsoft sign-in button is disabled (503), by design.
6. Once both environments sign in on their own app, **retire `QNSC Knowledge Base`**: set *Supported account types* to
   single tenant first (it stops accepting foreign tenants at once), remove its redirect URIs, delete its client
   secret, and delete the app after a grace period.

## Prove it

* `GET /api/v1/auth/entra/login` returns an `authorization_url` whose path starts `/<tenant GUID>/oauth2/v2.0/authorize`.
  If it says `common`, the environment is mis-set (it should now answer 503 instead).
* Sign in with an account from **another tenant** (or a personal Microsoft account): Entra answers with an `AADSTS`
  error on its own page and never calls us. If it somehow reaches the callback, the API answers 401
  "Microsoft sign-in could not be verified" and writes nothing.
* Sign in as an invited, assigned member of the tenant: you land on `/login?entra=success`.
* In the unit suite: `tests/unit/test_entra_tenant_pinning.py`.

## Does anything need multi-tenant?

Nothing in this repository does. Sign-in is invitation-based (`Invitation.company_domain`), `ALLOWED_EMAIL_DOMAINS` is
`qnsc.vn`, and a person from another organization can still be invited and use email + password. **If the product is
meant to let customers from other organizations sign in with their own Microsoft accounts**, that is an **owner
decision** and it is not made here: the options are (a) invite them as B2B guests of the QNSC tenant, which keeps this
single-tenant design, or (b) a separate multi-tenant sign-in app plus code that maps each accepted `tid` to a
`company_domain`, which this change deliberately does not build.
