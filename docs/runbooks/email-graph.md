# Application email through Microsoft 365 (Graph, certificate, one mailbox per environment)

Invitations and password resets are sent by the worker (`deliver_notification_queue`) through Microsoft Graph
`sendMail`, using the **application** permission, authenticated by a **certificate**, and limited by Exchange
Online **RBAC for Applications** to **one mailbox**. There are **two** mailboxes and **two** apps, one per
environment, and neither app can send as the other's mailbox. This is the owner's setup guide and the way to
prove it.

| | Develop | Production |
|---|---|---|
| Shared mailbox | `noreply-kb-dev@qnsc.vn` | `noreply-kb@qnsc.vn` |
| Entra app (a "Service app") | `KB Dev Service` | `KB Prod Service` |
| Exchange management scope | `KB mail sender (develop)` -> the dev mailbox only | `KB mail sender (prod)` -> the prod mailbox only |
| Mail credential (standalone secret) | `qnsc-kb/develop/mail-graph-certificate` | `qnsc-kb/production/mail-graph-certificate` |
| Terraform | `infra/live/develop/variables.tf` | `infra/live/prod/variables.tf` |
| Local | `EMAIL_PROVIDER=console` (the `.env.example` default): no mailbox, no app | |

**Rule (2026-10-18): a non-production environment never sends from the production mailbox.** Users cannot tell
the mail apart, a leaked develop credential would otherwise speak for production, and develop's mail would
count against production's sending reputation. The Terraform variables enforce the naming (develop's sender
must contain `-dev@`, production's must not), and step 8 proves it on the real tenant: **develop's app must get
403 when it sends from the PRODUCTION mailbox**, and production's app must get 403 from the dev mailbox.

## The design, and why

| Choice | Why |
|---|---|
| A **dedicated Entra app per environment**, not the SSO/connector app | Separate credentials and revocation; the connector app has other, broader permissions |
| **Certificate**, no client secret | A secret is a string anything that can read the environment can replay; the key lives in one standalone secret that only the worker's execution role can read, and the app proves possession by signing a short-lived assertion |
| **No Mail.Send consent in Entra at all** | Application `Mail.Send` consented in Entra is tenant-wide (send as ANY mailbox). Exchange RBAC for Applications grants are *independent* of Entra grants and the effective access is the **union**, so an unscoped Entra consent would silently cancel the scope. The app's only authority is the scoped Exchange role |
| A **management scope** naming exactly one mailbox, per app | A stolen key can send as that environment's mailbox and as nobody else, in particular not as the other environment's |
| **Shared mailboxes** | No licence, no sign-in; the sender address needs to exist, not to be a person |

Microsoft's documentation of the model: <https://learn.microsoft.com/exchange/permissions-exo/application-rbac>.

Who can do what: **Exchange**: Organization Management (mailboxes, scopes, role assignments). **Entra**:
Application Administrator or Cloud Application Administrator (register the app, upload the certificate).
**AWS**: write access to the `qnsc-kb/<env>/app` secret. **No Global Administrator consent is needed**, which is
the point.

Steps 1-2 create both mailboxes and scopes once. Steps 3-8 are done **once per environment**; wherever a name
or address differs, the table above says which.

## 1. The mailboxes

```powershell
Connect-ExchangeOnline
foreach ($m in @(@{Alias='noreply-kb';     Name='QNSC KB No-Reply';       Display='QNSC Knowledge Base'},
                 @{Alias='noreply-kb-dev'; Name='QNSC KB No-Reply (dev)'; Display='QNSC Knowledge Base (develop)'})) {
  New-Mailbox -Shared -Name $m.Name -DisplayName $m.Display -Alias $m.Alias `
    -PrimarySmtpAddress "$($m.Alias)@qnsc.vn"
  Set-Mailbox "$($m.Alias)@qnsc.vn" -HiddenFromAddressListsEnabled $true   # optional
}
Get-Mailbox noreply-kb@qnsc.vn, noreply-kb-dev@qnsc.vn | Format-List PrimarySmtpAddress, RecipientTypeDetails   # SharedMailbox
```

Mail flow for `qnsc.vn` is already Microsoft 365's, so SPF should include `spf.protection.outlook.com`; check
`nslookup -type=txt qnsc.vn` and that outbound DKIM for the domain is enabled in the Defender portal
(Email & collaboration > Policies > Email authentication settings > DKIM). The old SES DKIM records are
unrelated and are owned by qnsc-infra's `edge` stack. If the product is not `kb`, the names follow
`noreply-<product>[-dev]@qnsc.vn`.

## 2. The management scopes (one mailbox each)

```powershell
New-ManagementScope -Name "KB mail sender (prod)" `
  -RecipientRestrictionFilter "PrimarySmtpAddress -eq 'noreply-kb@qnsc.vn'"
New-ManagementScope -Name "KB mail sender (develop)" `
  -RecipientRestrictionFilter "PrimarySmtpAddress -eq 'noreply-kb-dev@qnsc.vn'"
```

## 3. The Entra app registration (per environment)

Entra admin center > App registrations > New registration: name `KB Dev Service` or `KB Prod Service` (the Service apps of plan §7.4), **single tenant**, no
redirect URI. Then:

* **API permissions: add NOTHING.** Do not add Microsoft Graph `Mail.Send`. (The default `User.Read`
  delegated entry can be removed; it is not used.)
* Record three IDs: **Application (client) ID** and **Directory (tenant) ID** from the app's Overview, and
  the **Object ID of the enterprise application** (Enterprise applications > the app > Object ID). Do not
  use the Object ID shown on the *App registrations* page; it is a different object and Exchange rejects it.
  If the enterprise application does not exist: `az ad sp create --id <client-id>`.

## 4. The certificate (per environment, its own pair)

On a trusted workstation (not a CI runner). Two years; put the renewal in the calendar. **Never reuse one
pair for both environments.**

```bash
umask 077
openssl req -x509 -newkey rsa:2048 -nodes -days 730 \
  -keyout mail-<env>.key -out mail-<env>.crt -subj "/CN=KB-<Env>-Service"
openssl x509 -in mail-<env>.crt -noout -enddate -fingerprint -sha256
```

* Upload **`mail-<env>.crt` only** to that environment's app: Certificates & secrets > Certificates >
  Upload certificate. The private key is never uploaded anywhere but its standalone secret (step 5).
* RSA only: the client assertion is RS256 and the application refuses an EC key.

## 5. The mail credential: a standalone secret (do this BEFORE setting `mail_graph_client_id`)

The credential is **one PEM holding the private key and the certificate** (`cat mail-<env>.key mail-<env>.crt`),
stored as the value of its own secret, `qnsc-kb/<env>/mail-graph-certificate`. It is **not** a key of the shared
`qnsc-kb/<env>/app` bundle: the bundle is granted to the API's execution role as well, so anything inside it is
readable by the API's role. This secret is referenced only by the **worker** container, granted only to the
**worker's execution role**, and a resource policy denies `GetSecretValue` to every other principal, you included.
No task role in the stack can read any secret. One value, one write: the key and the certificate cannot drift apart,
and there is no bundle JSON to merge or to clobber.

The code reads the credential ONLY from the **worker-only FILE** `AZURE_CLIENT_CERTIFICATE_PATH`
(ONPREM-PLAN §7.3): the same secret, synced and mounted into the worker's filesystem, never an env value.
Without the mounted file a worker told to send mail refuses to start — the deploy fails by name until the
mount exists.

The secret's **container** is created by Terraform as soon as the stack is applied (empty, without a version), in
every environment that is not the in-memory fake, **including develop while it is still on SES**; you only write the
value. Nothing is injected or granted until `mail_graph_client_id` is set. The deploy fails by name until you do (see below).

**Nothing secret goes on a command line.** Command lines are visible in the process list and saved in shell
history. The value goes in through a pipe (`file:///dev/stdin`):

```bash
ID=qnsc-kb/<env>/mail-graph-certificate
cat mail-<env>.key mail-<env>.crt \
  | aws secretsmanager put-secret-value --secret-id "$ID" --secret-string file:///dev/stdin
```

Verify without reading it back (reading it back is denied by design, which is the point). It must show one version
with `AWSCURRENT`:

```bash
aws secretsmanager list-secret-version-ids --secret-id "$ID" \
  --query 'Versions[].VersionStages' --output text          # AWSCURRENT
aws secretsmanager describe-secret --secret-id "$ID" --query '{Name:Name,Changed:LastChangedDate}'
```

The proof that the VALUE is right is step 8 (`scripts/graph_mail_check.py` reads the same two files from your
workstation and signs with them); the worker then loads the identical bytes. Check the file you piped first:
`openssl x509 -in mail-<env>.crt -noout -modulus | openssl md5` and `openssl rsa -in mail-<env>.key -noout -modulus |
openssl md5` must print the same hash.

**The per-secret deploy preflight.** `ci` runs, before every roll, `list-secret-version-ids` on every secret a task
definition injects and **fails by the secret's name** when one holds no version: `... injects secret
'qnsc-kb/<env>/mail-graph-certificate' but it holds no value`. That is metadata only (it needs no permission to read
the value), and it is the per-secret check that the keys inside the bundle never had. What it does *not* prove is
that the value is a valid PEM; step 8 does, and so does the worker, which refuses to start with a CRITICAL
`Mail configuration: ...` line when the key or the certificate block is missing.

Where the private key lives: generate it on an encrypted volume (FileVault is on a managed Mac) and keep the one
copy in an encrypted location (the team password manager, or an encrypted disk image). It is the only copy outside
AWS, because AWS will not give it back. Do not rely on `shred` or `rm -P`: on APFS and SSDs they do not overwrite
the blocks, so they give a false sense of having erased it. Keep the `.crt` and the expiry date. Each environment
has **its own** pair and its own secret: develop's in `qnsc-kb/develop/mail-graph-certificate`, production's in
`qnsc-kb/production/mail-graph-certificate`.

Break-glass: to read the value back, an administrator first deletes the secret's resource policy
(`aws secretsmanager delete-resource-policy`); Terraform restores it on the next apply.

## 6. Exchange RBAC for Applications (the scope)

Develop shown; for production use the prod app, the prod scope and `noreply-kb@qnsc.vn`.

```powershell
Connect-ExchangeOnline
New-ServicePrincipal -AppId <develop-client-id> -ObjectId <develop-enterprise-app-object-id> `
  -DisplayName "KB Dev Service"
New-ManagementRoleAssignment -App <develop-client-id> -Role "Application Mail.Send" `
  -CustomResourceScope "KB mail sender (develop)"

Get-ManagementRoleAssignment -RoleAssignee "QNSC KB mail develop" | Format-List Role, CustomResourceScope
Test-ServicePrincipalAuthorization -Identity <develop-client-id> -Resource noreply-kb-dev@qnsc.vn   # InScope : True
Test-ServicePrincipalAuthorization -Identity <develop-client-id> -Resource noreply-kb@qnsc.vn       # InScope : False  (PRODUCTION mailbox)
```

`Test-ServicePrincipalAuthorization` reads Exchange only and bypasses its permission cache; it **does not
see Entra grants**. Real Graph calls can lag a fresh assignment (Microsoft documents propagation of up to roughly two
hours); if step 8 returns 403 on the *allowed* mailbox straight after this, wait and retry (up to ~2 hours) before debugging.

## 7. Make sure there is NO unscoped grant in Entra (per app)

Entra admin center > Enterprise applications > the app > Permissions: **empty**. Or:

```powershell
Connect-MgGraph -Scopes "Application.Read.All"
$sp = Get-MgServicePrincipal -Filter "appId eq '<client-id>'"
Get-MgServicePrincipalAppRoleAssignment -ServicePrincipalId $sp.Id      # must return nothing
```

If anything is listed (especially Microsoft Graph `Mail.Send`), remove it. This is the step people skip, and
it is what makes the 403 negative test fail in step 8.

## 8. Prove it (including the cross-environment 403)

With the application's own sender, from a workstation that holds that environment's `.crt` and `.key`.

**Develop's app: its own mailbox works, the PRODUCTION mailbox is refused.**

```bash
poetry run python scripts/graph_mail_check.py \
  --tenant <tenant-id> --client-id <develop-client-id> --cert mail-develop.crt --key mail-develop.key \
  --sender noreply-kb-dev@qnsc.vn --to <your address> \
  --denied-mailbox noreply-kb@qnsc.vn --denied-mailbox <any other real mailbox>
```

```
PASS  TOKEN   issued for appid=<develop-client-id>; roles in the token: none (expected when authority comes from Exchange RBAC only)
PASS  SEND    sent as noreply-kb-dev@qnsc.vn to <you> (HTTP 202)
PASS  DENIED  as noreply-kb@qnsc.vn: HTTP 403 ErrorAccessDenied (request-id ...)      <- develop cannot speak for production
PASS  DENIED  as <other>: HTTP 403 ErrorAccessDenied (request-id ...)
```

**Production's app, the mirror image:** `--client-id <prod-client-id> --sender noreply-kb@qnsc.vn
--denied-mailbox noreply-kb-dev@qnsc.vn --denied-mailbox <other>`; run it before production's release.

* `DENIED` is the **negative test** and it is the proof the scope works. If it says the send was *accepted*,
  that app can send as the other environment (or anyone): go back to step 7.
* If the token carries **any** `roles` claim, the check fails: this app should hold no application role at all, so something was consented in Entra (step 7).
* `TOKEN` failing with `AADSTS700027` (invalid assertion signature) or `AADSTS700024`: the certificate
  uploaded to Entra is not the one whose key is in that environment's mail secret.

## 9. Configure the repository and deploy

The order matters, and `mail_graph_client_id` is the switch: while it is empty, **nothing** mail-related is injected
and develop keeps sending through SES exactly as before, so merging the code before these steps takes nothing
down.

1. Steps 1-8 done for the environment, including step 5 (the secret has a value) and step 8 (the proof).
2. In `infra/live/<env>/variables.tf` set `mail_graph_client_id` to that environment's Application (client) ID
   (a public identifier). `microsoft_graph_sender` already defaults to `noreply-kb-dev@qnsc.vn` in develop and
   `noreply-kb@qnsc.vn` in production, and `tofu plan` rejects a develop sender without `-dev@` and a production
   one with it. Merge that change.
3. Merging runs Infrastructure · Apply: develop's `email_provider` becomes `graph`, and the mail secret is injected
   into the **worker container only** and granted to the worker's execution role only (the API and beat never
   receive it, and no task role can read it). SES is removed in the same release window
   as the rova prod switch-on, after one real rova prod email has arrived (#216, see the release sequence in its PR
   body). Rollback is a redeploy of the previous release, not a retained SES path; this environment has no prod
   traffic yet, so production follows its own first prod release.
4. Deploy. The **worker** now refuses to start if the client id is set but the sender, certificate or key is
   missing, so a half-configured environment fails the deploy instead of quietly queueing mail. The API does not
   check mail at all. With the client id empty the worker starts and logs `Mail is not configured` on every
   delivery pass, and the `...-worker-mail-not-configured` alarm fires.
5. End to end: invite a user from the app; the message arrives from the environment's own mailbox
   (`noreply-kb-dev@qnsc.vn` from develop).

## What happens when delivery fails

Delivery is bounded by **age**, not by a count; a message is never retried forever and an outage never kills it early.

* **The message is refused** (HTTP 400, 413): the row becomes `status = 'dead'` at once and `Notification marked
  dead` is logged (alarm `...-worker-mail-dead`).
* **The sender is refused** (HTTP 403 or 404 on the send): this is a problem with the *sender*, not the message. The
  Exchange scope or role assignment is missing or still propagating (**up to ~2 hours** after step 6), or the
  mailbox does not exist or is not REST-enabled yet. The batch stops, **no row is charged an attempt or marked
  dead**, each is retried after 15 minutes, and `Mail sender refused` is logged (alarm `...-worker-mail-sender-refused`).
  Fix steps 1, 2 or 6 and mail resumes by itself.
* **Everything else that is transient** (5xx, network): exponential backoff, 30 s doubling to a 1 h cap. Honours
  `Retry-After` (seconds or an HTTP date) on 429/503.
* **Affects every message** (the token request is refused, certificate unusable, throttling, or a refused sender):
  the batch stops and the remaining rows are not charged an attempt.
* **A row older than 24 hours** that is still failing becomes `dead` (a reset link is worthless by then).
* Failing rows back off out of the way, so they cannot starve new mail.
* Re-queue dead rows after fixing the cause: `UPDATE notification_queue SET status='pending', attempts=0,
  next_attempt_at=NULL WHERE status='dead' AND type='email';` (rows older than 24 h will be marked dead again by
  the first failure; also refresh their `created_at` if they should be retried).
* Certificate expiry is checked **every night at 01:00 UTC and each time a worker starts** (so an environment
  redeployed more than daily is still checked): `Mail certificate expires soon` (30 days) and `Mail certificate
  has expired` raise the `...-worker-mail-certificate` alarm.
* A worker that is told to use the mail app (client id set) but lacks the sender, certificate or key **exits** at
  start with a CRITICAL `Mail configuration: ...` line, so the deploy fails instead of queueing mail.

## Rotation and revocation

* **Rotate (before the expiry date; the worker logs `Mail certificate expires soon` 30 days out):** generate a new
  pair, upload the new certificate in Entra **next to** the old one, write the new `key + certificate` PEM to the same secret (a new version, as in
  step 5), redeploy, verify with step 8, then delete the old certificate in Entra.
* **Revoke now:** delete the certificate in Entra (new tokens stop at once; an issued token lives up to an
  hour). To cut the app off from its mailbox as well: `Remove-ManagementRoleAssignment` for its assignment.

## Local development

`EMAIL_PROVIDER=console` (the `.env.example` default) writes the message, links included, to the log:
`docker compose logs worker`. It is refused outside development, and local runs use neither mailbox. To
exercise the real path locally, use the **develop** app and mailbox: set `EMAIL_PROVIDER=graph` and the four
`MAIL_GRAPH_*` / `MICROSOFT_GRAPH_SENDER` values in `.env`; never the production ones.
