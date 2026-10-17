# Application email through Microsoft 365 (Graph, certificate, one mailbox per environment)

Invitations and password resets are sent by the worker (`deliver_notification_queue`) through Microsoft Graph
`sendMail`, using the **application** permission, authenticated by a **certificate**, and limited by Exchange
Online **RBAC for Applications** to **one mailbox**. There are **two** mailboxes and **two** apps, one per
environment, and neither app can send as the other's mailbox. This is the owner's setup guide and the way to
prove it.

| | Develop | Production |
|---|---|---|
| Shared mailbox | `noreply-kb-dev@qnsc.vn` | `noreply-kb@qnsc.vn` |
| Entra app (a "Service app") | `qnsc-kb-mail-develop` | `qnsc-kb-mail-prod` |
| Exchange management scope | `KB mail sender (develop)` -> the dev mailbox only | `KB mail sender (prod)` -> the prod mailbox only |
| Secrets bundle | `qnsc-kb/develop/app` | `qnsc-kb/prod/app` |
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
| **Certificate**, no client secret | A secret is a string anything that can read the environment can replay; the key never leaves the secrets bundle, and the app proves possession by signing a short-lived assertion |
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

Entra admin center > App registrations > New registration: name `qnsc-kb-mail-<env>`, **single tenant**, no
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
  -keyout mail-<env>.key -out mail-<env>.crt -subj "/CN=qnsc-kb-mail-<env>"
openssl x509 -in mail-<env>.crt -noout -enddate -fingerprint -sha256
```

* Upload **`mail-<env>.crt` only** to that environment's app: Certificates & secrets > Certificates >
  Upload certificate. The private key is never uploaded anywhere but the secrets bundle.
* RSA only: the client assertion is RS256 and the application refuses an EC key.

## 5. The secrets bundle (do this BEFORE the next deploy)

The two PEMs are keys `mail-graph-certificate` and `mail-graph-private-key` of the bundle JSON
`qnsc-kb/<env>/app`. **Merge them into the existing JSON; do not replace it** (a plain `put-secret-value`
with only these keys erases `secret-key`, `data-encryption-key`, the database password and the rest).

```bash
SECRET_ID=qnsc-kb/<env>/app
CUR="$(aws secretsmanager get-secret-value --secret-id "$SECRET_ID" --query SecretString --output text)"
echo "$CUR" | jq 'keys'                       # note what is there; this prints key NAMES only
NEW="$(jq --rawfile c mail-<env>.crt --rawfile k mail-<env>.key \
  '. + {"mail-graph-certificate": $c, "mail-graph-private-key": $k}' <<<"$CUR")"
aws secretsmanager put-secret-value --secret-id "$SECRET_ID" --secret-string "$NEW"
aws secretsmanager get-secret-value --secret-id "$SECRET_ID" --query SecretString --output text | jq 'keys'
unset CUR NEW
```

The last command must list both new keys **and** all the old ones. Do not skip it: the deploy preflight
checks that the bundle container holds a value, not that each key exists. A missing key fails the task at boot
with `ResourceInitializationError` in a log that does not name the key (the circuit breaker then rolls the
deploy back). Keep `mail-<env>.key` only until the verification in step 8 has passed, then `shred -u` it
(or `rm -P` on macOS); keep the `.crt` and the expiry date. Make sure each environment's bundle got **its own**
pair: develop's key in `qnsc-kb/develop/app`, production's in `qnsc-kb/prod/app`.

## 6. Exchange RBAC for Applications (the scope)

Develop shown; for production use the prod app, the prod scope and `noreply-kb@qnsc.vn`.

```powershell
Connect-ExchangeOnline
New-ServicePrincipal -AppId <develop-client-id> -ObjectId <develop-enterprise-app-object-id> `
  -DisplayName "QNSC KB mail develop"
New-ManagementRoleAssignment -App <develop-client-id> -Role "Application Mail.Send" `
  -CustomResourceScope "KB mail sender (develop)"

Get-ManagementRoleAssignment -RoleAssignee "QNSC KB mail develop" | Format-List Role, CustomResourceScope
Test-ServicePrincipalAuthorization -Identity <develop-client-id> -Resource noreply-kb-dev@qnsc.vn   # InScope : True
Test-ServicePrincipalAuthorization -Identity <develop-client-id> -Resource noreply-kb@qnsc.vn       # InScope : False  (PRODUCTION mailbox)
```

`Test-ServicePrincipalAuthorization` reads Exchange only and bypasses its permission cache; it **does not
see Entra grants**. Real Graph calls can lag a fresh assignment (Microsoft's guidance is up to roughly half an
hour); if step 8 returns 403 on the *allowed* mailbox straight after this, wait and retry before debugging.

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
* If the token's `roles` lists `Mail.Send`, step 7 was not done.
* `TOKEN` failing with `AADSTS700027` (invalid assertion signature) or `AADSTS700024`: the certificate
  uploaded to Entra is not the one whose key is in that environment's bundle.

## 9. Configure the repository and deploy

1. In `infra/live/<env>/variables.tf` set `mail_graph_client_id` to that environment's Application (client) ID
   (a public identifier). `microsoft_graph_sender` already defaults to `noreply-kb-dev@qnsc.vn` in develop and
   `noreply-kb@qnsc.vn` in production, and `tofu plan` rejects a develop sender without `-dev@` and a production
   one with it. Commit to the pull request that is to be merged; do not merge before steps 1-8 are done.
2. Merging runs Infrastructure · Apply, which adds the two mail secrets and the three variables to the task
   definition baseline. Nothing sends yet: the running services keep their revision.
3. Deploy. The API now **refuses to boot** with `EMAIL_PROVIDER=graph` unless the sender, client ID and both PEMs
   are present, so a half-configured environment fails the deploy instead of quietly queueing mail.
4. End to end: invite a user from the app; the message arrives from the environment's own mailbox
   (`noreply-kb-dev@qnsc.vn` from develop). A failed delivery is recorded on the notification row
   (`status = failed`, `last_error`), for example `Graph sendMail as ... failed: HTTP 403 ErrorAccessDenied ...`.

## Rotation and revocation

* **Rotate (before the expiry date; the worker logs `Mail certificate expires soon` 30 days out):** generate a new
  pair, upload the new certificate in Entra **next to** the old one, replace the two bundle keys (merge, as in
  step 5), redeploy, verify with step 8, then delete the old certificate in Entra.
* **Revoke now:** delete the certificate in Entra (new tokens stop at once; an issued token lives up to an
  hour). To cut the app off from its mailbox as well: `Remove-ManagementRoleAssignment` for its assignment.

## Local development

`EMAIL_PROVIDER=console` (the `.env.example` default) writes the message, links included, to the log:
`docker compose logs worker`. It is refused outside development, and local runs use neither mailbox. To
exercise the real path locally, use the **develop** app and mailbox: set `EMAIL_PROVIDER=graph` and the four
`MAIL_GRAPH_*` / `MICROSOFT_GRAPH_SENDER` values in `.env`; never the production ones.
