# Application email through Microsoft 365 (Graph, certificate, one mailbox)

Invitations and password resets are sent by the worker (`deliver_notification_queue`) through Microsoft Graph
`sendMail`, using the **application** permission, authenticated by a **certificate**, and limited by Exchange
Online **RBAC for Applications** to **one mailbox**. This is the owner's setup guide and the way to prove it.

## The design, and why

| Choice | Why |
|---|---|
| A **dedicated Entra app per environment** (`qnsc-kb-mail-develop`, `qnsc-kb-mail-prod`), not the SSO/connector app | Separate credentials and separate revocation; the connector app has other, broader permissions |
| **Certificate**, no client secret | A secret is a string anything that can read the environment can replay; the key never leaves the secrets bundle, and the app proves possession by signing a short-lived assertion |
| **No Mail.Send consent in Entra at all** | Application `Mail.Send` consented in Entra is tenant-wide (send as ANY mailbox). Exchange RBAC for Applications grants are *independent* of Entra grants and the effective access is the **union**, so an unscoped Entra consent would silently cancel the scope. The app's only authority is the scoped Exchange role |
| **Management scope** naming the sender mailbox | A stolen key can send as `noreply-kb@qnsc.vn` and as nobody else |
| A **shared mailbox** | No licence, no sign-in; the sender address needs to exist, not to be a person |

Microsoft's documentation of the model: <https://learn.microsoft.com/exchange/permissions-exo/application-rbac>.

Who can do what: **Exchange**: Organization Management (to create the mailbox, scope, role assignment).
**Entra**: Application Administrator or Cloud Application Administrator (to register the app and upload the
certificate). **AWS**: write access to the `qnsc-kb/<env>/app` secret. **No Global Administrator consent is
needed**, which is the point.

Repeat steps 3-7 for each environment (develop, prod). The mailbox (steps 1-2) is shared unless you decide
develop should send as a different address (e.g. `noreply-kb-dev@qnsc.vn`); the address is the
`microsoft_graph_sender` variable in `infra/live/<env>/variables.tf`.

## 1. The mailbox

```powershell
Connect-ExchangeOnline
New-Mailbox -Shared -Name "QNSC KB No-Reply" -DisplayName "QNSC Knowledge Base" `
  -Alias noreply-kb -PrimarySmtpAddress noreply-kb@qnsc.vn
Set-Mailbox noreply-kb@qnsc.vn -HiddenFromAddressListsEnabled $true   # optional
Get-Mailbox noreply-kb@qnsc.vn | Format-List PrimarySmtpAddress, RecipientTypeDetails   # SharedMailbox
```

Mail flow for `qnsc.vn` is already Microsoft 365's, so SPF should include `spf.protection.outlook.com`; check
`nslookup -type=txt qnsc.vn` and that outbound DKIM for the domain is enabled in the Defender portal
(Email & collaboration > Policies > Email authentication settings > DKIM). The old SES DKIM records are
unrelated and are owned by qnsc-infra's `edge` stack.

## 2. The management scope (one mailbox)

```powershell
New-ManagementScope -Name "KB mail sender" `
  -RecipientRestrictionFilter "PrimarySmtpAddress -eq 'noreply-kb@qnsc.vn'"
```

## 3. The Entra app registration (per environment)

Entra admin center > App registrations > New registration:
name `qnsc-kb-mail-<env>`, **single tenant**, no redirect URI. Then:

* **API permissions: add NOTHING.** Do not add Microsoft Graph `Mail.Send`. (The default `User.Read`
  delegated entry can be removed; it is not used.)
* Record three IDs: **Application (client) ID** and **Directory (tenant) ID** from the app's Overview, and
  the **Object ID of the enterprise application** (Enterprise applications > the app > Object ID). Do not
  use the Object ID shown on the *App registrations* page; it is a different object and Exchange rejects it.
  If the enterprise application does not exist: `az ad sp create --id <client-id>`.

## 4. The certificate

On a trusted workstation (not a CI runner). Two years; put the renewal in the calendar.

```bash
umask 077
openssl req -x509 -newkey rsa:2048 -nodes -days 730 \
  -keyout mail.key -out mail.crt -subj "/CN=qnsc-kb-mail-<env>"
openssl x509 -in mail.crt -noout -enddate -fingerprint -sha256
```

* Upload **`mail.crt` only** to the app: Certificates & secrets > Certificates > Upload certificate.
  The private key is never uploaded anywhere but the secrets bundle.
* RSA only: the client assertion is RS256 and the application refuses an EC key.

## 5. The secrets bundle (do this BEFORE the next deploy)

The two PEMs are keys `mail-graph-certificate` and `mail-graph-private-key` of the bundle JSON
`qnsc-kb/<env>/app`. **Merge them into the existing JSON; do not replace it** (a plain `put-secret-value`
with only these keys erases `secret-key`, `data-encryption-key`, the database password and the rest).

```bash
SECRET_ID=qnsc-kb/<env>/app
CUR="$(aws secretsmanager get-secret-value --secret-id "$SECRET_ID" --query SecretString --output text)"
echo "$CUR" | jq 'keys'                       # note what is there; this prints key NAMES only
NEW="$(jq --rawfile c mail.crt --rawfile k mail.key \
  '. + {"mail-graph-certificate": $c, "mail-graph-private-key": $k}' <<<"$CUR")"
aws secretsmanager put-secret-value --secret-id "$SECRET_ID" --secret-string "$NEW"
aws secretsmanager get-secret-value --secret-id "$SECRET_ID" --query SecretString --output text | jq 'keys'
unset CUR NEW
```

The last command must list both new keys **and** all the old ones. Do not skip it: the deploy preflight
checks that the bundle container holds a value, not that each key exists. A missing key fails the task at boot
with `ResourceInitializationError` in a log that does not name the key (the circuit breaker then rolls the
deploy back). Keep `mail.key` only until the verification in step 8 has passed, then `shred -u mail.key`
(or `rm -P` on macOS); keep `mail.crt` and the expiry date.

## 6. Exchange RBAC for Applications (the scope)

```powershell
Connect-ExchangeOnline
New-ServicePrincipal -AppId <client-id> -ObjectId <enterprise-app-object-id> `
  -DisplayName "QNSC KB mail <env>"
New-ManagementRoleAssignment -App <client-id> -Role "Application Mail.Send" `
  -CustomResourceScope "KB mail sender"

Get-ManagementRoleAssignment -RoleAssignee "QNSC KB mail <env>" | Format-List Role, CustomResourceScope
Test-ServicePrincipalAuthorization -Identity <client-id> -Resource noreply-kb@qnsc.vn   # InScope : True
Test-ServicePrincipalAuthorization -Identity <client-id> -Resource <any other mailbox>  # InScope : False
```

`Test-ServicePrincipalAuthorization` reads Exchange only and bypasses its permission cache; it **does not
see Entra grants**. Real Graph calls can lag a fresh assignment (Microsoft's guidance is up to roughly half an
hour); if step 8 returns 403 on the *allowed* mailbox straight after this, wait and retry before debugging.

## 7. Make sure there is NO unscoped grant in Entra

Entra admin center > Enterprise applications > the app > Permissions: **empty**. Or:

```powershell
Connect-MgGraph -Scopes "Application.Read.All"
$sp = Get-MgServicePrincipal -Filter "appId eq '<client-id>'"
Get-MgServicePrincipalAppRoleAssignment -ServicePrincipalId $sp.Id      # must return nothing
```

If anything is listed (especially Microsoft Graph `Mail.Send`), remove it. This is the step people skip, and
it is what makes the 403 negative test fail in step 8.

## 8. Prove it (including the 403)

With the application's own sender, from a workstation that holds `mail.crt` and `mail.key`:

```bash
poetry run python scripts/graph_mail_check.py \
  --tenant <tenant-id> --client-id <client-id> --cert mail.crt --key mail.key \
  --sender noreply-kb@qnsc.vn --to <your address> --denied-mailbox <any other real mailbox>
```

Expected, in order:

```
PASS  TOKEN   issued for appid=<client-id>; roles in the token: none (expected when authority comes from Exchange RBAC only)
PASS  SEND    sent as noreply-kb@qnsc.vn to <you> (HTTP 202)
PASS  DENIED  as <other>: HTTP 403 ErrorAccessDenied (request-id ...)
```

* `DENIED` is the **negative test** and it is the proof the scope works. If it says the send was *accepted*,
  the app can send as anyone: go back to step 7.
* If the token's `roles` lists `Mail.Send`, step 7 was not done.
* `TOKEN` failing with `AADSTS700027` (invalid assertion signature) or `AADSTS700024`: the certificate
  uploaded to Entra is not the one whose key is in the bundle.

## 9. Configure the repository and deploy

1. In `infra/live/<env>/variables.tf` set `mail_graph_client_id` to the Application (client) ID (a public
   identifier) and, if it differs, `microsoft_graph_sender`. Commit it to the pull request that is to be
   merged; do not merge before steps 1-8 are done.
2. Merging runs Infrastructure · Apply, which adds the two mail secrets and the three variables to the task
   definition baseline. Nothing sends yet: the running services keep their revision.
3. Deploy. The API now **refuses to boot** with `EMAIL_PROVIDER=graph` unless the sender, client ID and both PEMs
   are present, so a half-configured environment fails the deploy instead of quietly queueing mail.
4. End to end: invite a user from the app; the message arrives from `noreply-kb@qnsc.vn`. A failed delivery is
   recorded on the notification row (`status = failed`, `last_error`), for example
   `Graph sendMail as ... failed: HTTP 403 ErrorAccessDenied ...`.

## Rotation and revocation

* **Rotate (before the expiry date; the worker logs `Mail certificate expires soon` 30 days out):** generate a new
  pair, upload the new certificate in Entra **next to** the old one, replace the two bundle keys (merge, as in
  step 5), redeploy, verify with step 8, then delete the old certificate in Entra.
* **Revoke now:** delete the certificate in Entra (new tokens stop at once; an issued token lives up to an
  hour). To cut the app off from the mailbox as well: `Remove-ManagementRoleAssignment` for its assignment.

## Local development

`EMAIL_PROVIDER=console` (the `.env.example` default) writes the message, links included, to the log:
`docker compose logs worker`. It is refused outside development. To exercise the real path locally, set
`EMAIL_PROVIDER=graph` and the four `MAIL_GRAPH_*` / `MICROSOFT_GRAPH_SENDER` values in `.env`.
