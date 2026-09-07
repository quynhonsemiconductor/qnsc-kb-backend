# qnsc-kb infrastructure

OpenTofu stacks for qnsc-kb, on the shared QNSC platform. Deploys are driven by
`.github/workflows/`; this document covers the things a pipeline cannot do for itself.

```
infra/
  live/_shared/     ECR repositories + every GitHub OIDC role. Applied first, always.
  live/develop/     values for develop
  live/prod/        values for production (pre-launch idle — read its header first)
  modules/stack/    the entire environment, shared by both
```

`live/*` hold **values only**. The stack lives in `modules/stack`, so the two
environments cannot drift structurally — only in what they feed in. Adding a resource
means editing the module once. Relocating an existing address needs a `moved {}` block,
or Terraform destroys and recreates it.

## What this stack consumes and does not create

| From | What |
| :--- | :--- |
| `qnsc-infra` `bootstrap` | state bucket, lock table, GitHub OIDC provider, KMS CMK |
| `qnsc-infra` `runtime-{dev,prod}` | VPC, subnets, NAT, security groups |
| `qnsc-infra` `storage-{dev,prod}` | the R2 sources bucket |
| `tf-modules` | every module, pinned per module version |

Those must exist before a product apply. The OIDC provider in particular is an account
singleton — AWS permits one per issuer URL — so this product consumes it and must never
create its own.

## Bootstrapping a new environment

Steps 1 and 2 are once per product, ever. Everything after them is pipeline-driven.

### 1. Apply `_shared` by hand

`infra-apply.yml` authenticates by assuming `qnsc-kb-github-infra-apply`, and that role
is created **by this stack**. Until it exists there is nothing for the workflow to
authenticate as, so the first apply needs admin credentials:

```bash
export AWS_PROFILE=qnsc-admin AWS_REGION=ap-southeast-1
cd infra/live/_shared
tofu init
tofu apply
```

Creates 3 ECR repositories and 5 IAM roles. Free until images are pushed.

### 2. Create the out-of-band objects

Terraform cannot mint these. The tunnel is NOT among them any more — the `cf-tunnel`
module creates it, names it `<product>-<env_slug>`, and writes its connector token
straight into its own secret. Nothing about a tunnel needs a dashboard visit.

**An R2 API token** scoped to `qnsc-kb-develop-sources`, giving an access key id and
secret.

**A Gemini API key**, and optionally Microsoft/Google OAuth applications whose redirect
URI is `https://<api_domain>/api/v1/connectors/oauth/callback`.

### 3. Populate the secret bundle

Terraform creates the container **empty** — values never enter Terraform state. The
deploy preflight refuses to roll while any injected secret is an empty container.

Every app secret lives in ONE Secrets Manager container, read per key by ECS through the
`<arn>:<key>::` form of `valueFrom`. Secrets Manager bills per SECRET regardless of
size, so this is seven secrets' worth of material for one secret's fee.

```bash
aws secretsmanager put-secret-value \
  --profile qnsc-admin --secret-id qnsc-kb/develop/app \
  --secret-string '{
    "secret-key":            "<32+ chars>",
    "data-encryption-key":   "<32+ chars, different from secret-key>",
    "app-db-password":       "<32+ chars>",
    "gemini-api-key":        "<from Google AI Studio>",
    "r2-access-key-id":      "<R2 token>",
    "r2-secret-access-key":  "<R2 token>"
  }'
```

`tunnel-token` is deliberately absent: it lives in its own Terraform-managed secret,
`qnsc-kb/<env>/tunnel-token`, because Terraform knows the value and would clobber this
JSON object if it had to write one key of it.

Add `microsoft-client-secret` / `google-client-secret` only when the matching client id
is set — the stack creates those keys only then, deliberately, because a secret that is
never populated and never injected still bills and shows up in every "which secrets are
empty?" audit as a permanent false positive.

> **Write every key in one call.** The preflight proves the CONTAINER holds a value, not
> that each key exists. A bundle missing one key passes CI and fails at task boot as a
> `ResourceInitializationError`, several minutes later, in a log that does not name the
> missing key.

The RDS master password needs nothing: RDS creates and rotates it (`rds!db-*`), and the
preflight skips those by design.

### 4. GitHub configuration

Almost everything is inherited from the organisation — `CLOUDFLARE_API_TOKEN` and
`RELEASE_BOT_PRIVATE_KEY` as org secrets, `AWS_ACCOUNT_ID`, `CLOUDFLARE_ACCOUNT_ID` and
`RELEASE_BOT_APP_ID` as org variables. If those use *selected repositories* visibility,
add `qnsc-kb-backend` and `qnsc-kb-frontend` to each.

Per repository:

- **Environments** — `shared`, `develop`, `production` on the backend; `develop`,
  `production` on the frontend. Add a required reviewer to both `production`
  environments; that reviewer IS the production gate.
- **The RELEASE_BOT App** installed on both repositories with **Variables: read and
  write**. The backend's apply publishes `PAGES_PROJECT` into the frontend repository,
  because the Pages project is an output of this stack and an input there.

Environment *variables* need no manual work — `infra-apply` publishes them from
Terraform outputs, so a rebuild that changes a subnet or security-group id cannot leave
the deploy pointing at something that no longer exists.

### 5. Merge, and let the pipeline take over

A merge to `main` applies `_shared` + `develop`, then deploys. A `v*.*.*` tag applies
`_shared` + `prod` behind the reviewer, promotes the exact image develop tested, and
deploys that.

### 6. After the first apply

**Run the migrator once.** It creates the `vector` and `pgcrypto` extensions, applies
every migration, builds the HNSW index, and creates the `qnsc_app` role. The deploy
pipeline runs it automatically on every deploy; the first one is worth watching.

**Create the first Admin user.** There is no self-registration and no SSO login path, so
nobody can get in until one exists.

**Confirm the alarm email subscriptions.** Terraform creates them `pending
confirmation` and cannot complete them — an unconfirmed address is silently no alerting
at all.

## Things that will bite

**An infra change alone does not take effect.** Terraform owns the task definition's
environment, but the `ecs-service` module sets `ignore_changes = [task_definition]`, so
the SERVICE is never moved onto the revision Terraform registers — the deploy pipeline
owns that. This is why `infra/**` is deliberately NOT in the deploy workflow's
`paths-ignore`, and why `wait-for-infra` sequences deploy-after-apply on the same
commit. Re-adding it there recreates a silent failure: the apply succeeds, the new
definition is correct, nothing rolls, and the old value stays live.

**Removing infrastructure the running code still uses needs the deploy FIRST.** The
normal order is apply-then-deploy, which is right for an addition. For a removal it is
backwards. Expand/contract: ship the code that no longer needs the resource, let it
roll, then remove the resource in a second change.

**The cache is the Celery broker, not a cache.** Disabling it does not degrade
background work, it stops all of it — ingestion, connector polling, the outbox relay. A
`check` block ties `cache.enabled = false` to zero service floors for that reason.

**Celery beat is a singleton riding in the worker task.** `max_count` is validated at 1;
two replicas double every scheduled job. Splitting beat into its own service is the
prerequisite for scaling the worker horizontally.

**develop sleeps.** It idles at 00:00 and 03:00 and wakes at 08:00 (Asia/Ho_Chi_Minh),
and any deploy wakes it. Two idle passes rather than one because a single stop cannot
hold against a wake signal that fires at any hour. While it is asleep nothing scheduled
runs, because beat is scaled down with the worker.

**production is pre-launch idle.** A deploy landing there today succeeds and starts no
tasks, because the floors are zero. Do not read a green prod deploy as "production is
up". `live/prod/main.tf` carries the go-live checklist and the reason a production
deploy fails against its stopped database.

**`db.t4g.micro` will not survive go-live.** Migration `20260802_03` builds an HNSW
index whose construction cost scales with the corpus. An instance class chosen against
an empty database is not a measurement. Under-sizing shows up as ingestion timeouts, not
as an obvious out-of-memory error.

**RDS never shrinks a volume.** Storage autoscales up to `max_allocated_storage_gb` and
the increase is permanent — coming back down needs the instance replaced. The
`rds_free_bytes` alarm is what makes growth visible before it is paid for.

## Local plan

```bash
export AWS_PROFILE=qnsc-admin AWS_REGION=ap-southeast-1
cd infra/live/develop
tofu init
tofu plan
```

A plan without `TF_VAR_cloudflare_account_id` shows the Pages project and DNS record as
absent; CI supplies it from the org variable in both plan and apply, which is what keeps
the two honest about count-gated resources.
