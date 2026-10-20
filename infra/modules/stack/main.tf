// qnsc-kb · the whole stack, for every environment.
//
// live/develop and live/prod hold VALUES ONLY and both instantiate this module, so the
// two environments cannot drift structurally — only in what they feed in. Adding a
// resource means editing this file once.
//
// Consumes, never creates: the VPC/subnets/SGs come from the platform runtime stack,
// the CMK and ECR repos from this product's own _shared stack, the R2 bucket from the
// platform storage stack. Relocating an existing address needs a `moved {}` block in
// live/*/moved.tf, or Terraform destroys and recreates it.

data "aws_caller_identity" "current" {}

// ── Product shared layer (ECR URLs, KMS, re-exported platform outputs) ───────
data "terraform_remote_state" "shared" {
  backend = "s3"
  config = {
    bucket = "qnsc-tofu-state"
    key    = var.shared_state_key
    region = "ap-southeast-1"
  }
}

// ── Platform runtime layer (shared VPC + NAT + SGs) ──────────────────────────
// One VPC per environment, shared by every product. NAT egress matters more here than
// for most products: the worker reaches Gemini, Microsoft Graph and Google APIs, and
// the api reaches Gemini on every question.
data "terraform_remote_state" "runtime" {
  backend = "s3"
  config = {
    bucket = "qnsc-tofu-state"
    key    = var.runtime_state_key
    region = "ap-southeast-1"
  }
}

// ── Platform storage layer (Cloudflare R2) ───────────────────────────────────
data "terraform_remote_state" "storage" {
  backend = "s3"
  config = {
    bucket = "qnsc-tofu-state"
    key    = var.storage_state_key
    region = "ap-southeast-1"
  }
}

locals {
  name         = "${var.product}-${var.env_slug}"
  app_base_url = "https://${var.app_domain}"
  api_base_url = "https://${var.api_domain}"

  kms_key_arn        = data.terraform_remote_state.shared.outputs.kms_key_arn
  cloudflare_zone_id = try(data.terraform_remote_state.shared.outputs.cloudflare_zone_id, "")

  // DKIM verifies the DOMAIN, not the mailbox, so this is the part after "@" — SES signs
  // for any address at it, and the local part of mail_from_email can change later without
  // a new identity. Keyed on mail_from_email, not on email_provider: the SES send grant must outlive the
  // switch to Graph (expand/contract: the switch ships first, the grant is deleted in a later change, so a
  // rollback to an EMAIL_PROVIDER=ses task definition still has the permission it needs).
  mail_from_domain = var.mail_from_email != "" ? replace(var.mail_from_email, "/^[^@]*@/", "") : ""

  // The mail app is "configured" once the owner has created it and put its client id in the live stack. Until
  // then NOTHING mail-related is injected, so merging this change before the owner's steps cannot make a task
  // reference a secret that has no value yet (ResourceInitializationError at boot, a circuit-breaker rollback,
  // and the next release failing the same way). Order of the owner's steps: the mail secret's value first, THEN the client id.
  mail_graph_enabled = var.email_provider == "graph" && var.mail_graph_client_id != ""

  // The secret CONTAINER and its resource policy exist for every provider that is not the in-memory fake, NOT only
  // once the client id is set. Develop is on "ses" until its client id is set, and the runbook tells the owner to put
  // the value first: put-secret-value on a secret that does not exist yet fails. The injection and the worker's IAM
  // grant stay gated on mail_graph_enabled, so an empty container references nothing and grants nothing.
  mail_secret_exists = var.email_provider != "fake"

  ecr_base         = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
  ecr_api_url      = "${local.ecr_base}/${var.product}-api:${var.image_tag}"
  ecr_worker_url   = "${local.ecr_base}/${var.product}-worker:${var.image_tag}"
  ecr_migrator_url = "${local.ecr_base}/${var.product}-migrator:${var.image_tag}"

  // Computed, not read from module.worker.log_group_name, to break a dependency cycle:
  // the beat and clamav sidecars need a log group, the worker service needs their
  // container definitions, and the worker service is what creates the log group.
  // `ecs-service` names it deterministically as /ecs/<cluster>-<service>.
  api_log_group    = "/ecs/${local.name}-api"
  worker_log_group = "/ecs/${local.name}-worker"

  // `rediss://`, never `redis://`: the cache module enables transit encryption
  // unconditionally. src/workers/celery_app.py keys its TLS settings off this exact
  // scheme, because Celery rejects a rediss:// URL that does not state ssl_cert_reqs.
  //
  // With the cache disabled this is a deliberately UNRESOLVABLE RFC 2606 name rather
  // than an empty string: REDIS_URL has a localhost default in config.py, so omitting
  // it would leave a deployed task silently pointing Celery at itself.
  # THREE cases; the middle one is the shared node in the runtime layer.
  #
  # The DATABASE INDEX is only appended when sharing — a dedicated node has nobody to
  # collide with, and appending `/0` there would rewrite REDIS_URL for no behaviour change.
  #
  # `try()` is load-bearing, not defensive: a runtime layer without a shared cache has no
  # `cache_endpoint` output, and a bare reference to a missing output fails the plan for
  # EVERY environment rather than only the one sharing. Missing output while sharing lands
  # on `.invalid`, which fails loudly at boot instead of pointing somewhere wrong.
  shared_cache_endpoint = try(data.terraform_remote_state.runtime.outputs.cache_endpoint, null)
  shared_cache_port     = try(data.terraform_remote_state.runtime.outputs.cache_port, 6379)

  redis_url = (
    !var.cache.enabled ? "rediss://cache-disabled.invalid:6379" :
    var.cache.shared ? (
      local.shared_cache_endpoint == null
      ? "rediss://shared-cache-missing.invalid:6379"
      : "rediss://${local.shared_cache_endpoint}:${local.shared_cache_port}/${var.cache.db_index}"
    ) :
    "rediss://${module.cache[0].endpoint}:${module.cache[0].port}"
  )

  // Every app secret this stack owns. Terraform creates the CONTAINER; values are
  // pasted in out of band and never enter state. The deploy preflight in ci
  // refuses to deploy while any injected secret is still empty.
  // A MAP of key → description, not a list: the module keys the bundle's JSON off these
  // names and uses the description on the secret itself.
  secret_names = merge({
    "secret-key"           = "JWT signing key (SECRET_KEY)"
    "data-encryption-key"  = "At-rest encryption of stored connector credentials (DATA_ENCRYPTION_KEY)"
    "app-db-password"      = "Password for the least-privilege application role (${var.app_db_role})"
    "gemini-api-key"       = "Gemini API key for answering and restructuring"
    "r2-access-key-id"     = "R2 credential scoped to the sources bucket"
    "r2-secret-access-key" = "R2 credential scoped to the sources bucket"
    },
    var.microsoft_client_id != "" ? { "microsoft-client-secret" = "Microsoft connector OAuth client secret" } : {},
    var.google_client_id != "" ? { "google-client-secret" = "Google connector OAuth client secret" } : {},
    // The mail app's certificate and key are NOT in this bundle: see aws_secretsmanager_secret.mail_graph.
  )
  // NOTE: "tunnel-token" is deliberately NOT in that map. The tunnel is created by
  // Terraform (module.tunnel below), which means the connector token is an attribute
  // rather than something a human copies out of a dashboard — so it is written straight
  // into its own secret, not pasted into this bundle. The bundle holds exactly the
  // values Terraform cannot know.

  // IAM resource list for the secret containers this stack owns.
  //
  // `secret_iam_arns`, NOT `secret_arns`: with use_bundle on, `secret_arns` returns
  // "<arn>:<key>::" — an ECS valueFrom reference, not an ARN — and an IAM statement
  // built from those matches nothing while still applying cleanly. The failure surfaces
  // at the next task start as "unable to pull secrets", long after apply reported
  // success.
  secret_iam_arns = module.secrets.secret_iam_arns

  // Non-secret connection parts. USER/PASSWORD arrive as injected secrets and the
  // application composes the URL (see Settings.model_post_init) — a full DATABASE_URL
  // cannot be assembled here without putting the password in plain task-definition env.
  db_env = [
    { name = "DATABASE_HOST", value = module.rds.address },
    { name = "DATABASE_PORT", value = tostring(module.rds.port) },
    { name = "DATABASE_NAME", value = module.rds.db_name },
    { name = "DATABASE_USER", value = var.app_db_role },
    { name = "APP_DATABASE_ROLE", value = var.app_db_role },
  ]

  // Configuration both the api and the worker must agree on. They run the same code
  // over the same database; a value set on one and not the other is a split brain —
  // most sharply EMBEDDING_MODEL, where a mismatch writes vectors the other half
  // cannot compare against.
  common_env = concat(local.db_env, [
    // production mode is deliberate in DEVELOP too. validate_production() is the only
    // thing enforcing AUTO_CREATE_SCHEMA=false, ENABLE_API_DOCS=false,
    // ALLOW_SELF_REGISTRATION=false, HTTPS-only origins and present R2 credentials, and
    // an environment that does not run those checks cannot prove production will pass
    // them.
    { name = "ENVIRONMENT", value = "production" },
    { name = "AWS_REGION", value = var.region },
    { name = "REDIS_URL", value = local.redis_url },

    // Celery, not in-process: JOB_MODE=inline would run ingestion inside the request
    // that uploaded the document.
    { name = "JOB_MODE", value = "celery" },

    // Schema comes from Alembic in the migrator task. AUTO_CREATE_SCHEMA would have
    // every booting task issue DDL of its own.
    { name = "AUTO_CREATE_SCHEMA", value = "false" },
    { name = "ENABLE_RLS", value = "true" },

    // The first administrator is created by scripts/create_admin.py as a one-off task,
    // which is how develop's was made. Leaving the startup bootstrap ENABLED would mean
    // storing a standing admin password in the secret bundle for an account that already
    // exists — and validate_production refuses to boot while that password is still the
    // development default, which is what took the API down after the SSO change landed.
    { name = "BOOTSTRAP_ADMIN_ENABLED", value = "false" },

    { name = "SOURCE_STORAGE_BACKEND", value = "r2" },
    { name = "SOURCE_STORAGE_BUCKET", value = data.terraform_remote_state.storage.outputs["${replace(var.product, "-", "_")}_sources_name"] },
    { name = "S3_ENDPOINT_URL", value = data.terraform_remote_state.storage.outputs["${replace(var.product, "-", "_")}_sources_endpoint"] },
    { name = "SOURCE_STORAGE_PREFIX", value = "qnsc-sources" },

    { name = "EMBEDDING_MODEL", value = var.embedding_model },
    { name = "EMBEDDING_VERSION", value = var.embedding_version },
    { name = "EMBEDDING_RUNTIME", value = var.embedding_runtime },

    // localhost is correct under awsvpc ONLY because BOTH tasks carry their own ClamAV
    // sidecar. The namespace is shared per TASK, not per cluster: the api and worker are
    // separate services, so the api reaching the worker's sidecar on localhost was never
    // possible. It resolved to nothing, every upload failed closed with "Malware scanning
    // is unavailable", and because connector ingestion runs in the worker — which did
    // have one — only the UI upload path was affected. See local.clamav_container.
    { name = "MALWARE_SCAN_ENABLED", value = tostring(var.malware_scan_enabled) },
    { name = "MALWARE_SCANNER_HOST", value = var.malware_scan_enabled ? "localhost" : "" },
    { name = "MALWARE_SCANNER_PORT", value = "3310" },

    { name = "FRONTEND_URL", value = local.app_base_url },
    { name = "CORS_ORIGINS", value = local.app_base_url },
    // The API ORIGIN, with no path. connectors.py builds the notification URL as
    // "<this>/api/v1/connectors/webhooks/<system>", so including the path here produced
    // .../api/v1/connectors/api/v1/connectors/webhooks/sharepoint — and Microsoft Graph
    // POSTs a validationToken to that URL while CREATING the subscription, so enabling
    // update notifications failed outright rather than going quietly stale.
    { name = "CONNECTOR_WEBHOOK_BASE_URL", value = local.api_base_url },

    { name = "MICROSOFT_CLIENT_ID", value = var.microsoft_client_id },
    { name = "MICROSOFT_TENANT_ID", value = var.microsoft_tenant_id },
    { name = "MICROSOFT_REDIRECT_URI", value = var.microsoft_client_id != "" ? "${local.api_base_url}/api/v1/connectors/oauth/callback" : "" },

    // Who application email is sent FROM, and the dedicated Entra app that sends as it (certificate
    // auth: the key + certificate is the WORKER-ONLY secret MAIL_GRAPH_PEM, see mail_secrets). The
    // API never refuses to boot over mail; the worker does the checking at its own start (tasks.py): client id
    // set with the rest missing refuses to start, client id empty logs an error that an alarm watches.
    { name = "MICROSOFT_GRAPH_SENDER", value = var.microsoft_graph_sender },
    { name = "MAIL_GRAPH_CLIENT_ID", value = var.mail_graph_client_id },

    // Only the worker actually calls get_email_sender() (deliver_notification_queue),
    // but this is stated in common_env rather than the worker-only block: api and worker
    // must agree on config the way EMBEDDING_MODEL does above, and a future path that
    // sends mail from the api would otherwise silently pick up a different transport.
    { name = "EMAIL_PROVIDER", value = var.email_provider },
    { name = "MAIL_FROM_EMAIL", value = var.mail_from_email },
    { name = "MAIL_FROM_NAME", value = var.mail_from_name },

    // SSO sign-in, a DIFFERENT callback from the connector one above: that consents to
    // SharePoint content, this authenticates a person. Both must be registered on the
    // Entra app.
    //
    // Set unconditionally, because validate_production() requires it with `if not all`
    // rather than only when SSO is switched on. Missing it took the develop API down —
    // "Microsoft Entra client, tenant, and redirect settings are required in production"
    // raised in the FastAPI lifespan, so uvicorn exited 3 before writing a request log
    // and the service crash-looped with an empty log stream.
    { name = "MICROSOFT_LOGIN_REDIRECT_URI", value = "${local.api_base_url}/api/v1/auth/entra/callback" },

    // Restricts admin-created accounts too. Empty accepts ANY domain, and under
    // company-scoped RLS a non-qnsc.vn address quietly creates a second tenant whose rows
    // nobody else can see.
    { name = "ALLOWED_EMAIL_DOMAINS", value = join(",", var.allowed_email_domains) },
    { name = "GOOGLE_CLIENT_ID", value = var.google_client_id },
    { name = "GOOGLE_REDIRECT_URI", value = var.google_client_id != "" ? "${local.api_base_url}/api/v1/connectors/oauth/callback" : "" },
  ])

  // Injected secrets shared by api and worker. Both encrypt and decrypt stored
  // connector credentials, so DATA_ENCRYPTION_KEY cannot be api-only.
  common_secrets = concat([
    { name = "SECRET_KEY", secret_arn = module.secrets.secret_arns["secret-key"] },
    { name = "DATA_ENCRYPTION_KEY", secret_arn = module.secrets.secret_arns["data-encryption-key"] },
    { name = "DATABASE_PASSWORD", secret_arn = module.secrets.secret_arns["app-db-password"] },
    { name = "GEMINI_API_KEY", secret_arn = module.secrets.secret_arns["gemini-api-key"] },
    { name = "R2_ACCESS_KEY_ID", secret_arn = module.secrets.secret_arns["r2-access-key-id"] },
    { name = "R2_SECRET_ACCESS_KEY", secret_arn = module.secrets.secret_arns["r2-secret-access-key"] },
    ], var.microsoft_client_id != "" ? [
    { name = "MICROSOFT_CLIENT_SECRET", secret_arn = module.secrets.secret_arns["microsoft-client-secret"] },
    ] : [], var.google_client_id != "" ? [
    { name = "GOOGLE_CLIENT_SECRET", secret_arn = module.secrets.secret_arns["google-client-secret"] },
  ] : [])

  // The mail credential (ONE PEM: private key + certificate) reaches the WORKER container only: it is the only
  // caller of get_email_sender() (deliver_notification_queue). The API and beat never need it, and a key that is
  // not in a container cannot be read out of it. It is a STANDALONE secret, not a key of the bundle above: the
  // bundle's ARN is on the api's execution role too, so anything in it is readable by the api's role. Empty until
  // mail_graph_client_id is set (mail_graph_enabled).
  mail_secrets = local.mail_graph_enabled ? [
    { name = "MAIL_GRAPH_PEM", secret_arn = aws_secretsmanager_secret.mail_graph[0].arn },
  ] : []

  tags = { Environment = var.env }
}

// ── Secrets ───────────────────────────────────────────────────────────────────
// One JSON container read per key, rather than one container per secret: Secrets
// Manager bills per SECRET regardless of size.
//
// Secrets Manager and not SSM Parameter Store, following the reasoning recorded in
// rova's stack: a Secrets Manager secret can exist while holding NO value, and that
// empty state is what makes "unpopulated" unambiguous. Parameter Store rejects an
// empty value, so the same guarantee needs a placeholder plus a version check plus a
// runtime guard — three mechanisms replacing one property. Revisit past ~30 secrets,
// where the per-secret fee starts to outweigh it. This stack has 9.
module "secrets" {
  # v2.1.1, not v2.1.0: the earlier output wrapped its ARNs in distinct(), which returns
  # an UNKNOWN-LENGTH list while the secrets do not exist yet — and ecs-service gates its
  # execution policy on `count = length(var.secret_arns) > 0`, so the plan failed outright
  # on a first apply. Only ever visible when creating a new environment.
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/secrets?ref=secrets-v2.1.1"

  prefix               = "${var.product}/${var.env}"
  kms_key_arn          = local.kms_key_arn
  recovery_window_days = var.secrets_recovery_window_days

  bundle_name  = var.secrets_bundle_name
  use_bundle   = var.secrets_use_bundle
  secret_names = local.secret_names

  tags = local.tags
}

// ── RDS PostgreSQL (pgvector) ─────────────────────────────────────────────────
// The extensions themselves (vector, pgcrypto) are created by migrations 20260802_00
// and 20260806_13, which is why the migrator connects as the MASTER user — creating an
// extension is not something the least-privilege application role may do.
module "rds" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/rds?ref=rds-v2.4.0"

  identifier        = local.name
  subnet_ids        = data.terraform_remote_state.runtime.outputs.data_subnet_ids
  security_group_id = data.terraform_remote_state.runtime.outputs.sg_rds_id
  kms_key_arn       = local.kms_key_arn

  engine_version           = var.rds.engine_version
  instance_class           = var.rds.instance_class
  allocated_storage_gb     = var.rds.allocated_storage_gb
  max_allocated_storage_gb = var.rds.max_allocated_storage_gb
  multi_az                 = var.rds.multi_az
  deletion_protection      = var.rds.deletion_protection
  backup_retention_days    = var.rds.backup_retention_days
  monitoring_interval      = var.rds.monitoring_interval

  # Major upgrade controls. Both default off; set together for the one apply that performs the upgrade.
  allow_major_version_upgrade = var.rds.allow_major_version_upgrade
  apply_immediately           = var.rds.apply_immediately
  auto_minor_version_upgrade  = var.rds.auto_minor_version_upgrade

  tags = local.tags
}

// ── Cache (Valkey) — Celery broker and result backend ────────────────────────
module "cache" {
  # NOT created when this product uses the shared node. Switching a live environment to
  # `shared` destroys the dedicated node — that is where the saving is — and issues a
  # different endpoint, so it is a task-definition revision and a rolling deploy.
  count  = var.cache.enabled && !var.cache.shared ? 1 : 0
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/cache?ref=cache-v1.1.0"

  name              = "${local.name}-cache"
  subnet_ids        = data.terraform_remote_state.runtime.outputs.data_subnet_ids
  security_group_id = data.terraform_remote_state.runtime.outputs.sg_cache_id
  kms_key_arn       = local.kms_key_arn

  mode      = var.cache.mode
  node_type = var.cache.node_type

  tags = local.tags
}

// ── ECS cluster ───────────────────────────────────────────────────────────────
module "ecs_cluster" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/ecs-cluster?ref=ecs-cluster-v2.0.0"

  name               = local.name
  container_insights = var.container_insights
  tags               = local.tags
}

// ── Cloudflare Tunnel ─────────────────────────────────────────────────────────
// Created by Terraform, not by hand. The provider exposes the tunnel's id, its CNAME
// target and its connector token as attributes, so nothing here needs a dashboard visit
// or a token pasted into a secret — which is what rova still does.
//
// Count-gated on the account id for the same reason as the Pages project: the AWS half
// of this stack must be able to apply before Cloudflare is wired up.
module "tunnel" {
  count  = var.tunnel_enabled && var.cloudflare_account_id != "" ? 1 : 0
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/cf-tunnel?ref=cf-tunnel-v0.2.1"

  account_id = var.cloudflare_account_id
  // One tunnel per product per environment. Sharing one across environments would let a
  // develop task serve production traffic, because a tunnel routes to whichever
  // connectors hold its token.
  name = local.name

  // Without these the connector runs, reports healthy, and answers 503 to everything —
  // "No ingress rules were defined". v0.1.1 is the first version that creates them.
  //
  // v0.2.0 made routing OPTIONAL so an existing tunnel could be adopted, so this now
  // depends on `config_src` keeping its default of "cloudflare". Setting it to "local"
  // hands routing to a config file on the connector, and this hostname/service pair
  // stops being applied — back to 503s, with nothing here changed to explain it.
  //
  // localhost is correct: under ECS awsvpc every container in a task shares one network
  // namespace, so cloudflared reaches the api without any port being exposed.
  hostname = var.api_domain
  service  = "http://localhost:8000"
}

// The connector token, in its own secret rather than in the bundle above.
//
// It cannot live in the bundle: Terraform would have to write the whole JSON object,
// clobbering the keys a human populated. Its own secret keeps the two ownership models
// apart — this one is Terraform's, the bundle is the operator's.
//
// The value IS in Terraform state, which is the trade the cf-tunnel module documents.
// The state bucket is KMS-encrypted and readable only by the infra-apply role, which
// already holds AdministratorAccess, so the token grants nothing that reading the state
// did not already imply.
resource "aws_secretsmanager_secret" "tunnel_token" {
  count = var.tunnel_enabled && var.cloudflare_account_id != "" ? 1 : 0

  name                    = "${var.product}/${var.env}/tunnel-token"
  description             = "Cloudflare Tunnel connector token (TUNNEL_TOKEN). Managed by Terraform — do not edit by hand."
  kms_key_id              = local.kms_key_arn
  recovery_window_in_days = var.secrets_recovery_window_days

  tags = local.tags
}

# The connector token's VALUE, written at apply time and never held as a Terraform attribute.
#
# The version resource that used to hold it read the value on every refresh, and that needs
# secretsmanager:GetSecretValue on the plan role. The plan role must not have it, so the
# container above stays managed here and this resource writes the token into a new version
# when it changes (a tunnel recreate changes the hash). No plan reads a secret value.
#
# The version resource that held the value is forgotten here, not destroyed. A bare `removed` block
# (OpenTofu 1.9: no `lifecycle` inside it; it forgets by default) takes the old version out of state on
# apply, and leaves the secret's current version alone. Without it, the removal would plan a DESTROY of
# the version, which can strip the AWSCURRENT stage the sidecar reads.
removed {
  from = aws_secretsmanager_secret_version.tunnel_token
}

resource "terraform_data" "tunnel_token_value" {
  count = var.tunnel_enabled && var.cloudflare_account_id != "" ? 1 : 0

  # A hash, so the token itself never lands in state or in a plan.
  triggers_replace = sha256(module.tunnel[0].token)

  # The token goes in through the environment and stdin, never argv, and is never echoed. Only the
  # new version id is printed.
  provisioner "local-exec" {
    environment = {
      TUNNEL_TOKEN = module.tunnel[0].token
      SECRET_ID    = aws_secretsmanager_secret.tunnel_token[0].id
    }
    command = "set -eu; command -v aws >/dev/null || { echo 'aws CLI missing on the runner' >&2; exit 1; }; printf '%s' \"$TUNNEL_TOKEN\" | aws secretsmanager put-secret-value --region ${var.region} --secret-id \"$SECRET_ID\" --secret-string file:///dev/stdin --query VersionId --output text"
  }
}

// ── The mail credential: a standalone secret, readable only by the worker's execution role ──────────────────
// ONE PEM holding the private key and its certificate, so the pair is written and read atomically and can never
// drift apart. Not a key of the secrets bundle: the bundle's ARN is granted to the api's execution role as well,
// which would make the key readable by the api's role. This secret is referenced only by the worker container
// (mail_secrets) and granted only to the worker's execution role (its secret_arns). No TASK role anywhere in
// this stack can read any secret (task_secret_arns is never passed to ecs-service), so application code, which
// runs as the task role, cannot read it either.
//
// The CONTAINER is created whenever the provider is not the in-memory fake (develop is still "ses" until its client id
// is set, and the owner writes the value before that), empty and without a version, so the owner can put the
// value BEFORE setting mail_graph_client_id; it is injected only once that is set (mail_graph_enabled). The value
// never enters Terraform state. The deploy preflight (ci backend-deploy) lists this secret's versions on every
// deploy and fails by NAME if it holds none: that is the per-secret check the bundle's keys never had.
resource "aws_secretsmanager_secret" "mail_graph" {
  count = local.mail_secret_exists ? 1 : 0

  name                    = "${var.product}/${var.env}/mail-graph-certificate"
  description             = "Mail app credential for Microsoft Graph: private key and certificate in ONE PEM (MAIL_GRAPH_PEM). Worker only. Value is set out of band: docs/runbooks/email-graph.md."
  kms_key_id              = local.kms_key_arn
  recovery_window_in_days = var.secrets_recovery_window_days

  tags = local.tags
}

// A boundary, not just an absence of grants: nobody may read the value back except the worker's execution role
// (the ECS agent, at task start). Administrators included; the owner keeps the key file, which is the only copy
// outside AWS. put-secret-value, describe-secret and list-secret-version-ids are not denied, so the owner can
// still write it, rotate it and see that it has a version. To read it for a break-glass, delete this policy first.
resource "aws_secretsmanager_secret_policy" "mail_graph" {
  count = local.mail_secret_exists ? 1 : 0

  secret_arn = aws_secretsmanager_secret.mail_graph[0].arn
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "OnlyTheWorkerExecutionRoleReadsTheMailCredential"
      Effect    = "Deny"
      Principal = "*"
      Action    = "secretsmanager:GetSecretValue"
      Resource  = "*"
      Condition = {
        ArnNotEquals = { "aws:PrincipalArn" = module.worker.execution_role_arn }
      }
    }]
  })
}

// ── ClamAV sidecar ───────────────────────────────────────────────────────────
// One definition, used by BOTH the api and the worker task. It is not a shared service:
// each task runs its own copy, because clamd is reached over localhost and awsvpc scopes
// that to a task. Duplicated memory is the price of a synchronous scan on the upload path
// — the API blocks on it in `_validate_source_bytes`, so it cannot wait on another
// service. If the API ever scales past a couple of tasks, replace this with one clamd
// behind Service Connect rather than paying ~2 GB per task.
locals {
  clamav_container = var.malware_scan_enabled ? [
    {
      name      = "clamav"
      image     = "clamav/clamav:1.4"
      essential = true
      // 2048, matching what the comment beside it always said: the signature database is
      // ~2 GB resident. At 1024 it did not fit, and the way that presented is worth
      // recording — clamd came up only on the runs where the signature UPDATE had
      // FAILED:
      //
      //   ERROR: Database test FAILED.  ...  Update failed.
      //   socket found, clamd started.
      //
      // A failed update leaves the older, smaller database, which fits; a successful one
      // does not. So the task cycled — two tasks killed for "failed container health
      // checks" before a third happened to get a failed update and went healthy. Roughly
      // 30 minutes per worker deploy, and it looked intermittent rather than like a
      // sizing error.
      cpu    = 256
      memory = 2048
      // clamd loads the whole database before it answers anything, so the start period
      // has to cover a cold load or the task is killed and restarted forever.
      //
      // 300 is the CEILING, not a choice: ECS rejects anything higher —
      //   ClientException: Health check startPeriod must be less than or equal to the
      //   maximum allowed value 300
      // I raised it to 600 on the theory that the ~3.4 GB image pull had eaten the
      // window; that was wrong twice over. The pull happens BEFORE the container starts,
      // so it never consumes the start period at all, and the limit forbids it regardless.
      // The memory above is the actual fix.
      healthCheck = {
        command     = ["CMD-SHELL", "clamdcheck.sh || exit 1"]
        interval    = 60
        timeout     = 10
        retries     = 3
        startPeriod = 300
      }
    },
  ] : []
}

// ── Cloudflare Tunnel sidecar (api only) ─────────────────────────────────────
// Ingress without an ALB: cloudflared dials out, so the task needs no inbound listener
// and no public IPv4. The worker has no HTTP surface and gets none.
module "tunnel_api" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/tunnel-agent?ref=tunnel-agent-v1.0.0"

  tunnel_token_secret_arn = length(aws_secretsmanager_secret.tunnel_token) > 0 ? aws_secretsmanager_secret.tunnel_token[0].arn : ""
  app_port                = 8000
  log_group               = local.api_log_group
  region                  = var.region

  // 512, not the module's 128 default. That default is a HARD limit — the module sets
  // `memory` and `memoryReservation` to the same value — and its own documentation says
  // to raise it for "a task holding many long-lived SSE connections", which is precisely
  // what this product is: the AI answer path streams tokens over SSE for the length of a
  // generation, and source uploads push up to 25 MB through the same connector.
  //
  // The sidecar is `essential = true`, so this is not a degraded-tunnel failure mode: if
  // cloudflared is OOM-killed the TASK dies and is replaced, and every request in flight
  // is reset with no HTTP response at all. In the browser that surfaces as a bare
  // "Network Error" with no status and no CORS headers, which is indistinguishable from
  // the API being down — and it reproduced from outside the browser as an intermittent
  // connection reset on large multipart POSTs while small requests stayed reliable.
  memory = 512
}

// ── API service ───────────────────────────────────────────────────────────────
module "api" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/ecs-service?ref=ecs-service-v2.3.2"

  service_name = "api"
  cluster_name = module.ecs_cluster.cluster_name
  cluster_arn  = module.ecs_cluster.cluster_arn
  region       = var.region
  image_uri    = local.ecr_api_url

  cpu    = var.api.cpu
  memory = var.api.memory

  vpc_id            = data.terraform_remote_state.runtime.outputs.vpc_id
  subnet_ids        = data.terraform_remote_state.runtime.outputs.private_subnet_ids
  security_group_id = data.terraform_remote_state.runtime.outputs.sg_app_id

  desired_count      = 1
  enable_autoscaling = var.api.enable_autoscaling
  min_count          = var.api.min_count
  max_count          = var.api.max_count
  use_spot           = var.api.use_spot
  cpu_target_pct     = var.api.cpu_target_pct
  memory_target_pct  = var.api.memory_target_pct
  log_retention_days = var.log_retention_days

  container_port = 8000

  attach_alb        = !var.tunnel_enabled
  alb_listener_arn  = try(data.terraform_remote_state.runtime.outputs.https_listener_arn, "")
  alb_priority      = 110
  alb_path_patterns = ["/*"]
  alb_host_headers  = [var.api_domain]

  // LIVENESS, deliberately — /health/live answers 200 without touching a dependency.
  // /health/ready checks Postgres and Redis, and a dependency-coupled probe here
  // deregisters or restarts the task whenever either blips, converting a hiccup into an
  // outage. Readiness is checked once after the roll, by the deploy pipeline.
  health_check_path = "/health/live"

  environment_vars = concat(local.common_env, [
    // Publishing the full endpoint inventory and every schema, unauthenticated, on a
    // public host. validate_production() also refuses to boot with this true.
    { name = "ENABLE_API_DOCS", value = "false" },
    // There is no self-service signup: users are created by an administrator. The
    // first Admin is created out of band — see infra/README.
    { name = "ALLOW_SELF_REGISTRATION", value = "false" },
  ])

  secrets = local.common_secrets

  // Execution-role read list. Includes the AWS-managed RDS master secret because the
  // migrator reuses this role and injects the master credential from it; omit it and
  // that task cannot start at all ("unable to pull secrets") — a boot failure, not a
  // runtime error.
  secret_arns = concat(local.secret_iam_arns, [module.rds.master_secret_arn], aws_secretsmanager_secret.tunnel_token[*].arn)
  kms_key_arn = local.kms_key_arn

  // The tunnel AND clamav. The api scans uploads synchronously in-process
  // (articles.py -> extract_source_pages -> _validate_source_bytes), so it needs a
  // clamd of its own on localhost; the worker's is in a different task.
  additional_containers = concat(
    module.tunnel_api.container_definitions,
    [for container in local.clamav_container : merge(container, {
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = local.api_log_group
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "clamav"
        }
      }
    })],
  )

  tags = local.tags
}

// ── Worker service — Celery worker + beat + ClamAV ───────────────────────────
// Three containers, one task, on purpose:
//   worker  the Celery consumer (embedding, OCR, connector sync)
//   beat    the scheduler. A SINGLETON — two of these double every scheduled job —
//           which is why var.worker caps max_count at 1.
//   clamav  the malware scanner the worker and api talk to over localhost
module "worker" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/ecs-service?ref=ecs-service-v2.3.2"

  service_name = "worker"
  cluster_name = module.ecs_cluster.cluster_name
  cluster_arn  = module.ecs_cluster.cluster_arn
  region       = var.region
  image_uri    = local.ecr_worker_url

  cpu    = var.worker.cpu
  memory = var.worker.memory

  vpc_id            = data.terraform_remote_state.runtime.outputs.vpc_id
  subnet_ids        = data.terraform_remote_state.runtime.outputs.private_subnet_ids
  security_group_id = data.terraform_remote_state.runtime.outputs.sg_app_id

  desired_count      = 1
  enable_autoscaling = var.worker.enable_autoscaling
  min_count          = var.worker.min_count
  max_count          = var.worker.max_count
  use_spot           = var.worker.use_spot
  log_retention_days = var.log_retention_days

  attach_alb     = false
  container_port = 8001

  // No HTTP surface: ask Celery whether the consumer is actually serving. `pgrep
  // celery` would pass for a process that is running but has lost its broker
  // connection, which is the failure worth catching.
  health_check_command = "celery -A src.workers.celery_app inspect ping -d celery@$HOSTNAME || exit 1"

  environment_vars = local.common_env
  secrets          = concat(local.common_secrets, local.mail_secrets)

  // The mail secret is granted here, on the WORKER's execution role, and nowhere else. No task role in this
  // stack gets any secretsmanager permission (task_secret_arns is never passed).
  secret_arns = concat(local.secret_iam_arns, [module.rds.master_secret_arn], local.mail_graph_enabled ? [aws_secretsmanager_secret.mail_graph[0].arn] : [])
  kms_key_arn = local.kms_key_arn

  additional_containers = concat([
    {
      name      = "beat"
      image     = local.ecr_worker_url
      essential = true
      command   = ["celery", "-A", "src.workers.celery_app", "beat", "--loglevel=info"]
      // Carved out of the task total, not added to it. Beat only computes due times
      // and enqueues — it neither embeds nor extracts.
      cpu         = 128
      memory      = 256
      environment = local.common_env
      secrets     = [for s in local.common_secrets : { name = s.name, valueFrom = s.secret_arn }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = local.worker_log_group
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "beat"
        }
      }
    },
    ], [for container in local.clamav_container : merge(container, {
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = local.worker_log_group
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "clamav"
        }
      }
  })])

  tags = local.tags
}

// The SES identity and its DKIM CNAME records are NOT declared here. qnsc-infra's `edge` stack owns the one
// identity for qnsc.vn that every product shares (an SES identity is per account and region), with the three
// DKIM records in Cloudflare. This stack declared them as well for a while, so two states owned the same
// objects and each apply rewrote the other's tags, comment and TTL. The product only needs permission to send
// as the identity, granted below by ARN.
//
// Grants the worker task (the only caller of get_email_sender()) permission to send
// through SES. Created whenever mail_from_email is set, NOT only while email_provider is "ses": this is the
// "expand" half of expand/contract (infra/README.md). Switching develop to Graph must not remove the grant in
// the same apply, because the ECS circuit breaker can roll the service back to a task definition that still
// says EMAIL_PROVIDER=ses. The grant is deleted in the follow-up change (kb-backend #216) once Graph is proven.
// The ecs-service module exposes task_role_arn but not a role NAME output, so the role name is derived from
// the ARN's final path segment. Scoped to the identity above, by ARN.
resource "aws_iam_role_policy" "worker_ses_send" {
  count = var.mail_from_email != "" ? 1 : 0
  name  = "ses-send"
  role  = split("/", module.worker.task_role_arn)[1]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "SendApplicationEmail"
      Effect = "Allow"
      Action = ["ses:SendEmail", "ses:SendRawEmail"]
      // The shared identity, by ARN (edge's `ses_identity_arn` output builds the same string).
      Resource = "arn:aws:ses:${var.region}:${data.aws_caller_identity.current.account_id}:identity/${local.mail_from_domain}"
    }]
  })
}

// ── Migrator — one-shot task run by the deploy pipeline before rolling services ──
module "migrator" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/oneshot-task?ref=oneshot-task-v2.0.0"

  name               = "${local.name}-migrator"
  container_name     = "migrator"
  image              = local.ecr_migrator_url
  cpu                = 512
  memory             = 1024
  execution_role_arn = module.api.execution_role_arn
  task_role_arn      = module.api.task_role_arn
  region             = var.region
  log_retention_days = var.log_retention_days

  environment = {
    ENVIRONMENT   = "production"
    AWS_REGION    = var.region
    DATABASE_HOST = module.rds.address
    DATABASE_PORT = tostring(module.rds.port)
    DATABASE_NAME = module.rds.db_name
    // Read by migration 20260802_05 to decide whether to enable row-level security and
    // whom to grant to, and by scripts/bootstrap_db_role.py to create that role.
    ENABLE_RLS        = "true"
    APP_DATABASE_ROLE = var.app_db_role

    // The migrations own the pgvector COLUMN WIDTH, and the width is derived from
    // EMBEDDING_MODEL. Left unset here, the migrator fell back to the code default while
    // the api and worker used this variable, and the column was created 1024 wide for a
    // model that emits 768 — every chunk then failed to index with
    // "expected 1024 dimensions, not 768". Same value as common_env, deliberately: the
    // task that shapes the column and the tasks that fill it must agree.
    EMBEDDING_MODEL   = var.embedding_model
    EMBEDDING_VERSION = var.embedding_version

    // The realign migrations' opt-in, and the migrator is the ONLY task that reads it —
    // it is the task that runs them. src/core/config.py grew the setting so the wipe
    // could be an explicit, reviewable step instead of an ad-hoc DELETE against a live
    // database, but nothing passed it here, so the sanctioned path past the guard did not
    // exist in a deployed environment. See variables.tf for what happened on develop.
    EMBEDDING_REALIGN_DISCARD_VECTORS = tostring(var.embedding_realign_discard_vectors)
  }

  secrets = {
    // The MASTER credential, and it stays master even though the api and worker do not
    // use it: migrations create extensions and grant privileges, and the role bootstrap
    // needs CREATEROLE. Narrowing this additionally requires transferring schema
    // ownership, which is a separate and more disruptive step.
    MIGRATION_DATABASE_USER     = "${module.rds.master_secret_arn}:username::"
    MIGRATION_DATABASE_PASSWORD = "${module.rds.master_secret_arn}:password::"
    // The password the app role is created WITH, so the role and the application's
    // injected DATABASE_PASSWORD can never disagree — both read this one secret.
    APP_DATABASE_PASSWORD = module.secrets.secret_arns["app-db-password"]
  }

  tags = local.tags
}

// ── SPA — Cloudflare Pages ────────────────────────────────────────────────────
// Deployed from the SEPARATE qnsc-kb-frontend repo. This creates the project and its
// custom domain; the project NAME is what that repo's deploy workflow needs, published
// to it as the PAGES_PROJECT environment variable by infra-apply.
//
// No production_env_vars: unlike rova, this SPA is not a Pages-Functions BFF. It calls
// the API directly with a bearer token, and VITE_API_BASE_URL is baked in at build
// time by the frontend's own pipeline.
module "web" {
  count  = var.cloudflare_account_id != "" ? 1 : 0
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/pages-web?ref=pages-web-v1.0.1"

  account_id  = var.cloudflare_account_id
  name        = "${local.name}-web"
  zone_id     = local.cloudflare_zone_id
  domain      = local.cloudflare_zone_id != "" ? var.app_domain : ""
  record_name = local.cloudflare_zone_id != "" ? var.web_record : ""
  comment     = "${local.name} web SPA → Cloudflare Pages (managed by ${var.product} infra, ${var.env})"
}

// ── DNS — api_domain → the tunnel ────────────────────────────────────────────
module "dns_api" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/dns-record?ref=dns-record-v1.1.0"

  enabled = local.cloudflare_zone_id != "" && length(module.tunnel) > 0
  zone_id = local.cloudflare_zone_id
  name    = var.api_record
  type    = "CNAME"
  // Read from the tunnel resource rather than assembled from its id: a Cloudflare-
  // internal name that resolves only through the edge, so the record CANNOT be unproxied
  // — orange cloud is the only way traffic reaches a connector.
  content = one(module.tunnel[*].cname)
  proxied = true
  comment = "${local.name} api → Cloudflare Tunnel"
}

// ── Plan-time guards ─────────────────────────────────────────────────────────
// Each of these encodes a failure that is invisible at apply time and only surfaces
// once a task tries to start.

check "tunnel_needs_cloudflare_account" {
  assert {
    condition     = !var.tunnel_enabled || var.cloudflare_account_id != ""
    error_message = "tunnel_enabled is true but cloudflare_account_id is empty, so no tunnel is created and the api has no ingress. Supply the account id (CI passes it from the org variable) or set tunnel_enabled = false."
  }
}

check "cache_required_for_jobs" {
  # SUPERSEDED 2026-09-12 — the assertion moved to a `validation` on var.cache in
  # variables.tf, because a `check` does not enforce: a violated assert emits
  # `Warning: Check block assertion failed` and the plan exits 0, so this block permitted
  # exactly the state it describes. Kept as a marker only; the condition below is now
  # redundant with the validation, which is deliberate — it fails at plan time instead.
  #
  # Safe to delete on the next edit to this file.
  assert {
    condition     = var.cache.enabled || (var.api.min_count == 0 && var.worker.min_count == 0)
    error_message = "The cache is the Celery broker: with it disabled no ingestion, connector sync or outbox relay runs at all. Scale both services to zero, or enable the cache."
  }
}

check "malware_scan_matches_production_mode" {
  assert {
    condition     = var.malware_scan_enabled
    error_message = "ENVIRONMENT is pinned to production in every environment, and validate_production() refuses to boot when MALWARE_SCAN_ENABLED is false. Disabling the scanner requires dropping out of production mode, which also unpins API docs and self-registration."
  }
}

// ── Idling ────────────────────────────────────────────────────────────────────
// Stops the database and takes both services to zero. This is the whole cost posture
// for a non-production environment: Fargate and RDS bill for time, and this environment
// is exercised by CI deploys and occasional manual checks rather than by users.
//
// The cache is NOT stopped, because ElastiCache has no stopped state — only delete —
// and it is the Celery broker rather than an optional cache. It is the one component of
// an idled environment that keeps billing.
resource "aws_iam_role" "idler" {
  count = var.idle_schedule == null ? 0 : 1
  name  = "${local.name}-idler"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
      // Confused-deputy guard: without it, a schedule in any other account could assume
      // this role. Scoped to this account's schedules only.
      Condition = { StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id } }
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "idler" {
  count = var.idle_schedule == null ? 0 : 1
  name  = "idle-environment"
  role  = aws_iam_role.idler[0].id

  // Stop only. Not Start, not Reboot: this schedule's entire job is to remove capacity,
  // and a role that can also start an instance turns a scheduling mistake into a cost
  // increase. Waking has its own role below, and the deploy pipeline has its own grant
  // in live/_shared.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "StopDatabase"
        Effect   = "Allow"
        Action   = "rds:StopDBInstance"
        Resource = module.rds.instance_arn
      },
      {
        // Scaling to zero as well as stopping the database, because stopping only the
        // database leaves Fargate tasks running against an instance they cannot reach:
        // still billed, unable to serve, and — since /health/live answers 200 without
        // touching a dependency — reporting themselves healthy the whole time.
        Sid    = "ScaleServicesToZero"
        Effect = "Allow"
        Action = "ecs:UpdateService"
        Resource = [
          module.api.service_arn,
          module.worker.service_arn,
        ]
      },
    ]
  })
}

resource "aws_scheduler_schedule" "rds_stop" {
  count       = var.idle_schedule == null ? 0 : 1
  name        = "${local.name}-rds-stop"
  description = "Stops ${module.rds.identifier}; see var.idle_schedule for why this exists"

  schedule_expression          = var.idle_schedule
  schedule_expression_timezone = "Asia/Ho_Chi_Minh"

  // OFF, not a window: this is not load-sensitive work, and an exact time keeps the
  // relationship between a run and its CloudTrail entry unambiguous.
  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:rds:stopDBInstance"
    role_arn = aws_iam_role.idler[0].arn
    input    = jsonencode({ DbInstanceIdentifier = module.rds.identifier })

    // No retries and no dead-letter queue, deliberately. The common outcome is
    // InvalidDBInstanceState because the instance is ALREADY STOPPED — the desired
    // state, not an error. Retrying would generate noise for a success and a DLQ would
    // collect messages nobody should act on. A real permissions failure still shows up
    // in CloudTrail and in the schedule's own metrics.
    retry_policy {
      maximum_retry_attempts = 0
    }
  }
}

// `desired_count` is under `ignore_changes` in the ecs-service module, so setting it out
// of band is the sanctioned, non-drifting mechanism — which is why this uses
// ecs:UpdateService rather than an Application Auto Scaling scheduled action. A
// scheduled action mutates the scalable target's min/max, and aws_appautoscaling_target
// has no ignore_changes on those, so every plan would show drift and any apply during
// the idle window would silently wake the environment.
//
// A floor of 0 on both services is what makes this hold: with a floor of 1, Application
// Auto Scaling restores the service within minutes and the scale-to-zero undoes itself.
resource "aws_scheduler_schedule" "ecs_scale_down" {
  for_each = var.idle_schedule == null ? {} : {
    api    = module.api.service_name
    worker = module.worker.service_name
  }

  name        = "${local.name}-${each.key}-scale-down"
  description = "Scales ${each.value} to zero; see var.idle_schedule"

  schedule_expression          = var.idle_schedule
  schedule_expression_timezone = "Asia/Ho_Chi_Minh"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ecs:updateService"
    role_arn = aws_iam_role.idler[0].arn
    input = jsonencode({
      Cluster      = module.ecs_cluster.cluster_name
      Service      = each.value
      DesiredCount = 0
    })

    // Idempotent — scaling an already-zero service to zero succeeds — so unlike the RDS
    // stop there is no expected-failure case here. Retries stay off for consistency; a
    // missed run is corrected by the next tick.
    retry_policy {
      maximum_retry_attempts = 0
    }
  }
}

// ── Waking ────────────────────────────────────────────────────────────────────
// The reverse of idling, on its own cron. See var.wake_schedule for why this exists
// even though every deploy already wakes the environment.
resource "aws_iam_role" "waker" {
  count = var.wake_schedule == null ? 0 : 1
  name  = "${local.name}-waker"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id } }
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "waker" {
  count = var.wake_schedule == null ? 0 : 1
  name  = "wake-environment"
  role  = aws_iam_role.waker[0].id

  // Start only, mirroring the idler's stop only. No rds:StopDBInstance, no Reboot, no
  // Delete: this role's entire job is to add capacity back.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "StartDatabase"
        Effect   = "Allow"
        Action   = "rds:StartDBInstance"
        Resource = module.rds.instance_arn
      },
      {
        Sid    = "RestoreServices"
        Effect = "Allow"
        Action = "ecs:UpdateService"
        Resource = [
          module.api.service_arn,
          module.worker.service_arn,
        ]
      },
    ]
  })
}

resource "aws_scheduler_schedule" "rds_start" {
  count       = var.wake_schedule == null ? 0 : 1
  name        = "${local.name}-rds-start"
  description = "Starts ${module.rds.identifier}; see var.wake_schedule for why this exists"

  schedule_expression          = var.wake_schedule
  schedule_expression_timezone = "Asia/Ho_Chi_Minh"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:rds:startDBInstance"
    role_arn = aws_iam_role.waker[0].arn
    input    = jsonencode({ DbInstanceIdentifier = module.rds.identifier })

    // Mirror of the stop schedule: starting an already-started instance fails with
    // InvalidDBInstanceState, which is the desired state rather than an error.
    retry_policy {
      maximum_retry_attempts = 0
    }
  }
}

resource "aws_scheduler_schedule" "ecs_scale_up" {
  for_each = var.wake_schedule == null ? {} : {
    api    = module.api.service_name
    worker = module.worker.service_name
  }

  name        = "${local.name}-${each.key}-scale-up"
  description = "Restores ${each.value} to one task; see var.wake_schedule"

  schedule_expression          = var.wake_schedule
  schedule_expression_timezone = "Asia/Ho_Chi_Minh"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ecs:updateService"
    role_arn = aws_iam_role.waker[0].arn
    input = jsonencode({
      Cluster      = module.ecs_cluster.cluster_name
      Service      = each.value
      DesiredCount = 1
    })

    // One task, never more, for BOTH services. The worker carries Celery beat as a
    // container, and beat is a singleton — restoring two worker tasks would double every
    // scheduled job. The api's own ceiling is var.api.max_count, which autoscaling would
    // handle if it were enabled.
    retry_policy {
      maximum_retry_attempts = 0
    }
  }
}

// ── Alarms ────────────────────────────────────────────────────────────────────
// CloudWatch alarms plus an SNS topic. Deliberately small: at $0.10 per alarm the cost
// is negligible, but an alarm nobody can act on is worse than none.
//
// `environment_idle` is DERIVED from the service floors rather than being its own
// switch, so it cannot disagree with reality. It suppresses every alarm whose premise is
// "this environment is serving traffic" — ECS CPU and memory, ALB 5xx, unhealthy hosts —
// because a service scaled to zero makes its CPU metric disappear, and the alarm then
// walks OK -> INSUFFICIENT_DATA -> OK on every wake. Raising the floors re-arms them
// automatically.
//
// The RDS alarms are NOT suppressed and need no special case: a stopped instance
// publishes nothing, which reads as INSUFFICIENT_DATA rather than ALARM, so the nightly
// stop does not page anyone.
//
// No ALB is passed. Ingress is a Cloudflare Tunnel, so there is no load balancer to
// measure and Cloudflare — not CloudWatch — sees the 5xx. Closing that gap needs an
// external health check, which belongs at go-live rather than against an environment
// deliberately running zero tasks.
module "observability" {
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/observability?ref=observability-v4.3.0"

  name             = local.name
  region           = var.region
  alarm_emails     = var.alarm_emails
  ecs_cluster_name = module.ecs_cluster.cluster_name

  ecs_service_names = [
    module.api.service_name,
    module.worker.service_name,
  ]

  rds_instance_id = module.rds.identifier

  // Node mode only, and only a node THIS product owns. qnsc-kb shares the runtime
  // layer's Valkey (cache.shared = true, db_index 1), and a shared node's alarms belong
  // to whoever creates it — otherwise every product borrowing an index would duplicate
  // alarms on the same cluster.
  //
  // Computed rather than left to the module defaults (false / "") so that flipping this
  // product to a dedicated node starts alarming it automatically. Hardcoding false would
  // hand the next person a node with no CPU, eviction or memory alarms and nothing saying
  // why. Mirrors rova and opshub, which compute the same two expressions.
  //
  // enable_cache_alarms is deliberately a SEPARATE condition from cache_cluster_id's
  // value: on a from-scratch environment the node's cluster_id is unknown until apply,
  // and gating a count on that directly is a hard OpenTofu error rather than a deferred
  // plan. This form is known at plan time regardless.
  enable_cache_alarms = var.cache.enabled && !var.cache.shared && var.cache.mode == "node"
  cache_cluster_id    = var.cache.enabled && !var.cache.shared && var.cache.mode == "node" ? module.cache[0].cluster_id : ""

  environment_idle = var.api.min_count == 0 && var.worker.min_count == 0

  thresholds = var.alarm_thresholds

  // Three dashboards are free per ACCOUNT and several environments already exist across
  // products, so a fourth is billable. The alarms carry the signal; a dashboard is for
  // looking at, and nobody is looking at an idle environment.
  create_dashboard = false

  // Treats "no registered targets" as breaching, which is right for an always-on
  // environment and wrong for one that is Spot-backed and scheduled to zero.
  monitor_target_health = false

  tags = local.tags
}

// ── Worker health: two signals ───────────────────────────────────────────────
// The worker went down because a container in its task could not pull its image. ECS kept retrying, the deploy
// only verified the API, the CPU/memory alarms have no data from a task that is not running, and the queue stood
// still until someone noticed.
//
// MEASURED in develop (2026-10-14, alarm test): with the worker at 0 running tasks, Container Insights published
// NO datapoint for RunningTaskCount and NO datapoint for DesiredTaskCount, and resumed only when a task was
// pending again. So "desired > running" in any form (including FILL) evaluates 0 > 0 during exactly the outage
// it was meant to catch. Missing data IS the outage signal. Two independent signals, therefore:
//
//  1. FAST, event based (EventBridge, no Container Insights needed): ECS says so itself the moment a worker task
//     fails to start (CannotPullContainerError, ResourceInitializationError, TaskFailedToStart), a service
//     cannot start or place tasks, or a deployment fails. Sent to the alarm topic with the service and reason.
//  2. SUSTAINED, metric based: RunningTaskCount < 1 for 10 of 10 minutes with missing data treated as BREACHING.
//     It is only correct for a service that is meant to be running ALL the time. An idle schedule scales the
//     worker to 0 on purpose, publishes no data, and would alarm every night, so it exists only while
//     `idle_schedule` is null. THE PAIRING: re-enabling idle_schedule disables this alarm (and removing it, as the
//     production go-live checklist does, arms it). It is derived from idle_schedule so the two cannot disagree.

locals {
  worker_always_on_alarm = var.worker_task_alarm && var.idle_schedule == null
}

resource "aws_cloudwatch_metric_alarm" "worker_not_running" {
  count = local.worker_always_on_alarm ? 1 : 0

  alarm_name          = "${local.name}-worker-not-running"
  alarm_description   = "The worker has had no running task (or no metric at all, which is what zero tasks looks like in Container Insights) for 10 minutes. Check the service events and the stopped tasks' stoppedReason (an image that cannot be pulled, a failing essential container, no capacity). Only exists while idle_schedule is null: an idle schedule scales the worker to zero on purpose."
  namespace           = "ECS/ContainerInsights"
  metric_name         = "RunningTaskCount"
  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = 10
  datapoints_to_alarm = 10
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"

  dimensions = {
    ClusterName = module.ecs_cluster.cluster_name
    ServiceName = module.worker.service_name
  }

  alarm_actions = [module.observability.alarm_topic_arn]
  ok_actions    = [module.observability.alarm_topic_arn]
  tags          = local.tags

  lifecycle {
    precondition {
      condition     = var.container_insights != "disabled"
      error_message = "worker_task_alarm needs Container Insights (RunningTaskCount). Set container_insights to \"enabled\", or worker_task_alarm = false."
    }
  }
}

// The alarm topic has no resource policy in the observability module (CloudWatch alarms publish under the
// account's default policy). EventBridge is a service principal and needs an explicit statement. The default
// statement is reproduced as it is, so alarms keep working, and one statement for EventBridge is added.
resource "aws_sns_topic_policy" "alarms_from_events" {
  count = var.worker_event_alerts ? 1 : 0

  arn = module.observability.alarm_topic_arn
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "__default_statement_ID"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Action = [
          "SNS:GetTopicAttributes", "SNS:SetTopicAttributes", "SNS:AddPermission", "SNS:RemovePermission",
          "SNS:DeleteTopic", "SNS:Subscribe", "SNS:ListSubscriptionsByTopic", "SNS:Publish",
        ]
        Resource  = module.observability.alarm_topic_arn
        Condition = { StringEquals = { "AWS:SourceOwner" = data.aws_caller_identity.current.account_id } }
      },
      {
        Sid       = "WorkerEventsPublish"
        Effect    = "Allow"
        Principal = { Service = "events.amazonaws.com" }
        Action    = "sns:Publish"
        Resource  = module.observability.alarm_topic_arn
        Condition = { ArnLike = { "aws:SourceArn" = "arn:aws:events:${var.region}:${data.aws_caller_identity.current.account_id}:rule/${local.name}-worker-*" } }
      },
    ]
  })
}

locals {
  worker_event_rules_all = {
    task-failed-to-start = {
      description = "A worker task was stopped because it could not start (image pull, resource initialization) or an essential container exited (crash loop)."
      pattern = {
        source        = ["aws.ecs"]
        "detail-type" = ["ECS Task State Change"]
        detail = {
          clusterArn = [module.ecs_cluster.cluster_arn]
          group      = ["service:${module.worker.service_name}"]
          lastStatus = ["STOPPED"]
          "$or" = [
            { stopCode = ["TaskFailedToStart"] },
            # A crash-looping worker: the task starts, an essential container (worker or beat) exits, ECS restarts it.
            # The sustained alarm may never see 10 minutes in a row with zero tasks when one is up for a minute in between.
            { stopCode = ["EssentialContainerExited"] },
            { stoppedReason = [{ wildcard = "*CannotPullContainerError*" }] },
            { stoppedReason = [{ wildcard = "*ResourceInitializationError*" }] },
          ]
        }
      }
      paths    = { service = "$.detail.group", reason = "$.detail.stoppedReason", at = "$.time" }
      template = "\"qnsc-kb worker task stopped (failed to start or crashed): <service>. Reason: <reason>. At <at>. Cluster ${module.ecs_cluster.cluster_name}.\""
    }
    service-impaired = {
      description = "The worker service cannot start or place tasks."
      pattern = {
        source        = ["aws.ecs"]
        "detail-type" = ["ECS Service Action"]
        resources     = [module.worker.service_arn]
        detail        = { eventName = ["SERVICE_TASK_START_IMPAIRED", "SERVICE_TASK_PLACEMENT_FAILURE"] }
      }
      paths    = { event = "$.detail.eventName", service = "$.resources[0]", at = "$.time" }
      template = "\"qnsc-kb worker service problem: <event> on <service>. At <at>. Cluster ${module.ecs_cluster.cluster_name}.\""
    }
    deployment-failed = {
      description = "A deployment of the worker service failed."
      pattern = {
        source        = ["aws.ecs"]
        "detail-type" = ["ECS Deployment State Change"]
        resources     = [module.worker.service_arn]
        detail        = { eventName = ["SERVICE_DEPLOYMENT_FAILED"] }
      }
      paths    = { service = "$.resources[0]", reason = "$.detail.reason", at = "$.time" }
      template = "\"qnsc-kb worker deployment FAILED: <service>. <reason>. At <at>. Cluster ${module.ecs_cluster.cluster_name}.\""
    }
  }
  worker_event_rules = { for name, rule in local.worker_event_rules_all : name => rule if var.worker_event_alerts }
}

// Event-driven, free, and independent of Container Insights. Noise: ECS backs off its own restarts of a task that
// keeps failing, so a persistent fault produces a trickle of task-stopped events rather than a flood; if it
// is still too chatty, narrow the first rule to `stopCode = TaskFailedToStart` only. There is no dedupe here.
resource "aws_cloudwatch_event_rule" "worker" {
  for_each = local.worker_event_rules

  name          = "${local.name}-worker-${each.key}"
  description   = each.value.description
  event_pattern = jsonencode(each.value.pattern)
  tags          = local.tags
}

resource "aws_cloudwatch_event_target" "worker_to_alarm_topic" {
  for_each = local.worker_event_rules

  rule = aws_cloudwatch_event_rule.worker[each.key].name
  arn  = module.observability.alarm_topic_arn

  input_transformer {
    input_paths    = each.value.paths
    input_template = each.value.template
  }

  depends_on = [aws_sns_topic_policy.alarms_from_events]
}

// ── Embedding model health ───────────────────────────────────────────────────
// Nothing else watches this. `/health/ready` is not on any load balancer's path here (ingress is
// a Cloudflare Tunnel), the deploy polls `/health/ready`, and a model that failed to load or took
// longer than EMBEDDING_WARMUP_TIMEOUT_SECONDS leaves the api answering 200 with vector search
// degraded. The api logs one of these two lines (src/api/main.py::_preload_embedding_model) when
// that happens; this turns them into a metric and an alarm on the shared alarm topic.
//
// Terms rather than a JSON selector so it matches whether the line is rendered as JSON or text.
// NOT TESTED against a live log group (no AWS access when written): after the first apply, run
// `aws logs test-metric-filter` with a real line, or provoke one in develop.
//
// treat_missing_data is notBreaching because a scheduled-to-zero environment publishes no logs, and
// an alarm that walks to INSUFFICIENT_DATA every night teaches people to ignore it.
resource "aws_cloudwatch_log_metric_filter" "embedding_model_unhealthy" {
  name           = "${local.name}-api-embedding-model-unhealthy"
  log_group_name = local.api_log_group
  pattern        = "?\"Embedding model preload failed\" ?\"Embedding model warm-up timed out\""

  metric_transformation {
    name      = "EmbeddingModelUnhealthy"
    namespace = "QNSC/${local.name}"
    value     = "1"
  }

  depends_on = [module.api]
}

resource "aws_cloudwatch_metric_alarm" "embedding_model_unhealthy" {
  alarm_name          = "${local.name}-api-embedding-model-unhealthy"
  alarm_description   = "The api could not load the embedding model, or took longer than EMBEDDING_WARMUP_TIMEOUT_SECONDS. Vector search is degraded to keyword search until it loads. Check the api log group for the warm-up lines."
  namespace           = "QNSC/${local.name}"
  metric_name         = "EmbeddingModelUnhealthy"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [module.observability.alarm_topic_arn]
  tags          = local.tags
}

// ── Mail alarms: three worker log lines, each an alarm on the shared topic ──────────────────────────────────
// The worker is where mail is sent (src/workers/tasks.py), so these watch the worker log group. Terms rather
// than a JSON selector, like embedding_model_unhealthy above, and notBreaching for the same reason (a
// scheduled-to-zero environment publishes no logs). NOT TESTED against a live log group (no AWS access when
// written): after the first apply run `aws logs test-metric-filter` with a real line.
locals {
  mail_alarms = {
    not-configured = {
      metric      = "MailNotConfigured"
      pattern     = "\"Mail is not configured\""
      description = "EMAIL_PROVIDER=graph but no mail app is configured (mail_graph_client_id is empty), or the provider is not allowed here: no notification email is being delivered. Logged on every 30 s delivery pass. See docs/runbooks/email-graph.md."
    }
    certificate = {
      metric      = "MailCertificateExpiring"
      pattern     = "?\"Mail certificate expires soon\" ?\"Mail certificate has expired\""
      description = "The mail certificate has under 30 days left, has expired, or cannot be read. Checked daily by the worker. Rotate it: docs/runbooks/email-graph.md."
    }
    sender-refused = {
      metric      = "MailSenderRefused"
      pattern     = "\"Mail sender refused\""
      description = "Graph refused the app the sender mailbox (HTTP 403 or 404): the Exchange management scope or role assignment is missing or still propagating (up to ~2 hours), or the mailbox does not exist or is not REST-enabled. Nothing is lost: the batch stops and is retried every 15 minutes. See docs/runbooks/email-graph.md, steps 1, 2 and 6."
    }
    dead = {
      metric      = "MailNotificationDead"
      pattern     = "\"Notification marked dead\""
      description = "A notification email was given up on: Graph refused the message itself (400/413) or it is older than 24 hours. Find the rows with status 'dead' in notification_queue; last_error says why."
    }
  }
}

resource "aws_cloudwatch_log_metric_filter" "mail" {
  for_each = local.mail_alarms

  name           = "${local.name}-worker-mail-${each.key}"
  log_group_name = local.worker_log_group
  pattern        = each.value.pattern

  metric_transformation {
    name      = each.value.metric
    namespace = "QNSC/${local.name}"
    value     = "1"
  }

  depends_on = [module.worker]
}

resource "aws_cloudwatch_metric_alarm" "mail" {
  for_each = local.mail_alarms

  alarm_name          = "${local.name}-worker-mail-${each.key}"
  alarm_description   = each.value.description
  namespace           = "QNSC/${local.name}"
  metric_name         = each.value.metric
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [module.observability.alarm_topic_arn]
  tags          = local.tags
}
