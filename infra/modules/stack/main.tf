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
    { name = "GEMINI_MODEL", value = var.gemini_model },

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

    // Who application email is sent FROM. Without it every invitation and password reset
    // queues a notification row that raises "MICROSOFT_GRAPH_SENDER is not configured"
    // every 30 seconds forever, and the recipient never gets a link. ENVIRONMENT is pinned
    // to "production" here, so the development FakeEmailSender is never selected and there
    // is no fallback. validate_production() does not check this, so stating it is the only
    // thing that makes outbound mail work.
    { name = "MICROSOFT_GRAPH_SENDER", value = var.microsoft_graph_sender },

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

    // Who may sign in, and who arrives as an administrator. Both stated here rather than
    // inherited from a code default: today develop's access rules happen to be correct
    // only because config.py defaults to "qnsc.vn", so a change to that default would
    // silently change who can get an account.
    //
    // Anyone in the pinned tenant with a matching domain is auto-provisioned at the
    // least-privileged Staff role; the addresses below arrive as global administrators
    // instead. That list is read only when an account is FIRST created, so it never
    // overrides a role set later in the admin UI.
    { name = "ENTRA_AUTO_PROVISION_DOMAIN", value = var.entra_auto_provision_domain },
    { name = "ENTRA_ADMIN_EMAILS", value = join(",", var.entra_admin_emails) },

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
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/rds?ref=rds-v2.1.2"

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

  tags = local.tags
}

// ── Cache (Valkey) — Celery broker and result backend ────────────────────────
module "cache" {
  # NOT created when this product uses the shared node. Switching a live environment to
  # `shared` destroys the dedicated node — that is where the saving is — and issues a
  # different endpoint, so it is a task-definition revision and a rolling deploy.
  count  = var.cache.enabled && !var.cache.shared ? 1 : 0
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/cache?ref=cache-v1.0.0"

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
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/cf-tunnel?ref=cf-tunnel-v0.1.1"

  account_id = var.cloudflare_account_id
  // One tunnel per product per environment. Sharing one across environments would let a
  // develop task serve production traffic, because a tunnel routes to whichever
  // connectors hold its token.
  name = local.name

  // Without these the connector runs, reports healthy, and answers 503 to everything —
  // "No ingress rules were defined". v0.1.1 is the first version that creates them.
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

resource "aws_secretsmanager_secret_version" "tunnel_token" {
  count = var.tunnel_enabled && var.cloudflare_account_id != "" ? 1 : 0

  secret_id     = aws_secretsmanager_secret.tunnel_token[0].id
  secret_string = module.tunnel[0].token
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
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/ecs-service?ref=ecs-service-v2.1.1"

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
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/ecs-service?ref=ecs-service-v2.1.1"

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
  secrets          = local.common_secrets

  secret_arns = concat(local.secret_iam_arns, [module.rds.master_secret_arn])
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

// Grants the worker task (the only caller of get_email_sender()) permission to send
// through SES. Only created when email_provider is actually "ses" — the ecs-service
// module exposes task_role_arn but not a role NAME output, so the role name is derived
// from the ARN's final path segment. Resource is "*" rather than a specific identity
// ARN because no SES identity is created by this stack; once mail_from_email's domain
// or address identity is verified some other way, scope this down to its ARN.
resource "aws_iam_role_policy" "worker_ses_send" {
  count = var.email_provider == "ses" ? 1 : 0
  name  = "ses-send"
  role  = split("/", module.worker.task_role_arn)[1]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "SendApplicationEmail"
      Effect   = "Allow"
      Action   = ["ses:SendEmail", "ses:SendRawEmail"]
      Resource = "*"
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
  source = "git::https://github.com/quynhonsemiconductor/tf-modules.git//modules/observability?ref=observability-v4.1.0"

  name             = local.name
  region           = var.region
  alarm_emails     = var.alarm_emails
  ecs_cluster_name = module.ecs_cluster.cluster_name

  ecs_service_names = [
    module.api.service_name,
    module.worker.service_name,
  ]

  rds_instance_id = module.rds.identifier

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
