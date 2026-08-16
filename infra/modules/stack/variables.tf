// Inputs an ENVIRONMENT chooses. Anything derived from these lives in locals in
// main.tf, so develop and prod cannot drift in how a value is assembled — only in
// what they feed in.

variable "product" {
  type    = string
  default = "qnsc-kb"
}

variable "env" {
  type        = string
  description = "Full environment name (develop | production). Used in tags and DEPLOYMENT_ENV."
}

variable "env_slug" {
  type        = string
  description = "Short environment name used in resource names (develop | prod)."
}

variable "region" {
  type    = string
  default = "ap-southeast-1"
}

// ── Remote state keys ────────────────────────────────────────────────────────
variable "shared_state_key" {
  type        = string
  description = "State key of this product's _shared stack (ECR + OIDC + re-exported platform outputs)."
}

variable "runtime_state_key" {
  type        = string
  description = "State key of the platform runtime stack (shared VPC/subnets/SGs). Must be applied first."
}

variable "storage_state_key" {
  type        = string
  description = "State key of the platform storage stack (Cloudflare R2 buckets)."
}

// ── Naming and ingress ───────────────────────────────────────────────────────
variable "app_domain" {
  type        = string
  description = "Public hostname of the SPA, e.g. kb-dev.qnsc.vn."
}

variable "api_domain" {
  type        = string
  description = "Public hostname of the API, e.g. kb-api-dev.qnsc.vn."
}

variable "web_record" {
  type        = string
  description = "DNS record name for the SPA (host part of app_domain)."
}

variable "api_record" {
  type        = string
  description = "DNS record name for the API (host part of api_domain)."
}

variable "image_tag" {
  type        = string
  description = "Container image tag. develop tracks \"latest\"; production pins the release tag."
}

// ── Ingress via Cloudflare Tunnel ────────────────────────────────────────────
variable "tunnel_enabled" {
  type    = bool
  default = true

  description = <<-EOT
    Serve the API through a cloudflared sidecar that dials OUT to Cloudflare, instead
    of an ALB target group.

    Default true, unlike rally's, because there is no shared ALB to fall back on:
    `enable_alb` is false on both platform runtime stacks (since 2026-08-02, after
    rally's own tunnel cutover — an idle load balancer with no target groups is
    $25.70/mo for nothing). Setting this false therefore requires turning that ALB
    back on first.
  EOT
}


// ── Sizing ───────────────────────────────────────────────────────────────────
variable "api" {
  type = object({
    cpu                = number
    memory             = number
    min_count          = number
    max_count          = number
    enable_autoscaling = optional(bool, false)
    use_spot           = optional(bool, false)
    cpu_target_pct     = optional(number, 65)
    memory_target_pct  = optional(number, 75)
  })

  description = <<-EOT
    API task sizing. Memory is the load-bearing number here, not CPU: the API answers
    search queries by embedding the query text, and src/lib/embeddings.py loads
    SentenceTransformer(EMBEDDING_MODEL) in-process. bge-m3 needs roughly 2 GB
    resident once loaded, so an under-sized task does not run slowly — it is OOM-killed
    on the first query, long after the deploy reported success.
  EOT

  validation {
    condition     = var.api.enable_autoscaling == false || var.api.min_count >= 1
    error_message = "Autoscaling with min_count = 0 cannot self-heal: nothing publishes a metric at zero tasks, so whatever scaled it down is permanent."
  }
}

variable "worker" {
  type = object({
    cpu                = number
    memory             = number
    min_count          = number
    max_count          = number
    enable_autoscaling = optional(bool, false)
    use_spot           = optional(bool, true)
    cpu_target_pct     = optional(number, 65)
    memory_target_pct  = optional(number, 75)
  })

  description = <<-EOT
    Worker task sizing. This task carries the most: the Celery worker (embedding +
    PaddleOCR extraction), the Celery beat scheduler, and the ClamAV daemon — whose
    signature database alone is ~2 GB resident.

    Container `memory` limits are carved out of the TASK total, never added to it, so
    this figure must cover all three.
  EOT

  validation {
    condition     = var.worker.enable_autoscaling == false || var.worker.min_count >= 1
    error_message = "Autoscaling with min_count = 0 cannot self-heal: nothing publishes a metric at zero tasks, so whatever scaled it down is permanent."
  }

  validation {
    condition     = var.worker.max_count <= 1
    error_message = "Celery beat runs as a container inside the worker task, and beat is a singleton — two replicas double every scheduled job. Split beat into its own service before raising max_count."
  }
}

variable "rds" {
  type = object({
    instance_class           = string
    allocated_storage_gb     = number
    max_allocated_storage_gb = number
    multi_az                 = bool
    deletion_protection      = bool
    backup_retention_days    = number
    monitoring_interval      = optional(number, 0)
    engine_version           = optional(string, "16")
  })

  description = <<-EOT
    Postgres settings. engine_version defaults to "16" rather than the rds module's
    own default, to match the pgvector/pgvector:pg16 image this application is
    developed and tested against.

    Sizing note: migration 20260802_03 builds an HNSW index over the embedding column.
    HNSW construction is memory-hungry and scales with corpus size, so an instance
    class chosen against an empty database will not stay right.
  EOT
}

variable "cache" {
  type = object({
    enabled   = optional(bool, true)
    mode      = optional(string, "node")
    node_type = optional(string, "cache.t4g.micro")

    # Use the SHARED node in the runtime layer instead of creating one for this product.
    # DEVELOP ONLY — see the description below. Created in QNSC-VN/qnsc-infra#69.
    shared = optional(bool, false)

    # Which Valkey database this product uses on the shared node. Ignored when
    # `shared = false`. Allocated centrally: 0 = rally, 1 = qnsc-kb.
    db_index = optional(number, 0)
  })
  default = {}

  description = <<-EOT
    Valkey cache. It is the Celery BROKER and result backend, not merely a cache, so
    disabling it stops all background work — ingestion, connector polling, the outbox
    relay — rather than degrading it.

    "node" mode, not "serverless": serverless has roughly a $90/mo floor against
    ~$12/mo for a cache.t4g.micro node.

    SHARING, in develop only. `shared = true` points this product at the ONE Valkey node
    in the runtime layer (qnsc-infra live/runtime-dev, module.shared_cache) instead of
    creating a node of its own. ElastiCache cannot be stopped — only deleted — so a
    per-product dev node bills all 730 hours of the month however little the environment
    runs, and develop now runs 55 hours a week. rally and qnsc-kb were paying $15.45 each
    for two nodes.

    `db_index` selects the Valkey database, NOT a key prefix: a prefix has to be honoured
    by every library touching the connection, while an index is enforced by the server.
    Cluster mode is disabled on the shared node, so all 16 databases exist and SELECT
    works. Allocated centrally — 0 is rally, 1 is qnsc-kb.

    THIS PRODUCT IS THE REASON THE EVICTION POLICY MATTERS. Celery's broker keys carry no
    TTL, so evicting one loses a QUEUED TASK rather than missing a cache. The shared node
    runs the default `volatile-lru`, which only evicts keys that have a TTL — rally's
    rate-limit counters and denylist entries go first and Celery's queue is never a
    candidate. Anyone setting `allkeys-lru` on that node to improve another product's hit
    rate would silently start dropping this product's background work.

    Production, when it exists, keeps its own node: a shared cache is a shared blast
    radius.
  EOT

  validation {
    condition     = !var.cache.shared || var.cache.enabled
    error_message = "cache.shared = true requires cache.enabled = true. `shared` selects WHERE the cache is, not WHETHER there is one — and with Celery on it, no cache means no background work at all."
  }

  validation {
    condition     = var.cache.db_index >= 0 && var.cache.db_index <= 15
    error_message = "cache.db_index must be 0-15: Valkey exposes 16 databases when cluster mode is disabled, which is what the shared node runs."
  }
}

// ── Secrets ──────────────────────────────────────────────────────────────────
variable "secrets_recovery_window_days" {
  type        = number
  default     = 0
  description = "0 in develop so a destroy+redeploy cycle does not hit \"secret scheduled for deletion\". Production keeps a real window."
}

variable "secrets_bundle_name" {
  type        = string
  default     = "app"
  description = "Bundle all app secrets into one Secrets Manager container read per key. Secrets Manager bills per SECRET regardless of size."
}

variable "secrets_use_bundle" {
  type    = bool
  default = true
}

// ── Application configuration ────────────────────────────────────────────────
variable "app_db_role" {
  type        = string
  default     = "qnsc_app"
  description = <<-EOT
    The least-privilege role the api and worker connect as. NOT a credential — a role
    name — so it travels as plain env; its password is the `app-db-password` secret.

    Created by scripts/bootstrap_db_role.py from the migrator's entrypoint, because
    migration 20260802_05 GRANTs to this role but never creates it.
  EOT
}

variable "malware_scan_enabled" {
  type    = bool
  default = true

  description = <<-EOT
    Run ClamAV as a sidecar in the worker task and scan uploads.

    Not freely optional: validate_production() in src/core/config.py REFUSES to boot
    when ENVIRONMENT is production and this is false. Turning it off therefore also
    means dropping out of production mode, which turns off every other guardrail in
    that function (API docs stay published, self-registration stays open).
  EOT
}

variable "embedding_model" {
  type        = string
  default     = "gemini-embedding-001"
  description = <<-EOT
    Determines EMBEDDING_DIMENSION, which fixes the pgvector column width and the HNSW
    index at migration time. Changing it later requires a migration AND a full re-embed
    of every chunk: vectors of different widths are not comparable, and a query embedded
    by one model against chunks embedded by another returns nonsense without erroring.

    This value must reach the MIGRATOR as well as the api and worker — the migrator is
    what creates the column. It did not, once, and the column was built 1024 wide for a
    model that emits 768.

    The deployed default uses a hosted embedding API, avoiding the local ONNX runtime and
    2.3 GB of model assets in a container whose job was to embed a search query.
  EOT
}

variable "embedding_version" {
  type        = string
  default     = "gemini-embedding-001-768-v1"
  description = "Stamped on every chunk, so a re-embed can be identified after the fact. Change it whenever embedding_model or the dimension changes."
}

variable "embedding_runtime" {
  type        = string
  default     = "onnx"
  description = <<-EOT
    HOW the embedding model executes — orthogonal to embedding_model. "onnx" runs the
    weights on the fp32 ONNX export baked into the image; parity against the torch
    reference was proven at cosine 1.000000 by tests/unit/test_embedding_backends.py,
    and int8 was measured and rejected (0.908-0.987 against the gate's 0.999). The
    default is onnx because NO SHIPPED IMAGE INSTALLS THE ML GROUP — an environment
    that sets "torch" gets a clean EmbeddingUnavailable at load, since the torch
    backend now exists only in a local checkout with `poetry install --with ml`,
    exactly what the parity gate re-enters when a model change needs re-proving.
    A hosted EMBEDDING_MODEL ignores this setting entirely.
  EOT
}

variable "gemini_model" {
  type        = string
  default     = "gemini-flash-lite-latest"
  description = <<-EOT
    The text GENERATION model (RAG answers), not the embedding model.

    A floating `-latest` alias on purpose. The previous default pinned
    gemini-2.5-flash-lite, and Google retired that whole family for new keys — every
    generateContent call returned 404 "no longer available to new users", which surfaces
    as broken AI features rather than as a deploy failure, because nothing validates a
    model name at apply time. An alias survives that; the cost is that the model can move
    under you, which retrieval quality is far more tolerant of than a dead endpoint.

    Verify a change before applying it:
      curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/MODEL:generateContent?key=KEY" \
        -H 'Content-Type: application/json' -d '{"contents":[{"parts":[{"text":"hi"}]}]}'
  EOT
}

variable "microsoft_tenant_id" {
  type        = string
  default     = "common"
  description = "Entra tenant for the Microsoft connector's OAuth flow (connectors only — this application has no SSO login path yet)."
}

variable "microsoft_client_id" {
  type        = string
  default     = ""
  description = "Public identifier, not a secret. Empty leaves the Microsoft connector dormant."
}

variable "google_client_id" {
  type        = string
  default     = ""
  description = "Public identifier, not a secret. Empty leaves the Google connector dormant."
}

variable "allowed_email_domains" {
  type        = list(string)
  default     = []
  description = "Comma-joined into ALLOWED_EMAIL_DOMAINS. Empty accepts any domain."
}

// ── Platform / misc ──────────────────────────────────────────────────────────
variable "cloudflare_account_id" {
  type        = string
  default     = ""
  description = "Cloudflare account that owns the Pages project. Empty skips the SPA module entirely."
}

variable "log_retention_days" {
  type    = number
  default = 7
}

variable "container_insights" {
  type        = string
  default     = "disabled"
  description = <<-EOT
    Stated, never inherited.

    "enhanced" adds per-task and per-container metrics that CloudWatch bills as CUSTOM
    metrics at $0.07 each — four clusters silently on that default produced 606
    metric-months (~$42) on the July 2026 bill, and the count grows with task churn
    rather than with traffic.

    ecs-cluster used to DEFAULT to "enhanced", which is how those clusters got there;
    v2.0.0 changed the default to "enabled" (cluster- and service-level metrics, in the
    free AWS/ECS namespace). So inheriting is no longer expensive — but it is still not
    queried by anything here, which is why this stays "disabled" and stays stated.
  EOT
}

// ── Cost schedules ───────────────────────────────────────────────────────────
variable "idle_schedule" {
  type    = string
  default = null

  description = <<-EOT
    EventBridge Scheduler cron, Asia/Ho_Chi_Minh, that stops the database and scales
    both services to zero. null disables idling entirely.

    Read this together with `wake_schedule` and with the deploy pipeline: they form a
    LOOP, not three independent switches. The deploy reusable's `ensure_rds` step starts
    a stopped database and restores a service left at zero, so ANY deploy wakes the
    environment regardless of the hour. A single nightly stop therefore does not hold —
    rally measured exactly that, with its develop database publishing CPU datapoints
    every hour of every night because deploys kept landing after the stop.

    Two passes are the fix (e.g. "cron(0 0,3 * * ? *)"): the first ends the working day,
    the second catches an environment woken by a late deploy.
  EOT

  // MUST FIRE AT LEAST DAILY, and this is the second half of the same lesson.
  //
  // The paragraph above explains why one pass a NIGHT is not enough. A pass a WEEK is
  // worse in a way that is easy to miss, because RDS run-state is not a Terraform concept:
  // the instance is stopped out of band, and AWS FORCE-STARTS a stopped instance after 7
  // days. A weekly re-stop therefore bounds that exposure at seven days rather than one —
  // a force-start landing on a Monday runs until the following Sunday.
  //
  // Both products shipped "cron(0 1 ? * SUN *)" in production on the reasoning that an
  // environment already at zero tasks with a stopped database only needs a backstop. That
  // holds for the ECS half and fails for RDS. Measured on rally-prod before the fix: 59 of
  // 168 hours in a week published CloudWatch datapoints — a pre-launch database with no
  // users, no tasks and no cache running 35% of the time.
  //
  // Checked rather than replaced by a named-posture enum deliberately: three call sites in
  // two repos do not justify inventing a vocabulary, and the constraint is a property of
  // the VALUE, so it belongs on the value.
  //
  // Fields are minute hour day-of-month month day-of-week year. Restricting either day
  // field means the schedule skips days; `*` and `?` are the only spellings that do not.
  validation {
    condition = var.idle_schedule == null || can(
      regex("^cron\\([^ ]+ [^ ]+ [*?] [*?] [*?] [^ )]+\\)$", var.idle_schedule)
    )
    error_message = <<-EOT
      idle_schedule must fire at least daily: day-of-month, month and day-of-week must all
      be "*" or "?". AWS force-starts a stopped RDS instance after 7 days, so a weekly
      schedule leaves it running for up to six of them — rally-prod measured 35% uptime
      under "cron(0 1 ? * SUN *)". Use "cron(0 1 * * ? *)" for a daily pass.
    EOT
  }
}

variable "wake_schedule" {
  type    = string
  default = null

  description = <<-EOT
    EventBridge Scheduler cron, Asia/Ho_Chi_Minh, that starts the database and restores
    both services to one task. null means the environment is woken only by a deploy.

    Worth having even though deploys wake it: that covers the days the environment is
    CHANGED, not the days it is merely USED. Someone who opens it on a morning nobody
    merged finds it stopped, and RDS takes minutes to reach `available`, so it cannot be
    waited out — it reads as an outage.

    Set it early enough that the database is serving before the working day rather than
    during its first minutes.
  EOT
}

// ── Alarms ───────────────────────────────────────────────────────────────────
variable "alarm_emails" {
  type    = list(string)
  default = []

  description = <<-EOT
    Addresses subscribed to the alarm SNS topic. Empty creates the alarms without a
    subscriber, which is not useless — the console still shows state — but nothing is
    delivered.

    Each subscription must be confirmed from the email itself; Terraform cannot do it.
  EOT
}

variable "alarm_thresholds" {
  type = object({
    rds_cpu_pct     = optional(number, 80)
    rds_free_bytes  = optional(number, 2147483648) // 2 GiB
    rds_connections = optional(number, 80)
  })
  default = {}

  description = <<-EOT
    Overrides for the RDS alarm thresholds.

    rds_free_bytes matters more here than in a typical product: this database stores
    embeddings and an HNSW index, storage AUTOSCALES up to max_allocated_storage_gb, and
    RDS refuses to shrink a volume afterwards. Growth is therefore permanent, and the
    alarm is the only thing that makes it visible before it is paid for.
  EOT
}

variable "entra_auto_provision_domain" {
  type        = string
  default     = "qnsc.vn"
  description = "Email domain auto-provisioned on Entra sign-in, at the Staff role. Stated explicitly so develop's access rules do not depend on a code default."
}

variable "entra_admin_emails" {
  type        = list(string)
  default     = []
  description = <<-EOT
    Addresses provisioned as GLOBAL administrators on first Entra sign-in, instead of
    Staff. Read only when the account is first created, so it never overrides a role set
    later in the admin UI, and removing an address here does not demote anyone.
  EOT
}
