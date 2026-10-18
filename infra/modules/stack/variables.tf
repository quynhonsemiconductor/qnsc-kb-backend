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
  description = <<-EOT
    Container image tag for the BASELINE task definitions. The deploy replaces it in every container on the
    service image, so in develop this is a placeholder that is never pulled; production pins the release
    tag. NEVER `latest`: the ECR repositories are immutable and the deploy pushes `<sha>` tags only, so
    `latest` does not exist and any container left on it cannot start (2026-10: qnsc-kb's Celery beat did
    exactly that and took the worker down).
  EOT

  validation {
    condition     = var.image_tag != "latest" && var.image_tag != ""
    error_message = "image_tag must not be \"latest\" or empty: that tag does not exist in the immutable ECR repositories. Use the release tag, or a clearly invalid placeholder in develop (the deploy sets the real one)."
  }
}

// ── Ingress via Cloudflare Tunnel ────────────────────────────────────────────
variable "tunnel_enabled" {
  type    = bool
  default = true

  description = <<-EOT
    Serve the API through a cloudflared sidecar that dials OUT to Cloudflare, instead
    of an ALB target group.

    Default true, unlike rova's, because there is no shared ALB to fall back on:
    `enable_alb` is false on both platform runtime stacks (since 2026-08-02, after
    rova's own tunnel cutover — an idle load balancer with no target groups is
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
    # DEVELOP ONLY — see the description below. Created in quynhonsemiconductor/infra#69.
    shared = optional(bool, false)

    # Which Valkey database this product uses on the shared node. Ignored when
    # `shared = false`.
    #
    # THE REGISTRY LIVES IN ONE PLACE: qnsc-infra `allocations.json`, key
    # `cache_db_index_allocations`. Do not copy the table here. The inline version that
    # used to sit on this line read "0 = rova, 1 = qnsc-kb" and silently omitted opshub
    # after it took index 2 — which is exactly how the same table drifted into three
    # different states across the three product stack modules. A registry with three
    # copies is not a registry.
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
    runs, and develop now runs 55 hours a week. rova and qnsc-kb were paying $15.45 each
    for two nodes.

    `db_index` selects the Valkey database, NOT a key prefix: a prefix has to be honoured
    by every library touching the connection, while an index is enforced by the server.
    Cluster mode is disabled on the shared node, so all 16 databases exist and SELECT
    works. Allocated centrally in qnsc-infra `allocations.json`, key
    `cache_db_index_allocations` — the registry lives in one place rather than being copied
    into each product's stack module, which is how it drifted into three different states
    before 2026-09-12.

    THIS PRODUCT IS THE REASON THE EVICTION POLICY MATTERS. Celery's broker keys carry no
    TTL, so evicting one loses a QUEUED TASK rather than missing a cache. The shared node
    runs the default `volatile-lru`, which only evicts keys that have a TTL — rova's
    rate-limit counters and denylist entries go first and Celery's queue is never a
    candidate. Anyone setting `allkeys-lru` on that node to improve another product's hit
    rate would silently start dropping this product's background work.

    Production, when it exists, keeps its own node: a shared cache is a shared blast
    radius.
  EOT

  # PROMOTED FROM A `check` BLOCK 2026-09-12. This condition previously lived in
  # `check "cache_required_for_jobs"` in main.tf, which did NOT enforce it: a violated
  # check emits `Warning: Check block assertion failed` and the plan exits 0 — measured on
  # OpenTofu 1.12.3 and recorded in rova's copy of this variable — so the forbidden
  # combination applied cleanly behind a warning nobody reads in CI output. A
  # cross-variable `validation` exits 1.
  #
  # rova and opshub already expressed this as a validation; this module was the last of the
  # three still relying on the inert form, which is the kind of divergence that survives
  # precisely because all three copies look reasonable in isolation.
  #
  # The stakes are higher here than in the other two products, which is why it is worth
  # fixing rather than noting: the error message below is not about a cache miss.
  validation {
    condition     = var.cache.enabled || (var.api.min_count == 0 && var.worker.min_count == 0)
    error_message = "cache.enabled = false requires min_count = 0 on BOTH services. The cache is the Celery broker: with it disabled no ingestion, connector sync or outbox relay runs at all, while health checks still answer 200. Scale both services to zero, or enable the cache."
  }

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

variable "embedding_realign_discard_vectors" {
  type        = bool
  default     = false
  description = <<-EOT
    Lets the realign migrations (20260810_51 / 20260828_64 / 20260912_77) rebuild the
    pgvector column when the stored vectors are at the old width.

    Those revisions refuse to run when the column width disagrees with the width derived
    from embedding_model AND embeddings are present, because there is no in-place
    conversion between widths — a 1024-dimension vector is not an extension of a
    384-dimension one. Refusing is the right default; the alternative is a migration that
    silently destroys every embedding.

    src/core/config.py grew EMBEDDING_REALIGN_DISCARD_VECTORS so that step could be
    explicit, reviewable and logged instead of an ad-hoc DELETE typed against a live
    database. Nothing carried it to the migrator, though, so the sanctioned way past the
    guard did not exist in a deployed environment at all:

        git grep EMBEDDING_REALIGN_DISCARD_VECTORS -- infra/   # nothing

    A blocked deploy therefore still had to be rescued by hand, which is what happened on
    develop on 2026-09-13: the chunks were cleared manually, the guard then saw zero
    populated rows, and 20260912_77 went through. This variable is the difference between
    that being a documented deploy input and being a DELETE typed against a live database.

    NOT NEEDED BY AN ENVIRONMENT AT THE CORRECT WIDTH. The guard only fires when the
    column width disagrees with the width derived from embedding_model AND embeddings are
    present, because there is no in-place conversion between widths — a 1024-dimension
    vector is not an extension of a 384-dimension one. Refusing is the right default; the
    alternative is a migration that silently destroys every embedding.

    SET IT ONLY FOR THE DEPLOY THAT PERFORMS THE REALIGN, then set it back. It is not a
    standing configuration: leaving it true means the next model change silently discards
    the corpus instead of stopping to ask. The wipe covers article_chunks/parent_chunks
    only — both DERIVED from articles.content — and the migration re-queues every published
    article for the API's startup sweep, so the cost is re-indexing time and degraded
    search until it drains, not lost source content.
  EOT
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

variable "microsoft_tenant_id" {
  type        = string
  default     = ""
  description = <<-EOT
    The Entra tenant (directory) GUID that Microsoft sign-in AND the connector's OAuth consent are pinned to. No
    default and no "common": under "common", "organizations" or "consumers" an account from any tenant (or a
    personal one) could complete sign-in or consent. The application also refuses to boot in production with
    anything but a GUID, and verifies every ID token's tid and iss against it (docs/runbooks/entra-apps.md).
  EOT

  validation {
    condition     = can(regex("^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", var.microsoft_tenant_id))
    error_message = "microsoft_tenant_id must be the Entra tenant GUID, never common, organizations or consumers, and never empty."
  }
}

variable "microsoft_client_id" {
  type        = string
  default     = ""
  description = <<-EOT
    Public identifier, not a secret. Application (client) ID of THIS environment's own single-tenant Entra app
    ("KB Dev" for develop, "KB Prod" for production; never shared between environments). It serves both the
    Microsoft sign-in and the user-delegated connector consent. Empty leaves both dormant.
    docs/runbooks/entra-apps.md.
  EOT
}

variable "microsoft_graph_sender" {
  type        = string
  default     = ""
  description = <<-EOT
    The mailbox Graph sends application email FROM: invitations and password resets. A UPN, e.g.
    "noreply-kb@qnsc.vn" in production and "noreply-kb-dev@qnsc.vn" in develop (a non-production environment
    never sends from the production mailbox; a shared mailbox is enough and needs no licence).

    The Entra app (mail_graph_client_id) holds NO Graph consent. Its only authority is the Exchange Online
    role "Application Mail.Send", assigned with a management scope that names this one mailbox, so the app
    can send as this address and as nobody else. See docs/runbooks/email-graph.md.

    Empty is no longer a silent failure: with email_provider "graph" the WORKER refuses to start when
    mail_graph_client_id is set and this or either secret is missing (the API never refuses to boot over mail).
  EOT
}

variable "mail_graph_client_id" {
  type        = string
  default     = ""
  description = <<-EOT
    Application (client) ID of the dedicated Entra app that sends mail (one per environment, separate from
    the SSO/connector app in microsoft_client_id). A public identifier, not a secret. Authenticates with the
    certificate whose PRIVATE KEY lives in a standalone secret (<product>/<env>/mail-graph-certificate); there is no client
    secret. Empty until the owner has created the app (docs/runbooks/email-graph.md).

    THIS IS THE SWITCH. While it is empty the (empty) secret container exists but nothing is injected, so this
    module can be applied before the owner's steps. The owner puts the value in the secret FIRST, then sets
    this, in that order. The deploy preflight names the secret if it still has no value.
  EOT
}

variable "email_provider" {
  type    = string
  default = "graph"

  description = <<-EOT
    Selects get_email_sender()'s transport (src/services/email.py). "graph" (default, and what every
    deployed environment uses) sends through Microsoft 365 with certificate auth (the mail secret is
    injected into the worker once mail_graph_client_id is set). "ses" is the legacy AWS path, kept only until Graph is proven in develop
    and then removed. "fake" is the in-memory sender for tests. "console" exists in the application but is
    refused outside development and is not accepted here.

    Switching away from "graph" destroys the empty-or-not mail secret container (recoverable for
    secrets_recovery_window_days); the plan guard stops that unless it is acknowledged.
  EOT

  validation {
    condition     = contains(["graph", "ses", "fake"], var.email_provider)
    error_message = "email_provider must be graph, ses, or fake."
  }
}

variable "mail_from_email" {
  type    = string
  default = ""

  description = <<-EOT
    Required when email_provider is "ses"; SesEmailSender raises without it. Must be an
    address SES has verified (directly, or via a verified domain identity), e.g.
    "no-reply@qnsc.vn".
  EOT
}

variable "mail_from_name" {
  type        = string
  default     = "QNSC Knowledge Base"
  description = "Display name on outbound mail sent through EMAIL_PROVIDER=ses."
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

variable "worker_task_alarm" {
  type        = bool
  default     = true
  description = <<-EOT
    The SUSTAINED worker alarm: RunningTaskCount < 1 for 10 of 10 minutes, missing data counted as breaching
    (Container Insights publishes nothing at zero tasks, so missing data IS the outage). Needs
    container_insights != "disabled".

    ONLY FOR A SERVICE THAT IS MEANT TO BE RUNNING ALL THE TIME. It is created only while idle_schedule is null:
    an idle schedule scales the worker to zero on purpose, which this alarm would report every night. So the
    pairing is: re-enabling idle_schedule disables this alarm, and removing it (the production go-live
    checklist) arms it. The fast event-based signal (worker_event_alerts) is independent of this and stays on.
  EOT
}

variable "worker_event_alerts" {
  type        = bool
  default     = true
  description = <<-EOT
    EventBridge rules that email the alarm topic when ECS reports that a worker task failed to start
    (CannotPullContainerError, ResourceInitializationError, TaskFailedToStart), that the service cannot start or
    place tasks (SERVICE_TASK_START_IMPAIRED / SERVICE_TASK_PLACEMENT_FAILURE), or that a deployment failed
    (SERVICE_DEPLOYMENT_FAILED). Fast, free, and needs neither Container Insights nor a running task. Also adds
    the EventBridge statement to the alarm topic's policy.
  EOT
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
    rova measured exactly that, with its develop database publishing CPU datapoints
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
  // holds for the ECS half and fails for RDS. Measured on rova-prod before the fix: 59 of
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
      schedule leaves it running for up to six of them — rova-prod measured 35% uptime
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

