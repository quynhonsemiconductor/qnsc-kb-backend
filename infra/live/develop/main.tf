// qnsc-kb · develop
//
// Values only. The whole stack lives in ../../modules/stack, so develop and production
// cannot drift structurally — only the numbers below differ. Develop leans on cheap,
// interruptible infrastructure (Fargate Spot, a small database, short log retention);
// production takes the durable settings.
terraform {
  required_version = ">= 1.9"
  required_providers {
    aws        = { source = "hashicorp/aws", version = "~> 5.0" }
    cloudflare = { source = "cloudflare/cloudflare", version = "~> 4.0" }
  }

  backend "s3" {
    bucket         = "qnsc-tofu-state"
    key            = "qnsc-kb/develop/terraform.tfstate"
    region         = "ap-southeast-1"
    encrypt        = true
    dynamodb_table = "qnsc-tofu-locks"
  }
}

provider "aws" {
  region = "ap-southeast-1"
  default_tags {
    tags = {
      Project     = "qnsc-kb"
      Environment = "develop"
      ManagedBy   = "opentofu"
    }
  }
}

// Reads TF_VAR_cloudflare_api_token. Pages and DNS resources are count-gated on the
// account id, so this stack applies before Cloudflare exists.
provider "cloudflare" {
  api_token = var.cloudflare_api_token != "" ? var.cloudflare_api_token : null
}

module "stack" {
  source = "../../modules/stack"

  product  = "qnsc-kb"
  env      = "develop"
  env_slug = "develop"
  region   = "ap-southeast-1"

  app_domain = "kb-dev.qnsc.vn"
  api_domain = "kb-api-dev.qnsc.vn"
  web_record = "kb-dev"
  api_record = "kb-api-dev"

  shared_state_key  = "qnsc-kb/shared/terraform.tfstate"
  runtime_state_key = "platform/runtime-dev/terraform.tfstate"
  storage_state_key = "platform/storage-dev/terraform.tfstate"

  // Develop tracks the newest build; production pins the release tag it was cut from.
  image_tag = "latest"

  // Cost-leaning: short retention, and immediate secret deletion so a
  // destroy+redeploy cycle does not trip "secret scheduled for deletion".
  log_retention_days           = 7
  secrets_recovery_window_days = 0

  // ── Ingress via Cloudflare Tunnel, not an ALB ──────────────────────────────
  // cloudflared runs as a sidecar in the api task and dials OUT, so this environment
  // needs no listener rule, no target group and no public IPv4. Both platform runtime
  // stacks have their ALB disabled anyway, so a tunnel is the only ingress available
  // without turning one back on.
  tunnel_enabled = true

  // ── Sizing ─────────────────────────────────────────────────────────────────
  // 2048 MB, sized from MEASUREMENT on EMBEDDING_RUNTIME=onnx (PR #51), not arithmetic:
  // the task held a steady 1,569 MB (38.3% of the 4096 it was given) across the whole
  // day it ran after the flip — the fp32 ONNX session resident, torch NOT loaded — and
  // a search request adds ~460 ms of one short query through it. 2048 leaves ~30%
  // headroom over that floor.
  //
  // The 4096 before this was the torch floor: the model plus the framework, sized when
  // a 1024 task failed the load and silently fell back to keyword-only search. ONNX is
  // what bought the headroom back — if this ever needs raising again, raise it on a
  // CloudWatch MemoryUtilization graph, not on a hunch, and remember 512 CPU caps a
  // task at 4096 MB.
  //
  // Spot with a floor of ZERO, and autoscaling off. Develop is exercised by CI deploys
  // and occasional manual checks, so paying for a task around the clock buys nothing.
  // The deploy pipeline scales a zero'd service back to 1 and starts a stopped
  // database, so every merge to main wakes this environment on its own.
  //
  // Autoscaling must stay off while the floor is 0: target tracking scales
  // proportionally, so from zero tasks there is no metric to scale out from, and from
  // one idle task it computes one. It would be inert while billing CloudWatch alarms —
  // and a floor of 1 instead would undo the scale-to-zero within minutes.
  api = {
    // Raised to carry the ClamAV sidecar. clamd carves out 256 CPU / 2048 MB of the
    // task, and the api scans uploads synchronously, so it needs one in ITS task —
    // the worker's is in a different network namespace. The remaining 768/2048 is
    // what the api and the tunnel had before this changed.
    cpu                = 1024
    memory             = 4096
    min_count          = 0
    max_count          = 2
    enable_autoscaling = false
    use_spot           = true
  }

  // Three containers share this task. Two carry hard container limits carved out of the
  // task total — beat 256, and clamav 2048 (its signature database is ~2 GB resident;
  // see the sizing post-mortem in the module) — so 2,304 MB is reserved against the
  // task and the Celery worker itself runs unlimited, bounded only by the task level.
  //
  // 4096, down from 6144 on the same ONNX evidence. Measured idle on the new task:
  // 1,278 MB total (clamav + beat + worker base, model not yet loaded). The embedding
  // session loads lazily on the first chunk and adds ~1.5 GB (the api holds the
  // identical session at 1,569 MB total), so ~2.8 GB steady while embedding, and
  // PaddleOCR is invoked per scanned file on top of that. 4096 leaves ~1.2 GB for the
  // OCR spike. 1024 CPU because 512 caps a task at 4096 MB — this is now the floor,
  // not the ceiling, and the first large scanned-file ingest is the thing to watch: if
  // it dies, raise this on the evidence of the killed task.
  //
  // max_count is 1 and cannot be raised while beat lives here — two beat containers
  // double every scheduled job. The stack module enforces that with a validation.
  worker = {
    cpu                = 2048
    memory             = 4096
    min_count          = 0
    max_count          = 1
    enable_autoscaling = false
    use_spot           = true
  }

  rds = {
    engine_version           = "16" // matches the pgvector/pgvector:pg16 image used in development
    instance_class           = "db.t4g.micro"
    allocated_storage_gb     = 20
    max_allocated_storage_gb = 100
    multi_az                 = false
    deletion_protection      = false // easy teardown in develop
    monitoring_interval      = 0     // Enhanced Monitoring off — saves CloudWatch cost

    // ZERO, which disables automated backups and point-in-time recovery here.
    // Develop holds nothing worth recovering: the migrator rebuilds the schema from
    // migrations on any deploy, and documents are re-ingestible from their sources.
    //
    // Applying a change from a non-zero retention to 0 DELETES every existing automated
    // snapshot for this instance and is not reversible. Take a manual snapshot first if
    // develop ever holds something real.
    backup_retention_days = 0
  }

  // ── Shared develop cache ─────────────────────────────────────────────────────
  // The ONE Valkey node in the runtime layer, not a node of qnsc-kb's own.
  //
  // ElastiCache has no stopped state — only delete — so this was the one component of an
  // idled environment that kept billing all 730 hours of the month, while the schedule
  // above now runs develop 55 hours a week. It could never be turned off either, because
  // it is the Celery broker rather than a cache that can be missed. rally-develop had the
  // same node for the same reason: two at $15.45 each.
  //
  // Saves $15.45/mo across the account. Node created in quynhonsemiconductor/qnsc-infra#69, and rally
  // moved onto it in quynhonsemiconductor/rally#448.
  //
  // DATABASE 1. rally holds 0. This is a Valkey database index, not a key prefix — a
  // prefix has to be honoured by every library touching the connection, while an index is
  // enforced by the server. Cluster mode is disabled on the shared node, so all 16
  // databases exist and SELECT works. Indexes are allocated centrally in the stack
  // variable's description; two products silently sharing one is the collision that
  // allocation prevents, and nothing catches it at plan time.
  //
  // CELERY IS WHY THE EVICTION POLICY MATTERS, and this is the product that carries the
  // risk. Broker keys have no TTL, so evicting one loses a QUEUED TASK rather than missing
  // a cache. The shared node runs the default `volatile-lru`, which evicts only keys that
  // HAVE a TTL — rally's rate-limit counters and denylist entries go first, and Celery's
  // queue is never a candidate. If anyone sets `allkeys-lru` on that node to improve
  // rally's hit rate, this product silently starts dropping background work.
  //
  // `mode` is deliberately NOT set here. It sizes a node this stack no longer creates —
  // the shared node's mode is decided in qnsc-infra's runtime layer — so passing it would
  // read as configuration and change nothing. Same for `node_type`.
  //
  // APPLIED 2026-08-17. qnsc-kb-develop-cache was destroyed and the endpoint changed, so
  // the cutover was a task-definition revision plus a rolling deploy, and in-flight Celery
  // tasks on the old node were lost — acceptable in develop, where the outbox replays and
  // connectors re-poll. Verified afterwards: /health/ready returned
  // {"database":"ok","redis":"ok","job_mode":"celery"} and beat resumed scheduling on the
  // new node. Production, when it exists, keeps its own node.
  cache = {
    enabled  = true
    shared   = true
    db_index = 1
  }

  // ClamAV runs here too, not only in production. It is not really optional: the
  // application is pinned to ENVIRONMENT=production in every environment (so the
  // validate_production() guardrails are actually exercised), and that function refuses
  // to boot when malware scanning is off.
  malware_scan_enabled = true

  // ── Off-hours idling ───────────────────────────────────────────────────────
  // Two passes, not one. A single nightly stop does not hold, because the deploy
  // pipeline's `ensure_rds` step wakes this environment whenever a deploy lands — so a
  // merge after the stop leaves everything running until the following night. rally
  // measured exactly that: its develop database published CPU datapoints every hour of
  // every night while a stop schedule fired correctly each evening.
  //
  // Midnight ends the working day; 03:00 catches an environment woken by a late deploy.
  //
  // KNOWN CONSEQUENCE, because it is not obvious: scaling the worker to zero stops
  // Celery beat with it, so nothing scheduled runs overnight — no outbox replay, no
  // cloud-connector polling. In develop that is the intended trade. Beat resumes at the
  // wake, and the outbox is a queue rather than a stream, so pending rows are replayed
  // then rather than lost. Production must not take this setting for that reason.
  //
  // THREE passes were tried (19:00/22:00/02:00, replacing 0,3) to move the money — develop
  // was up 08:00-00:00, so the 19:00-00:00 tail cost ~$8.13/mo of RDS and Fargate across
  // both develop environments. rally moved BACK to `0,3` on 2026-08-19 on request: a 19:00
  // stop cut the evening short, and develop being down while somebody is still working
  // costs more in interruption than the hours save. This repo follows the same reversal —
  // see rally's infra/live/develop/main.tf for the fuller history.
  //
  // TWO PASSES, and the second is not optional. 00:00 ends the day; 03:00 catches a deploy
  // that landed late and woke the environment, because nothing else would put it back down
  // until the next working day. Each pass is a no-op when develop is already down
  // (InvalidDBInstanceState, deliberately not retried).
  idle_schedule = "cron(0 0,3 * * ? *)"

  // 08:00 local, every day. Deploys already wake this environment, but that covers the
  // days it is CHANGED rather than the days it is USED — someone opening it on a
  // morning nobody merged would find it stopped, and RDS takes minutes to reach
  // `available`, which reads as an outage rather than something to wait out.
  //
  // 08:00 rather than 09:00 because the database needs those minutes and the API tasks
  // then have to pass a health check, so the environment is serving before the working
  // day rather than during its first minutes.
  // WEEKDAYS ONLY, 08:00 Asia/Ho_Chi_Minh. RESTORED after being removed in #53.
  //
  // #53 carried in the "make qnsc-kb develop on-demand" change, which deleted this line so
  // that only a deploy would ever start the environment. That was reverted as a decision,
  // not as a mistake in #53: develop is a SHARED environment, and on-demand only works if
  // waking it is trivial. It is not. There is no lightweight wake button — the options are
  // a full redeploy (15-20 minutes now that this repo builds without a registry cache) or
  // direct AWS CLI access. A tester or BA opening kb-dev.qnsc.vn on a quiet day would have
  // found it down with no obvious way to fix that, which is a poor trade for ~$8-12/mo.
  //
  // The saving that mattered was already taken in #46: 112 h/week -> 55 h/week, worth
  // $18.59/mo across both develop environments. This line is what keeps the remaining
  // hours PREDICTABLE, which is the property a shared environment needs.
  //
  // WITHOUT THIS LINE the environment is a one-way door: the idle passes above run daily
  // and nothing on a timer undoes them, so develop goes down at 19:00 and stays down.
  // That is correct for production, which is woken by a release, and wrong here.
  //
  // 08:00 rather than 09:00 because RDS takes ~7 minutes to reach `available` and the API
  // tasks then have to pass a health check, so the environment is serving before the
  // working day rather than during its first minutes.
  //
  // IF qnsc-kb ever goes properly dormant, build a one-click `Wake develop` workflow first
  // (qnsc-ci already has the `ensure-environment-awake` action; nothing exposes it as a
  // dispatchable workflow), THEN remove this line. In that order.
  wake_schedule = "cron(0 8 ? * MON-FRI *)"

  // MUST NAME THE MODEL THE IMAGE ACTUALLY CARRIES. This is not a preference: with
  // embedding_runtime = "onnx" the loader reads whatever export sits in
  // EMBEDDING_ONNX_DIR and ignores this value, while EMBEDDING_DIMENSION is derived FROM
  // it (src/core/config.py). Name a different model and the two disagree, so every embed
  // dies in src/lib/embeddings/base.py:
  //
  //   embedding backend returned 384 dimensions, but EMBEDDING_DIMENSION is 1024
  //
  // The Dockerfile bakes intfloat/multilingual-e5-small (384) via ARG EMBEDDING_MODEL,
  // and this must name the same model or EMBEDDING_DIMENSION resolves against the wrong
  // export. e5-small is retrieval-trained (query:/passage: prefixes applied in
  // src/lib/embeddings) and measured substantially better on Vietnamese retrieval.
  //
  // Fixes EMBEDDING_DIMENSION at 384, same as the previous MiniLM, so the pgvector
  // column and HNSW index are unchanged; adopting it required a one-time re-embed
  // migration because the 384-wide vector space is different.
  embedding_model   = "intfloat/multilingual-e5-small"
  embedding_version = "e5-small-v1"
  // Parity-gated ONNX flip (cosine 1.000000 vs torch, tests/unit/test_embedding_backends.py).
  // Rollback until the ml group leaves the images: set back to "torch" and redeploy.
  embedding_runtime = "onnx"

  alarm_emails           = var.alarm_emails
  cloudflare_account_id  = var.cloudflare_account_id
  microsoft_client_id    = var.microsoft_client_id
  microsoft_tenant_id    = var.microsoft_tenant_id
  microsoft_graph_sender = var.microsoft_graph_sender
  google_client_id       = var.google_client_id
  allowed_email_domains  = var.allowed_email_domains
  entra_admin_emails     = var.entra_admin_emails
}
