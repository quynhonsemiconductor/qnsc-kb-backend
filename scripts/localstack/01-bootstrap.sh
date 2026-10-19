#!/usr/bin/env bash
#
# LocalStack bootstrap: the one private bucket the KB API and worker store uploaded sources in, locally.
#
# Runs through /etc/localstack/init/ready.d every time LocalStack becomes ready (mounted from ./scripts/localstack in
# docker-compose.yml). A failure exits non-zero: the compose healthcheck then stays unhealthy, and `up --wait` says so. Only
# "the bucket already exists" is tolerated, so a re-run is safe.
#
# CORS allows the frontend under BOTH loopback names. The browser treats http://localhost:5173 and http://127.0.0.1:5173 as
# different origins, and the second frontend copy is opened as 127.0.0.1 (see the frontend README). The origin port follows
# FRONTEND_DEV_PORT through the compose environment (WEB_ORIGINS).
#
# Local stand-in for the Cloudflare R2 source bucket. The bucket is private; the browser reaches it only through a presigned
# URL the API signs. Production uses R2 and never sees this file.
#
# Names must match SOURCE_STORAGE_BUCKET in .env.example (test_local_compose.py checks it).
set -euo pipefail

REGION="us-east-1"
BUCKET="qnsc-kb-sources-local"
# Comma-separated origins, from compose. Defaults to the frontend on its default port, both loopback names.
WEB_ORIGINS="${WEB_ORIGINS:-http://localhost:5173,http://127.0.0.1:5173}"

echo "[localstack-init] provisioning the source bucket…"

if ! create_out=$(awslocal s3api create-bucket --bucket "$BUCKET" --region "$REGION" 2>&1); then
  case "$create_out" in
    *BucketAlreadyOwnedByYou*|*BucketAlreadyExists*) ;;
    *) echo "[localstack-init] create-bucket failed: $create_out" >&2; exit 1 ;;
  esac
fi

origins_json=""
IFS=',' read -r -a origins <<< "$WEB_ORIGINS"
for origin in "${origins[@]}"; do
  origins_json="${origins_json}\"${origin}\","
done
origins_json="${origins_json%,}"

awslocal s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration "{
  \"CORSRules\": [{
    \"AllowedMethods\": [\"PUT\", \"GET\", \"HEAD\"],
    \"AllowedOrigins\": [${origins_json}],
    \"AllowedHeaders\": [\"*\"],
    \"ExposeHeaders\": [\"ETag\"],
    \"MaxAgeSeconds\": 3600
  }]
}"

echo "[localstack-init]   bucket ready: $BUCKET (CORS for ${WEB_ORIGINS})"
