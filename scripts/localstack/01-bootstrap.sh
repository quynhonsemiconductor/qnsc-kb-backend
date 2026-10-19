#!/usr/bin/env bash
#
# LocalStack bootstrap: the one private bucket the KB API and worker store uploaded sources in, locally.
#
# Runs automatically through /etc/localstack/init/ready.d every time LocalStack becomes ready (mounted from
# ./scripts/localstack in docker-compose.yml). Idempotent: create-bucket's BucketAlreadyOwnedByYou is swallowed.
#
# Local stand-in for the Cloudflare R2 source bucket. The bucket is private; the browser reaches it only through a
# presigned URL the API signs. Production uses R2 and never sees this file.
#
# CORS is deliberately permissive in this one dev bucket (any header) because the browser PUTs the file directly.
# Production's R2 rules are set in infra, not here.
#
# Names must match SOURCE_STORAGE_BUCKET in .env.example (test_local_compose.py checks it).
set -euo pipefail

REGION="ap-southeast-1"
BUCKET="qnsc-kb-sources-local"
# The SPA origin the browser PUTs from. Follows FRONTEND_URL in .env through the compose environment.
WEB_ORIGIN="${WEB_ORIGIN:-http://localhost:5173}"

echo "[localstack-init] provisioning the source bucket…"

awslocal s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --create-bucket-configuration "LocationConstraint=$REGION" \
  >/dev/null 2>&1 || true

awslocal s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration "{
  \"CORSRules\": [{
    \"AllowedMethods\": [\"PUT\", \"GET\", \"HEAD\"],
    \"AllowedOrigins\": [\"${WEB_ORIGIN}\"],
    \"AllowedHeaders\": [\"*\"],
    \"ExposeHeaders\": [\"ETag\"],
    \"MaxAgeSeconds\": 3600
  }]
}" >/dev/null

echo "[localstack-init]   bucket ready: $BUCKET (CORS for $WEB_ORIGIN)"
