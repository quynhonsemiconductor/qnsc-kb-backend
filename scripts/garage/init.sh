#!/usr/bin/env bash
#
# One-shot setup of the local Garage node: layout, a fixed local-dev key, the source bucket, its grant, and CORS.
# Idempotent: every step checks before it changes anything. It exits non-zero on any failure, and the app containers wait
# for it (`service_completed_successfully`), so a broken CORS or bucket stops the stack instead of failing later.
#
# Local stand-in for the Cloudflare R2 source bucket. The bucket is private; the browser reaches it only through a presigned
# URL the API signs. Production uses R2 and never sees this file.
set -euo pipefail

ADMIN="http://garage:3903"
AUTH="Authorization: Bearer ${GARAGE_ADMIN_TOKEN}"
BUCKET="${SOURCE_STORAGE_BUCKET:-qnsc-kb-sources-local}"
WEB_ORIGINS="${WEB_ORIGINS:-http://localhost:5173,http://127.0.0.1:5173}"
export AWS_ACCESS_KEY_ID="${STORAGE_ACCESS_KEY_ID}" AWS_SECRET_ACCESS_KEY="${STORAGE_SECRET_ACCESS_KEY}" AWS_DEFAULT_REGION=us-east-1

# The amazon/aws-cli image has no python3, so pull fields out of Garage's compact JSON with grep and sed.
string_field() { grep -o "\"$1\" *: *\"[^\"]*\"" | head -1 | sed 's/.*: *"\([^"]*\)"$/\1/'; }
number_field() { grep -o "\"$1\" *: *[0-9]*" | head -1 | sed 's/.*: *//'; }

echo "[garage-init] waiting for the node…"
for _ in $(seq 1 60); do
  curl -sf -H "$AUTH" "$ADMIN/v1/status" >/dev/null && break
  sleep 2
done
curl -sf -H "$AUTH" "$ADMIN/v1/status" >/dev/null || { echo "[garage-init] node did not answer" >&2; exit 1; }

node_id=$(curl -sf -H "$AUTH" "$ADMIN/v1/status" | string_field node)
layout_version=$(curl -sf -H "$AUTH" "$ADMIN/v1/layout" | number_field version)
if [ "$layout_version" = "0" ]; then
  curl -sf -X POST -H "$AUTH" -H 'content-type: application/json' \
    -d "[{\"id\":\"$node_id\",\"zone\":\"dc1\",\"capacity\":1073741824,\"tags\":[]}]" "$ADMIN/v1/layout" >/dev/null
  curl -sf -X POST -H "$AUTH" -H 'content-type: application/json' -d '{"version":1}' "$ADMIN/v1/layout/apply" >/dev/null
  echo "[garage-init]   layout applied"
fi

if [ "$(curl -s -o /dev/null -w '%{http_code}' -H "$AUTH" "$ADMIN/v1/key?id=$STORAGE_ACCESS_KEY_ID")" != "200" ]; then
  curl -sf -X POST -H "$AUTH" -H 'content-type: application/json' \
    -d "{\"name\":\"local-dev\",\"accessKeyId\":\"$STORAGE_ACCESS_KEY_ID\",\"secretAccessKey\":\"$STORAGE_SECRET_ACCESS_KEY\"}" \
    "$ADMIN/v1/key/import" >/dev/null
  echo "[garage-init]   key imported"
fi

bucket_id=$(curl -sf -H "$AUTH" "$ADMIN/v1/bucket?globalAlias=$BUCKET" 2>/dev/null | string_field id || true)
if [ -z "$bucket_id" ]; then
  bucket_id=$(curl -sf -X POST -H "$AUTH" -H 'content-type: application/json' \
    -d "{\"globalAlias\":\"$BUCKET\"}" "$ADMIN/v1/bucket" | string_field id)
  echo "[garage-init]   bucket created: $BUCKET"
fi
curl -sf -X POST -H "$AUTH" -H 'content-type: application/json' \
  -d "{\"bucketId\":\"$bucket_id\",\"accessKeyId\":\"$STORAGE_ACCESS_KEY_ID\",\"permissions\":{\"read\":true,\"write\":true,\"owner\":true}}" \
  "$ADMIN/v1/bucket/allow" >/dev/null

# One CORS rule per origin: Garage joins origins in a single rule into one header, which browsers refuse.
rules=""
IFS=',' read -r -a origins <<< "$WEB_ORIGINS"
for origin in "${origins[@]}"; do
  rules="${rules}{\"AllowedOrigins\":[\"${origin}\"],\"AllowedMethods\":[\"PUT\",\"GET\",\"HEAD\"],\"AllowedHeaders\":[\"*\"],\"ExposeHeaders\":[\"ETag\"],\"MaxAgeSeconds\":3600},"
done
aws --endpoint-url http://garage:3900 s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration "{\"CORSRules\":[${rules%,}]}"

# Prove it, the way a browser asks: a preflight from each origin must get that origin back, with nothing else allowed.
for origin in "${origins[@]}"; do
  got=$(curl -s -o /dev/null -D - -X OPTIONS -H "Origin: $origin" -H "Access-Control-Request-Method: PUT" \
    "http://garage:3900/$BUCKET/probe" | tr -d '\r' | awk -F': ' 'tolower($1)=="access-control-allow-origin"{print $2}')
  if [ "$got" != "$origin" ]; then
    echo "[garage-init] CORS check failed for $origin (got '$got')" >&2
    exit 1
  fi
done
echo "[garage-init]   bucket ready: $BUCKET (CORS for ${WEB_ORIGINS})"
