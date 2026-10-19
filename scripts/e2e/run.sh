#!/usr/bin/env bash
# Runs the backend test suite (the same `pytest` CI runs) against an ISOLATED stack: its own compose project, ephemeral loopback
# ports (no fixed host port is used), and a migrated database.
#
#   scripts/e2e/run.sh [pytest args...]          # the whole suite, or the files/tests you name
#   E2E_KEEP=1 scripts/e2e/run.sh                # keep the stack up afterwards (default: `down -v`; the stack is disposable)
#   COMPOSE_PROJECT_NAME=qnsc-kb-e2e-alice scripts/e2e/run.sh   # a name for this run (default: random)
#
# Needs `poetry install` once. Never run this against a stack you did not start: the tests write to the database they point at.
set -euo pipefail
cd "$(dirname "$0")/../.."

files=(-f docker-compose.yml -f docker-compose.e2e.yml)
project=${COMPOSE_PROJECT_NAME:-qnsc-kb-e2e-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')}
redis_password=${KB_LOCAL_REDIS_PASSWORD:-devredis}

cleanup() {
  if [ "${E2E_KEEP:-0}" != 1 ]; then
    KB_LOCAL_REDIS_PASSWORD=$redis_password docker compose -p "$project" "${files[@]}" down -v
  fi
}
trap cleanup EXIT

KB_LOCAL_REDIS_PASSWORD=$redis_password docker compose -p "$project" "${files[@]}" up -d --wait db redis

# The containers' ephemeral ports, as Docker assigned them.
port_of() { docker compose -p "$project" "${files[@]}" port "$1" "$2" | awk -F: '{print $NF}'; }
db_port=$(port_of db 5432)
redis_port=$(port_of redis 6379)

export DATABASE_URL="postgresql+asyncpg://postgres:postgres@127.0.0.1:${db_port}/qnsc_kb"
export REDIS_URL="redis://:${redis_password}@127.0.0.1:${redis_port}/0"

poetry run alembic -c migrations/alembic.ini upgrade head
poetry run pytest -q "$@"
