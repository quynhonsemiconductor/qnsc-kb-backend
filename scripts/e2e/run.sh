#!/usr/bin/env bash
# Runs the backend test suite (the same `pytest` CI runs) against an ISOLATED stack: its own compose project, ephemeral loopback
# ports for Postgres and Valkey, and a migrated database. It matches CI's test job: the same database name, the same environment, and
# no .env (the settings read one from the working directory, and CI has none, so the run refuses to start beside a .env).
#
#   scripts/e2e/run.sh [pytest args...]                # the whole suite, or the files you name
#   E2E_KEEP=1 scripts/e2e/run.sh                      # keep the stack up afterwards and print its project name
#   E2E_PROJECT=qnsc-kb-e2e-alice scripts/e2e/run.sh   # a name for this run (default: random). Not COMPOSE_PROJECT_NAME.
#
# Needs `poetry install` once. The tests write to the database they point at: never point this at a stack you did not start.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -e .env ]; then
  echo "refusing to run: a .env is present, and the settings would read it. CI has no .env: run from a clean clone," \
       "or move .env aside for this run." >&2
  exit 1
fi

files=(-f docker-compose.yml -f docker-compose.e2e.yml)
project=${E2E_PROJECT:-qnsc-kb-e2e-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')}
if [ -n "$(docker ps -aq --filter "label=com.docker.compose.project=$project")" ]; then
  echo "refusing to run: compose project $project already has containers. Pick another E2E_PROJECT." >&2
  exit 1
fi

redis_password=${KB_LOCAL_REDIS_PASSWORD:-devredis}
export KB_LOCAL_REDIS_PASSWORD=$redis_password
compose() { docker compose -p "$project" "${files[@]}" "$@"; }

cleanup() {
  if [ "${E2E_KEEP:-0}" = 1 ]; then
    echo "kept compose project $project. Remove it with: KB_LOCAL_REDIS_PASSWORD=$redis_password docker compose -p $project ${files[*]} down -v"
  else
    compose down -v
  fi
}
trap cleanup EXIT

compose up -d --wait db redis

# The containers' ephemeral ports, as Docker assigned them.
port_of() { compose port "$1" "$2" | awk -F: '{print $NF}'; }
db_port=$(port_of db 5432)
redis_port=$(port_of redis 6379)

# CI's test database, created for this run.
compose exec -T db psql -U postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE qnsc_kb_test" >/dev/null

# The environment CI's test job sets, with the two ports pointing at this run's stack.
export PYTHONPATH=.
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@127.0.0.1:${db_port}/qnsc_kb_test"
export MIGRATION_DATABASE_URL="$DATABASE_URL"
export REDIS_URL="redis://:${redis_password}@127.0.0.1:${redis_port}/0"

poetry run alembic -c migrations/alembic.ini upgrade head
poetry run pytest -q "$@"
