from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migrations_have_one_current_head():
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "migrations" / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    # Pinned on purpose: a new revision must be a deliberate act, and a second head
    # means one was chained off the wrong parent. Bumped for 20260830_67, which keeps the
    # exception behind a 500 on the request metric row, so a production failure can be
    # diagnosed without an AWS role switch to read CloudWatch.
    assert script.get_heads() == ["20260830_67"]


def test_production_compose_is_explicitly_hardened():
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.production.yml").read_text(encoding="utf-8")
    assert "AUTO_CREATE_SCHEMA: \"false\"" in compose
    assert "ALLOW_SELF_REGISTRATION: \"false\"" in compose
    assert "JOB_MODE: celery" in compose
    assert "MIGRATION_DATABASE_URL" in compose
    assert "APP_DATABASE_ROLE" in compose
    assert "ENABLE_RLS: \"true\"" in compose
    assert "5432:5432" not in compose
    assert "6379:6379" not in compose
    assert "caddy:2.8-alpine" in compose
    assert "PUBLIC_HOSTNAME" in compose
    assert "GEMINI_MODEL" in compose
    assert "MICROSOFT_CLIENT_ID: ${MICROSOFT_CLIENT_ID:?set MICROSOFT_CLIENT_ID}" in compose
    assert "MICROSOFT_CLIENT_SECRET: ${MICROSOFT_CLIENT_SECRET:?set MICROSOFT_CLIENT_SECRET}" in compose
    assert "MICROSOFT_TENANT_ID: ${MICROSOFT_TENANT_ID:?set MICROSOFT_TENANT_ID}" in compose
    assert "MICROSOFT_LOGIN_REDIRECT_URI: ${MICROSOFT_LOGIN_REDIRECT_URI:?set MICROSOFT_LOGIN_REDIRECT_URI}" in compose
    assert compose.count("MICROSOFT_CLIENT_ID: ${MICROSOFT_CLIENT_ID:?set MICROSOFT_CLIENT_ID}") == 2
    assert compose.count("MICROSOFT_CLIENT_SECRET: ${MICROSOFT_CLIENT_SECRET:?set MICROSOFT_CLIENT_SECRET}") == 2
    assert compose.count("MICROSOFT_TENANT_ID: ${MICROSOFT_TENANT_ID:?set MICROSOFT_TENANT_ID}") == 2
    # The bootstrap administrator is a GLOBAL admin. An unset password must fail the
    # compose invocation rather than fall back to the one published in .env.example.
    assert "BOOTSTRAP_ADMIN_PASSWORD: ${BOOTSTRAP_ADMIN_PASSWORD:?set BOOTSTRAP_ADMIN_PASSWORD}" in compose
    assert "BOOTSTRAP_ADMIN_PASSWORD:-" not in compose
    # Migrations are not run from any service's entrypoint any more — the `migrator`
    # image target owns them, behind a compose profile, so a scale-out cannot fire N
    # concurrent migrations. Locally that means `alembic upgrade head` by hand.
    assert "target: migrator" in compose
    dev_compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    # No service has an entrypoint script any more: docker/entrypoint.sh is gone and
    # each service states its own `command`, so nothing can run migrations on start.
    assert "entrypoint:" not in dev_compose
    assert "AUTO_CREATE_SCHEMA=false" in dev_compose
    # Dev needs the same migrator, behind the same profile. Without it there was no
    # compose path to apply a migration at all, so a new table surfaced only as
    # `relation "..." does not exist` from a beat task, every 30 seconds.
    assert "target: migrator" in dev_compose
    assert 'profiles: ["migrate"]' in dev_compose
