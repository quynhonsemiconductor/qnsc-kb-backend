import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _env_example_defaults() -> dict[str, str]:
    """KEY=value for every host-port key, commented out or not: a copy of the file is the setup."""
    found = re.findall(r"(?m)^#?\s?([A-Z_]*_PORT)=(\S+)$", _read(".env.example"))
    return dict(found)


def test_every_published_port_binds_loopback_and_reads_its_env_default():
    compose = _read("docker-compose.yml")
    published = re.findall(r'^\s+- "([^"]+)"$', compose, flags=re.MULTILINE)
    assert published, "docker-compose.yml publishes no ports"
    for mapping in published:
        assert mapping.startswith(
            "127.0.0.1:"
        ), f"{mapping} is reachable beyond loopback"

    defaults = _env_example_defaults()
    interpolated = re.findall(r"\$\{([A-Z_]*PORT):-(\d+)\}", compose)
    assert {name for name, _ in interpolated} == {
        "KB_DB_PORT",
        "KB_REDIS_PORT",
        "KB_API_PORT",
        "LOCALSTACK_HOST_PORT",
        "FRONTEND_DEV_PORT",
    }
    for name, value in interpolated:
        assert (
            defaults.get(name) == value
        ), f"{name} default {value} is not what .env.example says"


def test_compose_names_no_fixed_container_and_sets_one_project_name():
    compose = _read("docker-compose.yml")
    assert "container_name:" not in compose
    assert re.search(r"(?m)^name: qnsc-kb-backend$", compose)


def test_compose_images_match_production_engines():
    compose = _read("docker-compose.yml")
    assert re.search(r"(?m)^\s+image: pgvector/pgvector:\d+\.\d+\.\d+-pg16$", compose)
    assert re.search(r"(?m)^\s+image: valkey/valkey:7\.2\.\d+-alpine$", compose)
    assert not re.search(
        r"image: \S+:(latest|pg16|7-alpine|7)\s*$", compose, flags=re.MULTILINE
    )


def test_ci_and_readme_address_services_not_containers():
    for name in (".github/workflows/backend-ci.yml", "README.md"):
        text = _read(name)
        assert not re.search(r"docker (exec|logs|rm|stop) qnsc-kb", text), name


def test_the_source_bucket_name_is_the_same_in_the_env_example_the_init_hook_and_the_healthcheck():
    env = dict(re.findall(r"(?m)^([A-Z_]+)=\"?([^\"\n]*)\"?$", _read(".env.example")))
    bootstrap = re.search(
        r'(?m)^BUCKET="([^"]+)"', _read("scripts/localstack/01-bootstrap.sh")
    )
    assert bootstrap, "BUCKET not found in scripts/localstack/01-bootstrap.sh"
    assert env["SOURCE_STORAGE_BUCKET"] == bootstrap.group(1)
    assert f'"--bucket", "{bootstrap.group(1)}"' in _read("docker-compose.yml")


def test_the_app_containers_reach_localstack_by_its_service_name_and_the_browser_by_the_host_port():
    compose = _read("docker-compose.yml")
    assert (
        compose.count("S3_ENDPOINT_URL=http://localstack:4566") == 3
    )  # api, worker, beat
    assert (
        "S3_PUBLIC_ENDPOINT_URL=http://localhost:${LOCALSTACK_HOST_PORT:-4570}"
        in compose
    )
