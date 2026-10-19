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
        assert mapping.startswith("127.0.0.1:"), f"{mapping} is reachable beyond loopback"

    defaults = _env_example_defaults()
    interpolated = re.findall(r"\$\{([A-Z_]*PORT):-(\d+)\}", compose)
    assert {name for name, _ in interpolated} == {
        "KB_DB_PORT",
        "KB_REDIS_PORT",
        "KB_API_PORT",
        "STORAGE_HOST_PORT",
        "FRONTEND_DEV_PORT",
    }
    for name, value in interpolated:
        assert defaults.get(name) == value, f"{name} default {value} is not what .env.example says"


def test_compose_names_no_fixed_container_and_sets_one_project_name():
    compose = _read("docker-compose.yml")
    assert "container_name:" not in compose
    assert re.search(r"(?m)^name: qnsc-kb-backend$", compose)


def test_compose_images_match_production_engines():
    compose = _read("docker-compose.yml")
    assert re.search(r"(?m)^\s+image: pgvector/pgvector:\d+\.\d+\.\d+-pg16$", compose)
    assert re.search(r"(?m)^\s+image: valkey/valkey:7\.2\.\d+-alpine$", compose)
    assert not re.search(r"image: \S+:(latest|pg16|7-alpine|7)\s*$", compose, flags=re.MULTILINE)


def test_ci_and_readme_address_services_not_containers():
    for name in (".github/workflows/backend-ci.yml", "README.md"):
        text = _read(name)
        assert not re.search(r"docker (exec|logs|rm|stop) qnsc-kb", text), name


def test_the_source_bucket_name_is_the_same_in_the_env_example_the_init_hook_and_the_compose_file():
    env = dict(re.findall(r"(?m)^([A-Z_]+)=\"?([^\"\n]*)\"?$", _read(".env.example")))
    init = re.search(r'(?m)^BUCKET="\$\{SOURCE_STORAGE_BUCKET:-([^}]+)\}"', _read("scripts/garage/init.sh"))
    assert init, "BUCKET default not found in scripts/garage/init.sh"
    assert env["SOURCE_STORAGE_BUCKET"] == init.group(1)
    assert f"SOURCE_STORAGE_BUCKET={init.group(1)}" in _read("docker-compose.yml")


def test_the_app_containers_reach_garage_by_its_service_name_and_the_browser_by_the_host_port():
    compose = _read("docker-compose.yml")
    assert compose.count("S3_ENDPOINT_URL=http://garage:3900") == 3  # api, worker, beat
    assert "S3_PUBLIC_ENDPOINT_URL=http://localhost:${STORAGE_HOST_PORT:-4570}" in compose
    # The app containers wait for the bucket and CORS to be in place, not just for the node to answer.
    assert compose.count("garage-init:\n") >= 1
    assert compose.count("condition: service_completed_successfully") >= 3


def test_the_bucket_cors_allows_both_loopback_names_of_the_frontend_and_the_init_proves_it():
    compose = _read("docker-compose.yml")
    init = _read("scripts/garage/init.sh")
    # localhost and 127.0.0.1 are different origins to the browser: both must be allowed, and the container must prove it.
    assert "http://127.0.0.1:" in compose and "http://localhost:" in compose
    assert "WEB_ORIGINS" in compose
    # One CORS rule per origin: a joined origin list is one invalid Access-Control-Allow-Origin header.
    assert "for origin in" in init and "put-bucket-cors" in init
    assert "CORS check failed" in init
    # Failures are loud: the create and CORS steps never swallow an error.
    for line in init.splitlines():
        if "put-bucket-cors" in line or "/v1/bucket\"" in line or "key/import" in line:
            assert "|| true" not in line, line


def test_the_garage_healthcheck_asks_the_node_and_the_init_is_a_one_shot():
    compose = _read("docker-compose.yml")
    garage = compose.split("  garage:\n", 1)[1].split("\n  garage-init:", 1)[0]
    assert '["CMD", "/garage", "status"]' in garage
    assert 'restart: "no"' in compose.split("  garage-init:\n", 1)[1].split("\n  api:", 1)[0]


def test_the_garage_secret_is_quoted_so_yaml_keeps_it_a_string():
    # An all-zero scalar without quotes is an integer to YAML: garage-init then imported the key "0" and Garage refused it.
    compose = _read("docker-compose.yml")
    assert re.search(r'(?m)^      STORAGE_SECRET_ACCESS_KEY: "[0-9a-f]{64}"$', compose)
