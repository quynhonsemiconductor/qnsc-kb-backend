import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_compose_takes_the_redis_password_from_env_not_a_literal():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "devredis" not in compose
    assert compose.count("${KB_LOCAL_REDIS_PASSWORD:?") == 5


def test_env_example_documents_the_local_redis_password():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert re.search(r'(?m)^KB_LOCAL_REDIS_PASSWORD="[^"]+"$', env_example)
