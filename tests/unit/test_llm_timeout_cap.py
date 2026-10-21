"""The non-streaming LLM timeout stays under Cloudflare's 100 s origin limit.

A proxied request past 100 s dies as a 524 with an empty body, so a longer
server-side wait is never observable by the client — the cap exists to keep the
answer reachable (ONPREM-PLAN §8.2 / §13.1 P0-14).
"""

from src.core.config import Settings


def test_default_llm_timeout_is_under_the_100s_edge_limit():
    settings = Settings()
    assert 0 < settings.LLM_TIMEOUT_SECONDS <= 60.0


def test_validate_production_refuses_a_timeout_that_would_524():
    """A copied 90 s env value is refused at boot in a deployed environment, not discovered in production."""
    import pytest

    settings = Settings(ENVIRONMENT="production")
    settings.LLM_TIMEOUT_SECONDS = 90.0  # as a stale task-definition value would arrive
    with pytest.raises(RuntimeError, match="60 s"):
        settings.validate_production()


def test_validate_production_admits_the_default_60s():
    settings = Settings(ENVIRONMENT="production")
    assert settings.LLM_TIMEOUT_SECONDS == 60.0
