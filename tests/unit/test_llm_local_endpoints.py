"""A developer may point the workspace at a model on their own machine — in dev only.

`llm_endpoint_security` exists because a configured endpoint is fetched SERVER-SIDE with
the workspace API key attached. A loopback or private address is therefore both a way to
reach whatever the deployment can reach, and a way to send a credential there. Allowing
it is a real reduction in defence, so it is gated twice: the operator opts in AND the
process must be a development environment.

Most of what follows pins the half that must not move — that the opt-in does nothing
once ENVIRONMENT is anything else.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.config import Settings, settings
from src.domain.llm_endpoint_security import (
    UnsafeLLMEndpointError,
    normalize_public_https_url,
    validate_public_endpoint_resolution,
)

OLLAMA = "http://localhost:11434/v1/chat/completions"


@pytest.fixture
def local_allowed(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "LLM_ALLOW_LOCAL_ENDPOINTS", True)
    return settings


# --- the gate itself -------------------------------------------------------


def test_both_conditions_are_required_to_allow_a_local_endpoint():
    """The flag alone is not enough, and neither is the environment alone."""
    assert not Settings(
        ENVIRONMENT="production", LLM_ALLOW_LOCAL_ENDPOINTS=True
    ).allows_local_llm_endpoints
    assert not Settings(
        ENVIRONMENT="development", LLM_ALLOW_LOCAL_ENDPOINTS=False
    ).allows_local_llm_endpoints
    assert Settings(
        ENVIRONMENT="development", LLM_ALLOW_LOCAL_ENDPOINTS=True
    ).allows_local_llm_endpoints


@pytest.mark.parametrize("environment", ["production", "prod", "staging", "uat", "demo", ""])
def test_the_opt_in_is_inert_outside_development(environment):
    """A flag travels in env files and task definitions; a copied value must not be able
    to switch the SSRF defence off on a deployed environment."""
    assert not Settings(
        ENVIRONMENT=environment, LLM_ALLOW_LOCAL_ENDPOINTS=True
    ).allows_local_llm_endpoints


def test_whitespace_in_the_environment_value_does_not_read_as_development():
    """What a Terraform join or a copied YAML line actually produces."""
    assert not Settings(
        ENVIRONMENT=" production ", LLM_ALLOW_LOCAL_ENDPOINTS=True
    ).allows_local_llm_endpoints
    assert Settings(
        ENVIRONMENT=" development ", LLM_ALLOW_LOCAL_ENDPOINTS=True
    ).allows_local_llm_endpoints


def test_the_default_is_off():
    assert Settings(ENVIRONMENT="development").allows_local_llm_endpoints is False


# --- what the default posture still rejects --------------------------------


@pytest.mark.parametrize(
    "url",
    [
        OLLAMA,
        "https://localhost/v1/chat/completions",
        "https://127.0.0.1/v1/chat/completions",
        "https://10.0.0.5/v1/chat/completions",
        "https://ollama.internal/v1/chat/completions",
        "https://api.openai.com:8443/v1/chat/completions",
    ],
)
def test_local_and_private_endpoints_are_rejected_by_default(url):
    assert not settings.allows_local_llm_endpoints, "default posture"
    with pytest.raises(UnsafeLLMEndpointError):
        normalize_public_https_url(url)


# --- what the opt-in permits, and what it still does not -------------------


@pytest.mark.parametrize(
    "url",
    [
        OLLAMA,
        "http://127.0.0.1:11434/v1/chat/completions",
        "http://host.docker.internal:11434/v1/chat/completions",
        "http://192.168.1.20:8000/v1/chat/completions",
        "https://localhost:8443/v1/chat/completions",
    ],
)
def test_a_development_opt_in_accepts_a_locally_hosted_model(local_allowed, url):
    assert normalize_public_https_url(url) == url.rstrip("/")


def test_the_port_survives_normalisation(local_allowed):
    """Ollama is 11434 and vLLM is 8000; losing the port reaches neither."""
    assert normalize_public_https_url(OLLAMA).endswith(":11434/v1/chat/completions")


@pytest.mark.parametrize(
    "url",
    [
        "http://user:password@localhost:11434/v1/chat/completions",
        "http://localhost:11434/v1/chat/completions?key=secret",
        "http://localhost:11434/v1/chat/completions#frag",
        "ftp://localhost:11434/v1/chat/completions",
        "file:///etc/passwd",
    ],
)
def test_the_opt_in_does_not_relax_anything_else(local_allowed, url):
    """A URL carrying credentials, a query or a non-HTTP scheme is a configuration
    mistake at any address, local included."""
    with pytest.raises(UnsafeLLMEndpointError):
        normalize_public_https_url(url)


def test_a_public_endpoint_still_works_while_the_opt_in_is_on(local_allowed):
    url = "https://api.openai.com/v1/chat/completions"
    assert normalize_public_https_url(url) == url


# --- resolution, which runs again at request time --------------------------


def test_resolution_pins_a_loopback_literal_when_permitted(local_allowed):
    resolved = asyncio.run(validate_public_endpoint_resolution("http://127.0.0.1:11434/v1"))
    assert (resolved.hostname, resolved.address) == ("127.0.0.1", "127.0.0.1")


def test_resolution_still_refuses_a_loopback_literal_by_default():
    assert not settings.allows_local_llm_endpoints
    with pytest.raises(UnsafeLLMEndpointError):
        asyncio.run(validate_public_endpoint_resolution("https://127.0.0.1/v1"))


def test_resolution_resolves_localhost_to_a_local_address_when_permitted(local_allowed):
    resolved = asyncio.run(validate_public_endpoint_resolution(OLLAMA))
    assert resolved.hostname == "localhost"
    assert resolved.address in {"127.0.0.1", "::1"}
