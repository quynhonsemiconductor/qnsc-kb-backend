"""Validation for administrator-configured outbound LLM endpoints."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


class UnsafeLLMEndpointError(ValueError):
    """The endpoint could expose provider credentials to an unsafe destination."""


@dataclass(frozen=True)
class ResolvedLLMEndpoint:
    hostname: str
    address: str


def _local_endpoints_allowed() -> bool:
    """Whether loopback/private LLM endpoints are permitted in THIS process.

    Read at call time rather than captured at import, so a test (or a reloaded dev
    process) that changes the setting is actually honoured. Requires both the operator
    opt-in and a development environment — see `Settings.allows_local_llm_endpoints`.
    """
    from src.core.config import settings

    return settings.allows_local_llm_endpoints


def normalize_public_https_url(value: str) -> str:
    """Normalize a credential-free LLM endpoint URL before it is persisted.

    Public HTTPS on port 443 is the only shape accepted by default. A development
    deployment may additionally opt into loopback/private endpoints so the workspace can
    be pointed at a locally hosted model (Ollama, vLLM, LM Studio); everything else here
    still applies, because a URL carrying credentials or a query string is a
    configuration mistake at any address.
    """
    candidate = value.strip().rstrip("/")
    allow_local = _local_endpoints_allowed()
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeLLMEndpointError("The LLM endpoint has an invalid port") from exc

    hostname = (parsed.hostname or "").lower().rstrip(".")
    allowed_schemes = {"https", "http"} if allow_local else {"https"}
    if (
        parsed.scheme not in allowed_schemes
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise UnsafeLLMEndpointError(
            "The LLM endpoint must be a credential-free "
            + ("HTTP(S)" if allow_local else "HTTPS")
            + " URL without a query or fragment"
        )
    if not allow_local and port not in (None, 443):
        raise UnsafeLLMEndpointError("The LLM endpoint must use port 443")
    if not allow_local:
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
            or hostname.endswith(".internal")
            or hostname.endswith(".home.arpa")
        ):
            raise UnsafeLLMEndpointError(
                "The LLM endpoint cannot target a local network host"
            )

        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise UnsafeLLMEndpointError(
                "The LLM endpoint cannot target a private or reserved address"
            )

    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


async def validate_public_endpoint_resolution(url: str) -> ResolvedLLMEndpoint:
    """Resolve once, reject mixed/private answers, and return an address to pin.

    Called again at request time, not just at save time: the saved hostname could have
    been re-pointed at a private address since (DNS rebinding), so the address the
    socket connects to is the one validated here.

    When a development deployment has opted into local endpoints, a private answer is
    the intended destination and is pinned like any other.
    """
    normalized = normalize_public_https_url(url)
    parsed = urlsplit(normalized)
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeLLMEndpointError("The LLM endpoint has no hostname")
    allow_local = _local_endpoints_allowed()
    # The URL's own port, not a fixed 443: a locally hosted model listens somewhere else
    # entirely (11434 for Ollama, 8000 for vLLM), and resolving for the wrong port
    # returns addresses for a service that is not the one about to be called.
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global and not allow_local:
            raise UnsafeLLMEndpointError(
                "The LLM endpoint cannot target a private or reserved address"
            )
        return ResolvedLLMEndpoint(hostname=hostname, address=str(literal))

    loop = asyncio.get_running_loop()
    try:
        answers = await loop.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise UnsafeLLMEndpointError(
            "The LLM endpoint hostname could not be resolved"
        ) from exc
    addresses: set[str] = {str(answer[4][0]).split("%", 1)[0] for answer in answers}
    if not addresses:
        raise UnsafeLLMEndpointError("The LLM endpoint hostname could not be resolved")
    if not allow_local and any(
        not ipaddress.ip_address(address).is_global for address in addresses
    ):
        raise UnsafeLLMEndpointError(
            "The LLM endpoint resolves to a private or reserved address"
        )
    return ResolvedLLMEndpoint(hostname=hostname, address=sorted(addresses)[0])
