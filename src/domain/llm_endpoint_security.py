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


def normalize_public_https_url(value: str) -> str:
    """Normalize a credential-free public HTTPS URL before it is persisted."""
    candidate = value.strip().rstrip("/")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeLLMEndpointError("The LLM endpoint has an invalid port") from exc

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise UnsafeLLMEndpointError(
            "The LLM endpoint must be a credential-free HTTPS URL without a query or fragment"
        )
    if port not in (None, 443):
        raise UnsafeLLMEndpointError("The LLM endpoint must use port 443")
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
    """Resolve once, reject mixed/private answers, and return an address to pin."""
    parsed = urlsplit(normalize_public_https_url(url))
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeLLMEndpointError("The LLM endpoint has no hostname")

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise UnsafeLLMEndpointError(
                "The LLM endpoint cannot target a private or reserved address"
            )
        return ResolvedLLMEndpoint(hostname=hostname, address=str(literal))

    loop = asyncio.get_running_loop()
    try:
        answers = await loop.getaddrinfo(
            hostname,
            443,
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
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise UnsafeLLMEndpointError(
            "The LLM endpoint resolves to a private or reserved address"
        )
    return ResolvedLLMEndpoint(hostname=hostname, address=sorted(addresses)[0])
