"""TLS for every asyncpg connection: API engine, worker engines, Alembic, scripts.

``DATABASE_SSL_CA`` is a path to the CA bundle database certificates must chain to —
the mounted CNPG ``ca.crt`` (the platform chart sets it).

- CA set: an ``SSLContext`` with ``CERT_REQUIRED`` and hostname checking — asyncpg's
  verify-full. The certificate must chain to the bundle AND match the host.
- CA unset with ``ENVIRONMENT=production``: the process raises. Production never talks to
  the database unverified; a silent ``ssl=None`` fallback would turn any missed wiring
  into traffic a man-in-the-middle can read.
- CA unset otherwise (local dev, CI): ``None`` — asyncpg connects without TLS,
  exactly as before.
"""

import ssl

from src.core.config import settings


def asyncpg_ssl(
    ca_file: str | None = None, environment: str | None = None
) -> ssl.SSLContext | None:
    """The verify-full context, or None where TLS is deliberately off."""
    ca_file = settings.DATABASE_SSL_CA if ca_file is None else ca_file
    environment = (settings.ENVIRONMENT if environment is None else environment) or ""
    if ca_file:
        context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH, cafile=ca_file
        )
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    if (environment or "").strip().lower() == "production":
        raise RuntimeError(
            "DATABASE_SSL_CA is required when ENVIRONMENT=production: refusing a database "
            "connection that is not verified. Set it to a CA bundle file (the CNPG ca.crt), or set "
            "ENVIRONMENT=development for local work."
        )
    return None


def ssl_connect_args(
    ca_file: str | None = None, environment: str | None = None
) -> dict:
    """asyncpg/SQLAlchemy ``connect_args`` entries; empty when TLS is off."""
    context = asyncpg_ssl(ca_file, environment)
    return {"ssl": context} if context else {}
