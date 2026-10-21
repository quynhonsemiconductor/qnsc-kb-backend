"""TLS for every asyncpg connection: API engine, worker engines, Alembic, scripts.

``DATABASE_SSL_CA`` is a path to the CA bundle database certificates must chain to —
the RDS global bundle (baked into the image at ``/etc/ssl/rds-global-bundle.pem``)
while the platform is on AWS, the mounted CNPG ``ca.crt`` on-prem.

- CA set: an ``SSLContext`` with ``CERT_REQUIRED`` and hostname checking — asyncpg's
  verify-full. The certificate must chain to the bundle AND match the host.
- CA unset with ``ENVIRONMENT=production``: the bundle baked into the image
  (``BAKED_CA_PATH``, the rds-ca stage) is used, and if even that is missing the
  process raises. Production never talks to the database unverified; a silent
  ``ssl=None`` fallback would turn any missed wiring into traffic a
  man-in-the-middle can read.
- CA unset otherwise (local dev, CI): ``None`` — asyncpg connects without TLS,
  exactly as before. The baked path is deliberately NOT a container-level
  ``ENV`` default: that would leak verify-full into dev compose, whose database
  has no TLS.
"""

import os
import ssl

from src.core.config import settings

BAKED_CA_PATH = "/etc/ssl/rds-global-bundle.pem"


def asyncpg_ssl(
    ca_file: str | None = None, environment: str | None = None
) -> ssl.SSLContext | None:
    """The verify-full context, or None where TLS is deliberately off."""
    ca_file = settings.DATABASE_SSL_CA if ca_file is None else ca_file
    environment = (settings.ENVIRONMENT if environment is None else environment) or ""
    if (
        not ca_file
        and environment.strip().lower() == "production"
        and os.path.exists(BAKED_CA_PATH)
    ):
        ca_file = BAKED_CA_PATH
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
            "connection that is not verified. Set it to a CA bundle file (the RDS global "
            "bundle is baked into the image; on-prem use the CNPG ca.crt), or set "
            "ENVIRONMENT=development for local work."
        )
    return None


def ssl_connect_args(
    ca_file: str | None = None, environment: str | None = None
) -> dict:
    """asyncpg/SQLAlchemy ``connect_args`` entries; empty when TLS is off."""
    context = asyncpg_ssl(ca_file, environment)
    return {"ssl": context} if context else {}
