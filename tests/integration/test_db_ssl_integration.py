"""Proves the DATABASE_SSL_CA decision against a REAL Postgres speaking TLS.

CI (backend-ci.yml) and the local harness generate a self-signed CA with SAN
localhost and start Postgres with ssl=on. Skipped entirely when PG_TLS_URL is
unset, so ordinary local runs are untouched.
"""

import os
import ssl

import asyncpg
import pytest

from src.core.db_ssl import asyncpg_ssl

pytestmark = pytest.mark.skipif(
    not os.environ.get("PG_TLS_URL"), reason="PG_TLS_URL not set"
)


async def _connect(ca_file: str):
    context = asyncpg_ssl(ca_file, "production")
    assert context is not None
    return await asyncpg.connect(os.environ["PG_TLS_URL"], ssl=context)


@pytest.mark.asyncio
async def test_right_ca_connects_with_verify_full() -> None:
    connection = await _connect(os.environ["PG_TLS_CA_FILE"])
    try:
        assert await connection.fetchval("select 1") == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_wrong_hostname_is_refused() -> None:
    # A second server (PG_TLS_WRONG_HOST_URL) presents a certificate whose SAN has
    # no IP entry, so connecting to 127.0.0.1 exercises the hostname check: the
    # right CA must not make a wrong name acceptable.
    url = os.environ.get("PG_TLS_WRONG_HOST_URL")
    if not url:
        pytest.skip("PG_TLS_WRONG_HOST_URL not set")
    context = asyncpg_ssl(os.environ["PG_TLS_CA_FILE"], "production")
    with pytest.raises(ssl.SSLCertVerificationError, match="certificate verify failed"):
        await asyncpg.connect(url, ssl=context)


@pytest.mark.asyncio
async def test_wrong_ca_is_refused(tmp_path) -> None:
    # A second, valid CA the server does not chain to: the real failure mode of a
    # swapped or stale bundle, and the one verify-full exists to catch.
    rogue = os.environ["PG_TLS_ROGUE_CA_FILE"]
    with pytest.raises(ssl.SSLCertVerificationError, match="certificate verify failed"):
        await _connect(rogue)
