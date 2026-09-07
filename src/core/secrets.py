"""Rotatable envelope-encryption boundary for application secrets.

Production keeps this material separate from JWT signing and can decrypt with
short-lived previous keys during a controlled rotation. A KMS/secret manager
can supply the same values without changing callers.
"""
from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import settings


def _fernet_for(material: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(material.encode("utf-8")).digest())
    return Fernet(key)


def _key_materials() -> list[str]:
    """Primary key followed by rotation fallbacks, without duplicates.

    DECRYPTION still falls back to SECRET_KEY: installs that predate
    DATA_ENCRYPTION_KEY hold ciphertexts made with the signing key, and dropping that
    fallback would make their stored connector tokens and LLM keys unreadable. Remove
    SECRET_KEY from PREVIOUS_DATA_ENCRYPTION_KEYS only once everything is re-encrypted.
    """
    values = [settings.DATA_ENCRYPTION_KEY or settings.SECRET_KEY]
    values.extend(key.strip() for key in settings.PREVIOUS_DATA_ENCRYPTION_KEYS.split(",") if key.strip())
    if settings.DATA_ENCRYPTION_KEY:
        values.append(settings.SECRET_KEY)
    return list(dict.fromkeys(values))


def encrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    # A NEW ciphertext must never be made with the JWT signing key. validate_production
    # requires DATA_ENCRYPTION_KEY outside development, so this only ever fires on a
    # local or test run — but that was exactly the run where connector OAuth tokens and
    # the workspace LLM API key ended up encrypted under the key that also signs access
    # tokens, collapsing the separation the whole module exists to provide. Raising here
    # names the missing setting instead of silently producing that state.
    if not settings.DATA_ENCRYPTION_KEY:
        raise RuntimeError(
            "DATA_ENCRYPTION_KEY is required to store a secret; it must be a strong "
            "value distinct from SECRET_KEY, which signs tokens and must not encrypt "
            "data at rest"
        )
    return _fernet_for(settings.DATA_ENCRYPTION_KEY).encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    for material in _key_materials():
        try:
            return _fernet_for(material).decrypt(value.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError, UnicodeEncodeError, ValueError):
            continue
    return None
