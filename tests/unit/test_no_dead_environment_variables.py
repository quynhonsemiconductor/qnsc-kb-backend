"""Every environment variable the project sets must be read by something.

Dead variables are not harmless. `ENTRA_ADMIN_EMAILS` and `ENTRA_AUTO_PROVISION_DOMAIN` stayed in the
Terraform task definitions, with comments saying who "arrives as an administrator", long after the code that
read them was deliberately removed: the infrastructure described an access rule that no longer existed.
`.env.example` told people the chat provider keys were "read as defaults for that page" when nothing read them.

Two guards:
* every Settings field is read somewhere outside its declaration (by code, a script, a migration, or by
  config.py itself in a property or validator);
* every variable Terraform puts in a task definition is a Settings field or is read through os.getenv/os.environ.
"""
from __future__ import annotations

import pathlib
import re

from src.core.config import Settings

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _text(*globs: str, exclude: tuple[str, ...] = ()) -> str:
    chunks = []
    for pattern in globs:
        for path in ROOT.glob(pattern):
            if path.is_file() and path.name not in exclude and "__pycache__" not in path.parts:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _word(name: str, text: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\b", text))


def test_every_setting_is_read_somewhere_outside_its_declaration():
    elsewhere = _text("src/**/*.py", "scripts/**/*.py", "migrations/**/*.py", exclude=("config.py",))
    own = (ROOT / "src/core/config.py").read_text(encoding="utf-8")
    dead = [
        name
        for name in Settings.model_fields
        if _word(name, elsewhere) == 0 and _word(name, own) <= 1  # the declaration is the one mention
    ]
    assert not dead, (
        f"Settings fields nothing reads: {sorted(dead)}. Delete them (and their .env.example, compose and "
        "Terraform entries) rather than leave a knob that does nothing."
    )


def test_every_variable_terraform_sets_is_read():
    settings_fields = set(Settings.model_fields)
    read_directly = set(
        re.findall(
            r"os\.(?:getenv|environ\.get|environ)\(?\[?[\"']([A-Z][A-Z0-9_]+)[\"']",
            _text("src/**/*.py", "scripts/**/*.py", "migrations/**/*.py"),
        )
    )
    terraform = (ROOT / "infra/modules/stack/main.tf").read_text(encoding="utf-8")
    names = set(re.findall(r'name\s*=\s*"([A-Z][A-Z0-9_]+)"\s*,\s*(?:value|secret_arn)', terraform))
    assert names, "no environment found in the stack module; has its layout changed?"
    unread = sorted(n for n in names if n not in settings_fields and n not in read_directly)
    assert not unread, f"Terraform sets variables nothing reads: {unread}"
