"""Every environment variable the project sets must be read by something.

Dead variables are not harmless. `ENTRA_ADMIN_EMAILS` and `ENTRA_AUTO_PROVISION_DOMAIN` stayed in the
Terraform task definitions, with comments saying who "arrives as an administrator", long after the code that
read them was deliberately removed: the infrastructure described an access rule that no longer existed.
`.env.example` told people the chat provider keys were "read as defaults for that page" when nothing read them.

Reads are counted in CODE only: comments are stripped first (a name that survives only in a comment is not
read), and the Terraform side understands both ways a task definition names a variable, the
`{ name = "X", value = ... }` list entries and the `environment = { X = ... }` map the migrator uses.

Two guards:
* every Settings field is read somewhere outside its declaration (by code, a script, a migration, or by
  config.py itself in a property or validator);
* every variable Terraform puts in a task definition is a Settings field or is read through os.getenv/os.environ.
"""
from __future__ import annotations

import io
import pathlib
import re
import tokenize

from src.core.config import Settings

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _strip_python_comments(source: str) -> str:
    """Drop `#` comments (tokenize, so a `#` inside a string is left alone); docstrings and strings stay,
    because `os.getenv("NAME")` is a string and a real read."""
    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(tokens)
    except (tokenize.TokenError, IndentationError):
        return source  # unparseable: count it as written rather than hide a read


def _text(*globs: str, exclude: tuple[str, ...] = ()) -> str:
    chunks = []
    for pattern in globs:
        for path in ROOT.glob(pattern):
            if path.is_file() and path.name not in exclude and "__pycache__" not in path.parts:
                source = path.read_text(encoding="utf-8", errors="ignore")
                chunks.append(_strip_python_comments(source) if path.suffix == ".py" else source)
    return "\n".join(chunks)


def _strip_terraform_comments(source: str) -> str:
    # `//` and `#` only where a comment can start (line start or after whitespace), so `https://` survives.
    return re.sub(r"(^|\s)(//|#).*$", r"\1", source, flags=re.M)


def _without_provisioners(source: str) -> str:
    """Drop `provisioner "..." { ... }` blocks. A local-exec `environment` is the environment of that one
    shell command on the runner, never a container's: it is not a task variable and must not count as one."""
    kept, cursor = [], 0
    for match in re.finditer(r'\bprovisioner\s+"[^"]*"\s*\{', source):
        if match.start() < cursor:  # inside a block already removed
            continue
        kept.append(source[cursor : match.start()])
        depth, i = 1, match.end()
        while i < len(source) and depth:
            depth += {"{": 1, "}": -1}.get(source[i], 0)
            i += 1
        cursor = i
    kept.append(source[cursor:])
    return "".join(kept)


def terraform_variable_names(source: str) -> set[str]:
    """Every environment variable a stack module puts in a task: list entries and `environment = { ... }` maps."""
    source = _without_provisioners(_strip_terraform_comments(source))
    names = set(re.findall(r'name\s*=\s*"([A-Z][A-Z0-9_]+)"\s*,\s*(?:value|secret_arn)', source))
    for match in re.finditer(r"\benvironment\s*=\s*\{", source):
        depth, i = 1, match.end()
        while i < len(source) and depth:
            depth += {"{": 1, "}": -1}.get(source[i], 0)
            i += 1
        block = source[match.end() : i - 1]
        # depth-1 keys only: a nested map/object inside a value is not a variable name
        level, flat = 0, []
        for line in block.splitlines():
            if level == 0:
                key = re.match(r"\s*([A-Z][A-Z0-9_]+)\s*=", line)
                if key:
                    names.add(key.group(1))
            level += line.count("{") - line.count("}")
            flat.append(line)
    return names


def _word(name: str, text: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\b", text))


def test_every_setting_is_read_somewhere_outside_its_declaration():
    elsewhere = _text("src/**/*.py", "scripts/**/*.py", "migrations/**/*.py", exclude=("config.py",))
    own = _strip_python_comments((ROOT / "src/core/config.py").read_text(encoding="utf-8"))
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
    names: set[str] = set()
    for path in sorted((ROOT / "infra/modules/stack").glob("*.tf")):
        names |= terraform_variable_names(path.read_text(encoding="utf-8"))
    assert names, "no environment found in the stack module; has its layout changed?"
    assert {"ENABLE_RLS", "APP_DATABASE_ROLE"} <= names, "the migrator's `environment = {...}` map was not parsed"
    unread = sorted(n for n in names if n not in settings_fields and n not in read_directly)
    assert not unread, f"Terraform sets variables nothing reads: {unread}"


# ── the guard itself ─────────────────────────────────────────────────────────────────────────
def test_a_name_that_survives_only_in_a_comment_is_not_a_read():
    code = "x = 1  # DEFAULT_LANGUAGE is documented here\ny = os.getenv('REAL_ONE')\n"
    stripped = _strip_python_comments(code)
    assert _word("DEFAULT_LANGUAGE", stripped) == 0
    assert _word("REAL_ONE", stripped) == 1, "a string read must survive comment stripping"
    assert _word("A_HASH", _strip_python_comments("v = '# A_HASH inside a string'\n")) == 1


def test_terraform_names_are_found_in_lists_and_in_environment_maps_and_not_in_comments():
    source = '''
      common_env = [
        { name = "LISTED", value = "1" },
        // { name = "COMMENTED_OUT", value = "1" },
        { name = "A_SECRET", secret_arn = "arn" },
      ]
      environment = {
        MAPPED = "x"
        # COMMENTED_MAP = "y"
        NESTED = { INNER = "z" }
        URL    = "https://example.test/path"
      }
    '''
    assert terraform_variable_names(source) == {"LISTED", "A_SECRET", "MAPPED", "NESTED", "URL"}


def test_a_provisioner_environment_is_not_a_task_variable_but_a_task_map_next_to_it_still_is():
    source = '''
      resource "terraform_data" "writer" {
        provisioner "local-exec" {
          environment = {
            TUNNEL_TOKEN = module.tunnel[0].token
            SECRET_ID    = aws_secretsmanager_secret.x[0].id
          }
          command = "set -eu; { echo ok; }; printf '%s' \\"$TUNNEL_TOKEN\\" | aws secretsmanager put-secret-value"
        }
      }
      environment = {
        REAL_TASK_VAR = "1"
      }
    '''
    assert terraform_variable_names(source) == {"REAL_TASK_VAR"}
