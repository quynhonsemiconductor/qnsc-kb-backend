"""The shipped images install the `onnx` group, never `ml` (torch, transformers, scipy).

PR #54 took `ml` out of the images, on the reasoning that the deployed runtime is ONNX and
torch is ~0.8 GB of site-packages (torch 695 MB, scipy 83, transformers 53 by the team's own
measurement in docs/embedding-runtime-onnx.md) that nothing imports. A later rewrite of the
Dockerfile (#69) put `--only main,ml` back, silently: nothing failed, the images just grew
again, while the docs and infra/modules/stack/variables.tf kept saying `ml` was gone.

These tests pin the three things that make dropping it safe, and the one that keeps it dropped:

  * no `poetry install` in the Dockerfile asks for `ml`;
  * everything the ONNX runtime and the build-time quantiser import still arrives, via `onnx`
    or `main`, so the swap cannot drop a package the loader needs;
  * nothing in src/ imports torch / sentence-transformers / transformers at import time, so an
    image without them can still start (the two backends that need them import lazily and
    raise a clear error when the flag that selects them is set).

Read as text, like the other Dockerfile tests: CI builds the images with the model bake off,
and a regression here is a heavier image rather than a failed build.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).parents[2]
DOCKERFILE = (REPO / "Dockerfile").read_text(encoding="utf-8")
LOCK = (REPO / "poetry.lock").read_text(encoding="utf-8")

# What src/lib/embeddings/local_onnx.py imports, plus what scripts/quantize_embedding.py's
# build stage imports from the same set (it adds `onnx` by pip, which is not in the lock).
RUNTIME_NEEDS = ("onnxruntime", "tokenizers", "huggingface-hub", "numpy")
HEAVY = ("torch", "sentence_transformers", "transformers")


def _poetry_installs() -> list[list[str]]:
    """The group list of every `poetry install ... --only a,b,c` in the Dockerfile.

    Matched anywhere on a line, not only after RUN: the base `--only main` install sits inside
    a multi-line `RUN pip install ... && poetry install ...`, and an install of `ml` hidden the
    same way must not escape. Comment lines are dropped so prose cannot match or hide one.
    """
    code = "\n".join(
        line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#")
    )
    return [
        match.split(",")
        for match in re.findall(r"poetry install\b[^\n]*?--only\s+([\w,]+)", code)
    ]


def _lock_groups(package: str) -> set[str]:
    block = re.search(
        rf'(?ms)^\[\[package\]\]\nname = "{re.escape(package)}"\n(.*?)(?=^\[\[package\]\])', LOCK
    )
    assert block, f"{package} is not in poetry.lock"
    groups = re.search(r"(?m)^groups = \[(.*?)\]", block.group(1))
    assert groups, f"{package} has no groups line in poetry.lock"
    return set(re.findall(r'"([^"]+)"', groups.group(1)))


def test_the_dockerfile_installs_poetry_groups_at_all():
    """Guards the parser: a regex that matches nothing would make every test below vacuous."""
    assert len(_poetry_installs()) >= 3, _poetry_installs()


def test_no_shipped_image_installs_the_ml_group():
    for groups in _poetry_installs():
        assert "ml" not in groups, (
            f"`poetry install --only {','.join(groups)}` brings torch, transformers and scipy "
            "back into the image. The runtime is ONNX; use the `onnx` group."
        )


def test_the_images_that_embed_install_the_onnx_group():
    embedding = [groups for groups in _poetry_installs() if "main" in groups and len(groups) > 1]
    assert embedding, "no stage installs anything beyond main"
    assert all("onnx" in groups for groups in embedding), embedding


def test_torch_arrives_only_through_the_ml_group():
    """If something in main or onnx started depending on torch, dropping `ml` would stop
    removing it and nothing here would notice."""
    assert _lock_groups("torch") == {"ml"}
    assert _lock_groups("sentence-transformers") == {"ml"}
    assert _lock_groups("transformers") == {"ml"}


def test_everything_the_onnx_runtime_needs_survives_without_ml():
    for package in RUNTIME_NEEDS:
        groups = _lock_groups(package)
        assert groups & {"main", "onnx"}, (
            f"{package} is only in {sorted(groups)}, so an image that installs main + onnx "
            "would not have it and the embedding model would fail to load"
        )


def test_nothing_imports_torch_at_import_time():
    """An image without `ml` must still start. The torch backend and the cross-encoder import
    it inside the function that needs it and raise a clear error otherwise; a module-level
    import anywhere in src/ would turn that into a crash on startup."""
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only: function-local imports are the lazy ones
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            if any(name in HEAVY for name in names):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, offenders
