"""What the shipped images contain, pinned as string assertions against the Dockerfile.

ECR bills per GB stored, per image, and KB's was the fastest-growing cost line. The images were ~6.6 GB (api) and
~9.3 GB (worker) uncompressed because of four things, each pinned here so it cannot come back unnoticed:

  * the `ml` group (torch, sentence-transformers, transformers, scipy, scikit-learn) was installed although
    EMBEDDING_RUNTIME defaults to onnx and docs/embedding-runtime-onnx.md says the group left the images;
  * the main libraries were copied into a base stage and again, with the rest, into the next one, so a layer held
    them twice (a layer cannot delete what a lower layer holds);
  * poetry and its own dependencies were pip-installed into the directory the runtime copied wholesale;
  * the embedding export was downloaded into the Hub cache and then copied again: a 4.5 GB layer for 2.2 GB of weights.

String assertions, not a build, for the reason test_embedding_config_matches_image.py gives: the invariant is a
property of how the file is written, and a build would take twenty minutes to say less. They pin WHAT ships (which
dependency groups, how many copies, nothing build-only), not how a step is spelled: the spelling of the download, the
cache directory and the poetry venv is free to change, and the stage graph of the embedding exports is
test_embedding_quantization.py's.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).parents[2]
TEXT = (REPO / "Dockerfile").read_text(encoding="utf-8")


def stage(name: str) -> str:
    match = re.search(rf"(?ms)^FROM\s+\S+\s+AS\s+{re.escape(name)}\s*$(.*?)(?=^FROM\s|\Z)", TEXT)
    assert match, f"Dockerfile has no `{name}` stage"
    return match.group(1)


def code(body: str) -> str:
    """The stage with its comments removed, so a sentence about `ml` is not mistaken for installing it."""
    return "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))


def install_groups(name: str) -> list[set[str]]:
    """The dependency groups each `poetry install --only ...` in a stage selects, as sets (order and spelling free)."""
    return [set(groups.split(",")) for groups in re.findall(r"poetry install\b[^\n]*--only\s+(\S+)", code(stage(name)))]


def test_no_shipped_stage_installs_the_ml_group():
    for name in ("deps", "deps-onnx", "deps-onnx-ocr"):
        for groups in install_groups(name):
            assert "ml" not in groups, f"{name} installs the ml group (torch): {groups}"
    assert not re.search(r"--(with|only)[ =]\S*\bml\b", code(TEXT)), "the ml group is installed somewhere"


def test_each_target_installs_exactly_the_groups_it_runs():
    assert install_groups("deps") == [{"main"}]  # migrator
    assert install_groups("deps-onnx") == [{"main", "onnx"}]  # api: embeds the search query
    assert install_groups("deps-onnx-ocr") == [{"main", "onnx", "ocr"}]  # worker: the only reader of scanned files


def test_each_target_copies_exactly_one_virtualenv_and_no_system_libraries():
    expected = {"api": "deps-onnx", "worker": "deps-onnx-ocr", "migrator": "deps"}
    for target, source in expected.items():
        venvs = re.findall(r"COPY --from=(\S+) /opt/venv /opt/venv", code(stage(target)))
        assert venvs == [source], f"{target} copies {venvs}, wanted exactly [{source}]"
    # Substring-matching "/usr/local/lib" is too wide: runtime-base now DELETES the
    # system pip tree (its vendored certifi trips the CRITICAL gate), so a removal
    # line mentions the path. The invariant is that nothing system-side is SHIPPED:
    # no COPY may pull from /usr/local.
    assert not re.search(r"COPY --from=\S+ /usr/local", code(TEXT)), (
        "libraries are shipped as one venv, not by copying /usr/local"
    )


def test_the_base_stage_carries_no_libraries_to_be_stored_twice():
    base = code(stage("runtime-base"))
    assert "COPY --from" not in base and "poetry" not in base


def test_nothing_that_builds_is_in_a_shipped_stage():
    for name in ("runtime-base", "api", "worker", "migrator"):
        body = code(stage(name))
        assert "build-essential" not in body and "libpq-dev" not in body and "pip install" not in body, name
        assert "poetry" not in body, name
    assert "build-essential" in code(stage("builder"))


def test_only_the_api_and_worker_carry_an_embedding_export():
    for target in ("api", "worker"):
        assert re.search(r"COPY --from=(api-embedding|model|embedding-int8) /opt/embedding-export /opt/embedding-onnx", code(stage(target))), target
    assert "embedding" not in code(stage("migrator")).replace("migrate-entrypoint", "")


def test_the_exports_are_readable_by_any_uid():
    """onnxruntime writes the quantised weights 0600 and root-owned; the api runs as appuser. Unreadable weights fail
    deep inside InferenceSession and the api silently serves keyword-only search (found building #223 on top of #221:
    main avoided it with `COPY --chown`; the exports are now chmod'ed in their own stages instead)."""
    for name in ("embedding-int8", "model"):
        assert re.search(r"(?m)^RUN chmod -R a\+rX ", code(stage(name))), name


def test_application_code_is_still_the_last_layer_of_every_target():
    for target in ("api", "worker", "migrator"):
        copies = re.findall(r"(?m)^(COPY\b[^\n]*)", code(stage(target)))
        assert copies[-1].endswith(". ."), f"{target}: code must be copied last, or every commit re-stores the layers above it"


def test_the_interpreter_image_and_the_model_revision_are_pinned():
    """A rebuild must not change the interpreter, and the Hub's owner must not be able to change the weights under a
    corpus that is already embedded with the old ones (nothing would raise: the vectors would just stop matching)."""
    assert re.search(r"(?m)^FROM python:3\.13-slim@sha256:[0-9a-f]{64} AS python-base$", TEXT)
    assert len(re.findall(r"(?m)^FROM python:", TEXT)) == 1, "name the interpreter image in ONE place"
    assert re.search(r"(?m)^ARG EMBEDDING_REVISION=[0-9a-f]{40}$", TEXT)


def test_dockerignore_keeps_ci_and_infra_out_of_the_context():
    ignore = (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for entry in (".github/", "infra/", "tests/", "docs/"):
        assert entry in ignore, entry
    entries = [l for l in ignore if l.strip() and not l.startswith("#")]
    assert len(entries) == len(set(entries)), "duplicate .dockerignore entries"
