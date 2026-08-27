"""The configured embedding model must be the one the image actually carries.

RAG had never once worked in the deployed environment, and nothing failed loudly enough
to say so. `infra/live/*/main.tf` set `embedding_model = "BAAI/bge-m3"`, from which
`config.py` derives `EMBEDDING_DIMENSION = 1024`. But with `embedding_runtime = "onnx"`
the loader reads whatever export sits in `EMBEDDING_ONNX_DIR` and never looks at
`EMBEDDING_MODEL` at all — and what the Dockerfile bakes there is
paraphrase-multilingual-MiniLM-L12-v2, which emits 384.

So every single embed raised in `src/lib/embeddings/base.py`:

    embedding backend returned 384 dimensions, but EMBEDDING_DIMENSION is 1024
    and the pgvector column is fixed at that width

Search fell back to keyword-only and indexing failed, quietly, in both environments.

bge-m3 could not have been in the image even in principle: the Dockerfile takes it as
`ARG EMBEDDING_MODEL`, and the deploy pipeline has no way to pass one — qnsc-ci's
`build-push-ecr` action exposes no `build-args` input. The two halves of the
configuration had no mechanism that could ever have agreed.

These tests pin the three declarations to each other. They are deliberately string
comparisons against the source files rather than against a running container: the bug
was a mismatch BETWEEN files, so reading one of them twice would prove nothing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.core.config import Settings

REPO = Path(__file__).parents[2]
LIVE_ENVIRONMENTS = ("develop", "prod")


def _dockerfile_baked_model() -> str:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"(?m)^ARG\s+EMBEDDING_MODEL=(\S+)\s*$", text)
    assert match, "Dockerfile no longer declares ARG EMBEDDING_MODEL"
    return match.group(1)


def _infra_value(environment: str, key: str) -> str:
    text = (REPO / "infra" / "live" / environment / "main.tf").read_text(encoding="utf-8")
    match = re.search(rf'(?m)^\s*{key}\s*=\s*"([^"]+)"\s*$', text)
    assert match, f"{environment}/main.tf no longer sets {key}"
    return match.group(1)


@pytest.mark.parametrize("environment", LIVE_ENVIRONMENTS)
def test_infra_names_the_model_the_image_bakes(environment):
    """The mismatch that made every embed raise."""
    assert _infra_value(environment, "embedding_model") == _dockerfile_baked_model()


def test_the_code_default_also_matches_the_image():
    """A deployment that sets nothing must still agree with the baked export."""
    assert Settings.model_fields["EMBEDDING_MODEL"].default == _dockerfile_baked_model()


def test_both_environments_use_the_same_model():
    """Same width but different models is WORSE than a mismatch: the vectors are
    comparable in type and meaningless in substance, so nothing errors."""
    models = {_infra_value(env, "embedding_model") for env in LIVE_ENVIRONMENTS}
    assert len(models) == 1, models


def test_the_version_stamp_tracks_the_model():
    """EMBEDDING_VERSION is what hybrid_search filters on, so a stamp naming a model that
    did not produce the vectors makes the corpus invisible rather than merely mislabelled.
    """
    for environment in LIVE_ENVIRONMENTS:
        model = _infra_value(environment, "embedding_model").lower()
        version = _infra_value(environment, "embedding_version").lower()
        assert "minilm" in model, model
        assert "minilm" in version, f"{environment}: {version} does not name {model}"
        assert "bge" not in version, f"{environment}: stale stamp {version}"


@pytest.mark.parametrize("environment", LIVE_ENVIRONMENTS)
def test_the_derived_dimension_is_384(environment):
    """EMBEDDING_DIMENSION is derived, never configured, and is what base.py enforces."""
    settings = Settings(EMBEDDING_MODEL=_infra_value(environment, "embedding_model"))
    assert settings.EMBEDDING_DIMENSION == 384
