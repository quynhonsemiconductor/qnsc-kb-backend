"""The configured embedding model must be the one the image actually carries.

RAG had never once worked in the deployed environment, and nothing failed loudly enough
to say so. the AWS-era `infra/live/*/main.tf` set `embedding_model = "BAAI/bge-m3"`, from which
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

These tests pin the code default to the Dockerfile's baked model (the deployment values
now live in the gitops repository, outside this one). They are deliberately string
comparisons against the source files rather than against a running container.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.core.config import Settings

REPO = Path(__file__).parents[2]


def _dockerfile_baked_model() -> str:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"(?m)^ARG\s+EMBEDDING_MODEL=(\S+)\s*$", text)
    assert match, "Dockerfile no longer declares ARG EMBEDDING_MODEL"
    return match.group(1)


def test_the_code_default_also_matches_the_image():
    """A deployment that sets nothing must still agree with the baked export."""
    assert Settings.model_fields["EMBEDDING_MODEL"].default == _dockerfile_baked_model()


def test_the_derived_dimension_is_1024():
    """EMBEDDING_DIMENSION is derived, never configured, and is what base.py enforces.

    1024, not 384: multilingual-e5-large-instruct (XLM-R-large-based) is a wider model
    than the e5-small/MiniLM generation this repo shipped before it -- the pgvector
    column and HNSW index this pins to are correspondingly wider (see the
    ALTER-COLUMN-style migration, not the same-width delete+re-embed one).
    """
    settings = Settings(EMBEDDING_MODEL=_dockerfile_baked_model())
    assert settings.EMBEDDING_DIMENSION == 1024
