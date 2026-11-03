"""The api serves queries from an int8 export; the worker keeps writing fp32 vectors.

That split is only safe while it stays a split. An int8 query against stored fp32 chunks
moves each score by ~0.002 on average, but an int8 export that reached the worker would
write vectors into the same column as fp32 ones, and nothing would raise — retrieval would
just get quietly worse. So these tests pin the shape of the Dockerfile's stage graph, and
the two gates in scripts/quantize_embedding.py that stop a degraded export being shipped.

They read the Dockerfile as text rather than building it: CI builds the images with
BAKE_EMBEDDING_MODEL=false, so the quantisation step itself only ever runs where the
weights are baked, and a broken stage graph would otherwise be found by a failed deploy.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
DOCKERFILE = (REPO / "Dockerfile").read_text(encoding="utf-8")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "quantize_embedding", REPO / "scripts" / "quantize_embedding.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


quantize_embedding = _load_script()


# ── The Dockerfile's stage graph ────────────────────────────────────────────────


def _parse():
    """Global ARG defaults, and for each stage: its base and every COPY --from it makes."""
    globals_: dict[str, str] = {}
    stages: dict[str, dict] = {}
    order: list[str] = []
    current = None
    for raw in DOCKERFILE.splitlines():
        line = raw.strip()
        from_match = re.match(r"(?i)FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", line)
        if from_match:
            name = from_match.group(2) or from_match.group(1)
            stages[name] = {"base": from_match.group(1), "copies": []}
            order.append(name)
            current = name
            continue
        if current is None:
            arg = re.match(r"ARG\s+(\w+)=(\S+)\s*$", line)
            if arg:
                globals_[arg.group(1)] = arg.group(2)
            continue
        copy = re.match(r"COPY\s+(?:--\S+\s+)*--from=(\S+)", line)
        if copy:
            stages[current]["copies"].append(copy.group(1))
    return globals_, stages, order


def _expand(text: str, values: dict[str, str]) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: values[m.group(1)], text)


def _chain(target: str, **overrides: str) -> set[str]:
    """Every stage that ends up inside `target`, following FROM and COPY --from."""
    globals_, stages, _ = _parse()
    values = {**globals_, **overrides}
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen or name not in stages:
            return  # an external image such as python:3.13-slim
        seen.add(name)
        visit(_expand(stages[name]["base"], values))
        for source in stages[name]["copies"]:
            visit(_expand(source, values))

    visit(target)
    return seen


@pytest.mark.parametrize("api_embedding", ["int8", "fp32"])
def test_every_stage_reference_resolves_to_an_earlier_stage(api_embedding):
    """A FROM or COPY --from naming a stage that does not exist, or one defined further
    down, is a build error that CI's BAKE_EMBEDDING_MODEL=false build would still hit —
    but only for the stages it builds, and this covers both values of the switch."""
    globals_, stages, order = _parse()
    values = {**globals_, "API_EMBEDDING": api_embedding}
    for position, name in enumerate(order):
        for reference in [stages[name]["base"], *stages[name]["copies"]]:
            reference = _expand(reference, values)
            if reference in stages:
                assert order.index(reference) < position, f"{name} uses {reference} before it is defined"
            else:
                assert ":" in reference or reference == "scratch", (
                    f"{name} references {reference!r}, which is neither a stage nor an image"
                )


def test_the_api_serves_int8_by_default():
    api = _chain("api")
    assert "embedding-int8" in api
    assert "model" not in api and "embedding-fp32" not in api, "the api would carry the fp32 export as well, and gain nothing"


def test_the_api_can_fall_back_to_fp32():
    """Rollback is a rebuild with this arg; it must keep working."""
    api = _chain("api", API_EMBEDDING="fp32")
    assert "embedding-fp32" in api and "model" in api
    assert "embedding-int8" not in api


@pytest.mark.parametrize("api_embedding", ["int8", "fp32"])
def test_the_worker_always_writes_fp32_vectors(api_embedding):
    """The switch is named API_EMBEDDING because it must never reach the worker."""
    worker = _chain("worker", API_EMBEDDING=api_embedding)
    assert "model" in worker
    assert not {"embedding-int8", "api-embedding", "embedding-fp32"} & worker


def test_the_quantise_stage_needs_no_torch():
    """Quantising needs onnxruntime, tokenizers, numpy and huggingface_hub: the onnx group, not the ml group."""
    assert "deps-onnx" in _chain("embedding-int8")
    assert not {"deps-ml", "deps-ml-ocr"} & _chain("embedding-int8")


def test_the_migrator_carries_no_embedding_weights():
    migrator = _chain("migrator")
    assert not {"embedding-int8", "embedding-fp32", "api-embedding", "model", "hub"} & migrator


def test_the_embedding_model_default_is_declared_once_before_the_first_from():
    """Two defaults are two places to forget when the model changes. The stage-level
    redeclarations carry none and inherit this one."""
    defaults = re.findall(r"(?m)^ARG\s+EMBEDDING_MODEL=\S+\s*$", DOCKERFILE)
    assert len(defaults) == 1, defaults
    assert DOCKERFILE.index(defaults[0]) < DOCKERFILE.index("\nFROM ")


def test_the_quantise_stage_downloads_nothing_when_baking_is_off():
    """CI builds with BAKE_EMBEDDING_MODEL=false to avoid pushing 2.3 GB through the cache
    twice; the new stage must honour that or every CI run starts downloading the model."""
    stage = DOCKERFILE.split("AS embedding-int8", 1)[1].split("\nFROM ", 1)[0]
    assert 'if [ "$BAKE_EMBEDDING_MODEL" = "true" ]' in stage
    # The command is wrapped across lines, so find the flag rather than the whole command.
    assert stage.index("--download") > stage.index('if [ "$BAKE_EMBEDDING_MODEL"')


# ── The gates that stop a degraded export being shipped ──────────────────────


def test_an_output_directory_that_has_been_used_is_refused(tmp_path):
    """onnxruntime APPENDS to an existing external-data file, so re-running into a used
    directory silently doubles it (measured: 562 MB became 1,121 MB)."""
    (tmp_path / "model.onnx.data").write_bytes(b"x")
    with pytest.raises(quantize_embedding.QuantisationError, match="not empty"):
        quantize_embedding.require_empty(tmp_path)


def test_a_missing_or_empty_output_directory_is_fine(tmp_path):
    quantize_embedding.require_empty(tmp_path / "does-not-exist")
    quantize_embedding.require_empty(tmp_path)


def test_the_size_gate_passes_the_measured_export_and_catches_the_fp32_embedding_table():
    fp32 = 2_236_000_000
    assert quantize_embedding.check_size(fp32, 562_000_000) < quantize_embedding.MAX_SIZE_RATIO
    # MatMul-only quantisation leaves the ~1 GB word-embedding table in fp32: 1,331 MB.
    with pytest.raises(quantize_embedding.QuantisationError, match="embedding table"):
        quantize_embedding.check_size(fp32, 1_331_000_000)


def test_the_parity_gate_separates_the_recipes_it_was_measured_on():
    """Per-channel + reduce_range measured 0.9914 at worst; per-tensor + reduce_range
    0.977. The limit has to sit between them or the gate cannot tell a good recipe from a
    worse one."""
    limit = quantize_embedding.MIN_COSINE
    assert 0.977 < limit < 0.9914
    assert quantize_embedding.check_parity([0.9955, 0.9936, 0.9914]) == 0.9914
    with pytest.raises(quantize_embedding.QuantisationError, match="degrades retrieval"):
        quantize_embedding.check_parity([0.9955, 0.977])


def test_a_used_output_directory_fails_before_anything_is_downloaded(tmp_path, monkeypatch, capsys):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "model.onnx").write_bytes(b"x")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("downloaded 2.2 GB before noticing the output directory was used")

    monkeypatch.setattr(quantize_embedding, "download_fp32", must_not_run)

    code = quantize_embedding.main(
        ["--download", "some/model", "--fp32-dir", str(tmp_path / "fp32"), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 1
    assert "not empty" in capsys.readouterr().err
