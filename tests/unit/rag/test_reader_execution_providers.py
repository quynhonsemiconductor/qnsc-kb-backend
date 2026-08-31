"""The reader must be CPU-only unless someone deliberately says otherwise.

Production runs without a GPU. That guarantee is worth a test rather than a
comment, because the failure mode is silent in the direction that matters: a
default that quietly preferred CUDA would work on a developer's machine and only
surface in an environment that has no GPU to fall back from.

The provider-resolution logic is tested directly rather than through a real
session: it decides what gets passed to InferenceSession, and a real session
would additionally require the weights.
"""
from src.core.config import Settings, settings


def _resolve(requested: str, available: set[str]) -> list[str]:
    """The provider filter from src/lib/reader/local_onnx.py._load.

    Duplicated deliberately: the original runs inside a function that also builds
    a session and loads a tokenizer, and the decision under test is this list.
    """
    requested_names = [name.strip() for name in (requested or "").split(",") if name.strip()]
    providers = [name for name in requested_names if name in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def test_the_default_is_cpu_only() -> None:
    """The deployment guarantee. If this fails, production may try to use a GPU."""
    assert Settings.model_fields["READER_ONNX_PROVIDERS"].default == "CPUExecutionProvider"

def test_the_project_declares_no_gpu_runtime() -> None:
    """The guarantee lives in pyproject, not in whatever is installed locally.

    `onnxruntime-gpu` must never become a project dependency: the images are built
    from these declarations, so a CPU-only image is CPU-only by construction. A
    developer may install the GPU wheel into their own virtualenv to make a
    40-minute benchmark finish in minutes -- that is why this asserts on the
    manifest rather than on `onnxruntime.get_available_providers()`, which would
    fail on exactly the machine doing the measuring.
    """
    from pathlib import Path

    manifest = (Path(__file__).parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    assert "onnxruntime-gpu" not in manifest
    assert "onnxruntime" in manifest


def test_cpu_is_appended_when_only_cuda_is_requested() -> None:
    # A GPU that disappears mid-flight must degrade to slow, not to broken.
    providers = _resolve("CUDAExecutionProvider", {"CUDAExecutionProvider", "CPUExecutionProvider"})
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_unavailable_providers_are_dropped_rather_than_passed_through() -> None:
    # InferenceSession raises on an unknown provider name, so filtering is what
    # keeps a CUDA-naming env var from taking the reader down on a CPU-only wheel.
    providers = _resolve(
        "CUDAExecutionProvider,CPUExecutionProvider", {"CPUExecutionProvider"}
    )
    assert providers == ["CPUExecutionProvider"]


def test_order_is_preserved_because_it_is_a_priority_list() -> None:
    available = {"CUDAExecutionProvider", "CPUExecutionProvider"}
    assert _resolve("CPUExecutionProvider,CUDAExecutionProvider", available)[0] == (
        "CPUExecutionProvider"
    )


def test_empty_configuration_still_yields_cpu() -> None:
    assert _resolve("", {"CPUExecutionProvider"}) == ["CPUExecutionProvider"]


def test_live_settings_are_cpu_only_in_this_process() -> None:
    # Guards against a stray .env or exported variable during a test run.
    assert "CUDA" not in settings.READER_ONNX_PROVIDERS
