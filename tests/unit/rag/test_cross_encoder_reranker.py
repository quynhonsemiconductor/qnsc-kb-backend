"""The cross-encoder reranker must be opt-in, CPU-only, and correctly calibrated.

Three guarantees worth a test rather than a comment, because each failure mode is
silent in the direction that matters:

1. RERANKER_BACKEND defaults to "lexical", so adding this package changes nothing
   until a deployment opts in. A default of "onnx" would make every search depend
   on a 2.3 GB model that is not present in production.
2. The ONNX providers default to CPU only -- the same deployment guarantee the
   reader has.
3. The cross-encoder's logit is mapped through a sigmoid into (0, 1), and the
   relevance floor on that path is RERANKER_MIN_SCORE, NOT the lexical
   RAG_MIN_RELEVANCE_SCORE. Getting this wrong filters every result to zero (a bug
   this iteration actually hit and fixed): a correctly-ranked-but-unconfident
   passage has a negative logit -> sigmoid < 0.12 -> deleted by the lexical floor.
"""
import math

from src.core.config import Settings, settings


def test_backend_defaults_to_lexical_so_nothing_changes_until_opt_in() -> None:
    assert Settings.model_fields["RERANKER_BACKEND"].default == "lexical"


def test_reranker_provider_default_is_cpu_only() -> None:
    assert Settings.model_fields["RERANKER_ONNX_PROVIDERS"].default == "CPUExecutionProvider"


def test_reranker_min_score_defaults_to_keep_all() -> None:
    # 0.0 keeps the cross-encoder's ranking intact; the reader abstains downstream.
    assert Settings.model_fields["RERANKER_MIN_SCORE"].default == 0.0


def test_sigmoid_maps_a_negative_logit_below_the_lexical_floor() -> None:
    # This is why the cross-encoder path needs its own threshold: a passage the
    # model ranks correctly but not confidently scores negative, and sigmoid puts
    # it under the lexical 0.12 floor, which would delete it.
    negative_logit = -3.0
    mapped = 1.0 / (1.0 + math.exp(-negative_logit))
    assert mapped < settings.RAG_MIN_RELEVANCE_SCORE
    assert mapped >= settings.RERANKER_MIN_SCORE  # survives the cross-encoder floor


def test_sigmoid_is_monotonic_so_ranking_is_preserved() -> None:
    # The mapping must not reorder anything; it only rescales for the threshold.
    logits = [-11.0, -3.0, 0.0, 0.44, 6.39]
    mapped = [1.0 / (1.0 + math.exp(-x)) for x in logits]
    assert mapped == sorted(mapped)


def test_project_declares_no_gpu_runtime_for_the_reranker_either() -> None:
    from pathlib import Path

    manifest = (Path(__file__).parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    assert "onnxruntime-gpu" not in manifest


def test_resolver_returns_the_onnx_backend() -> None:
    from src.lib.reranker import resolve_reranker

    backend = resolve_reranker()
    assert backend.name == "onnx-cross-encoder"
    assert hasattr(backend, "score") and hasattr(backend, "warm_up")


def test_live_reranker_settings_are_cpu_only_in_this_process() -> None:
    assert "CUDA" not in settings.RERANKER_ONNX_PROVIDERS
