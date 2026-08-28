"""glm must be told explicitly not to think, or the answer budget is spent before the answer.

An assistant reply ended mid-word:

    Theo kiến thức chung trong ngành: Verilog là một ngôn ngữ mô t

`_payload` sends glm's `thinking` field ONLY when the caller passes something other than
None. The RAG answer path passed nothing, so glm-4.5 used its own default — reasoning
enabled — and those hidden tokens come out of the same `max_tokens` budget as the reply.

That budget is shared twice over: one generation produces the grounded answer AND the
EXTENDED section, split afterwards on a sentinel. Reasoning ate the front, the grounded
half took what was left, and the extended half — generated last — was cut off part-way
through a word, with nothing marking it incomplete.

`content_restructure` has always passed `thinking=False`. Only this path did not, and
only glm was affected: the Gemini branch always writes `thinkingConfig`.
"""
from __future__ import annotations



from src.domain.llm_client import Provider, _payload


def _glm() -> Provider:
    return Provider(
        name="glm",
        model="glm-4.5-flash",
        url="https://example.invalid/v1/chat/completions",
        api_key="k",
        native_gemini=False,
    )


def _messages():
    return [{"role": "user", "content": "hỏi gì đó"}]


def test_thinking_false_disables_it_for_glm():
    payload = _payload(_glm(), _messages(), 0.0, thinking=False)
    assert payload["thinking"] == {"type": "disabled"}


def test_thinking_true_enables_it_for_glm():
    payload = _payload(_glm(), _messages(), 0.0, thinking=True)
    assert payload["thinking"] == {"type": "enabled"}


def test_omitting_thinking_leaves_glm_on_its_own_default():
    """The gap that caused the truncation: no field sent means the provider decides, and
    glm-4.5 decides to reason — inside the answer's token budget."""
    payload = _payload(_glm(), _messages(), 0.0)
    assert "thinking" not in payload


def test_the_answer_path_disables_thinking():
    """The call site, not just the mechanism. Source-level because the generation sits
    inside a long streaming function with no seam worth carving one for."""
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "src" / "domain" / "ai_service.py"
    ).read_text(encoding="utf-8")
    call = source[source.index("max_tokens=settings.RAG_MAX_ANSWER_TOKENS") :][:1200]
    # The ARGUMENT, not the substring: the comment above it also says "thinking=False",
    # and matching that made this test pass with the argument deleted.
    passed_explicitly = any(
        line.strip() == "thinking=False," for line in call.splitlines()
    )
    assert passed_explicitly, (
        "the RAG answer generation must pass thinking explicitly; without it glm spends "
        "the answer budget on hidden reasoning"
    )


def test_the_answer_budget_covers_two_sections():
    """One generation emits the grounded answer and the extended section, and Vietnamese
    costs roughly twice the tokens per character that English does."""
    from src.core.config import settings

    assert settings.RAG_MAX_ANSWER_TOKENS >= 4096
