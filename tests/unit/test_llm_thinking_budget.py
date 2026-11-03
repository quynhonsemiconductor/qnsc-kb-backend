"""glm must be told to keep deliberation low, or the answer budget is spent before the answer.

NOTE: the original fix here sent `{"type": "disabled"}`, which newer GLM models reject
with HTTP 400 code 1210 — see `test_thinking_false_asks_for_the_least_deliberation_glm_allows`
below. The field is now always `level`. The budget reasoning that follows is unchanged;
only the wire shape is.


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


def test_thinking_false_asks_for_the_least_deliberation_glm_allows():
    """`level`, never `{"type": "disabled"}`.

    Newer GLM models cannot turn thinking off and reject that payload outright:

        HTTP 400 {"error":{"code":"1210","message":"This model always engages in
        thinking and cannot be disabled; please use low, high, or max"}}

    `thinking=False` is what every non-chat caller passes, so the rejection hit all of
    them. Document preparation was the visible casualty: every section failed, the
    lossless fallback always won, and an upload could never reach review AI-formatted.

    Verified against the live provider — `{"type": "disabled"}` and `{"type": "low"}`
    are both refused with 1210, while `level` is accepted by this model AND by the
    previous generation.
    """
    payload = _payload(_glm(), _messages(), 0.0, thinking=False)
    assert payload["thinking"] == {"level": "low"}


def test_thinking_true_enables_it_for_glm():
    payload = _payload(_glm(), _messages(), 0.0, thinking=True)
    assert payload["thinking"] == {"level": "high"}


def test_no_glm_thinking_payload_tries_to_disable_thinking():
    """The shape itself, pinned against reintroduction.

    A model that always thinks treats "off" as a bad request, so neither value may ever
    be `disabled` again — for either flag state.
    """
    for requested in (True, False):
        thinking = _payload(_glm(), _messages(), 0.0, thinking=requested)["thinking"]
        assert "type" not in thinking
        assert "disabled" not in thinking.values()


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
    """One generation emits the grounded answer AND the extended section, and Vietnamese
    costs roughly twice the tokens per character English does. 2048 cut replies in half.

    None means the cap is not sent at all and the model stops when it is finished, which
    is the intended default; any explicit value must still clear the two-section floor.
    """
    from src.core.config import settings

    cap = settings.RAG_MAX_ANSWER_TOKENS
    assert cap is None or cap >= 4096


def test_no_cap_sends_no_max_tokens_field():
    """Uncapped must mean the field is absent, not zero — a zero would truncate to
    nothing rather than let the model finish."""
    payload = _payload(_glm(), _messages(), 0.0, thinking=False, max_tokens=None)
    assert "max_tokens" not in payload
