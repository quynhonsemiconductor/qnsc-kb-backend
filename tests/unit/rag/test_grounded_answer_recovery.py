"""Three ways a correct, sourced answer was thrown away before reaching the reader.

All three ended at the same screen — "Không thể tạo câu trả lời có căn cứ từ các nguồn
được cấp quyền trong Cơ sở tri thức" — which reads as "the knowledge base has nothing",
while the knowledge base had the answer and the model had written it.
"""
from __future__ import annotations

from src.domain.ai_service import _is_grounding_refusal
from src.domain.llm_client import _extract_openai_text
from src.rag.answer_sections import strip_citation_markers
from src.rag.citations import extract_citation_ids


def test_an_answer_left_in_the_reasoning_channel_is_still_the_answer():
    """Observed on Groq openai/gpt-oss-120b: finish_reason "stop", 659 reasoning
    tokens, a complete answer inside message.reasoning, and message.content empty.
    Reading only content produced a blank answer that the citation guard then reported
    as a grounding failure."""
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": (
                        "We need to produce answer in Vietnamese. Look for the "
                        "definition in C4. Let's craft.\n\n"
                        "<<<GROUNDED>>>\nHDL là ngôn ngữ mô tả phần cứng. [C4]\n"
                    ),
                },
            }
        ]
    }

    recovered = _extract_openai_text(response)

    assert recovered.startswith("<<<GROUNDED>>>"), (
        "deliberation before the sentinel must never reach the reader"
    )
    assert "[C4]" in recovered
    assert "We need to produce" not in recovered


def test_reasoning_without_an_answer_stays_empty():
    """No sentinel means there is no answer in there — only thinking.

    Returning it anyway would show the user the model's private deliberation.
    """
    response = {
        "choices": [
            {"message": {"content": "", "reasoning": "Hmm, the context is unclear."}}
        ]
    }

    assert _extract_openai_text(response) == ""


def test_content_wins_when_the_provider_sends_both():
    response = {
        "choices": [
            {"message": {"content": "<<<GROUNDED>>>\nreal [C1]", "reasoning": "notes"}}
        ]
    }

    assert _extract_openai_text(response) == "<<<GROUNDED>>>\nreal [C1]"


def test_a_citation_carrying_its_review_date_is_still_a_citation():
    """The prompt tells the model to surface review dates and owners, so it writes them
    into what it considers one citation. Requiring the bracket to close right after the
    id read this as NO citation, and an uncited grounded answer is refused wholesale."""
    answer = (
        "HDL là ngôn ngữ mô tả phần cứng.\n"
        "[C4: 2026-08-08T16:34:46.563958, admin.manager@qnsc.vn]"
    )

    assert extract_citation_ids(answer) == ["C4"]


def test_several_ids_in_one_bracket_are_all_extracted():
    assert extract_citation_ids("Quy định này áp dụng chung [C1, C2].") == ["C1", "C2"]


def test_a_year_in_brackets_is_not_a_citation():
    """Fails closed: promoting `[2024]` to a citation would attach a source that never
    supported the claim."""
    assert extract_citation_ids("Ban hành năm [2024] theo quyết định.") == []


def test_an_enriched_marker_is_stripped_from_the_extended_section():
    """Nothing in the extended section is attributable to the knowledge base."""
    cleaned = strip_citation_markers("Thông thường [C4: 2026-08-08, a@b.c] là vậy.")

    assert "C4" not in cleaned
    assert "a@b.c" not in cleaned


def test_the_refusal_wording_the_model_actually_uses_is_recognised():
    """The prompt asks for "the language-specific equivalent" and leaves the wording to
    the model, so a fixed-string check was never going to hold. An unrecognised refusal
    skips the recovery path that shows the retrieved passage."""
    assert _is_grounding_refusal("Không tìm thấy trong Cơ sở Kiến thức.")
    assert _is_grounding_refusal("Không tìm thấy thông tin trong Cơ sở tri thức.")
    assert _is_grounding_refusal("Not found in the Knowledge Base.")


def test_a_real_answer_that_mentions_the_phrase_is_not_a_refusal():
    """Otherwise a long, correct answer gets replaced by a retrieved snippet."""
    answer = (
        "Quy trình lưu trữ yêu cầu mọi tài liệu phải được duyệt trước khi công bố. "
        "Nếu tài liệu không tìm thấy trong Cơ sở tri thức, người dùng gửi yêu cầu bổ "
        "sung nội dung tới bộ phận quản trị, kèm mã tài liệu và lý do sử dụng. Bộ phận "
        "quản trị phản hồi trong vòng ba ngày làm việc và cập nhật trạng thái yêu cầu "
        "trên hệ thống để người gửi theo dõi tiến độ xử lý. [C1]"
    )

    assert not _is_grounding_refusal(answer)


def test_a_quota_refusal_carries_how_long_to_wait():
    """Groq's free tier allows 8,000 tokens per minute and one grounded answer costs
    about 5,000, so the second question inside a minute is refused. It is a wait, not a
    failure, and the wait is only in the body — there is no Retry-After header."""
    import httpx

    from src.domain.llm_client import _retry_after_seconds

    response = httpx.Response(
        429,
        json={
            "error": {
                "message": (
                    "Rate limit reached for model `openai/gpt-oss-120b` ... "
                    "Please try again in 27.1725s."
                ),
                "code": "rate_limit_exceeded",
            }
        },
    )

    assert _retry_after_seconds(response) == 27


def test_the_retry_after_header_wins_when_the_provider_sends_one():
    import httpx

    from src.domain.llm_client import _retry_after_seconds

    assert _retry_after_seconds(httpx.Response(429, headers={"retry-after": "12"})) == 12
    assert _retry_after_seconds(httpx.Response(429, text="busy")) is None


def test_glm_style_reasoning_content_is_read_too():
    """GLM and DeepSeek name the field `reasoning_content`, Groq names it `reasoning`.

    glm-4.5-flash was observed returning `content: ""` with the text in
    `reasoning_content` on the very first token.
    """
    response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "Thinking…\n<<<GROUNDED>>>\nCâu trả lời [C2]",
                }
            }
        ]
    }

    assert _extract_openai_text(response) == "<<<GROUNDED>>>\nCâu trả lời [C2]"


def test_a_pasted_key_keeps_working_when_the_copy_carried_a_space():
    """A Zhipu key is {32 hex}.{16 alnum}. Saved as "{32}. {16}" — a copy that wrapped —
    the provider answers `{"code":"1000","message":"Authentication Failed"}`, which
    reads as an invalid key rather than as a space inside a valid one."""
    from src.domain.llm_config import normalize_api_key

    assert normalize_api_key("abc123def456. XyZ9") == "abc123def456.XyZ9"
    assert normalize_api_key("  sk-live-42\n") == "sk-live-42"
    assert normalize_api_key(None) is None
