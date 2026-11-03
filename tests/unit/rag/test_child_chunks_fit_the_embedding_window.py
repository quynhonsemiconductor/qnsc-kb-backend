"""A retrieval chunk must fit the embedding model's context window.

`_pack` measures CHARACTERS. The embedding model measures TOKENS, and past its window it
TRUNCATES rather than failing — so an oversized chunk loses its tail with no error, no
log line, and no visible symptom beyond retrieval quietly getting worse.

Child chunks were 500 characters against MiniLM-L12-v2's 128-token window
(EMBEDDING_MAX_TOKENS). 500 characters is roughly 125 tokens of English and appreciably
more of Vietnamese, where diacritics cost extra tokens, so the tail of most chunks was
being embedded as nothing at all — in a corpus that is largely Vietnamese.

The ratio below is a deliberately CONSERVATIVE estimate, not a measurement: the point is
to keep a margin against the worst-case language in this corpus, and to fail loudly if
someone raises the chunk size without also moving to a model with a longer window.
"""
from __future__ import annotations

from src.core.config import settings
from src.rag.chunker import create_parent_child_chunks, sliding_chunks

#: Characters per token, worst case. Vietnamese under a multilingual (XLM-R style)
#: tokenizer runs denser than English; 2.5 leaves margin under either.
CONSERVATIVE_CHARS_PER_TOKEN = 2.5


def _child_size_limit() -> int:
    """The character size the chunker actually asks for, read from its own output."""
    text = "\n\n".join(f"Cau {n}. " + "tai lieu noi bo " * 40 for n in range(6))
    children = [child for parent in create_parent_child_chunks(text) for child in parent["children"]]
    assert children, "chunker produced no children"
    return max(len(child) for child in children)


def test_a_child_chunk_fits_the_model_window():
    """The regression: at 500 characters this exceeded 128 tokens and truncated."""
    worst_case_tokens = _child_size_limit() / CONSERVATIVE_CHARS_PER_TOKEN
    assert worst_case_tokens <= settings.EMBEDDING_MAX_TOKENS, (
        f"child chunks reach {_child_size_limit()} chars, about "
        f"{worst_case_tokens:.0f} tokens, over EMBEDDING_MAX_TOKENS="
        f"{settings.EMBEDDING_MAX_TOKENS} — the tail is silently truncated"
    )


def test_the_budget_is_size_plus_overlap():
    """The trap that made the first attempt at this fix insufficient.

    _pack flushes BEFORE the piece that would overflow, then seeds the next chunk with the
    last `overlap` characters of the one it just closed and appends that piece anyway. So
    the real bound is size + overlap + 2, not size — setting size alone still overflows.
    """
    text = "tai lieu noi bo cua cong ty " * 200
    chunks = sliding_chunks(text, size=300, overlap=100)
    assert chunks
    assert max(len(chunk) for chunk in chunks) > 300, (
        "if this ever holds, _pack now caps at `size` and the callers above may be "
        "needlessly conservative"
    )
    assert max(len(chunk) for chunk in chunks) <= 300 + 100 + 2


def test_children_still_cover_the_parent():
    """Shrinking the child size must not start dropping content."""
    text = "\n\n".join(f"Muc {n}. " + "noi dung quan trong " * 30 for n in range(4))
    for parent in create_parent_child_chunks(text):
        joined = " ".join(parent["children"])
        # Every non-trivial word of the parent must appear somewhere in its children.
        missing = [w for w in set(parent["parent_text"].split()) if len(w) > 3 and w not in joined]
        assert not missing, missing[:5]
