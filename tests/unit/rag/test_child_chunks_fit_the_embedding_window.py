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

TWO BOUNDS, NOT ONE. The upper bound catches silent truncation. The LOWER bound catches
the opposite failure, which is just as silent: after the move to e5-large-instruct the
window went from 128 to 512 tokens and the chunker kept asking for 250 characters, so
three quarters of a window we pay for on every single embed went unused, and children
were shorter than they needed to be to carry a whole thought. Nothing failed, nothing
logged, and retrieval was quietly worse than the model allowed. A window change should
break a test, in whichever direction it moves.

The upper bound is measured against what the EMBEDDER actually receives, which is not
the child: indexing.py prepends a per-section context header, up to MAX_HEADER_CHARS,
before embedding. A test that checked the bare child would pass while the real input
overflowed.
"""
from __future__ import annotations

from src.core.config import settings
from src.rag.chunker import create_parent_child_chunks, sliding_chunks
from src.rag.contextual_header import MAX_HEADER_CHARS, apply_header

#: Characters per token, worst case. Vietnamese under a multilingual (XLM-R style)
#: tokenizer runs denser than English; 2.5 leaves margin under either.
CONSERVATIVE_CHARS_PER_TOKEN = 2.5

#: How much of the window the child itself is allowed to leave unused before this is a
#: bug rather than a margin. 0.6 passes at the current 802-of-1204 characters and fails
#: if someone widens the window again without revisiting the chunker.
MIN_WINDOW_UTILISATION = 0.6


def _child_size_limit() -> int:
    """The character size the chunker actually asks for, read from its own output."""
    text = "\n\n".join(f"Cau {n}. " + "tai lieu noi bo " * 40 for n in range(6))
    children = [child for parent in create_parent_child_chunks(text) for child in parent["children"]]
    assert children, "chunker produced no children"
    return max(len(child) for child in children)


def _embedder_input_limit() -> int:
    """What the EMBEDDER receives at worst: a full-length header plus the largest child.

    indexing.py embeds `apply_header(section_header, child)`, never the bare child, and
    generate_section_context truncates its result to MAX_HEADER_CHARS. So the worst case
    is a header at exactly that cap joined to the largest child the chunker emits.
    """
    return len(apply_header("h" * MAX_HEADER_CHARS, "c" * _child_size_limit()))


def test_the_embedder_input_fits_the_model_window():
    """The regression: at 500 characters this exceeded 128 tokens and truncated.

    Measured on header + child, because that is what gets tokenised.
    """
    worst_case_tokens = _embedder_input_limit() / CONSERVATIVE_CHARS_PER_TOKEN
    assert worst_case_tokens <= settings.EMBEDDING_MAX_TOKENS, (
        f"header + child reaches {_embedder_input_limit()} chars, about "
        f"{worst_case_tokens:.0f} tokens, over EMBEDDING_MAX_TOKENS="
        f"{settings.EMBEDDING_MAX_TOKENS} — the tail is silently truncated. Either lower "
        f"the child size in chunker.py or lower MAX_HEADER_CHARS ({MAX_HEADER_CHARS})"
    )


def test_children_actually_use_the_window_they_are_given():
    """The other half of the bound: a child must not waste the window we pay for.

    This is the failure that followed the e5-large-instruct move — the window tripled to
    512 tokens and the chunker still asked for 250 characters, so most of every embed was
    padding. Silent, like truncation, but in the opposite direction.
    """
    budget_chars = settings.EMBEDDING_MAX_TOKENS * CONSERVATIVE_CHARS_PER_TOKEN
    used = _embedder_input_limit() / budget_chars
    assert used >= MIN_WINDOW_UTILISATION, (
        f"header + child reaches only {_embedder_input_limit()} of about "
        f"{budget_chars:.0f} usable characters ({used:.0%} of the window). "
        f"EMBEDDING_MAX_TOKENS is {settings.EMBEDDING_MAX_TOKENS}; raise the child size "
        f"in chunker.py to match the model actually in use"
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
