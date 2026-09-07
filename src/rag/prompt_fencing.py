"""Neutralise prompt-envelope delimiters in untrusted text.

Every LLM surface in this codebase fences untrusted input inside XML-ish tags and then
tells the model that anything within those tags is DATA, never instruction. That contract
holds only while the untrusted text cannot close the tag it sits inside. A passage
containing the literal `</untrusted-passage>` ends the envelope early, and everything
after it is read as prompt: the document can forge `<authorized-document>` blocks that
were never retrieved, and the answer then carries citations to sources that do not exist.

The fix is deliberately not sanitisation-by-blocklist. Instead of guessing which phrases
are hostile, the closing delimiter itself is made unrepresentable by inserting a
zero-width space after the `<` of any tag-like run. The model still reads the words --
nothing is deleted, so a document legitimately discussing `</untrusted-passage>` is
quoted faithfully -- but the tokeniser can no longer see a tag boundary there.

Applies to the tag SHAPE, not to a list of tag names: a future prompt that adds a new
fence gets the same protection without editing this module.
"""

import re

# Any `<...>` run that could be read as a tag: optional slash, a name, and anything up to
# the closing bracket. Deliberately broader than the fences in use — a forged OPENING tag
# such as `<authorized-document id="9">` is as dangerous as a forged closing one.
_TAG_LIKE = re.compile(r"<(/?[A-Za-z][A-Za-z0-9_:-]*)")

# Zero-width space. Invisible to a reader, and it breaks `</tag` into a non-tag for the
# tokeniser. U+200B is preserved verbatim through JSON transport, unlike a stripped
# control character.
_ZERO_WIDTH_SPACE = "\u200b"


def fence_untrusted(text: str | None) -> str:
    """Return `text` with tag-like delimiters made inert for prompt assembly.

    Safe on None and on text with no tags, so call sites need no branching. The result is
    only ever placed INSIDE a fence; it is not valid to use this as HTML escaping.
    """
    if not text:
        return ""
    return _TAG_LIKE.sub(lambda match: f"<{_ZERO_WIDTH_SPACE}{match.group(1)}", text)
