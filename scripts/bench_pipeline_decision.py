"""End-to-end check: does a real document now avoid the LLM restructuring pass?

Runs the production extraction path over a file, then asks the production skip
predicate what would happen next. Prints the decision and what it saves.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.content_restructure import (  # noqa: E402
    _chunk_heading_count,
    is_already_structured,
    split_into_chunks,
)
from src.domain.source_extraction import extract_source_markdown  # noqa: E402


def main() -> None:
    for argument in sys.argv[1:]:
        path = Path(argument)
        data = path.read_bytes()
        start = time.perf_counter()
        markdown = extract_source_markdown(path.name, data)
        elapsed = time.perf_counter() - start

        headings = _chunk_heading_count(markdown)
        skipped = is_already_structured(markdown)
        chunks = len(split_into_chunks(markdown))
        # The pass sends the document and regenerates it, so ~2x its tokens.
        approximate_tokens = len(markdown) // 4

        print(f"\n=== {path.name} ({len(data):,} bytes) ===")
        print(f"  extraction            {elapsed:.2f}s -> {len(markdown):,} chars")
        print(f"  chunk headings        {headings}")
        print(f"  chunks                {chunks}")
        print(f"  LLM pass skipped      {skipped}")
        if skipped:
            print(
                f"  avoided              ~{approximate_tokens * 2:,} tokens "
                f"(~{approximate_tokens:,} in + ~{approximate_tokens:,} out)"
            )
        else:
            print("  -> still sent to the model (no usable structure found)")


if __name__ == "__main__":
    main()
