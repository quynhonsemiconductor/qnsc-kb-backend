"""Run the deterministic F23 splitter against a local pilot corpus.

Usage:
    python scripts/run_splitter_pilot.py --path .. --output ../splitter-pilot-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.domain.document_splitter import splitter_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    roots: list[Path] = []
    for path in args.path:
        roots.extend([path] if path.is_file() else sorted(path.rglob("*.md")))
    roots = list(dict.fromkeys(roots))
    documents = [
        (path.name, path.read_text(encoding="utf-8", errors="replace"))
        for path in roots
    ]
    report = splitter_metrics(documents)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
