"""Download and normalize MLQA and UIT-ViQuAD 2.0 into a common eval format.

Both benchmarks are SQuAD-shaped, but they ship differently: MLQA as a zip of
per-language-pair JSON files, ViQuAD 2.0 as parquet on the Hub. This script lands
them in one JSONL schema so the harness never has to care which is which.

Output schema, one object per question:

    {"qid", "dataset", "split", "lang_q", "lang_c", "question",
     "answers": [str, ...], "is_impossible": bool,
     "context": str, "title": str, "doc_id": str}

`doc_id` is a stable hash of the context, so several questions that share a
paragraph share one document -- that is what makes retrieval a real task rather
than one-document lookup.

MLQA note: the corpus ships `test` and `dev` per language PAIR. `dev` is the
tuning split, `test` is the gate. ViQuAD 2.0 ships train/validation/test, but its
test split has NO public answers (all answer lists are empty) -- so the gate for
Vietnamese reads on `validation`, and `train` is available for tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request
import zipfile

CACHE = pathlib.Path(__file__).resolve().parents[1] / ".eval-cache"
DATA = CACHE / "datasets"
MLQA_ZIP_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
HF = "https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0/resolve/main/data"
VIQUAD_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
# MLQA pairs worth measuring: monolingual en, monolingual vi, and both
# cross-lingual directions between them.
MLQA_PAIRS = [("en", "en"), ("vi", "vi"), ("en", "vi"), ("vi", "en")]


def _fetch(url: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=900) as response, open(part, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    part.replace(dest)
    return dest


def _doc_id(context: str) -> str:
    return hashlib.sha1(context.strip().encode("utf-8")).hexdigest()[:16]


def _squad_records(payload: dict, dataset: str, split: str, lang_c: str, lang_q: str) -> list[dict]:
    records: list[dict] = []
    for article in payload.get("data", []):
        title = article.get("title") or ""
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context") or ""
            if not context.strip():
                continue
            document = _doc_id(context)
            for qa in paragraph.get("qas", []):
                answers = [a["text"] for a in (qa.get("answers") or []) if a.get("text")]
                plausible = [
                    a["text"] for a in (qa.get("plausible_answers") or []) if a.get("text")
                ]
                impossible = bool(qa.get("is_impossible", False))
                records.append(
                    {
                        "qid": str(qa.get("id")),
                        "dataset": dataset,
                        "split": split,
                        "lang_q": lang_q,
                        "lang_c": lang_c,
                        "question": (qa.get("question") or "").strip(),
                        "answers": sorted(set(answers)),
                        "plausible_answers": sorted(set(plausible)),
                        "is_impossible": impossible,
                        "context": context,
                        "title": title,
                        "doc_id": document,
                    }
                )
    return records


def prepare_mlqa() -> list[dict]:
    archive = _fetch(MLQA_ZIP_URL, DATA / "MLQA_V1.zip")
    target = DATA / "mlqa"
    if not target.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    records: list[dict] = []
    for split in ("dev", "test"):
        for lang_c, lang_q in MLQA_PAIRS:
            name = f"{split}-context-{lang_c}-question-{lang_q}.json"
            matches = list(target.rglob(name))
            if not matches:
                print(f"  MLQA missing: {name}", file=sys.stderr)
                continue
            payload = json.loads(matches[0].read_text(encoding="utf-8"))
            found = _squad_records(payload, "mlqa", split, lang_c, lang_q)
            records.extend(found)
            print(f"  mlqa {split} c={lang_c} q={lang_q}: {len(found)} questions")
    return records


def prepare_viquad() -> list[dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        print("pyarrow is required for ViQuAD parquet files", file=sys.stderr)
        raise
    records: list[dict] = []
    for split, filename in VIQUAD_FILES.items():
        path = _fetch(f"{HF}/{filename}", DATA / f"viquad2_{split}.parquet")
        table = parquet.read_table(path).to_pylist()
        for row in table:
            context = (row.get("context") or "").strip()
            if not context:
                continue
            answers_field = row.get("answers") or {}
            texts = list(answers_field.get("text") or [])
            impossible = not texts
            records.append(
                {
                    "qid": str(row.get("id")),
                    "dataset": "viquad2",
                    "split": split,
                    "lang_q": "vi",
                    "lang_c": "vi",
                    "question": (row.get("question") or "").strip(),
                    "answers": sorted({t for t in texts if t}),
                    "plausible_answers": [],
                    "is_impossible": impossible,
                    "context": context,
                    "title": row.get("title") or "",
                    "doc_id": _doc_id(context),
                }
            )
        answered = sum(1 for r in records if r["split"] == split and not r["is_impossible"])
        total = sum(1 for r in records if r["split"] == split)
        print(f"  viquad2 {split}: {total} questions ({answered} answerable)")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DATA / "normalized")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("MLQA:")
    mlqa = prepare_mlqa()
    print("UIT-ViQuAD 2.0:")
    viquad = prepare_viquad()

    written = 0
    for dataset, records in (("mlqa", mlqa), ("viquad2", viquad)):
        by_split: dict[str, list[dict]] = {}
        for record in records:
            by_split.setdefault(record["split"], []).append(record)
        for split, rows in by_split.items():
            path = args.out / f"{dataset}_{split}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += len(rows)
            docs = len({row["doc_id"] for row in rows})
            print(f"wrote {path.name}: {len(rows)} questions, {docs} unique contexts")
    print(f"total {written} questions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
