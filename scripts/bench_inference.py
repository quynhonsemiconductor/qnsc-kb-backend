"""Measure the CPU inference paths that decide how a search FEELS: model load, query embedding,
the rerankers, and the pipeline they make together.

Why it exists. `develop` runs the api on 2 vCPU shared with clamd, loads
multilingual-e5-large-instruct (24 layers, ~2.2 GB fp32 ONNX) in-process, and a partner
reported searches that "look stuck". Every number that justified a sizing decision before
this was either the worker's or a guess. This measures the real code (`src.lib.embeddings`,
`src.rag.reranker`, `src.rag.cross_encoder`) on the real model, so the numbers can be
reproduced and compared across thread counts and instance types.

Run it where the CPU is the thing you want to know about, pinned to the cores you want, and ONE
thread count per process (the ONNX session is a process-wide singleton):

    taskset -c 0,1 python scripts/bench_inference.py --threads 2 --out result-2.json

What it does NOT measure, on purpose and said here so no one reads more into it:
  * the database. `--db-dsn` adds a pgvector micro-benchmark on SYNTHETIC vectors (shape only);
    the search pipeline numbers use an in-memory brute-force stand-in for `hybrid_search`;
  * the LLM. Nothing here calls a provider.
  * retrieval QUALITY on your corpus. The quality section is a small public proxy (Vietnamese
    Wikipedia introductions, queries written from a chunk with the title words removed). It
    ranks configurations against each other; it is not a score for your documents.

The corpus is fetched from the public Wikipedia API on first use and cached in `--cache`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Titles of Vietnamese Wikipedia articles with a substantial introduction. A title that does not
# exist or has a short intro is dropped, so this list is a ceiling and not a promise.
TITLES = [
    "Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng", "Huế", "Quy Nhơn", "Nha Trang", "Cần Thơ",
    "Hải Phòng", "Sông Mê Kông", "Sông Hồng", "Vịnh Hạ Long", "Phú Quốc", "Đà Lạt", "Sa Pa",
    "Hồ Chí Minh", "Nguyễn Huệ", "Trần Hưng Đạo", "Lý Thường Kiệt", "Quang Trung", "Võ Nguyên Giáp",
    "Phở", "Bánh mì", "Áo dài", "Tết Nguyên Đán", "Cà phê Việt Nam", "Nước mắm", "Bún chả",
    "Trí tuệ nhân tạo", "Học máy", "Mạng nơ-ron nhân tạo", "Máy tính", "Internet", "Hệ điều hành",
    "Cơ sở dữ liệu", "Bán dẫn", "Transistor", "Vi mạch", "Pin mặt trời", "Điện toán đám mây",
    "Quang hợp", "Hệ mặt trời", "Sao Hỏa", "Trái Đất", "Mặt Trăng", "Núi lửa", "Động đất",
    "Hồ Hoàn Kiếm", "Văn Miếu - Quốc Tử Giám", "Hội An", "Phong Nha-Kẻ Bàng", "Cố đô Huế",
    "Vịnh Cam Ranh", "Đồng bằng sông Cửu Long", "Tây Nguyên", "Biển Đông", "Dãy Trường Sơn",
    "Công nghệ sinh học", "Năng lượng tái tạo", "Biến đổi khí hậu", "Kinh tế Việt Nam",
]


# ── small helpers ────────────────────────────────────────────────────────────────────────────
def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))
    return ordered[index]


def stats_ms(samples: list[float]) -> dict[str, float]:
    """Summary of a list of seconds, in milliseconds."""
    ms = [s * 1000 for s in samples]
    return {
        "n": len(ms),
        "p50_ms": round(percentile(ms, 0.50), 1),
        "p95_ms": round(percentile(ms, 0.95), 1),
        "max_ms": round(max(ms), 1) if ms else float("nan"),
        "mean_ms": round(statistics.fmean(ms), 1) if ms else float("nan"),
    }


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def environment() -> dict[str, Any]:
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        model = platform.processor()
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 0
    import onnxruntime

    return {
        "cpu_model": model,
        "machine": platform.machine(),
        "cpus_usable_by_this_process": usable,
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
    }


# ── corpus ───────────────────────────────────────────────────────────────────────────────────
def fetch_articles(cache: Path, limit: int) -> list[dict[str, str]]:
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))[:limit]
    import httpx

    found: list[dict[str, str]] = []
    with httpx.Client(timeout=30.0, headers={"User-Agent": "qnsc-kb-bench/1.0 (performance benchmark)"}) as client:
        for start in range(0, len(TITLES), 20):
            batch = TITLES[start : start + 20]
            response = client.get(
                "https://vi.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "exlimit": 20,
                    "format": "json",
                    "titles": "|".join(batch),
                },
            )
            response.raise_for_status()
            for page in (response.json().get("query", {}).get("pages", {}) or {}).values():
                text = (page.get("extract") or "").strip()
                if len(text) >= 500:
                    found.append({"title": page["title"], "text": text})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    return found[:limit]


def build_pool(articles: list[dict[str, str]]) -> tuple[list[SimpleNamespace], list[dict[str, Any]]]:
    """Chunk with the REAL chunker and wrap the children the way the ORM objects look to the reranker."""
    from src.rag.chunker import create_parent_child_chunks

    chunks: list[SimpleNamespace] = []
    for article_index, article in enumerate(articles):
        for parent in create_parent_child_chunks(article["text"], heading=article["title"]):
            parent_ref = SimpleNamespace(text=parent["parent_text"], section_ref=parent.get("heading") or article["title"])
            for child in parent["children"]:
                text = str(child)
                if len(text) < 80:
                    continue
                chunks.append(
                    SimpleNamespace(
                        id=len(chunks),
                        article_index=article_index,
                        chunk_text=text,
                        article=SimpleNamespace(title=article["title"]),
                        parent_chunk=parent_ref,
                    )
                )
    queries: list[dict[str, Any]] = []
    for chunk in chunks[:: max(1, len(chunks) // 80)]:
        title_words = set(re.findall(r"\w+", chunk.article.title.lower()))
        words = [w for w in chunk.chunk_text.split() if re.sub(r"\W+", "", w).lower() not in title_words]
        if len(words) >= 12:
            queries.append({"query": " ".join(words[2:14]), "gold_chunk": chunk.id, "gold_article": chunk.article_index})
    return chunks, queries


# ── sections ─────────────────────────────────────────────────────────────────────────────────
def bench_load() -> dict[str, Any]:
    from src.lib.embeddings import warm_up

    before = rss_mb()
    started = time.perf_counter()
    warm_up()
    seconds = time.perf_counter() - started
    return {"load_seconds": round(seconds, 2), "rss_before_mb": before, "rss_after_mb": rss_mb()}


def bench_query_embedding(queries: list[str], repeats: int) -> dict[str, Any]:
    from src.lib.embeddings import get_bge_embedding

    first = []
    for q in queries[:3]:  # the first call after load pays one-off allocation; report it, then discard
        t = time.perf_counter()
        get_bge_embedding(q)
        first.append(time.perf_counter() - t)
    samples = []
    for _ in range(repeats):
        for q in queries:
            t = time.perf_counter()
            get_bge_embedding(q)
            samples.append(time.perf_counter() - t)
    return {"first_calls_ms": [round(x * 1000, 1) for x in first], **stats_ms(samples)}


def bench_lexical_rerank(chunks: list[SimpleNamespace], queries: list[str], sizes: list[int]) -> dict[str, Any]:
    from src.rag.reranker import rerank_chunks_with_scores

    out = {}
    for size in sizes:
        pool = (chunks * math.ceil(size / max(1, len(chunks))))[:size]
        samples = []
        for q in queries[:20]:
            t = time.perf_counter()
            rerank_chunks_with_scores(q, pool, limit=size)
            samples.append(time.perf_counter() - t)
        out[str(size)] = stats_ms(samples)
    return out


def bench_cross_encoder(chunks: list[SimpleNamespace], queries: list[str], pools: list[int], threads: int) -> dict[str, Any]:
    import torch

    torch.set_num_threads(threads)
    from src.rag import cross_encoder
    from src.rag.reranker import chunk_passage

    started = time.perf_counter()
    cross_encoder._model.get()  # noqa: SLF001 - this IS the cold load being measured
    load = time.perf_counter() - started
    model = cross_encoder._model.get()  # noqa: SLF001
    out: dict[str, Any] = {
        "load_seconds": round(load, 2),
        "model": getattr(model, "model", None).__class__.__name__,
        "max_length_attr": getattr(model, "max_length", None),
        "rss_after_mb": rss_mb(),
    }
    for pool in pools:
        passages = [chunk_passage(c) for c in chunks[:pool]]
        samples = []
        for q in queries[:8]:
            t = time.perf_counter()
            cross_encoder.score(q, passages)
            samples.append(time.perf_counter() - t)
        out[f"pool_{pool}"] = stats_ms(samples)
    return out


def bench_quality(
    chunks: list[SimpleNamespace], queries: list[dict[str, Any]], pools: list[int], cross_pools: list[int], with_cross: bool
) -> dict[str, Any]:
    """Rank the corpus with the real embedder, then see what each pool size / reranker does to the gold chunk."""
    import numpy

    from src.lib.embeddings import get_bge_embedding, get_bge_embeddings
    from src.rag.reranker import chunk_passage, rerank_chunks_with_scores

    started = time.perf_counter()
    matrix = numpy.array(get_bge_embeddings([c.chunk_text for c in chunks]), dtype="float32")
    index_seconds = time.perf_counter() - started
    query_vectors = numpy.array([get_bge_embedding(q["query"]) for q in queries], dtype="float32")
    similarity = query_vectors @ matrix.T

    cross_scorer = None
    if with_cross:
        from src.rag.cross_encoder import score as cross_scorer  # type: ignore[assignment]

    def metrics(rank_lists: list[list[int]]) -> dict[str, float]:
        hit5 = hit10 = rr = 0.0
        for q, ranking in zip(queries, rank_lists):
            gold = q["gold_chunk"]
            if gold in ranking[:5]:
                hit5 += 1
            if gold in ranking[:10]:
                hit10 += 1
            if gold in ranking:
                rr += 1.0 / (ranking.index(gold) + 1)
        n = max(1, len(queries))
        return {"R@5": round(hit5 / n, 3), "R@10": round(hit10 / n, 3), "MRR": round(rr / n, 3)}

    out: dict[str, Any] = {
        "corpus_chunks": len(chunks),
        "queries": len(queries),
        "embedding_index_seconds": round(index_seconds, 1),
    }
    out["dense_only"] = metrics([list(numpy.argsort(-row)) for row in similarity])
    for pool in pools:
        ranks = []
        for q, row in zip(queries, similarity):
            top = [int(i) for i in numpy.argsort(-row)[:pool]]
            reranked = rerank_chunks_with_scores(q["query"], [chunks[i] for i in top], limit=pool)
            ranks.append([c.id for c, _ in reranked])
        out[f"lexical_pool_{pool}"] = metrics(ranks)
    if cross_scorer is not None:
        for pool in cross_pools:
            ranks, seconds = [], []
            for q, row in zip(queries, similarity):
                top = [int(i) for i in numpy.argsort(-row)[:pool]]
                lexical = [c for c, _ in rerank_chunks_with_scores(q["query"], [chunks[i] for i in top], limit=pool)]
                t = time.perf_counter()
                scores = cross_scorer(q["query"], [chunk_passage(c) for c in lexical])
                seconds.append(time.perf_counter() - t)
                order = sorted(range(len(lexical)), key=lambda k: scores[k], reverse=True)
                ranks.append([lexical[k].id for k in order])
            out[f"cross_encoder_pool_{pool}"] = {**metrics(ranks), "latency": stats_ms(seconds)}
    return out


def bench_pipeline(chunks: list[SimpleNamespace], queries: list[str], pool_size: int, with_cross: bool, cross_pool: int) -> dict[str, Any]:
    """A search without the database: embed -> candidate pool -> lexical rerank (-> cross-encoder).

    The pool comes from a brute-force dot product over the in-memory corpus, standing in for
    `hybrid_search`'s two SQL legs. It is NOT a database timing.
    """
    import numpy

    from src.lib.embeddings import get_bge_embedding, get_bge_embeddings
    from src.rag.reranker import chunk_passage, rerank_chunks_with_scores

    matrix = numpy.array(get_bge_embeddings([c.chunk_text for c in chunks]), dtype="float32")
    if with_cross:
        from src.rag.cross_encoder import score as cross_scorer
    totals, parts = [], {"embed": [], "pool": [], "lexical": [], "cross": []}
    for q in queries[:30]:
        t0 = time.perf_counter()
        vec = numpy.array(get_bge_embedding(q), dtype="float32")
        t1 = time.perf_counter()
        top = [int(i) for i in numpy.argsort(-(matrix @ vec))[:pool_size]]
        pool = [chunks[i] for i in top]
        t2 = time.perf_counter()
        ranked = rerank_chunks_with_scores(q, pool, limit=pool_size)
        t3 = time.perf_counter()
        if with_cross:
            cross_scorer(q, [chunk_passage(c) for c, _ in ranked[:cross_pool]])
        t4 = time.perf_counter()
        parts["embed"].append(t1 - t0)
        parts["pool"].append(t2 - t1)
        parts["lexical"].append(t3 - t2)
        parts["cross"].append(t4 - t3)
        totals.append(t4 - t0)
    return {"total": stats_ms(totals), **{k: stats_ms(v) for k, v in parts.items() if v}}


def bench_db(dsn: str, sizes: list[int], ef_values: list[int]) -> dict[str, Any]:
    """pgvector on SYNTHETIC unit vectors: the cost of `ORDER BY embedding <=> q LIMIT 240` at an
    ef_search, with 10% of rows passing a filter (the permission/status predicates cut into the scan)."""
    import numpy
    import psycopg2
    from psycopg2.extras import execute_values

    out: dict[str, Any] = {}
    rng = numpy.random.default_rng(7)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for size in sizes:
        cur.execute("DROP TABLE IF EXISTS bench_chunks")
        cur.execute("CREATE TABLE bench_chunks (id int primary key, bucket int, embedding vector(1024))")
        vectors = rng.standard_normal((size, 1024)).astype("float32")
        vectors /= numpy.linalg.norm(vectors, axis=1, keepdims=True)
        rows = [(i, i % 10, "[" + ",".join(f"{x:.5f}" for x in v) + "]") for i, v in enumerate(vectors)]
        for start in range(0, len(rows), 2000):
            execute_values(cur, "INSERT INTO bench_chunks (id, bucket, embedding) VALUES %s", rows[start : start + 2000],
                           template="(%s, %s, %s::vector)")
        t = time.perf_counter()
        cur.execute("CREATE INDEX ON bench_chunks USING hnsw (embedding vector_cosine_ops)")
        build = time.perf_counter() - t
        cur.execute("ANALYZE bench_chunks")
        section: dict[str, Any] = {"hnsw_build_seconds": round(build, 1)}
        queries = vectors[rng.integers(0, size, 25)]
        for ef in ef_values:
            samples, returned = [], []
            cur.execute("SET hnsw.ef_search = %s", (ef,))
            cur.execute("SET hnsw.iterative_scan = relaxed_order")
            for q in queries:
                literal = "[" + ",".join(f"{x:.5f}" for x in q) + "]"
                t = time.perf_counter()
                cur.execute(
                    "SELECT id FROM bench_chunks WHERE bucket = 3 ORDER BY embedding <=> %s::vector LIMIT 240", (literal,)
                )
                returned.append(len(cur.fetchall()))
                samples.append(time.perf_counter() - t)
            section[f"ef_search_{ef}"] = {**stats_ms(samples), "rows_returned_mean": round(statistics.fmean(returned), 1)}
        out[str(size)] = section
    cur.execute("DROP TABLE IF EXISTS bench_chunks")
    conn.close()
    return out


# ── main ─────────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", default=os.getenv("EMBEDDING_ONNX_DIR", "/opt/embedding-onnx"))
    parser.add_argument("--threads", type=int, required=True, help="ONNX / torch intra-op threads for this run")
    parser.add_argument("--articles", type=int, default=60)
    parser.add_argument(
        "--sections",
        default="load,embed,lexical,cross,quality,pipeline,db",
        help="which parts to run; `cross` also needs --cross-encoder, `db` needs --db-dsn",
    )
    parser.add_argument("--repeats", type=int, default=3, help="passes over the query set for the embedding latency")
    parser.add_argument("--lexical-pools", default="48,80,120,240")
    parser.add_argument("--cross-encoder", action="store_true", help="also measure the optional cross-encoder (needs the ml group)")
    parser.add_argument("--cross-pools", default="10,20,30")
    parser.add_argument("--db-dsn", default=None)
    parser.add_argument("--db-sizes", default="5000,30000")
    parser.add_argument("--ef-values", default="40,200,1000")
    parser.add_argument("--cache", type=Path, default=Path(os.getenv("BENCH_CACHE", "/tmp/bench-corpus.json")))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    os.environ["EMBEDDING_ONNX_DIR"] = args.onnx_dir
    os.environ["EMBEDDING_ONNX_THREADS"] = str(args.threads)
    os.environ.setdefault("ENVIRONMENT", "development")

    result: dict[str, Any] = {"threads": args.threads, "environment": environment()}
    articles = fetch_articles(args.cache, args.articles)
    chunks, queries = build_pool(articles)
    query_texts = [q["query"] for q in queries]
    result["corpus"] = {"articles": len(articles), "chunks": len(chunks), "queries": len(queries),
                        "chunk_chars_mean": round(statistics.fmean(len(c.chunk_text) for c in chunks))}
    print(f"corpus: {result['corpus']}", flush=True)

    sections = set(args.sections.split(","))
    lexical_pools = [int(x) for x in args.lexical_pools.split(",")]
    cross_pools = [int(x) for x in args.cross_pools.split(",")]
    if "load" in sections:
        result["model_load"] = bench_load()
        print(f"load: {result['model_load']}", flush=True)
    if "embed" in sections:
        result["query_embedding"] = bench_query_embedding(query_texts[:40], args.repeats)
        print(f"query embedding: {result['query_embedding']}", flush=True)
    if "lexical" in sections:
        result["lexical_rerank"] = bench_lexical_rerank(chunks, query_texts, lexical_pools)
        print(f"lexical rerank: {result['lexical_rerank']}", flush=True)
    if "cross" in sections and args.cross_encoder:
        result["cross_encoder"] = bench_cross_encoder(chunks, query_texts, cross_pools, args.threads)
        print(f"cross-encoder: {result['cross_encoder']}", flush=True)
    if "quality" in sections:
        result["quality_proxy"] = bench_quality(chunks, queries, lexical_pools, cross_pools, args.cross_encoder)
        print(f"quality proxy: {json.dumps(result['quality_proxy'])}", flush=True)
    if "pipeline" in sections:
        result["pipeline_no_db"] = bench_pipeline(chunks, query_texts, max(lexical_pools), args.cross_encoder, max(cross_pools))
        print(f"pipeline (no db): {result['pipeline_no_db']}", flush=True)
    if args.db_dsn and "db" in sections:
        result["db_vector_synthetic"] = bench_db(
            args.db_dsn, [int(x) for x in args.db_sizes.split(",")], [int(x) for x in args.ef_values.split(",")]
        )
        print(f"db: {json.dumps(result['db_vector_synthetic'])}", flush=True)
    result["rss_end_mb"] = rss_mb()
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
