# Multimodal evidence and cited analytics

## Decision

The first production slice uses the existing permission-aware multilingual text retrieval to select **evidence artifacts**: bounded tables and visual document regions with an Article, original source, page, and optional bounding box. Only selected visual artifacts are sent to a vision-capable provider, and numerical answers are calculated by a closed deterministic DSL.

This deliberately does not deploy ColPali/Qdrant multivectors yet. ColPali's late-interaction page representation is attractive for visually rich retrieval, but its model card demonstrates roughly 1,000 128-dimensional vectors per rendered page plus VLM inference at indexing and query time. That is a new GPU/index service, not a safe incremental change. The current design retains the provenance needed to benchmark that phase later without blocking useful table/image retrieval now.

## Research basis

- [ColPali](https://arxiv.org/abs/2407.01449) and the [model card](https://huggingface.co/vidore/colpali-v1.3) motivate preserving rendered visual evidence instead of flattening every page.
- [Qdrant multivectors](https://qdrant.tech/documentation/concepts/vectors/#multivectors) provide the likely MaxSim index for a later ColPali evaluation.
- [Docling](https://arxiv.org/abs/2408.09869) is the preferred extraction adapter to benchmark later for richer layout, table, formula, and figure understanding; this slice reuses already-shipped parsers to avoid a large unmeasured dependency.
- [ChartQA](https://arxiv.org/abs/2203.10244) shows chart questions require both visual grounding and arithmetic/logical reasoning.
- [FinQA](https://arxiv.org/abs/2109.00122) motivates explicit reasoning programs: the application validates and executes a small operation plan rather than trusting model arithmetic.
- [OWASP prompt injection guidance](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) motivates treating every file, OCR string, table cell, caption, and image as data rather than instructions.

## Data and retrieval flow

1. Article indexing loads the newest private `DocumentSource` and extracts bounded `DocumentEvidence` records.
2. Evidence types are `table`, `figure`, and `page_visual`. Each record keeps Article/source ancestry, page, optional box/page dimensions, bounded search text, bounded table JSON, content hash, and extraction version.
3. The evidence text projection is embedded through the existing local model and stored as ordinary parent/child chunks carrying `evidence_id`.
4. Existing Article SQL authorization remains the retrieval gate. Evidence has no independent visibility or tenant field that could drift from its Article.
5. Citations add evidence metadata only when the cited authorized chunk carries an artifact.
6. Preview requests are nested under the Article route and re-check Article, evidence, and source ancestry before loading private bytes. Visuals become bounded PNG; tables become bounded JSON. Responses are private/no-store and nosniff.

Evidence extraction is enhancement-only. A malformed visual/table parser does not remove ordinary document text from search. Re-index replacement of evidence and chunks shares one transaction, so a failed embed cannot publish a partial evidence index.

## Analytics boundary

Allowed operations are `count`, `sum`, `average`, `median`, `min`, `max`, `difference`, and `percentage_change`. Plans reference one server-provided evidence ID, known columns, exact-match filters, and explicit row indices for comparisons. Execution uses `Decimal` and returns operation, source marker, operands, formula, result, rows used, and omitted-cell warnings.

There is no `eval`, SQL, Python source, filesystem, network, dynamic import, document formula execution, macro execution, or arbitrary operator. Row/column/cell/plan limits are settings. The live conversational bridge is deliberately conservative: it runs only an unambiguous aggregate naming exactly one known column. The richer DSL exists for a future strictly validated planner but ambiguous prose never silently selects rows.

## Vision boundary

Vision is off by default because the workspace may configure GLM/Groq or a custom OpenAI-compatible provider that does not support image blocks. When enabled, only 1–2 server-rendered authorized PNGs within a total-byte ceiling are sent. Native Gemini and standard OpenAI-compatible content blocks are supported. The model is told that image text/instructions are untrusted and may only describe relevant visible facts; calculation remains deterministic. Provider rejection degrades to text/table evidence and reports `evidence_status=text_fallback` rather than failing the grounded answer or claiming the image was analyzed.

## Threat model

- **Guessed artifact ID:** nested Article route plus triple Article/evidence/source ownership predicates; unauthorized Article lookup is fail-closed.
- **Cross-tenant retrieval:** evidence enters the process only through Article-authorized chunk retrieval; visual rehydration repeats the full Article authorization predicate.
- **Indirect prompt injection:** evidence is fenced as untrusted; visual prompts prohibit following image instructions; no privileged tool is exposed.
- **Resource exhaustion:** artifact, row, column, cell, text, image dimension, image count, and byte ceilings.
- **Active content:** no document formula, macro, embedded object, URL, or instruction is executed. Browser receives only inert JSON/PNG from evidence preview.
- **Stale history/cache:** historical citations are rehydrated through currently authorized chunks; retrieval version is bumped and evidence-enriched questions bypass the legacy text-only cache.

## Operations and rollout

Settings begin with `EVIDENCE_`. Monitor extraction failures, evidence count/type distribution, evidence chunk retrieval rate, preview errors/bytes, analytics validation failures, vision fallback rate, vision latency/tokens, and answer feedback. Enable vision only for a provider/model proven to accept image blocks.

Phase two requires an internal Vietnamese/semiconductor evaluation set with text-heavy, chart, table, scan, and diagram questions. Compare current artifact retrieval against Docling extraction and ColPali/Qdrant using retrieval recall, answer citation correctness, numerical exact match, indexing time/storage, query p95, GPU/memory, and permission-negative tests. Adopt the heavier service only if the measured quality gain justifies its operating cost.
