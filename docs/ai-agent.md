# The AI assistant as an agent

`AIService.ask` still owns permissions, context selection, the prompt, citations and the
refusal rules. The agent (`src/domain/agent/`) changes three things inside it.

```
question
  │
  ▼  AgentRun.start ── off (AGENT_ENABLED=false, or flag `ai.agent` off for this user)
  │                      └──► the plain one-pass pipeline, exactly as before
  ▼
 understand   one model call → JSON plan: intent, sub-questions, doc types, departments,
  │           wants chart / figure, needs clarification.
  │           Skipped for a short single-topic question ("fast path").
  ├─ needs clarification ──► the question is asked back (never twice in a row)
  ▼
 search       one `SearchService.search` per sub-question (permissions are applied in SQL
  │           before ranking, every time). Plan filters are only a first guess.
  ▼
 review       a sub-question is covered when a result scores ≥ RAG_MIN_CONTEXT_SCORE.
  │           Not covered → retry without the plan's filters → ask the model to reword it
  │           (≤ AGENT_MAX_RETRIEVAL_ROUNDS). Still nothing solid → the existing refusal,
  │           and the question is recorded in the Gap Queue.
  ▼
 existing pipeline: context selection, conflicts, tables' verified calculations
  ▼
 chart        (wants chart) the model picks table + columns + chart type; the SERVER reads the
  │           values from the stored table. The browser draws it from JSON.
 figure       (wants figure, EVIDENCE_VISION_ENABLED) a vision model returns rectangles as
  │           fractions of the figure; the server validates them; the browser draws them over
  │           the untouched original. Refused above AGENT_IMAGE_MAX_SENSITIVITY.
  ▼
 answer       the existing prompt and citation guards
  ▼
 verify       every quantity in the answer must occur in the sources, a verified calculation
              or a chart; the ones that do not are returned as `unverified_numbers`.
```

## Events

`POST /ai/ask/stream` adds `{"type":"step","step":…,"status":"running|done|skipped","detail":…,"data":{…}}`
events ahead of the tokens, and the `done` event carries `artifacts`, `unverified_numbers`,
`agent_trace` and `agent_plan`. The same fields are stored with the message (inside
`action_data`), so reopening a conversation shows the chart and the steps, and they are hidden
together with the answer if a cited source stops being readable.

## Safety properties

* No model-written code is run. A chart is a JSON description plus numbers the server read from
  a table the user may read; a marked figure is rectangles over the original image.
* A plan can only narrow a search, never widen what the user may read; unknown document types
  and departments are dropped.
* The agent's own model calls are bounded (`AGENT_MAX_LLM_CALLS`, `AGENT_MAX_RETRIEVAL_ROUNDS`).
  Any step that fails is skipped and the answer is still given.
* An answer carrying a chart or a marked figure is not cached.

## Rolling it out and back

`AGENT_ENABLED` is the global switch; the `ai.agent` feature flag (admin → feature flags) limits
it by role, department or percentage of users. Both off → the pre-agent behaviour.
