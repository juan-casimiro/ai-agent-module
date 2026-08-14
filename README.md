# AI Agent Module

A LangGraph agent that classifies a user's question and routes it to one
of three tools: document-grounded retrieval (RAG), precise calculation,
or direct general-knowledge answering.

## Architecture
```
Question
│
▼
condense_question — resolves follow-up questions against conversation
history into a standalone question (skipped if no prior history)
│
▼
classify_question — LLM classifies intent via structured output
│
▼
route_decision — routes based on classification
│
├── BIOMED ──────► answer_from_document
│   (calls the Research Assistant RAG service)
│   │
│   ├─ context_sufficient=False, first attempt → retry_document_lookup
│   │    (retries with query rewriting)
│   │    ├─ still insufficient → answer_general (ungrounded-fallback caveat)
│   │    └─ sufficient → record_turn
│   │
│   └─ context_sufficient=True → record_turn
│
├── CALCULATION ───► answer_with_calculation
│ (LLM extracts expression + precision,
│ evaluated safely via simpleeval)
│
└── GENERAL ───────► answer_general
(LLM answers directly)
│
▼
record_turn — appends {question, answer} to conversation history
(runs on every path before END)

Conversation state persists across calls via LangGraph's `MemorySaver`
checkpointer, keyed by `thread_id`. Each `thread_id` is an isolated
conversation — no state is shared between threads.
```
## Tech stack

- **LangGraph** — graph-based agent orchestration (nodes, edges, state)
- **LangChain** (`langchain-anthropic`) — LLM abstraction; `with_structured_output`
  works identically across supported providers, not tied to Anthropic
- **simpleeval** — safe evaluation of LLM-extracted arithmetic expressions
- **httpx** — calls the separately running RAG service (Portfolio Project 2)
- **langgraph-checkpoint** (`MemorySaver`) — in-memory conversation
  checkpointing, keyed by `thread_id`
## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env`:
```
ANTHROPIC_API_KEY=your-key-here
```
**Requires the Research Assistant RAG service running separately** on
`localhost:8000` (see the `ai-research-assistant` project) for the
document-lookup path to work.

## Running

```bash
python hello_langgraph.py
```

Runs a scripted 3-turn conversation on a single `thread_id`, demonstrating:
- Turn 1: a standalone BIOMED question (no history yet, condensation
  skipped)
- Turn 2: a follow-up ("What about the fall transition?") resolved
  against turn 1's topic via `condense_question`
- Turn 3: a follow-up referencing a specific number from turn 2's
  answer, resolved and correctly routed to CALCULATION

Each turn prints the original question, the resolved (condensed)
question, a short reasoning note explaining the resolution, the
classification, and the answer. A final check confirms
`len(history) == 3` — one entry per turn, not duplicated.

The script also runs a fourth, isolated check on a second `thread_id`,
reusing turn 2's exact follow-up phrasing with no prior history, to
confirm conversation state does not leak across threads.

A fifth, isolated scenario (`demo-thread-retry-scenario`) triggers the
retry → fallback/recovery loop against the real, running RAG service,
using a genuinely off-corpus question (HCR-FISH imaging in *M.
tuberculosis*-infected lung tissue — not present in the ingested
biomedical corpus; see ADR-001). The script reports whichever outcome
actually occurs rather than assuming one; `retried` and
`context_sufficient` are captured mid-graph via
`app.stream(..., stream_mode="values")`, since `record_turn` resets
both fields before the final state is returned.

## Design decisions

See [ADR-001](./adr/001-agent-routing-and-structured-outputs.md) for the
reasoning behind: structured outputs over prompted JSON, `simpleeval`
over raw `eval()`, `GraphState` type consistency, and the fail-loud
routing check.

See [ADR-002](./adr/002-recursive-retry-loop.md) for the retry loop's
design: the `context_sufficient` trigger, why query rewriting is the
retry lever despite `ai-research-assistant`'s own null result on it,
the fallback/retry-cap mechanics, and what's proven vs. assumed about
recovery.

## Known limitations

- The document-lookup path is BIOMED-specific right now
(the underlying RAG service's corpus is scoped to diabetes, cardiology, and oncology — see `corpus_manifest.json` in the `ai-research-assistant` project).
A non-biomed document question will either get misrouted to `GENERAL` or hit empty/irrelevant retrieval.
- The document-lookup path inherits all known limitations of the underlying RAG service (see that project's ADR), including sensitivity to exact query phrasing.
- The retry trigger (`context_sufficient`) is an LLM's self-assessment of
  whether retrieved context was enough to answer — not perfectly
  reliable, and not a guarantee against false positives/negatives. See
  ADR-001 for the reasoning and the trigger it replaced.
- Conversation history is unbounded and passed as raw text into every
  `condense_question` call. Fine for short demo conversations; a long-
  running conversation would grow the condensation prompt (and its
  token cost) linearly with turn count, with no truncation or
  summarization of older turns.
- Classification is a single LLM call with no retry/fallback if it
  returns an unexpected result (mitigated by the fail-loud check in
  `route_decision`, but not gracefully recovered from).
- `MemorySaver` is in-memory only — conversation history does not
  survive a process restart. Deliberate choice for a portfolio project,
  not a production-readiness gap (see ADR-001 update).
- The retry mechanism (conditional edges, retry-cap termination, state
  tracking) was built to practice self-correcting agent-loop design —
  that architecture is what this exercise tests, and it's what
  `tests/test_graph_retry_integration.py` and JUA-15's live demo verify.
  The specific recovery lever it reuses (`ai-research-assistant`'s
  query rewriting) was chosen for consistency with an existing
  capability, not for expected efficacy: that project's own evaluation
  (ADR-001, 111 scored queries, two corpus sizes) already showed
  rewriting produces zero verdict changes on this corpus. A different,
  purpose-built recovery strategy was considered but not implemented,
  in favor of keeping this project's scope on the orchestration
  pattern rather than retrieval-quality tuning.
  Separately: the recovery branch is proven correct only via a mocked
  test (`test_insufficient_then_sufficient_recovers`) and has never been
  observed succeeding on a real, non-mocked query — the one live run on
  record (JUA-15) used an off-corpus question no retrieval strategy
  could have recovered. See ADR-002 for the full reasoning.

## Possible future improvements

- Mocked classification tests / end-to-end graph integration tests
  (unit tests for routing, calculation safety, and the retry/fallback
  edges exist; the HTTP-calling nodes and prompt-assembly branches are
  currently only verified manually via the demo script)
- Additional tools (e.g., web search)
- Hybrid (BM25 + dense) search on the retry path specifically. Query
  rewriting is already implemented as the retry lever
  (`retry_document_lookup`, `use_query_rewriting=True`); BM25 exists in
  the underlying RAG service but isn't currently exposed as a retry
  option. Not evaluated for this use case — see
  `ai-research-assistant`'s ADR-001, where BM25 alone caused one
  attributable regression on the golden QA set.
- This project will grow biomed-specific capability (tools, prompts, maybe a dedicated BIOMED_TASK category)