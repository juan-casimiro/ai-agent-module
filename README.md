# AI Agent Module

A LangGraph agent that classifies a user's question and routes it to one
of three tools: document-grounded retrieval (RAG), precise calculation,
or direct general-knowledge answering.

## Architecture
```
Question
│
▼
classify_question — LLM classifies intent via structured output
│
▼
route_decision — routes based on classification
│
├── BIOMED ──────► answer_from_document
│ (calls the Research Assistant RAG service, BIOMED specific content)
│
├── CALCULATION ───► answer_with_calculation
│ (LLM extracts expression + precision,
│ evaluated safely via simpleeval)
│
└── GENERAL ───────► answer_general
(LLM answers directly)
```
## Tech stack

- **LangGraph** — graph-based agent orchestration (nodes, edges, state)
- **LangChain** (`langchain-anthropic`) — LLM abstraction; `with_structured_output`
  works identically across supported providers, not tied to Anthropic
- **simpleeval** — safe evaluation of LLM-extracted arithmetic expressions
- **httpx** — calls the separately running RAG service (Portfolio Project 2)

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

Runs three example questions, one per path (general knowledge, document
lookup, calculation), printing the classification and answer for each.

## Design decisions

See [ADR-001](./adr/001-agent-routing-and-structured-outputs.md) for the
reasoning behind: structured outputs over prompted JSON, `simpleeval`
over raw `eval()`, `GraphState` type consistency, and the fail-loud
routing check.

## Known limitations

- The document-lookup path is BIOMED-specific right now 
(the underlying RAG service's corpus is scoped to diabetes, cardiology, and oncology — see `corpus_manifest.json` in the `ai-research-assistant` project). 
A non-biomed document question will either get misrouted to `GENERAL` or hit empty/irrelevant retrieval.
- The document-lookup path inherits all known limitations of the underlying RAG service (see that project's ADR), including sensitivity to exact query phrasing. Multi-turn phrasing/reference issues (e.g. follow-up questions relying on earlier context) are not addressed — there is no conversation memory yet (see below).
- No conversation memory yet — each invocation is stateless.
- Classification is a single LLM call with no retry/fallback if it
  returns an unexpected result (mitigated by the fail-loud check in
  `route_decision`, but not gracefully recovered from).

## Possible future improvements

- Conversation memory via LangGraph checkpointing
- Additional tools (e.g., web search)
- Query rewriting or hybrid search on the document-lookup path, to
  address the phrasing-sensitivity limitation
- This project will grow biomed-specific capability (tools, prompts, maybe a dedicated BIOMED_TASK category)
