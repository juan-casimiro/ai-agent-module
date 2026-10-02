# Agent guide — ai-agent-module

LangGraph agent routing questions to biomedical RAG, safe arithmetic, or general knowledge, with conversation memory and a bounded document retry.

## Shared guidance

Follow the [development guidance index](https://github.com/juan-casimiro/development-config/blob/main/AGENTS.md)
for shared process and environment instructions. If already loaded in this chat,
reuse its completed startup and instructions.

## Sources

- [README](README.md): setup, demo, architecture, measured limitations, and future scope.
- Relevant ADRs: [routing, structured outputs, and memory](adr/001-agent-routing-and-structured-outputs.md); [retry and fallback](adr/002-recursive-retry-loop.md).
- [Test-quality audit](docs/test-quality-audit.md): behavioral protection, evidence, and remaining risks.

## Project map

- `hello_langgraph.py`: the entire agent, including `GraphState`, categories, structured response models, node functions, HTTP boundary, graph wiring, and compiled `app`. There is no separate graph module.
- `requirements.txt`: pinned Python dependencies, including LangGraph, LangChain's Anthropic integration, Pydantic, simpleeval, httpx, and pytest.
- `eval_classification.py`, `classification_set.json`, and `eval_results/classification_results.json`: routing evaluation and recorded results, separate from deterministic tests.
- `tests/`: routing, arithmetic, document contracts, structured-output failures, retry control flow, compiled conversations, and evaluation scoring. Root `conftest.py` isolates checkpoint state, disables credentials/tracing, and blocks unexpected live model/HTTP calls.

## Contracts to preserve

- Nodes return only fields they set, never `{**state, ...}`. `history` uses an append reducer; re-emitting existing history duplicates it. `record_turn` appends the raw question and final answer exactly once; downstream nodes use `resolved_question`, and condensation reasoning stays out of history.
- Persist classification as `Category.value` strings. Keep the enum, prompts, and routing map aligned, and retain the fail-loud check for unmapped classifications. Use Pydantic structured outputs for classification, condensation, and calculation; absent output and parsing failures stop the turn.
- Keep arithmetic behind the character allow-list and `simpleeval`; reject names, function calls, `**`, and `//`. Apply rounding separately in trusted Python, never via evaluated code. General and calculation paths clear document sources, including calculation errors.
- Document lookup calls the sibling `ai-research-assistant` service directly at `http://localhost:8000/query`; it does not use `spring-mcp-gateway`. The current corpus is biomedical (diabetes, cardiology, oncology).
- Preserve HTTP-status and strict response validation: `answer` is required; missing legacy `sources`, `context_sufficient`, and `insufficiency_reason` default to `[]`, `True`, and `None`. Malformed supplied fields, transport failures, and protocol failures abort the turn without insufficiency retry, general fallback, or recording a completed answer.
- Retry only on insufficient context, once, with `use_query_rewriting=True`; source emptiness is not the trigger. Continued insufficiency routes to general answering with the explicit ungrounded caveat and cleared sources. `insufficiency_reason` is diagnostic only.
- `record_turn` resets `retried`, `context_sufficient`, and `retry_reason`. Observe retry outcomes mid-graph with `app.stream(..., stream_mode="values")`, before those resets.
- `MemorySaver` is deliberately in-memory and isolated by `thread_id`; durable storage and history truncation are not current requirements. Control-flow tests do not establish model accuracy or real retrieval recovery; consult README and ADR-002 before making quality claims.

## Verification

From the repository root, run `.venv/bin/python -m pytest -q` for offline regressions after the README setup. Use focused node/HTTP tests for local contracts and compiled-graph tests for routing, history, thread isolation, and retry behavior; retain the live-call guards in `conftest.py`.

`python hello_langgraph.py` is a live, paid-model demo requiring the separately running RAG service. `python eval_classification.py` is a separate paid Anthropic evaluation, not an offline test or CI gate. Obtain explicit approval before running paid external verification.

Keep this guide concise and current when project structure, commands, contracts, or durable agent guidance change. Put detailed rationale and measurements in README, ADRs, or the audit; keep transient task status in the tracker.
