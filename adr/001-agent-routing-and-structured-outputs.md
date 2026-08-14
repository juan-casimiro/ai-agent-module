# ADR-001: Agent Routing, Structured Outputs, and Type Consistency

## Context

This project builds a LangGraph agent that routes a user's question to
one of three tools — document-grounded RAG lookup, precise calculation,
or direct general-knowledge answering — based on an LLM's classification
of the question's intent.

## Decision: structured outputs over prompted JSON

Initial versions asked the LLM to return JSON as plain text, parsed via
`json.loads()`. This failed unpredictably — Claude would wrap valid JSON
in markdown code fences (` ```json ... ``` `) despite explicit
instructions not to, breaking the parser.

Switched to `llm.with_structured_output(PydanticModel)` (a LangChain
abstraction, not an Anthropic-specific mechanism — the same call works
identically across supported providers) for both the classification and
calculation-extraction nodes. This constrains the model's output at the
API/schema level rather than hoping a text-formatting instruction is
followed, eliminating an entire class of parsing failures.

## Decision: simpleeval over raw eval()

The calculation tool needs to evaluate an LLM-extracted arithmetic
expression. Raw `eval()` on any LLM-generated text is a serious security
risk — it can execute arbitrary code, not just arithmetic. A regex
allow-list (digits, operators, parentheses only) was added as a first
line of defense, and `simpleeval` (a library purpose-built for safely
evaluating untrusted arithmetic expressions) was used instead of Python's
built-in `eval()`, with explicit exception handling for division-by-zero
and malformed expressions.

**Deliberately avoided**: expanding the regex to allow function-call
syntax (e.g., to let the LLM directly emit `round(x, 2)`) — allow-listing
specific "safe-looking" function calls via regex is fragile and hard to
get right. Instead, rounding is requested as a separate structured field
(`decimal_places`) and applied by trusted Python code (`round()`) after
the expression is safely evaluated — never inside the evaluated
expression itself.

## Decision: GraphState field types must match actual runtime values

`GraphState.classification` was initially typed as `str` while the code
stored a `Category` enum member directly — a type contract violation
that happened to work by coincidence, since `route_decision`'s mapping
also used `Category` members as keys throughout. This was caught via
external review (a second LLM's code review) and corrected: `Category`
is converted to its `.value` (a plain string) at the point
`classify_question` writes to `GraphState`, and `route_decision`'s
mapping uses `.value` keys consistently, matching the honest `str` type.

This choice specifically anticipates LangGraph's checkpointing/memory
capability, which serializes `GraphState` directly — a plain `str` is
universally serializable, while a Python-specific `Enum` object is not,
without custom serialization logic. The short-lived `Classification`
Pydantic model (used only during the LLM call, never itself persisted)
correctly keeps the richer `Category` enum type, since it never crosses
a serialization boundary.

**Principle**: internal, short-lived objects can use rich types freely;
objects that flow through the whole system (and are candidates for
future persistence/serialization) should use the plainest type that
honestly reflects their declared contract.

## Decision: fail loudly on an unmapped classification

Added an explicit check in `route_decision` that raises `ValueError` if
`classification` isn't a recognized category — even though this should
be structurally impossible given `Classification.category`'s enum-typed
field and `with_structured_output`'s schema enforcement. This is
deliberate defense against future refactoring drift (e.g., a new
category added to `Category` but not to the routing map), not a response
to an observed failure. Cheap insurance against a silent failure mode
that would otherwise produce a plausible-but-wrong answer rather than a
clear error.

## Consequences

- Structured outputs and `simpleeval` both required no
  Anthropic-specific code — consistent with a stated goal of building
  provider-agnostic solutions.
- Prompts follow a consistent Role/Task/Constraints/Examples/Input
  template; category names are referenced via the `Category` enum
  rather than duplicated as literal strings, preventing prompt text and
  routing logic from silently drifting out of sync.
- The document-path tool calls a separately running RAG service
  (Portfolio Project 2) over HTTP — this surfaced a real, already-
  documented RAG limitation (query phrasing sensitivity) directly
  through agent-level testing, confirming the same issue found in
  isolated RAG evaluation also affects end-to-end agent behavior.

## Update: Corpus scope narrowed to BIOMED, category introduced

The `ai-research-assistant` corpus has been narrowed to biomedical
content only (diabetes, cardiology, oncology — see that project's
`corpus_manifest.json`). The document-lookup path is therefore
BIOMED-specific: a non-biomed document question either misroutes to
`GENERAL` or hits empty/irrelevant retrieval.

**Decision:** introduce a dedicated `BIOMED` category rather than
routing biomed document questions through a generically-named category.
This is deliberate, not just a rename:

- It keeps `Category` an honest contract (per this ADR's own principle
  above — plain types/names that reflect actual runtime meaning),
  rather than a generic category quietly meaning "biomed document"
  everywhere it's read.
- It leaves room for biomed-specific behavior beyond routing — e.g. a
  prompt tuned for clinical terminology, or tool-specific handling —
  without overloading a generic category name.
- It matches the RAG service's own trajectory: that project is a
  domain-agnostic pipeline currently pointed at a biomed corpus — same
  distinction being drawn here at the agent-routing level.

**Implemented (2026-08-12):** `Category` includes `BIOMED`,
`classify_question`'s prompt and examples route biomed document
questions to it, and `route_decision`'s mapping is extended
accordingly. The fail-loud check (see above) covers any future gap in
that mapping automatically.

**Still open — biomed-specific behavior beyond routing.** Introducing
the category is the routing-level change only. The biomed-specific
*behavior* the decision anticipated is not yet built: a clinical-
terminology-tuned prompt on the document path, tool-specific handling,
or a finer `BIOMED_TASK` split distinguishing biomed document lookup
from a future biomed-specific task category. Tracked as future work in
the README's "Possible future improvements"; recorded here so the
category's introduction isn't mistaken for the full domain-specific
capability it leaves room for.

## Update: Conversation memory, query condensation, and the reducer trap

### Decision: `MemorySaver`, keyed by `thread_id`

In-memory checkpointing only — deliberate for a portfolio project, not
a durability requirement. Verified thread isolation directly: reusing
an established follow-up phrase as the first message on a fresh
`thread_id` correctly failed to resolve, confirming state isn't shared
across threads.

### Bug: `{**state, ...}` silently duplicates accumulating fields

`history` uses `Annotated[list[dict], add]` so LangGraph concatenates
returned lists rather than overwriting. But every node was returning
`{**state, ...}`, which re-includes the already-accumulated `history`
on every hop — the `add` reducer concatenates it again each time.
Observed: history length grew 1 → 5 → 21 instead of 1 → 2 → 3.

**Fix:** nodes now return only the fields they actually set, never a
full `**state` spread. **Principle:** any field with a non-default
reducer must be treated as write-only by nodes that don't modify it.

### Decision: `question` vs `resolved_question`

`condense_question` rewrites follow-ups into standalone questions;
downstream nodes read `resolved_question`. `record_turn` writes the
raw `question` into `history` instead — history also feeds future
condensation calls, so it should reflect the user's actual phrasing,
not an LLM-rewritten version.

### Decision: condensation reasoning is debug-only

`condense_question` returns structured output (`resolved_question` +
`reasoning`), consistent with this project's existing preference for
structured outputs over parsed text. `reasoning` is not written into
`history` — history's only consumer is future condensation prompts, and
debug commentary has no value there while adding token cost.


## Update: Retry trigger replaced — `context_sufficient` flag over `sources`

### Problem: the original trigger never fired

Retry (JUA-9/10) fired on `if not sources`. Testing showed this is
dead code: Chroma's `collection.query()` returns the *n* nearest
neighbours with no relevance cutoff, so there's always a "nearest"
chunk, however irrelevant. Confirmed in `eval_results_baseline.json` —
`unanswerable` queries return fully-populated sources at both n=3 and
n=8. Real failure mode is low relevance, not zero results.

### Decision: structured `context_sufficient` flag from the RAG service

The answering LLM already judges sufficiency in prose when it refuses
to answer — the fix captures that judgement as structured output
(`with_structured_output`, `GroundedAnswer` schema) instead of
discarding it.

`ai-agent-module` routes on the flag instead of source count:

- Both `answer_from_document` and `retry_document_lookup` write it —
  required, since `should_fallback_to_general` reads state *after*
  the retry node runs.
- Both edge functions default a missing key to `True`
  (never-retry / keep-the-answer). Load-bearing default, asserted in
  `tests/test_retry_stopping_condition.py`.
- `record_turn` resets the flag every turn to prevent leakage across
  questions.

**Verified** on an off-corpus question (HCR-FISH imaging, not in the
corpus): retry fired, rewrite still insufficient, fell back to
`answer_general` with the ungrounded caveat — no fabricated findings.

### Alternatives rejected

- **Reranker score** (already computed, discarded in `retrieve()`) —
  needs a calibration pass to find a threshold; corpus-dependent.
  Deferred, not ruled out.
- **Agent-side classification** of the answer text — asks a model
  that never saw the retrieved chunks to re-infer a judgement the
  RAG service already made and threw away. Extra LLM call, wrong
  layer.

### Known limitation

`context_sufficient` is an LLM self-assessment — not perfectly
reliable in either direction. Better than the alternatives evaluated,
not guaranteed.