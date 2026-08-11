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

## Update: Corpus scope narrowed to BIOMED, category split planned

The `ai-research-assistant` corpus has been narrowed to biomedical
content only (diabetes, cardiology, oncology — see that project's
`corpus_manifest.json`). The current `Category` enum (`BIOMED`,
`CALCULATION`, `GENERAL`) still treats document-lookup as
domain-agnostic, which is no longer an honest reflection of what the
underlying service can actually answer.

**Decision:** introduce a dedicated `BIOMED` category rather than
continuing to route biomed document questions through a generically-
named `BIOMED` category. This is deliberate, not just a rename:

- It keeps `Category` an honest contract (per this ADR's own principle
  above — plain types/names that reflect actual runtime meaning),
  rather than `BIOMED` quietly meaning "biomed document" everywhere
  it's read.
- It leaves room for biomed-specific behavior beyond routing — e.g. a
  prompt tuned for clinical terminology, or tool-specific handling —
  without overloading a generic category name.
- It matches the RAG service's own trajectory: that project is a
  domain-agnostic pipeline currently pointed at a biomed corpus — same
  distinction being drawn here at the agent-routing level.

**Not yet implemented** — this is a recorded decision for upcoming
work, not a completed change. When implemented: `Category` gains
`BIOMED`, `classify_question`'s prompt and examples are updated to
distinguish biomed document questions from general document questions
(if `GENERAL` document-lookup returns), and `route_decision`'s mapping
is extended accordingly. The fail-loud check (see above) should catch
any gap in that mapping automatically, per its original purpose.