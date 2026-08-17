# ADR-002: Recursive Self-Correcting Retry Loop

## Context

The document-lookup path calls out to a separately-evaluated RAG
service (`ai-research-assistant`). Retrieval sometimes returns content
that isn't enough to answer the question confidently. This ADR covers
how the agent detects that, what it does about it, and — because the
retry's chosen strategy has a genuine cross-repo contradiction attached
to it — why that strategy was kept anyway.

## Decision: trigger — `context_sufficient` over source-emptiness

The retry fires on a structured `context_sufficient` flag returned by
the RAG service alongside the answer, not on `sources` being empty.

This replaced an earlier `not sources` trigger that turned out to be
empirically unreachable: Chroma's `collection.query()` returns the *n*
nearest neighbours with no relevance cutoff, so there is always a
"nearest" chunk, however irrelevant. Confirmed directly against
`ai-research-assistant`'s own eval data — `unanswerable` queries return
fully-populated `sources` at both n=3 and n=8. The real failure mode is
low relevance, not zero results, and no source-count check can see that.

`context_sufficient` works because it captures a judgement the
answering LLM was already making in prose when it refused to answer —
the fix turns that into structured output instead of discarding it. See
`ai-research-assistant`'s ADR-001 for the RAG-service-side
implementation; this repo only consumes the flag.

## Decision: single retry via query rewriting — and the cross-repo tension

`retry_document_lookup` retries exactly once, with `use_query_rewriting=True`.

This needs to be stated plainly rather than glossed over: `ai-research-assistant`'s
own golden QA evaluation found query rewriting to be a **measured no-op**
on its corpus — zero verdict changes across 111 scored queries, at both
n=3 and n=8, re-confirmed after the corpus grew from 16 to 19 documents (outliner cluster).
Rewriting is consequently opt-in there and defaults to `False`. This
project turns the same flag on unconditionally on every retry. Read
both repos back to back and that's a contradiction with nothing
bridging it — so here's the bridge.

**Primary reasoning: the eval doesn't measure what the retry depends on.**
`eval_golden.py`'s `score_query` compares *document identity* —
`expected_doc in sources`. That's what "verdict" means in "zero verdict
changes." `context_sufficient` is a different, finer-grained signal:
the answering model's judgement over the retrieved *chunk text and
composition*, not which documents came back. `ai-research-assistant`'s
own ADR-001 already flags that the harness compares verdicts, not raw
`retrieved_sources` ordering — so "rewriting didn't change which
documents surface" says nothing about whether it changed chunk
selection or ordering *within* a correctly-identified document, which
is exactly what `context_sufficient` reads. The null result and the
retry's dependency are measuring two different things.

**This cuts both ways, and that has to be said honestly.** The eval
can't confirm rewriting helps `context_sufficient` — but by the same
logic, it equally can't confirm it doesn't. This is an **unmeasured**
lever, not a vindicated one. The actual way to settle it is to measure
`context_sufficient` accuracy directly against golden QA ground truth
(tracked as future work — see the project's open issues) rather than
inferring an answer from a metric that was never testing this question.

**Secondary reasoning, explicitly weaker and labelled as inference, not
fact.** Golden QA queries were authored from each document's
`abstract_summary` and full text — so their vocabulary is anchored to
the sources themselves. That plausibly *under-samples* register or
phrasing mismatches (typos, casual wording vs. academic phrasing) —
the specific failure mode rewriting is meant to catch. This is a real
sampling bias in how the eval set was built. It is not proof that
rewriting would help on such queries, only that the eval set wasn't
built to test that case. Presented here as a plausible reason the null
result might not generalize, not as a rebuttal of it.

**What the retry actually demonstrates, stated without inflation:**
control flow and graceful degradation — a bounded single retry, clean
termination, a fallback to `answer_general` carrying an explicit
ungrounded caveat, and no fabricated findings. That is proven, by both
a mocked graph-level test suite and one live run against the real
service. It is *not* a demonstrated retrieval-quality improvement, and
this document does not claim it is one.

## Decision: fallback and the structural retry cap

Both `should_retry_document_lookup` and `should_fallback_to_general`
default a missing `context_sufficient` key to `True` — never-retry,
keep-the-answer. This default is load-bearing, not incidental: it's
what stops a missing or malformed RAG response from looping or
silently discarding a possibly-good answer. Asserted directly in
`tests/test_retry_stopping_condition.py`.

`record_turn` resets both `retried` and `context_sufficient` at the end
of every turn, so a flag from one question can't leak into the next.

## Alternatives considered and rejected

- **LLM-judged answer quality.** A fresh model call judging the
  generated answer text, without access to the retrieved chunks. This
  re-infers a judgement the RAG service already made — with strictly
  *less* information than the service had (text only, not the chunks
  it was grounded in) — for the cost of an extra LLM call. Wrong layer:
  the component that read the chunks should own the sufficiency
  judgement, not a downstream consumer of its output.
- **A second retry, or a different second-attempt strategy.** Rejected
  for scope discipline. Adding an unvalidated new component on top of
  an already-unmeasured lever compounds uncertainty instead of
  resolving it.
- **Reranker score threshold.** The cross-encoder score is already
  computed during retrieval and discarded. Using it as a sufficiency
  signal was considered, but needs a calibration pass to find a
  workable threshold, and that threshold would be corpus-dependent.
  Deferred, not ruled out — worth revisiting if `context_sufficient`'s
  own accuracy turns out to be weak once measured.

## Verification status: two distinct claims, not one

- **"The loop terminates correctly and degrades gracefully."** Proven.
  Confirmed by mocked graph-level tests covering the fallback,
  recovery, and no-retry-needed paths (`tests/test_graph_retry_integration.py`),
  and by one live run against the real RAG service, which observed the
  fallback path end to end with no mocking.
- **"The retry can recover a real, previously-insufficient query."**
  Not observed in production. The one real (non-mocked) verification
  run used an off-corpus question (HCR-FISH imaging) that no retrieval
  strategy could have recovered — it isn't in the corpus, by
  construction. The recovery branch is proven correct only in
  isolation, via a mocked test that hardcodes a sufficient response on
  the second attempt. Whether that branch fires on a real,
  genuinely-recoverable query has not been tested.

These are different claims and this document does not let the first
stand in for the second.

## Cross-references

- `ai-research-assistant`'s ADR-001 — the rewriting evaluation this
  section's primary argument responds to (zero verdict changes, 111
  scored queries, two corpus sizes, opt-in and default `False` there).
- `tests/test_graph_retry_integration.py` — mocked proof of loop
  termination and both branch outcomes.
- `tests/test_retry_stopping_condition.py` — the load-bearing
  missing-key defaults.
- The live demo scenario in `hello_langgraph.py` — the one real-world
  verification run on record.