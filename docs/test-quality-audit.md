# Bounded test audit — JUA-81

Audited 2026-09-09 against baseline `7196eb0` (21 passing tests).
Final suite: **62 passing tests**, with fake model/RAG boundaries, fresh
in-memory checkpoints per test, and guards against accidental live calls.
No paid evaluations were run. No runtime dependencies, CI gates, or mutation
platform were added; coverage.py and Radon were installed locally for analysis.

## Scope and findings

The academic project's main value is orchestration. The broad issue wording
does not justify exhaustive CLI/evaluation coverage, production resilience
infrastructure, or replicating JUA-80. Reviewed the current Linear Working
Agreement and Repo Guide, completed routing/calculation/retry issues (including
JUA-6/7/10/16/17), JUA-20's evaluation, and ADR-001/002.

| Behavior | Protection and outcome |
| --- | --- |
| BIOMED routing and bounded retry | Reused existing compiled-graph success, recovery, and fallback tests. They assert call counts, query rewriting, retained/discarded sources, and the ungrounded fallback prompt. Existing predicate tests cover the retry cap and legacy missing-state defaults. |
| GENERAL and CALCULATION routing | New compiled multi-turn conversation exercises both paths, including a successful calculation and ordinary general answers without the fallback caveat. Unmapped classification also fails through the compiled graph. |
| History and turn state | Follow-up condensation receives prior questions/answers; the resolved question reaches classification/calculation. Standalone questions pass through sensibly. History records raw questions and final answers exactly once, excludes debug reasoning, and stays isolated by thread. A second document turn can retry after the first fell back. |
| Calculation | Reused/expanded tests for arithmetic, rounding (including zero decimal places), division by zero, malformed syntax, and unsafe expressions. Fixed stale document sources on all calculation error returns. Reject `**` and `//`, which previously passed the character allow-list despite exceeding the prompt's arithmetic contract. |
| RAG boundary | Added status checking and a small strict response model using existing Pydantic. Required `answer` must be a string; supplied sources and sufficiency fields must have the expected types. Tests cover non-success statuses, malformed JSON, missing answer, invalid field types, connection failures, and timeouts. Transport failures at either document attempt abort without general fallback or recording a completed turn. |
| Structured model output | Classification, condensation, and calculation now raise descriptive `ValueError`s for absent (`None`) structured output. Parsing exceptions continue to propagate. Compiled-graph tests assert that neither failure mode records an answer or invokes general fallback. No duplicate Pydantic/LangChain validation tests. |
| Evaluation | One small deterministic scoring test protects correct/misroute labels and confusion-matrix direction/zero filling. No live model calls or output-format polishing. |

RAG compatibility remains deliberate: absent `sources` defaults to `[]`, absent
`context_sufficient` to `True`, and absent `insufficiency_reason` to `None`.
Unknown extra response fields are ignored. These preserve the JUA-16 contract;
explicit malformed values are rejected. Errors propagate without adding
transport retries, circuit breakers, or an integration abstraction.

## Coverage and hotspots

Python 3.12; pytest 9.1.1; coverage.py 7.16.0; Radon 6.0.1. Statements and
branches below are reported separately. Coverage includes the entire two
production files, including demo/CLI code; only tests and conftest are omitted.

| File | Baseline statements | Final statements | Baseline branches | Final branches |
| --- | ---: | ---: | ---: | ---: |
| `hello_langgraph.py` | 122/180 (67.8%) | 142/193 (73.6%) | 13/30 (43.3%) | 21/36 (58.3%) |
| `eval_classification.py` | 0/76 (0%) | 25/76 (32.9%) | 0/18 (0%) | 3/18 (16.7%) |

All 12 core agent functions (the nodes, routing predicates, history formatter,
and query boundary) now have 100% measured statement/branch coverage. This is
a consequence of the selected behavioral tests, not a coverage target or proof
of exhaustive paths. Remaining agent misses are the unused demo health probe
and `__main__` demonstration. Exception paths and boolean operands are not
fully described by coverage.py branch counts; the negative tests matter too.

CRAP-style score: `CC² × (1 − statement_coverage_fraction)³ + CC`, using Radon
cyclomatic complexity and coverage.py's per-function statement coverage.
Nesting is maximum explicit Python control-block depth (if/loop/try/except/
with/match); comprehensions and boolean expressions are excluded. It is a
simple structural measure, not a claim to implement cognitive complexity.

| Function | CC before → after | Statement coverage before → after | CRAP before → after | Final nesting |
| --- | ---: | ---: | ---: | ---: |
| `answer_with_calculation` | 5 → 8 | 92.3% → 100% | 5.01 → 8.00 | 2 |
| `condense_question` | 2 → 3 | 50% → 100% | 2.50 → 3.00 | 1 |
| `format_history` | 2 → 2 | 0% → 100% | 6.00 → 2.00 | 0 |
| `query_document_service` | 1 → 1 | 0% → 100% | 2.00 → 1.00 | 0 |
| `check_document_service` | 4 → 4 | 0% → 0% | 20.00 → 20.00 | 2 |
| eval `run_one` | 1 → 1 | 0% → 100% | 2.00 → 1.00 | 0 |
| eval `build_confusion_matrix` | 4 → 4 | 0% → 100% | 20.00 → 4.00 | 1 |
| eval `print_confusion_matrix` | 4 → 4 | 0% → 0% | 20.00 → 20.00 | 1 |
| eval `print_summary` | 10 → 10 | 0% → 0% | 110.00 → 110.00 | 2 |
| eval `main` | 6 → 6 | 0% → 0% | 42.00 → 42.00 | 1 |

All other agent functions have CC ≤3, full statement coverage, and CRAP ≤3.
The calculation score rises because explicit defensive checks add branches;
there is no reason to refactor them merely to lower a metric. The highest
complexity/low-coverage hotspots are evaluation printing and CLI orchestration,
which are secondary to this audit. Maximum explicit nesting is 2 in both files.

## Representative effectiveness verification

Each selected test passed, then failed after the production change below,
then passed again after restoration. Four changes only; no exhaustive score
or claim of mutation adequacy. No intentional breakage is committed.

| Temporary breakage | Expected failure observed |
| --- | --- |
| Remove the `retried` guard from `should_retry_document_lookup` | Existing cap case returned `retry` instead of `no_retry` (1 failed, 3 passed). |
| Re-emit existing history in `record_turn` | Multi-turn history contained four extra entries instead of exactly three turns (1 failed). |
| Remove `response.raise_for_status()` | All four non-success response cases failed with “DID NOT RAISE”; an answer-shaped error body was accepted (4 failed). |
| Bypass the calculation allow-list | Seven unsafe/unsupported expression cases returned calculation/parser results instead of the required rejection (7 failed). |

## Reproduce lightweight analysis

From the repository root after installing `requirements.txt` into `.venv`:

```sh
.venv/bin/python -m pip install coverage==7.16.0 radon==6.0.1
mkdir -p /Users/juancasimiro/development/.agent-tmp
export COVERAGE_FILE=/Users/juancasimiro/development/.agent-tmp/jua81.coverage
.venv/bin/python -m coverage run --branch --source=. --omit='tests/*,conftest.py' -m pytest -q
.venv/bin/python -m coverage report -m
.venv/bin/python -m coverage json -o /Users/juancasimiro/development/.agent-tmp/jua81.json
.venv/bin/radon cc -s hello_langgraph.py eval_classification.py
```

For per-function hotspot scores and the stated nesting measure:

```sh
.venv/bin/python - <<'PY'
import ast, json
from pathlib import Path
from radon.complexity import cc_visit
data = json.loads(Path('/Users/juancasimiro/development/.agent-tmp/jua81.json').read_text())
blocks = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
          ast.ExceptHandler, ast.With, ast.AsyncWith, ast.Match)
def nesting(node, depth=0):
    depth += isinstance(node, blocks)
    return max([depth] + [nesting(child, depth) for child in ast.iter_child_nodes(node)])
for filename in ('hello_langgraph.py', 'eval_classification.py'):
    code = Path(filename).read_text()
    functions = {n.name: n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    for block in cc_visit(code):
        if block.name not in functions:
            continue
        summary = data['files'][filename]['functions'][block.name]['summary']
        fraction = summary['percent_statements_covered'] / 100
        crap = block.complexity ** 2 * (1 - fraction) ** 3 + block.complexity
        print(filename, block.name, 'CC', block.complexity,
              'CRAP', round(crap, 2), 'nesting', nesting(functions[block.name]))
PY
```

## Deliberately remaining risks

- Fake condensation/classification/calculation outputs prove data flow, not that
  an actual model resolves language correctly. Existing JUA-19/20 measurements
  remain separate evidence; real retry recovery remains unobserved.
- The fallback caveat is tested as a prompt instruction, not a guarantee that
  a live model obeys it or avoids hallucination.
- RAG/model failures abort the invocation. Checkpoints may contain partial
  node state from a failed turn; recovery/resumption UX is not redesigned.
- Legacy missing sufficiency defaults can accept an answer without an explicit
  sufficiency judgment. Intentional compatibility, not a grounding guarantee.
- No arithmetic resource-budget/numeric-overflow policy, model timeout-policy
  overhaul, persistent memory, or history truncation was added.
- The demo health probe has incomplete HTTP-error handling and no automated
  coverage. Evaluation printing/CLI remains untested; malformed datasets or a
  model failure abort evaluation before the final results artifact is written.
  These do not justify broadening this orchestration audit.
