import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import pytest

from hello_langgraph import (
    Node,
    should_fallback_to_general,
    should_retry_document_lookup,
)


@pytest.mark.parametrize(
    "state, expected",
    [
        # Insufficient context, not yet retried — the one case that retries.
        ({"context_sufficient": False}, "retry"),
        # Retry cap: insufficient again after a retry must not loop.
        ({"context_sufficient": False, "retried": True}, "no_retry"),
        # Sufficient context — no reason to retry.
        ({"context_sufficient": True}, "no_retry"),
        # Missing key degrades to never-retry. Load-bearing default.
        ({}, "no_retry"),
    ],
)
def test_should_retry_document_lookup(state, expected):
    assert should_retry_document_lookup(state) == expected


@pytest.mark.parametrize(
    "state, expected",
    [
        # Rewrite didn't help — discard the ungrounded answer, fall back.
        ({"context_sufficient": False}, Node.GENERAL_PATH.value),
        # Rewrite found sufficient context — keep the grounded answer.
        ({"context_sufficient": True}, "record_turn"),
        # Missing key degrades to keeping the answer. Load-bearing default.
        ({}, "record_turn"),
    ],
)
def test_should_fallback_to_general(state, expected):
    assert should_fallback_to_general(state) == expected