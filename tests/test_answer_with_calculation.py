# tests/test_answer_with_calculation.py
from unittest.mock import MagicMock

import pytest

from hello_langgraph import answer_with_calculation, CalculationRequest


def _fake_llm(expression, decimal_places):
    fake_request = CalculationRequest(expression=expression, decimal_places=decimal_places)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.return_value = fake_request
    return fake_llm


def test_regex_rejects_unsafe_expression():
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    unsafe_expression = "__import__('os').system('ls')"
    result = answer_with_calculation(
        state, llm=_fake_llm(unsafe_expression, None)
    )
    assert result == {
        "answer": f"Unsafe or unparseable expression: {unsafe_expression}"
    }


def test_malformed_expression_returns_error_message():
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    malformed_expression = "(1 + 2"
    result = answer_with_calculation(state, llm=_fake_llm(malformed_expression, None))
    assert result["answer"].startswith("Invalid expression:")


@pytest.mark.parametrize("expression, decimal_places, expected_answer", [
    ("10 / 3", 2, "The calculation 10 / 3 = 3.33"),
    ("10 / 4", None, "The calculation 10 / 4 = 2.5"),
])
def test_decimal_places_rounding(expression, decimal_places, expected_answer):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    result = answer_with_calculation(state, llm=_fake_llm(expression, decimal_places))
    assert result == {"answer": expected_answer, "sources": []}