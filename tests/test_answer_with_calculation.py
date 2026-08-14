from unittest.mock import MagicMock

import pytest

from hello_langgraph import answer_with_calculation, CalculationRequest
import hello_langgraph


def _fake_llm(expression, decimal_places):
    fake_request = CalculationRequest(expression=expression, decimal_places=decimal_places)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.return_value = fake_request
    return fake_llm


def test_regex_rejects_unsafe_expression(monkeypatch):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    unsafe_expression = "__import__('os').system('ls')"
    monkeypatch.setattr(hello_langgraph, "llm", _fake_llm(unsafe_expression, None))
    result = answer_with_calculation(state)
    assert result == {
        "answer": f"Unsafe or unparseable expression: {unsafe_expression}"
    }


def test_malformed_expression_returns_error_message(monkeypatch):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    malformed_expression = "(1 + 2"
    monkeypatch.setattr(hello_langgraph, "llm", _fake_llm(malformed_expression, None))

    result = answer_with_calculation(state)
    assert result["answer"].startswith("Invalid expression:")


@pytest.mark.parametrize("expression, decimal_places, expected_answer", [
    ("10 / 3", 2, "The calculation 10 / 3 = 3.33"),
    ("10 / 4", None, "The calculation 10 / 4 = 2.5"),
])
def test_decimal_places_rounding(expression, decimal_places, expected_answer, monkeypatch):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    monkeypatch.setattr(hello_langgraph, "llm", _fake_llm(expression, decimal_places))

    result = answer_with_calculation(state)
    assert result == {"answer": expected_answer, "sources": []}