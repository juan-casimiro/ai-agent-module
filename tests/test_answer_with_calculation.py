from unittest.mock import MagicMock

import pytest

from hello_langgraph import answer_with_calculation, CalculationRequest
import hello_langgraph


def _fake_llm(expression, decimal_places):
    fake_request = CalculationRequest(expression=expression, decimal_places=decimal_places)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.return_value = fake_request
    return fake_llm


@pytest.mark.parametrize("unsafe_expression", [
    "__import__('os').system('ls')", "round(1.25, 1)", "24%", "some_name + 1",
    "2 ** 8", "7 // 2", "",
])
def test_regex_rejects_unsafe_expression(monkeypatch, unsafe_expression):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    monkeypatch.setattr(hello_langgraph, "llm", _fake_llm(unsafe_expression, None))
    result = answer_with_calculation(state)
    assert result == {
        "answer": f"Unsafe or unparseable expression: {unsafe_expression}",
        "sources": [],
    }


def test_malformed_expression_returns_error_message(monkeypatch):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    malformed_expression = "(1 + 2"
    monkeypatch.setattr(hello_langgraph, "llm", _fake_llm(malformed_expression, None))

    result = answer_with_calculation(state)
    assert result["answer"].startswith("Invalid expression:")
    assert result["sources"] == []


def test_division_by_zero(monkeypatch):
    monkeypatch.setattr(hello_langgraph, "llm", _fake_llm("10 / 0", None))
    assert answer_with_calculation({"resolved_question": "test arithmetic question"}) == {
        "answer": "Error: division by zero", "sources": [],
    }


@pytest.mark.parametrize("expression, decimal_places, expected_answer", [
    ("10 / 3", 2, "The calculation 10 / 3 = 3.33"),
    ("10 / 4", None, "The calculation 10 / 4 = 2.5"),
    ("(2 + 3) * -4", None, "The calculation (2 + 3) * -4 = -20"),
    ("10 / 3", 0, "The calculation 10 / 3 = 3.0"),
])
def test_decimal_places_rounding(expression, decimal_places, expected_answer, monkeypatch):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    monkeypatch.setattr(hello_langgraph, "llm", _fake_llm(expression, decimal_places))

    result = answer_with_calculation(state)
    assert result == {"answer": expected_answer, "sources": []}
