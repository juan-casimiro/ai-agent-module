from unittest.mock import MagicMock

import pytest

from hello_langgraph import classify_question, Classification, Category


def _fake_llm(category: Category) -> MagicMock:
    fake_classification = Classification(category=category)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.return_value = fake_classification
    return fake_llm


@pytest.mark.parametrize("category", [
    Category.BIOMED,
    Category.CALCULATION,
    Category.GENERAL,
])
def test_classify_question_writes_category_value(category):
    state = {"resolved_question": "irrelevant, LLM call is faked"}
    result = classify_question(state, llm=_fake_llm(category))

    # The node must persist the plain .value string, never the raw enum
    # (ADR-001: the string-vs-enum bug this test exists to guard).
    assert result["classification"] == category.value
    assert isinstance(result["classification"], str)
    assert not isinstance(result["classification"], Category)