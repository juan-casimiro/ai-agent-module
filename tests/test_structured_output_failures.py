from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.exceptions import OutputParserException
import pytest

import hello_langgraph as agent


@pytest.mark.parametrize("stage", ["condensation", "classification", "calculation"])
@pytest.mark.parametrize("failure", [None, OutputParserException("test malformed model output")])
def test_structured_failure_stops_turn_without_recording_answer(monkeypatch, stage, failure):
    fake_llm = MagicMock()
    outputs = [agent.Classification(category=agent.Category.CALCULATION)] if stage == "calculation" else []
    fake_llm.with_structured_output.return_value.invoke.side_effect = outputs + [failure]
    monkeypatch.setattr(agent, "llm", fake_llm)
    prior_history = [{"question": "test prior question", "answer": "test prior answer"}] if stage == "condensation" else []
    config = {"configurable": {"thread_id": "test-structured-failure"}}
    expected_error = ValueError if failure is None else OutputParserException
    message = f"No structured {stage} output" if failure is None else "test malformed model output"

    with pytest.raises(expected_error, match=message):
        agent.app.invoke({"question": "test new question", "history": prior_history}, config=config)

    state = agent.app.get_state(config).values
    assert state.get("history", []) == prior_history
    assert "answer" not in state
    fake_llm.invoke.assert_not_called()


def test_unexpected_classification_fails_in_compiled_graph(monkeypatch):
    fake_llm = MagicMock()
    # Simulate schema/routing drift without asking Pydantic to accept invalid data.
    fake_llm.with_structured_output.return_value.invoke.return_value = SimpleNamespace(
        category=SimpleNamespace(value="TEST_UNMAPPED_CATEGORY"))
    monkeypatch.setattr(agent, "llm", fake_llm)
    with pytest.raises(ValueError, match="Unexpected classification 'TEST_UNMAPPED_CATEGORY'"):
        agent.app.invoke({"question": "test question"},
                         config={"configurable": {"thread_id": "test-unmapped"}})
    fake_llm.invoke.assert_not_called()
