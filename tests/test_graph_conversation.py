from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import hello_langgraph as agent


def test_followups_use_history_without_duplicating_it_and_threads_are_isolated(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.side_effect = [
        agent.Classification(category=agent.Category.GENERAL),
        agent.CondensedQuestion(resolved_question="What is 24 multiplied by 3?", reasoning="Use prior value"),
        agent.Classification(category=agent.Category.CALCULATION),
        agent.CalculationRequest(expression="24 * 3"),
        agent.CondensedQuestion(resolved_question="What is the capital of France?", reasoning="Already standalone"),
        agent.Classification(category=agent.Category.GENERAL),
        agent.Classification(category=agent.Category.GENERAL),
    ]
    fake_llm.invoke.side_effect = [SimpleNamespace(content="The increase was 24%."),
                                   SimpleNamespace(content="Paris."), SimpleNamespace(content="Please clarify.")]
    monkeypatch.setattr(agent, "llm", fake_llm)
    config = {"configurable": {"thread_id": "test-conversation"}}

    first = agent.app.invoke({"question": "What was the increase?"}, config=config)
    second = agent.app.invoke({"question": "What's that times 3?"}, config=config)
    third = agent.app.invoke({"question": "What is the capital of France?"}, config=config)
    fresh = agent.app.invoke({"question": "What's that times 3?"},
                             config={"configurable": {"thread_id": "test-fresh"}})

    assert first["resolved_question"] == "What was the increase?"
    assert second["resolved_question"] == "What is 24 multiplied by 3?"
    assert second["condensation_reasoning"] == "Use prior value"
    assert second["answer"] == "The calculation 24 * 3 = 72"
    assert third["resolved_question"] == "What is the capital of France?"
    assert third["history"] == [
        {"question": "What was the increase?", "answer": "The increase was 24%."},
        {"question": "What's that times 3?", "answer": "The calculation 24 * 3 = 72"},
        {"question": "What is the capital of France?", "answer": "Paris."},
    ]
    assert fresh["history"] == [{"question": "What's that times 3?", "answer": "Please clarify."}]
    assert fresh["resolved_question"] == "What's that times 3?"
    prompts = fake_llm.with_structured_output.return_value.invoke.call_args_list
    assert "Q: What was the increase?\nA: The increase was 24%." in prompts[1].args[0]
    assert "What's that times 3?" in prompts[1].args[0]
    assert "What is 24 multiplied by 3?" in prompts[2].args[0]
    assert "What is 24 multiplied by 3?" in prompts[3].args[0]
    assert "Q: What's that times 3?\nA: The calculation 24 * 3 = 72" in prompts[4].args[0]
    assert all(agent.UNGROUNDED_FALLBACK_PROMPT not in invocation.args[0][0].content
               for invocation in fake_llm.invoke.call_args_list)


@pytest.mark.parametrize("expression, answer_prefix", [
    ("10 / 0", "Error: division by zero"),
    ("some_name", "Unsafe or unparseable expression:"),
    ("(1 + 2", "Invalid expression:"),
])
def test_calculation_error_clears_previous_document_sources(monkeypatch, expression, answer_prefix):
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.side_effect = [
        agent.Classification(category=agent.Category.BIOMED),
        agent.CondensedQuestion(resolved_question="test calculation", reasoning="test resolution"),
        agent.Classification(category=agent.Category.CALCULATION),
        agent.CalculationRequest(expression=expression),
    ]
    monkeypatch.setattr(agent, "llm", fake_llm)
    rag_query = MagicMock(return_value={"answer": "test grounded answer", "sources": ["test.pdf"]})
    monkeypatch.setattr(agent, "query_document_service", rag_query)
    config = {"configurable": {"thread_id": "test-stale-sources"}}
    agent.app.invoke({"question": "test biomedical question"}, config=config)
    result = agent.app.invoke({"question": "test arithmetic question"}, config=config)
    assert result["answer"].startswith(answer_prefix)
    assert result["sources"] == []
    assert len(result["history"]) == 2
    assert result["history"][-1] == {"question": "test arithmetic question", "answer": result["answer"]}
    rag_query.assert_called_once()
    fake_llm.invoke.assert_not_called()


def test_retry_flags_reset_so_next_document_turn_can_retry(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.side_effect = [
        agent.Classification(category=agent.Category.BIOMED),
        agent.CondensedQuestion(resolved_question="test second resolved question", reasoning="test resolution"),
        agent.Classification(category=agent.Category.BIOMED),
    ]
    fake_llm.invoke.return_value = SimpleNamespace(content="test ungrounded fallback")
    monkeypatch.setattr(agent, "llm", fake_llm)
    insufficient = {"answer": "test partial answer", "context_sufficient": False,
                    "sources": ["test.pdf"], "insufficiency_reason": "test missing evidence"}
    rag_query = MagicMock(side_effect=[insufficient, insufficient, insufficient,
                                       {"answer": "test recovered answer", "context_sufficient": True,
                                        "sources": ["recovered.pdf"]}])
    monkeypatch.setattr(agent, "query_document_service", rag_query)
    config = {"configurable": {"thread_id": "test-two-document-turns"}}
    states = list(agent.app.stream({"question": "test first question"}, config=config, stream_mode="values"))
    assert any(s.get("retried") is True and s.get("context_sufficient") is False for s in states)
    first = states[-1]
    assert first["retried"] is False
    assert first["context_sufficient"] is True
    assert first["retry_reason"] is None
    second = agent.app.invoke({"question": "test second question"}, config=config)
    assert second["answer"] == "test recovered answer"
    assert second["sources"] == ["recovered.pdf"]
    assert len(second["history"]) == 2
    assert rag_query.call_args_list == [
        call("test first question"), call("test first question", use_query_rewriting=True),
        call("test second resolved question"), call("test second resolved question", use_query_rewriting=True),
    ]
    fake_llm.invoke.assert_called_once()
