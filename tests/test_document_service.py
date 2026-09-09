from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

import hello_langgraph as agent


def response(payload, status=200):
    return httpx.Response(status, json=payload,
                          request=httpx.Request("POST", "http://test-rag/query"))


@pytest.mark.parametrize("rewrite", [False, True])
def test_document_request_and_legacy_response_defaults(monkeypatch, rewrite):
    rag_post = MagicMock(return_value=response({"answer": "test grounded answer"}))
    monkeypatch.setattr(agent.httpx, "post", rag_post)
    monkeypatch.setattr(agent, "RAG_SERVICE_URL", "http://test-rag")

    result = agent.query_document_service("test resolved question", rewrite)

    rag_post.assert_called_once_with(
        "http://test-rag/query",
        json={"question": "test resolved question", "n_results": 8,
              "use_query_rewriting": rewrite}, timeout=60.0,
    )
    assert result == {"answer": "test grounded answer", "sources": [],
                      "context_sufficient": True, "insufficiency_reason": None}


@pytest.mark.parametrize("status", [302, 400, 500, 503])
def test_non_success_status_is_not_accepted_as_an_answer(monkeypatch, status):
    # Even an answer-shaped error body must not become a successful graph turn.
    monkeypatch.setattr(agent.httpx, "post", MagicMock(return_value=response(
        {"answer": "test error body"}, status)))
    with pytest.raises(httpx.HTTPStatusError):
        agent.query_document_service("test question")


@pytest.mark.parametrize("payload", [
    {}, {"answer": None}, {"answer": 42}, [],
    {"answer": "test answer", "context_sufficient": "false"},
    {"answer": "test answer", "context_sufficient": None},
    {"answer": "test answer", "sources": "test.pdf"},
    {"answer": "test answer", "sources": [42]},
])
def test_malformed_response_fields_fail_at_boundary(monkeypatch, payload):
    monkeypatch.setattr(agent.httpx, "post", MagicMock(return_value=response(payload)))
    with pytest.raises(ValidationError):
        agent.query_document_service("test question")


def test_malformed_json_fails_at_boundary(monkeypatch):
    malformed = httpx.Response(200, content=b"not JSON",
                              request=httpx.Request("POST", "http://test-rag/query"))
    monkeypatch.setattr(agent.httpx, "post", MagicMock(return_value=malformed))
    with pytest.raises(ValueError):
        agent.query_document_service("test question")


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
@pytest.mark.parametrize("on_retry", [False, True])
def test_transport_failure_aborts_graph_without_general_fallback(monkeypatch, failure, on_retry):
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.invoke.return_value = agent.Classification(
        category=agent.Category.BIOMED)
    monkeypatch.setattr(agent, "llm", fake_llm)
    first_attempt = response({"answer": "test insufficient answer", "sources": ["test.pdf"],
                              "context_sufficient": False, "insufficiency_reason": "test gap"})
    rag_post = MagicMock(side_effect=([first_attempt] if on_retry else []) + [failure("test failure")])
    monkeypatch.setattr(agent.httpx, "post", rag_post)
    config = {"configurable": {"thread_id": "test-transport-failure"}}

    with pytest.raises(failure, match="test failure"):
        agent.app.invoke({"question": "test biomedical question"}, config=config)

    assert rag_post.call_count == (2 if on_retry else 1)
    fake_llm.invoke.assert_not_called()
    assert agent.app.get_state(config).values.get("history", []) == []
