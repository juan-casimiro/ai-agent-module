import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

from types import SimpleNamespace
from unittest.mock import MagicMock

import hello_langgraph
from hello_langgraph import Category, Classification


FALLBACK_ANSWER = (
    "This is not grounded in the document corpus. Based on general "
    "knowledge only: [placeholder answer]."
)
RECOVERED_ANSWER = "Grounded answer from the reworded retry."
DIRECT_ANSWER = "Grounded answer, first attempt."


def _fake_llm(invoke_content: str = FALLBACK_ANSWER) -> MagicMock:
    """BIOMED classification always; llm.invoke() only matters when
    answer_general actually runs (fallback scenario)."""
    fake = MagicMock()
    fake.with_structured_output.return_value.invoke.return_value = Classification(
        category=Category.BIOMED
    )
    fake.invoke.return_value = SimpleNamespace(content=invoke_content)
    return fake


def test_insufficient_twice_falls_back_to_general(monkeypatch):
    """Proves the loop terminates and the caveat reaches answer_general.

    Insufficient on the first attempt, still insufficient on the retry.
    side_effect has exactly two entries, so a third call raises
    StopIteration — that's the termination proof, not just an assertion.
    Checks: two service calls, second uses query rewriting, grounded
    content is discarded (sources == []), one history entry, and the
    UNGROUNDED_FALLBACK_PROMPT text is present in the system message
    passed to the LLM.
    """
    fake_llm = _fake_llm(invoke_content=FALLBACK_ANSWER)
    monkeypatch.setattr(hello_langgraph, "llm", fake_llm)

    service = MagicMock(
        side_effect=[
            {
                "answer": "partial attempt 1",
                "sources": ["a.pdf"],
                "context_sufficient": False,
                "insufficiency_reason": "no direct coverage",
            },
            {
                "answer": "partial attempt 2",
                "sources": ["a.pdf"],
                "context_sufficient": False,
                "insufficiency_reason": "still insufficient",
            },
        ]
    )
    monkeypatch.setattr(hello_langgraph, "query_document_service", service)

    result = hello_langgraph.app.invoke(
        {"question": "What does the corpus say about imaging biomarkers in cardiology?"},
        config={"configurable": {"thread_id": "test-fallback-v1"}},
    )

    assert service.call_count == 2
    assert service.call_args_list[1].kwargs.get("use_query_rewriting") is True

    assert result["answer"] == FALLBACK_ANSWER
    assert result["sources"] == []
    assert len(result["history"]) == 1

    system_message = fake_llm.invoke.call_args.args[0][0]
    assert hello_langgraph.UNGROUNDED_FALLBACK_PROMPT in system_message.content


def test_insufficient_then_sufficient_recovers(monkeypatch):
    """Proves the retry can actually succeed — never observed in a real run.

    Insufficient first, sufficient on the reworded retry. The grounded
    answer from the retry must be kept, answer_general must never run
    (llm.invoke untouched), and the real sources must survive.
    """
    fake_llm = _fake_llm()
    monkeypatch.setattr(hello_langgraph, "llm", fake_llm)

    service = MagicMock(
        side_effect=[
            {
                "answer": "partial attempt 1",
                "sources": [],
                "context_sufficient": False,
                "insufficiency_reason": "no direct coverage",
            },
            {
                "answer": RECOVERED_ANSWER,
                "sources": ["b.pdf"],
                "context_sufficient": True,
            },
        ]
    )
    monkeypatch.setattr(hello_langgraph, "query_document_service", service)

    result = hello_langgraph.app.invoke(
        {"question": "What does the corpus say about diabetic retinopathy screening intervals?"},
        config={"configurable": {"thread_id": "test-recovery-v1"}},
    )

    assert service.call_count == 2
    assert service.call_args_list[1].kwargs.get("use_query_rewriting") is True

    assert result["answer"] == RECOVERED_ANSWER
    assert result["sources"] == ["b.pdf"]
    assert len(result["history"]) == 1

    # answer_general must never have run on the recovery path
    assert fake_llm.invoke.call_count == 0


def test_sufficient_first_time_no_retry(monkeypatch):
    """Proves the happy path doesn't retry when it doesn't need to.

    Sufficient context on the first attempt: exactly one service call,
    no retry, answer_general never invoked.
    """
    fake_llm = _fake_llm()
    monkeypatch.setattr(hello_langgraph, "llm", fake_llm)

    service = MagicMock(
        return_value={
            "answer": DIRECT_ANSWER,
            "sources": ["c.pdf"],
            "context_sufficient": True,
        }
    )
    monkeypatch.setattr(hello_langgraph, "query_document_service", service)

    result = hello_langgraph.app.invoke(
        {"question": "What does the corpus say about oncology biomarker panels?"},
        config={"configurable": {"thread_id": "test-no-retry-v1"}},
    )

    assert service.call_count == 1
    assert result["answer"] == DIRECT_ANSWER
    assert result["sources"] == ["c.pdf"]
    assert len(result["history"]) == 1
    assert fake_llm.invoke.call_count == 0