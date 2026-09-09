import os
from types import SimpleNamespace

import httpx
import pytest

# Set before test modules import the agent; no developer credentials or tracing.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def isolated_boundaries(monkeypatch):
    import hello_langgraph
    from langgraph.checkpoint.memory import MemorySaver

    def unexpected_call(*args, **kwargs):
        pytest.fail("Unexpected live model or HTTP call: mock the boundary")

    monkeypatch.setattr(hello_langgraph, "llm", SimpleNamespace(
        invoke=unexpected_call, with_structured_output=unexpected_call))
    monkeypatch.setattr(httpx.Client, "send", unexpected_call)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_call)
    monkeypatch.setattr(hello_langgraph, "app", hello_langgraph.graph.compile(checkpointer=MemorySaver()))
