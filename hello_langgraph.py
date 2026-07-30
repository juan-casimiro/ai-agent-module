from enum import Enum
from typing import TypedDict, NotRequired
import httpx
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class Node(Enum):
    CLASSIFY = "classify"
    DOCUMENT_PATH = "document_path"
    GENERAL_PATH = "general_path"

class GraphState(TypedDict):
    question: str
    classification: NotRequired[str]
    answer: NotRequired[str]


def classify_question(state: GraphState) -> GraphState:
    response = llm.invoke(
        f"Does answering this question require looking up specific facts from "
        f"a document, or can it be answered from general knowledge? "
        f"Respond with exactly one word: DOCUMENT or GENERAL.\n\n"
        f"Question: {state['question']}"
    )
    return {**state, "classification": response.content.strip()}


def answer_general(state: GraphState) -> GraphState:
    response = llm.invoke(state["question"])
    return {**state, "answer": response.content}


def answer_from_document(state: GraphState) -> GraphState:
    response = httpx.post(
        "http://localhost:8000/query",
        json={"question": state["question"], "n_results": 8},
    )
    result = response.json()
    return {**state, "answer": result["answer"]}

def route_decision(state: GraphState) -> str:
    return Node.DOCUMENT_PATH.value if "DOCUMENT" in state["classification"] else Node.GENERAL_PATH.value


graph = StateGraph(GraphState)
graph.add_node(Node.CLASSIFY.value, classify_question)
graph.add_node(Node.DOCUMENT_PATH.value, answer_from_document)
graph.add_node(Node.GENERAL_PATH.value, answer_general)

graph.set_entry_point(Node.CLASSIFY.value)
graph.add_conditional_edges(
    Node.CLASSIFY.value,
    route_decision,
    {
        Node.DOCUMENT_PATH.value: Node.DOCUMENT_PATH.value,
        Node.GENERAL_PATH.value: Node.GENERAL_PATH.value,
    }
)
graph.add_edge(Node.DOCUMENT_PATH.value, END)
graph.add_edge(Node.GENERAL_PATH.value, END)

app = graph.compile()

if __name__ == "__main__":
    for q in [
        "What is the capital of France?",
        "What did the University of Minnesota Law School study find about ChatGPT?",
    ]:
        result = app.invoke({"question": q})
        print(
            f"Q: {q}\nClassification: {result['classification']}\nA: {result['answer']}\n"
        )
