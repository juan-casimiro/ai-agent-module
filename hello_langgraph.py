from enum import Enum
from operator import add
import re
from textwrap import dedent
from typing import Annotated, Literal, Optional, TypedDict, NotRequired
import httpx
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from simpleeval import simple_eval
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class Node(Enum):
    CLASSIFY = "classify"
    DOCUMENT_PATH = "document_path"
    GENERAL_PATH = "general_path"
    CALCULATION_PATH = "calculation_path"

class Category(Enum):
    BIOMED = "BIOMED"
    CALCULATION = "CALCULATION"
    GENERAL = "GENERAL"    

class GraphState(TypedDict):
    question: str
    resolved_question: NotRequired[str]
    classification: NotRequired[str]
    answer: NotRequired[str]
    sources: NotRequired[list[str]]
    history: Annotated[list[dict], add]

class CalculationRequest(BaseModel):
    expression: str
    decimal_places: Optional[int] = None

class Classification(BaseModel):
    category: Category

def condense_question(state: GraphState) -> GraphState:
    history = state.get("history", [])
    if not history:
        return {"resolved_question": state["question"]}

    CONDENSE_PROMPT = dedent(f"""\
        # Role
        You resolve a follow-up question into a fully standalone question, \
        using the conversation history for context. You do not answer the \
        question.

        # Task
        Given the conversation history and a new follow-up question, rewrite \
        the follow-up into a standalone question that makes sense with NO \
        prior context — resolving pronouns, ellipsis ("what about X?"), and \
        implicit references. If the new question is already standalone, \
        return it unchanged. If the follow-up references a specific number \
        or fact stated in a previous answer, include that concrete value \
        explicitly in the rewritten question.

        # History
        {format_history(history)}

        # New question
        {state['question']}

        # Output
        Return ONLY the rewritten standalone question, nothing else.
        """)

    resolved = llm.invoke(CONDENSE_PROMPT).content
    return {"resolved_question": resolved}

def classify_question(state: GraphState) -> GraphState:
    CLASSIFICATION_PROMPT = dedent(f"""\
        # Role
        You are a routing assistant for a multi-tool system. Your job is to decide \
        which specialized tool should handle a user's question — not to answer it \
        yourself.

        # Task
        Classify the question into exactly one category:
        - {Category.BIOMED.value}: the question asks about specific findings, data, \
        or content from biomedical/clinical research literature (e.g., a named \
        study, trial, or clinical review) — not something you'd know from general \
        knowledge, and specifically biomedical in subject matter.
        - {Category.CALCULATION.value}: the question requires precise arithmetic or \
        numeric computation to answer correctly.
        - {Category.GENERAL.value}: the question can be answered confidently from \
        general knowledge, with no document lookup or precise computation needed.

        # Constraints
        - Choose exactly one category, even if the question could arguably fit \
        more than one — pick the category that best represents the PRIMARY task \
        needed to answer it.
        - Do not attempt to answer the question. Only classify it.

        # Examples
        Question: "What is the capital of France?"
        -> {Category.GENERAL.value}

        Question: "What AUC did Attia et al.'s CNN model achieve for identifying \
        patients with prevalent AF during sinus rhythm from standard 12-lead ECGs?"
        -> {Category.BIOMED.value}

        Question: "What did the company's Q3 earnings report say about iPhone \
        sales?"
        # Not biomedical, and not in any ingested corpus — document-shaped
        # phrasing alone doesn't mean a document lookup is possible.
        -> {Category.GENERAL.value}

        Question: "What is 15% of 340, rounded to the nearest whole number?"
        -> {Category.CALCULATION.value}

        # Input
        Question: {state['resolved_question']}
        """)

    classification = llm.with_structured_output(Classification).invoke(CLASSIFICATION_PROMPT)

    return {"classification": classification.category.value}

def record_turn(state: GraphState) -> GraphState:
    return {
        "history": [{"question": state["question"], "answer": state["answer"]}],
    }

def format_history(history: list[dict]) -> str:
    return "\n\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history
    )

def answer_general(state: GraphState) -> GraphState:
    response = llm.invoke(state["resolved_question"])
    return {"answer": response.content, "sources": []}


def answer_from_document(state: GraphState) -> GraphState:
    response = httpx.post(
        "http://localhost:8000/query",
        json={"question": state["resolved_question"], "n_results": 8},
    )
    result = response.json()
    return {"answer": result["answer"], "sources": result.get("sources", [])}

def answer_with_calculation(state: GraphState) -> GraphState:
    CALCULATION_PROMPT = dedent(f"""\
        # Role
        You are a precise calculation-extraction assistant. Your job is to parse \
        natural language questions into a structured arithmetic request, not to \
        perform the calculation yourself.

        # Task
        Given a user's question, extract:
        1. The mathematical expression, written using standard arithmetic \
        operators only (+, -, *, /, parentheses) — never words like "times" or \
        "plus".
        2. Any rounding or decimal-precision instruction explicitly stated in \
        the question.

        # Constraints
        - The expression must contain ONLY numbers, +, -, *, /, and parentheses. \
        Never include function calls, variable names, or any other syntax.
        - If the question does not mention rounding or precision at all, set \
        decimal_places to null — do not assume a default.

        # Examples
        Question: "What is 15% of 200?"
        -> expression: "200 * 0.15", decimal_places: null

        Question: "What's 91 times 1.15, rounded to two decimal places?"
        -> expression: "91 * 1.15", decimal_places: 2

        # Input
        Question: {state['resolved_question']}
        """)

    calc_request = llm.with_structured_output(CalculationRequest).invoke(CALCULATION_PROMPT)

    if not re.fullmatch(r"[\d\s+\-*/().]+", calc_request.expression):
        return {"answer": f"Unsafe or unparseable expression: {calc_request.expression}"}

    try:
        result = simple_eval(calc_request.expression)
    except ZeroDivisionError:
        return {"answer": "Error: division by zero"}
    except Exception as e:
        return {"answer": f"Invalid expression: {e}"}
    
    if calc_request.decimal_places is not None:
        result = round(result, calc_request.decimal_places)

    return {"answer": f"The calculation {calc_request.expression} = {result}", "sources": []}

def route_decision(state: GraphState) -> str:
    mapping = {
        Category.BIOMED.value: Node.DOCUMENT_PATH.value,
        Category.CALCULATION.value: Node.CALCULATION_PATH.value,
        Category.GENERAL.value: Node.GENERAL_PATH.value,
    }
    classification = state["classification"]
    if classification not in mapping:
        raise ValueError(
            f"Unexpected classification '{classification}' — this should be "
            f"impossible given the Category enum constraint, indicating a bug."
        )
    return mapping[classification]

graph = StateGraph(GraphState)
graph.add_node("condense_question", condense_question)
graph.add_node(Node.CLASSIFY.value, classify_question)
graph.add_node(Node.DOCUMENT_PATH.value, answer_from_document)
graph.add_node(Node.GENERAL_PATH.value, answer_general)
graph.add_node(Node.CALCULATION_PATH.value, answer_with_calculation)

graph.set_entry_point("condense_question")
graph.add_edge("condense_question", Node.CLASSIFY.value)

graph.add_conditional_edges(
    Node.CLASSIFY.value,
    route_decision,
    {
        Node.DOCUMENT_PATH.value: Node.DOCUMENT_PATH.value,
        Node.GENERAL_PATH.value: Node.GENERAL_PATH.value,
        Node.CALCULATION_PATH.value: Node.CALCULATION_PATH.value,
    }
)
graph.add_node("record_turn", record_turn)

graph.add_edge(Node.DOCUMENT_PATH.value, "record_turn")
graph.add_edge(Node.GENERAL_PATH.value, "record_turn")
graph.add_edge(Node.CALCULATION_PATH.value, "record_turn")
graph.add_edge("record_turn", END)

app = graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    for q in [
        "I want to visit the capital of France, how's this city called? Brief response, only the city name",
        "What AUC did the CCTA-derived nomogram achieve for predicting MACE, in both the derivation and external validation cohorts?",
        "What is 91 times 1.15, rounded to two decimal places?",
        "How tall is its most famous iron landmark in the city I want to visit, in metres? Brief response, tower name and height"
    ]:
        config = {"configurable": {"thread_id": "demo-session-1"}}
        result = app.invoke({"question": q}, config=config)
        print(f"resolved_question: {result.get('resolved_question')}")
        output = (
            f"Original Q: {q}\n"
            f"Resolved Q: {result.get('resolved_question')}\n"
            f"Classification: {result['classification']}\n"
            f"A: {result['answer']}\n"
        )
        if result.get("sources"):
            output += f"Sources: {', '.join(result['sources'])}\n"
        print(output)