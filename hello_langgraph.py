from enum import Enum
import re
from textwrap import dedent
from typing import Literal, Optional, TypedDict, NotRequired
import httpx
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from simpleeval import simple_eval

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class Node(Enum):
    CLASSIFY = "classify"
    DOCUMENT_PATH = "document_path"
    GENERAL_PATH = "general_path"
    CALCULATION_PATH = "calculation_path"

class Category(Enum):
    DOCUMENT = "DOCUMENT"
    CALCULATION = "CALCULATION"
    GENERAL = "GENERAL"    

class GraphState(TypedDict):
    question: str
    classification: NotRequired[str]
    answer: NotRequired[str]

class CalculationRequest(BaseModel):
    expression: str
    decimal_places: Optional[int] = None

class Classification(BaseModel):
    category: Category

def classify_question(state: GraphState) -> GraphState:
    CLASSIFICATION_PROMPT = dedent(f"""\
        # Role
        You are a routing assistant for a multi-tool system. Your job is to decide \
        which specialized tool should handle a user's question — not to answer it \
        yourself.

        # Task
        Classify the question into exactly one category:
        - {Category.DOCUMENT.value}: the question asks about specific facts, findings, \
        or content that would only exist in a particular reference document (e.g., a \
        study, report, or paper) — not something you'd know from general knowledge.
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

        Question: "What did the University of Minnesota Law School study find \
        about ChatGPT's exam performance?"
        -> {Category.DOCUMENT.value}

        Question: "What is 15% of 340, rounded to the nearest whole number?"
        -> {Category.CALCULATION.value}

        # Input
        Question: {state['question']}
        """)
    
    classification = llm.with_structured_output(Classification).invoke(CLASSIFICATION_PROMPT)

    return {**state, "classification": classification.category.value}


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
        Question: {state['question']}
        """)

    calc_request = llm.with_structured_output(CalculationRequest).invoke(CALCULATION_PROMPT)

    if not re.fullmatch(r"[\d\s+\-*/().]+", calc_request.expression):
        return {**state, "answer": f"Unsafe or unparseable expression: {calc_request.expression}"}

    try:
        result = simple_eval(calc_request.expression)
    except ZeroDivisionError:
        return {**state, "answer": "Error: division by zero"}
    except Exception as e:
        return {**state, "answer": f"Invalid expression: {e}"}
    
    if calc_request.decimal_places is not None:
        result = round(result, calc_request.decimal_places)

    return {**state, "answer": f"The calculation {calc_request.expression} = {result}"}

def route_decision(state: GraphState) -> str:
    mapping = {
        Category.DOCUMENT.value: Node.DOCUMENT_PATH.value,
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
graph.add_node(Node.CLASSIFY.value, classify_question)
graph.add_node(Node.DOCUMENT_PATH.value, answer_from_document)
graph.add_node(Node.GENERAL_PATH.value, answer_general)
graph.add_node(Node.CALCULATION_PATH.value, answer_with_calculation)

graph.set_entry_point(Node.CLASSIFY.value)
graph.add_conditional_edges(
    Node.CLASSIFY.value,
    route_decision,
    {
        Node.DOCUMENT_PATH.value: Node.DOCUMENT_PATH.value,
        Node.GENERAL_PATH.value: Node.GENERAL_PATH.value,
        Node.CALCULATION_PATH.value: Node.CALCULATION_PATH.value,
    }
)
graph.add_edge(Node.DOCUMENT_PATH.value, END)
graph.add_edge(Node.GENERAL_PATH.value, END)
graph.add_edge(Node.CALCULATION_PATH.value, END)

app = graph.compile()

if __name__ == "__main__":
    for q in [
        "What is the capital of France?",
        "What did the University of Minnesota Law School study find about ChatGPT?",
        "What is 91 times 1.15, rounded to two decimal places?",
    ]:
        result = app.invoke({"question": q})
        print(
            f"Q: {q}\nClassification: {result['classification']}\nA: {result['answer']}\n"
        )
