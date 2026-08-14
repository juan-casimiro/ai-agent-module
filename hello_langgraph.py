from enum import Enum
from operator import add
import re
from textwrap import dedent
from typing import Annotated, Optional, TypedDict, NotRequired
import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from simpleeval import simple_eval
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

MAX_RETRIES = 1  # hard cap — enforced defensively in should_retry_document_lookup,
                  # not just relied on via graph topology

GENERAL_SYSTEM_PROMPT = dedent("""\
    Answer in plain prose, under 150 words. No markdown headers, tables,
    or bullet lists — the output is printed to a terminal. Be direct and
    concrete; do not pad with structure.
    """)

UNGROUNDED_FALLBACK_PROMPT = dedent("""\
    This question was routed to document retrieval, but the document corpus
    did not contain sufficient information to answer it. You are answering
    from general knowledge only.

    Begin your answer by stating plainly that it is not grounded in the
    document corpus. Do not invent specific findings, figures, gene names,
    or study results. If you do not know, say so.
    """)

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
    condensation_reasoning: NotRequired[str]
    classification: NotRequired[str]
    answer: NotRequired[str]
    context_sufficient: NotRequired[bool]
    sources: NotRequired[list[str]]
    retried: NotRequired[bool]
    retry_reason: NotRequired[str]  # for debugging purposes
    history: Annotated[list[dict], add]

class CalculationRequest(BaseModel):
    expression: str
    decimal_places: Optional[int] = None

class Classification(BaseModel):
    category: Category

class CondensedQuestion(BaseModel):
    resolved_question: str
    reasoning: str  # short debug note: why the question was rewritten this way

def condense_question(state: GraphState) -> GraphState:
    history = state.get("history", [])
    if not history:
        return {"resolved_question": state["question"]}

    CONDENSE_PROMPT = dedent(f"""\
        # Role
        You resolve a follow-up question into a fully standalone question, using \
        the conversation history for context. You do not answer the question.

        # Task
        Given the conversation history and a new follow-up question, rewrite the \
        follow-up into a standalone question that makes sense with NO prior \
        context — resolving pronouns, ellipsis ("what about X?"), and implicit \
        references. If the new question is already standalone, return it \
        unchanged. If the follow-up references a specific number or fact stated \
        in a previous answer, include that concrete value explicitly in the \
        rewritten question.

        # Constraints
        - When a referenced value is a percentage, write it as a plain number \
        with no "%" sign (e.g. "50", not "50%") — this avoids ambiguity in any \
        downstream calculation on that value.

        # Examples
        History:
        Q: What was the reported increase?
        A: The study found a 70% increase in incidents.
        New question: What's that number times 5?
        -> What is 70 multiplied by 5?

        # History
        {format_history(history)}

        # New question
        {state['question']}

        # Output
        Return ONLY the rewritten standalone question, nothing else.
        """)

    condensed = llm.with_structured_output(CondensedQuestion).invoke(CONDENSE_PROMPT)
    return {
        "resolved_question": condensed.resolved_question,
        "condensation_reasoning": condensed.reasoning,
    }

def classify_question(state: GraphState, llm=llm) -> GraphState:
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
        "retry_reason": None,
        "retried": False,
        "context_sufficient": True,
    }

def format_history(history: list[dict]) -> str:
    return "\n\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history
    )

def answer_general(state: GraphState) -> GraphState:
    question = state["resolved_question"]
    system_prompt = GENERAL_SYSTEM_PROMPT

    if state.get("retried"):
        system_prompt = f"{system_prompt}\n{UNGROUNDED_FALLBACK_PROMPT}"

        question = dedent(f"""\
            The following question was routed to document retrieval, but the
            document corpus did not contain sufficient information to answer it.
            You are answering from general knowledge only.

            Begin your answer by stating plainly that this answer is not grounded
            in the document corpus. Do not invent specific findings, figures, gene
            names, or study results. If you do not know, say so.

            Question: {question}
            """)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    return {"answer": response.content, "sources": []}


def query_document_service(question: str, use_query_rewriting: bool = False) -> dict:
    response = httpx.post(
        "http://localhost:8000/query",
        json={
            "question": question,
            "n_results": 8,
            "use_query_rewriting": use_query_rewriting,
        },
        timeout=60.0,  # RAG service does retrieval + reranking + an LLM call
    )
    return response.json()


def answer_from_document(state: GraphState) -> GraphState:
    result = query_document_service(state["resolved_question"])
    context_sufficient = result.get("context_sufficient", True)

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "context_sufficient": context_sufficient,
        # debug/demo label only — not read by routing
        "retry_reason": None if context_sufficient else result.get("insufficiency_reason"),
    }


def retry_document_lookup(state: GraphState) -> GraphState:
    result = query_document_service(state["resolved_question"], use_query_rewriting=True)
    sources = result.get("sources", [])
    context_sufficient = result.get("context_sufficient", True)
    return {
        "answer": result["answer"],
        "sources": sources,
        "context_sufficient": context_sufficient,
        "retried": True,
    }

def should_retry_document_lookup(state: GraphState) -> str:
    if not state.get("context_sufficient", True) and not state.get("retried", False):
        return "retry"
    return "no_retry"


def should_fallback_to_general(state: GraphState) -> str:
    if not state.get("context_sufficient", True):
        return Node.GENERAL_PATH.value
    return "record_turn"


def answer_with_calculation(state: GraphState, llm=llm) -> GraphState:
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
        Never include function calls, variable names, or any other syntax, \
        including "%".
        - If the question refers to a percentage figure by its number (e.g. \
        "that 24% increase" or "a 24 percent rise"), treat the number itself \
        (24) as the value to use in the expression — do NOT convert it to a \
        decimal fraction (0.24) unless the question explicitly asks for "24% of" \
        some other quantity, which is a different operation (multiplication by \
        0.24) than "24 times" something.
        - If the question does not mention rounding or precision at all, set \
        decimal_places to null — do not assume a default.

        # Examples
        Question: "What is 15% of 200?"
        -> expression: "200 * 0.15", decimal_places: null

        Question: "What's 91 times 1.15, rounded to two decimal places?"
        -> expression: "91 * 1.15", decimal_places: 2

        Question: "What is 26 multiplied by 7, rounded to 1 decimal place?"
        -> expression: "26 * 7", decimal_places: 1

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
graph.add_node("retry_document_lookup", retry_document_lookup)

graph.add_conditional_edges(
    Node.CLASSIFY.value,
    route_decision,
    {
        Node.DOCUMENT_PATH.value: Node.DOCUMENT_PATH.value,
        Node.GENERAL_PATH.value: Node.GENERAL_PATH.value,
        Node.CALCULATION_PATH.value: Node.CALCULATION_PATH.value,
    }
)
graph.add_conditional_edges(
    Node.DOCUMENT_PATH.value,
    should_retry_document_lookup,
    {
        "retry": "retry_document_lookup",
        "no_retry": "record_turn",
    },
)
graph.add_conditional_edges(
    "retry_document_lookup",
    should_fallback_to_general,
    {
        Node.GENERAL_PATH.value: Node.GENERAL_PATH.value,
        "record_turn": "record_turn",
    },
)
graph.add_node("record_turn", record_turn)

graph.add_edge(Node.GENERAL_PATH.value, "record_turn")
graph.add_edge(Node.CALCULATION_PATH.value, "record_turn")
graph.add_edge("record_turn", END)

app = graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    thread_id = "demo-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    conversation = [
        "What effect did the spring daylight saving transition have on MI "
            "rates, according to the Sadhu et al. analysis in the diabetes "
            "cardiovascular outcomes review?",
        "What about the fall transition?",
        "And what's that spring percentage increase times 3, rounded to 1 decimal place?",
    ]

    final_state = None
    for i, q in enumerate(conversation, start=1):
        result = app.invoke({"question": q}, config=config)
        final_state = result

        print(f"--- Turn {i} ---")
        print(f"Question:          {q}")
        print(f"Resolved question: {result.get('resolved_question')}")
        print(f"Condensation reason: {result.get('condensation_reasoning', '(no history — skipped)')}")
        print(f"Classification:    {result['classification']}")
        print(f"Answer:            {result['answer']}")
        if result.get("sources"):
            print(f"Sources:           {', '.join(result['sources'])}")
        print()

    print(f"Final history length: {len(final_state['history'])}")