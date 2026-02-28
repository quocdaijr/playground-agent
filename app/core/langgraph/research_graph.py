"""
Research Pipeline — LangGraph StateGraph
=========================================
A 5-stage sequential research pipeline that demonstrates a longer agent cycle
with deterministic, structured stages (no conditional branching).

Flow:
  START → plan → research → synthesize → review → finalize → END

Stages:
  1. plan       — LLM breaks the query into 2-3 targeted sub-questions
  2. research   — Calls web_search for each sub-question (uses existing tool)
  3. synthesize — LLM merges search results into a structured draft report
  4. review     — LLM self-critiques the draft and identifies gaps
  5. finalize   — LLM produces the polished final answer addressing the critique
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, TypedDict

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.core.langgraph.tools import web_search
from app.services.llm import get_llm

logger = structlog.get_logger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """
    Accumulates data as it flows through all 5 pipeline stages.

    query:          Original user question (immutable after init)
    sub_questions:  2-3 sub-questions produced by plan_node
    search_results: Dict mapping sub-question → raw web search result
    draft_report:   Synthesized markdown report (after synthesize_node)
    critique:       Self-critique bullet points (after review_node)
    final_answer:   Polished final answer (after finalize_node)
    stage:          Current stage name — read by SSE streaming layer
    messages:       Full message log for DB persistence and streaming
    provider:       LLM provider to use across all nodes
    model:          Specific model name — overrides provider default when set
    """
    query: str
    sub_questions: list[str]
    search_results: dict[str, str]
    draft_report: str
    critique: str
    final_answer: str
    stage: str
    messages: Annotated[list[BaseMessage], add_messages]
    provider: str
    model: str | None


# ── Stage prompts ──────────────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a research planner. Break the user's query into exactly 2-3 focused \
sub-questions that together fully answer the query.
Respond with a JSON array of strings ONLY — no explanation, no markdown fences.
Example: ["What is X?", "How does Y affect Z?", "What are the latest trends in W?"]"""

_SYNTHESIZE_PROMPT = """\
You are a research analyst. Using the search results provided, write a structured \
markdown report that answers the original query.
Use clear headings (##), bullet points, and note which sub-question each section \
addresses. Be factual and comprehensive."""

_REVIEW_PROMPT = """\
You are a critical reviewer. Read the draft research report and identify:
1. Factual gaps or unsupported claims
2. Missing perspectives or important angles
3. Areas needing clarification or expansion
Output exactly 3-5 bullet points. Be concise."""

_FINALIZE_PROMPT = """\
You are a professional writer. Given the draft report and reviewer critique, \
produce a final, polished answer that:
1. Addresses all gaps identified by the reviewer
2. Is well-structured with clear headings
3. Directly and fully answers the user's original query
Write in clear, professional English."""


@lru_cache(maxsize=8)
def _get_research_llm(provider: str | None, model: str | None) -> BaseChatModel:
    """Return a cached LLM for the given (provider, model) pair (no tools needed)."""
    return get_llm(provider, model=model)


# ── Nodes ──────────────────────────────────────────────────────────────────────

async def plan_node(state: ResearchState) -> dict:
    """Stage 1: Decompose the query into 2-3 research sub-questions."""
    llm = _get_research_llm(state.get("provider"), state.get("model"))
    response = await llm.ainvoke([
        SystemMessage(content=_PLAN_PROMPT),
        HumanMessage(content=state["query"]),
    ])

    try:
        raw = response.content.strip()
        # Robustly strip markdown fences (handles ```json\n...\n``` and ``` ... ```)
        if raw.startswith("```"):
            lines = raw.splitlines()
            # Drop the opening fence line (e.g. ```json) and closing ``` line
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            raw = "\n".join(inner).strip()
        sub_questions: list[str] = json.loads(raw)
        if not isinstance(sub_questions, list):
            raise TypeError("Expected JSON array")
        sub_questions = [str(q).strip() for q in sub_questions[:3] if q]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("plan_node_parse_failed", error=str(exc), raw=response.content[:200])
        # Fallback: treat whole query as single sub-question
        sub_questions = [state["query"]]

    logger.info("plan_node_complete", sub_questions=sub_questions)
    return {
        "sub_questions": sub_questions,
        "stage": "plan",
        "messages": [
            HumanMessage(content=state["query"]),
            AIMessage(content=f"Research plan:\n{json.dumps(sub_questions, indent=2)}"),
        ],
    }


async def research_node(state: ResearchState) -> dict:
    """Stage 2: Web search for each sub-question using the existing web_search tool."""
    results: dict[str, str] = {}

    for question in state.get("sub_questions", []):
        try:
            result = await web_search.ainvoke({"query": question})
            results[question] = result
            logger.info("research_node_search", question=question[:80])
        except Exception:
            logger.exception("research_node_search_failed", question=question[:80])
            results[question] = "[Search failed]"

    search_summary = "\n\n".join(
        f"### Sub-question: {q}\n{r}" for q, r in results.items()
    )
    return {
        "search_results": results,
        "stage": "research",
        "messages": [AIMessage(content=f"Search results:\n{search_summary}")],
    }


async def synthesize_node(state: ResearchState) -> dict:
    """Stage 3: Synthesize search results into a structured draft report."""
    llm = _get_research_llm(state.get("provider"), state.get("model"))

    search_text = "\n\n".join(
        f"## Sub-question: {q}\n{r}"
        for q, r in state.get("search_results", {}).items()
    )
    user_msg = f"Original query: {state['query']}\n\nSearch results:\n{search_text}"

    response = await llm.ainvoke([
        SystemMessage(content=_SYNTHESIZE_PROMPT),
        HumanMessage(content=user_msg),
    ])
    logger.info("synthesize_node_complete", draft_len=len(response.content))
    return {
        "draft_report": response.content,
        "stage": "synthesize",
        "messages": [AIMessage(content=f"Draft report:\n{response.content}")],
    }


async def review_node(state: ResearchState) -> dict:
    """Stage 4: Self-critique of the draft — identify gaps and weaknesses."""
    llm = _get_research_llm(state.get("provider"), state.get("model"))

    review_input = (
        f"Original query: {state['query']}\n\n"
        f"Draft report:\n{state.get('draft_report', '')}"
    )
    response = await llm.ainvoke([
        SystemMessage(content=_REVIEW_PROMPT),
        HumanMessage(content=review_input),
    ])
    logger.info("review_node_complete")
    return {
        "critique": response.content,
        "stage": "review",
        "messages": [AIMessage(content=f"Critique:\n{response.content}")],
    }


async def finalize_node(state: ResearchState) -> dict:
    """Stage 5: Produce the final polished answer incorporating the critique."""
    llm = _get_research_llm(state.get("provider"), state.get("model"))

    final_input = (
        f"Original query: {state['query']}\n\n"
        f"Draft report:\n{state.get('draft_report', '')}\n\n"
        f"Reviewer critique:\n{state.get('critique', '')}"
    )
    response = await llm.ainvoke([
        SystemMessage(content=_FINALIZE_PROMPT),
        HumanMessage(content=final_input),
    ])
    logger.info("finalize_node_complete", answer_len=len(response.content))
    return {
        "final_answer": response.content,
        "stage": "finalize",
        "messages": [AIMessage(content=response.content)],
    }


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_research_graph():
    """Build and compile the 5-stage research pipeline StateGraph."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", "review")
    graph.add_edge("review", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled once at import time — provider is passed in state, not baked into the graph
research_graph = build_research_graph()
