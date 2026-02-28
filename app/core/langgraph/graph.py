"""
PlaygroundAgent — LangGraph StateGraph

Flow:
  START → agent → (tools → agent)* → END

The agent node injects a summary context into the system prompt when available,
giving the model lightweight long-term memory without expanding the context window.
"""

from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.core.langgraph.tools import TOOLS
from app.core.prompts import SYSTEM_PROMPT


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    summary_context: str  # injected from DB before each invocation


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_system_prompt(summary_context: str) -> str:
    """Render the system prompt, optionally injecting a prior-conversation summary."""
    ctx = f"\n## Prior conversation summary\n{summary_context}\n" if summary_context else ""
    return SYSTEM_PROMPT.replace("{summary_context}", ctx)


def _make_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.DEFAULT_LLM_MODEL,
        temperature=settings.DEFAULT_LLM_TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        anthropic_api_key=settings.ANTHROPIC_API_KEY or None,
    ).bind_tools(TOOLS)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def _agent_node(llm):
    def call_model(state: AgentState) -> AgentState:
        system = _build_system_prompt(state.get("summary_context", ""))
        messages = [SystemMessage(content=system), *state["messages"]]
        response = llm.invoke(messages)
        return {
            "messages": [response],
            "turn_count": state.get("turn_count", 0) + 1,
            "summary_context": state.get("summary_context", ""),
        }
    return call_model


def _should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the PlaygroundAgent LangGraph StateGraph."""
    llm = _make_llm()
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node(llm))
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compiled once at import time — reused across all requests
agent_graph = build_graph()
