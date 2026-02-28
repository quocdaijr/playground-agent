"""
PlaygroundAgent — LangGraph StateGraph

Flow:
  START → agent → (tools → agent)* → END

Each compiled graph is tied to a (provider, model) pair:
- The provider-specific system prompt is baked into the agent node via closure.
- The LLM (with tool bindings) is instantiated once and reused.

Multi-LLM: call `get_graph(provider, model)` to get a cached compiled graph.
The default `agent_graph` uses the configured LLM_PROVIDER and its default model.
"""

from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.core.langgraph.tools import TOOLS
from app.core.prompts import load_system_prompt
from app.services.llm import get_llm


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    summary_context: str  # injected from DB before each invocation


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_system_prompt(provider: str, summary_context: str) -> str:
    """Render the provider-specific system prompt, injecting the summary context."""
    template = load_system_prompt(provider)
    ctx = f"\n## Prior conversation summary\n{summary_context}\n" if summary_context else ""
    return template.replace("{summary_context}", ctx)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def _agent_node(llm, provider: str):
    """Return an async agent node closed over *llm* and *provider*."""
    async def call_model(state: AgentState) -> AgentState:
        system = _build_system_prompt(provider, state.get("summary_context", ""))
        messages = [SystemMessage(content=system), *state["messages"]]
        response = await llm.ainvoke(messages)
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

def _build_graph(provider: str, model: str | None) -> object:
    """Build and compile a StateGraph for the given (provider, model) pair."""
    llm = get_llm(provider, model=model, tools=TOOLS)
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node(llm, provider))
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


@lru_cache(maxsize=16)
def get_graph(provider: str = "anthropic", model: str | None = None) -> object:
    """
    Return a compiled LangGraph cached by (provider, model).

    Args:
        provider: "anthropic" | "openai" | "google"
        model:    Specific model name; None uses the provider's configured default.

    Supports up to 16 distinct (provider, model) combinations in-process.
    """
    return _build_graph(provider, model)


# Default graph — compiled once at import time using the configured defaults
agent_graph = get_graph(settings.LLM_PROVIDER)
