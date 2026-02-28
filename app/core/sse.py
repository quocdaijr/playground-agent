"""
SSE Streaming Helpers
======================
Adapts LangGraph .astream_events() output to Server-Sent Events (SSE) frames.

Wire format (RFC 8895):
    event: <type>\\n
    data: <json>\\n
    \\n

Supports both the chat graph (AgentState) and the research pipeline (ResearchState).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any

import structlog
from langchain_core.messages import AIMessage

logger = structlog.get_logger(__name__)

# Stage labels shown to the client for each graph node
_STAGE_LABELS: dict[str, str] = {
    # Chat graph
    "agent": "Thinking...",
    "tools": "Running tools...",
    # Research pipeline
    "plan": "Planning research sub-questions...",
    "research": "Searching the web...",
    "synthesize": "Synthesizing findings...",
    "review": "Reviewing the draft...",
    "finalize": "Producing final answer...",
}

# Nodes that should emit a "stage" event (filter out LangGraph internals)
_STAGE_NODES = {"agent", "tools", "plan", "research", "synthesize", "review", "finalize"}


def format_sse(event: str, data: dict | str) -> str:
    """Return a single SSE frame string (ending with double newline)."""
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_graph_events(
    graph: Any,
    initial_state: dict,
    session_id: str,
    on_complete: Callable[[dict], Coroutine] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Consume LangGraph ``.astream_events(version="v2")`` and yield SSE frame strings.

    Emitted events:
      ``stage``      — when a named graph node starts
      ``token``      — each LLM text chunk
      ``tool_call``  — when a tool is invoked
      ``tool_result``— when a tool returns (first 500 chars)
      ``done``       — final completion with reply, session_id, turn_count
      ``error``      — on exception

    Args:
        graph:         Compiled LangGraph (chat graph or research graph)
        initial_state: Starting state dict passed to .astream_events()
        session_id:    Session ID echoed in the ``done`` event
        on_complete:   Optional async callback invoked with final_state before
                       emitting ``done`` (use for DB persistence)
    """
    final_state: dict = {}
    current_node: str = ""

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind: str = event.get("event", "")
            name: str = event.get("name", "")
            data: dict = event.get("data", {})

            # ── Stage start ────────────────────────────────────────────────
            if kind == "on_chain_start" and name in _STAGE_NODES:
                current_node = name
                label = _STAGE_LABELS.get(name, f"Running {name}...")
                yield format_sse("stage", {"stage": name, "label": label})

            # ── LLM token ─────────────────────────────────────────────────
            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield format_sse("token", {"token": chunk.content, "node": current_node})

            # ── Tool invocation ────────────────────────────────────────────
            elif kind == "on_tool_start":
                tool_input = data.get("input", {})
                yield format_sse("tool_call", {
                    "tool": name,
                    "args": tool_input,
                    "node": current_node,
                })

            # ── Tool result ────────────────────────────────────────────────
            elif kind == "on_tool_end":
                output = data.get("output", "")
                yield format_sse("tool_result", {
                    "tool": name,
                    "output": str(output)[:500],
                    "node": current_node,
                })

            # ── Capture final graph output ─────────────────────────────────
            elif kind == "on_chain_end" and name == "LangGraph":
                final_state = data.get("output", {})

    except Exception as exc:
        logger.error("sse_stream_error", session_id=session_id, error=str(exc))
        yield format_sse("error", {"message": str(exc)})
        return

    # Optional DB persistence callback before emitting done
    if on_complete:
        try:
            await on_complete(final_state)
        except Exception as exc:
            logger.error("sse_on_complete_error", session_id=session_id, error=str(exc))

    # ── Done event ─────────────────────────────────────────────────────────
    reply = _extract_reply(final_state)
    turn_count = final_state.get("turn_count", 0)
    yield format_sse("done", {
        "reply": reply,
        "turn_count": turn_count,
        "session_id": session_id,
    })


def _extract_reply(state: dict) -> str:
    """Extract the final AI reply from either graph's finished state."""
    # Research graph
    if state.get("final_answer"):
        return state["final_answer"]
    # Chat graph — last AIMessage with content
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""
