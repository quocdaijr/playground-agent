"""Utility helpers for building and transforming LangGraph state."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def build_initial_state(user_message: str, summary_context: str = "") -> dict:
    """Create a fresh AgentState for the first turn of a session."""
    return {
        "messages": [HumanMessage(content=user_message)],
        "turn_count": 0,
        "summary_context": summary_context,
    }


def extract_reply(state: dict) -> str:
    """Pull the last AIMessage text from a completed graph state."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def append_user_message(state: dict, text: str, summary_context: str = "") -> dict:
    """Append a human message to an existing state for multi-turn continuation."""
    return {
        **state,
        "messages": [*state["messages"], HumanMessage(content=text)],
        "summary_context": summary_context or state.get("summary_context", ""),
    }


def state_messages_to_dicts(old_count: int, state: dict) -> list[dict]:
    """Extract only the messages added during this turn (delta) as role/content dicts."""
    new_msgs = state["messages"][old_count:]
    result = []
    for m in new_msgs:
        if isinstance(m, HumanMessage):
            result.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            result.append({"role": "ai", "content": m.content})
    return result


def db_messages_to_state(records, turn_count: int, summary_context: str = "") -> dict:
    """Reconstruct a LangGraph AgentState from database message rows."""
    messages = []
    for r in records:
        if r.role == "human":
            messages.append(HumanMessage(content=r.content))
        elif r.role == "ai":
            messages.append(AIMessage(content=r.content))
    return {
        "messages": messages,
        "turn_count": turn_count,
        "summary_context": summary_context,
    }
