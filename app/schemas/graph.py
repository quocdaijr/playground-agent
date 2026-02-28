from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class GraphState(BaseModel):
    """Pydantic representation of the LangGraph AgentState (for documentation / validation)."""

    class Config:
        arbitrary_types_allowed = True

    messages: list[BaseMessage] = []
    turn_count: int = 0
    summary_context: str = ""
