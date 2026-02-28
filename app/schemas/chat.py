from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["anthropic", "openai", "google"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = None
    provider: Provider = "anthropic"
    model: str | None = None  # overrides provider default, e.g. "claude-opus-4-5", "gpt-4o-mini"


class ChatResponse(BaseModel):
    reply: str
    session_id: str | None = None
    turn_count: int = 0
    provider: Provider = "anthropic"
    model: str | None = None


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    session_id: str | None = None
    provider: Provider = "anthropic"
    model: str | None = None  # overrides provider default


class ResearchResponse(BaseModel):
    final_answer: str
    sub_questions: list[str]
    session_id: str
    stages_completed: int = 5
    provider: Provider = "anthropic"
    model: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
