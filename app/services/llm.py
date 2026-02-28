"""
LLM Service
===========
Thin wrapper around ChatAnthropic for direct (non-graph) completions,
with tenacity retry logic for production reliability.

Also exports `get_llm()` — a provider-aware factory used by both the chat
graph and the research pipeline.
"""

import structlog
from anthropic import APIStatusError, APITimeoutError, RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    tools: list | None = None,
) -> BaseChatModel:
    """
    Factory that returns a LangChain chat model for the requested provider.

    Args:
        provider: "anthropic" | "openai" | "google" — defaults to settings.LLM_PROVIDER
        model:    Specific model name to use — overrides the provider default from settings.
                  Examples: "claude-opus-4-5", "gpt-4o-mini", "gemini-1.5-pro"
        tools:    Optional list of tools to bind via .bind_tools()

    Returns:
        Instantiated (and optionally tool-bound) BaseChatModel.
    """
    p = (provider or settings.LLM_PROVIDER).lower()

    if p == "openai":
        from langchain_openai import ChatOpenAI
        llm: BaseChatModel = ChatOpenAI(
            model=model or settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY or None,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
    elif p == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model or settings.GOOGLE_MODEL,
            google_api_key=settings.GOOGLE_API_KEY or None,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            max_output_tokens=settings.MAX_TOKENS,
        )
    else:  # anthropic (default)
        llm = ChatAnthropic(
            model=model or settings.DEFAULT_LLM_MODEL,
            anthropic_api_key=settings.ANTHROPIC_API_KEY or None,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )

    return llm.bind_tools(tools) if tools else llm


class LLMService:
    def __init__(self) -> None:
        self._llm = ChatAnthropic(
            model=settings.DEFAULT_LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            anthropic_api_key=settings.ANTHROPIC_API_KEY or None,
        )

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=lambda rs: logger.warning(
            "llm_retry",
            attempt=rs.attempt_number,
            error=str(rs.outcome.exception()),
        ),
    )
    async def complete(self, messages: list[BaseMessage]) -> str:
        response = await self._llm.ainvoke(messages)
        return response.content

    async def stream(self, messages: list[BaseMessage]):
        async for chunk in self._llm.astream(messages):
            if chunk.content:
                yield chunk.content


llm_service = LLMService()
