"""
LLM Service
===========
Thin wrapper around ChatAnthropic for direct (non-graph) completions,
with tenacity retry logic for production reliability.
"""

import structlog
from anthropic import APIStatusError, APITimeoutError, RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)


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
