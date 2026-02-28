"""
Conversation Summary Service
=============================
After every SUMMARY_EVERY turns, automatically summarise the conversation
and persist it to the `summaries` table.

On subsequent turns, the latest summary is injected into the system prompt
so PlaygroundAgent maintains context across long conversations without expanding the
context window with the full message history.
"""

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.session import Message, Summary

logger = structlog.get_logger(__name__)

SUMMARY_EVERY = 10
SUMMARISER_PROMPT = """\
You are a conversation summariser. Given the transcript below, write a concise \
summary (3-5 sentences) covering:
- The main topics discussed
- Any important facts, decisions, or outcomes
- Relevant tool results (e.g. calculations, weather, exchange rates)

Be factual and brief. Write in third person. Output ONLY the summary text."""


async def maybe_summarise(
    session: AsyncSession,
    session_id: str,
    turn_count: int,
    messages: list[Message],
) -> str | None:
    """
    Generate and persist a summary when turn_count is a multiple of SUMMARY_EVERY.
    Returns the summary text if one was created, otherwise None.
    """
    if turn_count == 0 or turn_count % SUMMARY_EVERY != 0:
        return None

    turn_start = turn_count - SUMMARY_EVERY + 1
    turn_end = turn_count

    transcript = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in messages
        if m.role in ("human", "ai")
    )

    llm = ChatAnthropic(
        model=settings.DEFAULT_LLM_MODEL,
        max_tokens=512,
        anthropic_api_key=settings.ANTHROPIC_API_KEY or None,
    )
    response = await llm.ainvoke([
        SystemMessage(content=SUMMARISER_PROMPT),
        HumanMessage(content=transcript),
    ])
    summary_text = response.content.strip()

    session.add(Summary(
        session_id=session_id,
        content=summary_text,
        turn_start=turn_start,
        turn_end=turn_end,
    ))
    await session.commit()
    logger.info("summary_created", session_id=session_id, turns=f"{turn_start}-{turn_end}")
    return summary_text


async def get_latest_summary(session: AsyncSession, session_id: str) -> str | None:
    """Return the most recent summary text for context injection."""
    result = await session.execute(
        select(Summary)
        .where(Summary.session_id == session_id)
        .order_by(Summary.created_at.desc())
        .limit(1)
    )
    summary = result.scalar_one_or_none()
    return summary.content if summary else None


async def get_all_summaries(session: AsyncSession, session_id: str) -> list[Summary]:
    """Return all summaries for a session in chronological order."""
    result = await session.execute(
        select(Summary)
        .where(Summary.session_id == session_id)
        .order_by(Summary.created_at)
    )
    return list(result.scalars().all())
