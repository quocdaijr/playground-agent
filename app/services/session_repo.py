"""
Session Repository
==================
Centralises all database interactions for sessions and messages.
Keeps the API layer free of query logic.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.session import Message, Session


async def get_or_create_session(db: AsyncSession, session_id: str) -> Session:
    result = await db.execute(
        select(Session)
        .options(selectinload(Session.messages))
        .where(Session.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        session = Session(id=session_id, turn_count=0)
        db.add(session)
        await db.flush()
    return session


async def save_turn(
    db: AsyncSession,
    session: Session,
    turn_count: int,
    new_messages: list[dict],
) -> None:
    """Persist new messages and update the session turn counter atomically."""
    session.turn_count = turn_count
    for m in new_messages:
        db.add(Message(session_id=session.id, role=m["role"], content=m["content"]))
    await db.commit()
    await db.refresh(session)


async def get_session_with_messages(db: AsyncSession, session_id: str) -> Session | None:
    result = await db.execute(
        select(Session)
        .options(selectinload(Session.messages), selectinload(Session.summaries))
        .where(Session.id == session_id)
    )
    return result.scalar_one_or_none()


async def delete_session(db: AsyncSession, session_id: str) -> bool:
    """Delete a session and all its messages / summaries (CASCADE). Returns False if not found."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        return False
    await db.delete(session)
    await db.commit()
    return True
