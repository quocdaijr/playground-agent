"""
Async PostgreSQL service — SQLAlchemy 2.0 async engine + session factory.
The engine and session factory are initialised once at startup via init_db()
and torn down gracefully on shutdown via close_db().
"""

from contextlib import asynccontextmanager

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

logger = structlog.get_logger(__name__)

_engine = None
_session_factory: async_sessionmaker | None = None


async def init_db() -> None:
    global _engine, _session_factory
    url = (
        f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    _engine = create_async_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=settings.DEBUG,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    logger.info("db_ready", host=settings.POSTGRES_HOST, db=settings.POSTGRES_DB)


async def close_db() -> None:
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = _session_factory = None
        logger.info("db_closed")


@asynccontextmanager
async def get_session() -> AsyncSession:
    """Async context manager that yields a database session with auto-rollback on error."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
