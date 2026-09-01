"""Async database engine and session management.

Ye module SQLAlchemy async engine banata hai aur sessions provide karta hai.
Default backend SQLite (aiosqlite) hai — local, zero-setup. Postgres par jaana
ho to sirf `DATABASE_URL` badalna hai; is file ka koi code change nahi hota,
kyunki hum sirf async SQLAlchemy API use kar rahe hain (backend-agnostic).

Models isi module ke `Base` ko inherit karenge, aur repositories/routes
`get_session()` ke through DB access lenge.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base class for all ORM models.

    Har model (Business, Contact, Call, ...) is class ko inherit karega. Ek hi
    Base hone se SQLAlchemy saari tables ko ek metadata registry mein rakhta
    hai, jisse migrations aur create-all ek jagah se hote hain.
    """


def _create_engine() -> AsyncEngine:
    """Build the async SQLAlchemy engine from settings.

    SQLite ke liye `check_same_thread` disable karna padta hai kyunki async
    context mein connection alag threads se touch ho sakta hai. Ye handling
    sirf SQLite URLs par apply hoti hai.

    Returns:
        AsyncEngine: Configured async engine instance.
    """
    settings = get_settings()
    connect_args: dict[str, object] = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        future=True,
        connect_args=connect_args,
    )


# Module-level singletons — poori app ek hi engine/session-factory share karti hai.
engine: AsyncEngine = _create_engine()

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session, closing it afterwards.

    FastAPI dependency ke roop mein use hone ke liye designed. Har request ko
    apna fresh session milta hai; block khatam hone par session guaranteed
    close hota hai (chahe error aaye).

    Yields:
        AsyncSession: An active async session bound to the engine.
    """
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables that don't yet exist.

    Development/bootstrap convenience — models ko import karke unki tables
    banata hai. Production mein iski jagah proper migrations (alembic) use
    honge; abhi local ke liye ye kaafi hai.

    Note:
        Ye function tabhi tables banayega jab models import ho chuke hon,
        kyunki create_all `Base.metadata` mein registered tables par chalta hai.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)