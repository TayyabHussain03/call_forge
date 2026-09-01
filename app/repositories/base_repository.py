"""Generic async repository base.

Repositories saari DB access ek jagah rakhte hain, taake queries business logic
mein bikhri na hon. Ye base sirf common CRUD deta hai; domain-specific lookups
subclasses mein aate hain.

BOUNDARY (guide ke mutabiq): repositories mein LLM/conversation/state-machine
    logic NAHI aati — sirf persistence operations.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import ModelBase

ModelT = TypeVar("ModelT", bound=ModelBase)


class BaseRepository(Generic[ModelT]):
    """Common create/read/update helpers over one model type.

    Ye raw SQLAlchemy session ko encapsulate karta hai taake application layer
    seedha session par depend na kare. Har repository ek session ke saath bandhi
    hoti hai (per-request).

    Attributes:
        model: The ORM model class this repository manages.
        session: The active async session.
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an active async session.

        Args:
            session: The async session to run queries against.
        """
        self.session = session

    async def get_by_id(self, entity_id: str) -> ModelT | None:
        """Fetch a single row by primary key.

        Args:
            entity_id: The UUID string primary key.

        Returns:
            ModelT | None: The row, or None if not found.
        """
        return await self.session.get(self.model, entity_id)

    async def add(self, entity: ModelT) -> ModelT:
        """Stage a new entity for insertion (no commit).

        Commit caller ki responsibility hai — taake multiple operations ek
        transaction mein atomically ho sakein.

        Args:
            entity: The model instance to add.

        Returns:
            ModelT: The same instance (id available after flush/commit).
        """
        self.session.add(entity)
        return entity

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[ModelT]:
        """Return a page of rows.

        Args:
            limit: Max rows to return.
            offset: Rows to skip.

        Returns:
            list[ModelT]: The requested page of rows.
        """
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())