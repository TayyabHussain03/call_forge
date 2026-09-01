"""Shared ORM foundation: base class, ID strategy, timestamps.

Ye module saare models ka common foundation deta hai. Isme koi business logic
NAHI hai — sirf persistence-level conventions:

    - String UUID primary keys (Postgres migration ke liye clean; DB-agnostic).
    - Timezone-aware UTC timestamps (local machine time kabhi mix nahi hoti).
    - Consistent constraint naming (Alembic migrations baad mein predictable
      rahengi).

DESIGN NOTE (guide ke mutabiq): ORM models Pydantic contracts ki exact copy
    NAHI hain. Contracts communication boundaries define karte hain; ye models
    database consistency aur relationships ke liye hain. Dono ke beech conversion
    repository layer mein explicit hota hai.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def generate_uuid() -> str:
    """Generate a new random string UUID for a primary key.

    String UUIDs use karne ki wajah: SQLite aur Postgres dono par same tarah
    kaam karte hain, application-level generate hote hain (DB round-trip ke
    bina), aur distributed/concurrent inserts mein collision-safe hain.

    Returns:
        str: A new UUID4 in hex-with-dashes string form.
    """
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Poora system isi function se time leta hai taake kahin bhi naive/local
    timestamps na ghusein. DB mein sab kuch UTC mein persist hota hai.

    Returns:
        datetime: Current instant, tz-aware, in UTC.
    """
    return datetime.now(UTC)


class TimestampMixin:
    """Mixin adding created_at / updated_at columns to a model.

    `created_at` insert par set hota hai; `updated_at` har update par DB dwara
    refresh hota hai. Dono UTC tz-aware hain.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )


class UUIDMixin:
    """Mixin providing a string-UUID primary key named `id`."""

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )


class ModelBase(UUIDMixin, TimestampMixin, Base):
    """Abstract base for all application ORM models.

    Har concrete model isi ko inherit karega, jisse har table ko automatically
    string-UUID `id` aur UTC `created_at`/`updated_at` mil jaate hain.

    Note:
        `__abstract__ = True` ka matlab ye khud koi table nahi banata — sirf
        shared columns/behaviour inherit karwata hai.
    """

    __abstract__ = True