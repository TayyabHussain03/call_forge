"""Transcript ORM model — one record per conversation turn.

Guide ke mutabiq: transcript ek giant string ki tarah store NAHI hota. Har turn
apna record hai (speaker, text, order, confidence), taake baad mein analyze/
replay/audit kiya ja sake. Call → TranscriptTurn one-to-many.

PII NOTE: `text` mein poora client speech hota hai — sensitive. __repr__ isko
expose nahi karta, aur logging layer isko default INFO par log nahi karegi.

TRUST: client turns ka text UNTRUSTED hai (STT output). Ye persist to hota hai,
lekin kabhi instruction ki tarah treat nahi hota.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import ModelBase

if TYPE_CHECKING:
    from app.models.call import Call


class TranscriptTurn(ModelBase):
    """A single speaker turn within a call transcript.

    Attributes:
        call_id: Parent call (FK).
        turn_index: 0-based order within the call.
        speaker: "agent" ya "client".
        text: Bola gaya text. Client ke liye UNTRUSTED (STT output).
        stt_confidence: Speech-recognition confidence (0.0–1.0), sirf client
            turns ke liye meaningful. None agar available nahi.
        spoken_at: Turn ka UTC waqt, agar available.
        call: Parent call relationship.
    """

    __tablename__ = "transcript_turns"
    __table_args__ = (
        Index("ix_transcript_turns_call_id", "call_id"),
        # Ek call ke andar turn_index unique hona chahiye — duplicate/out-of-order
        # protection.
        Index("ix_transcript_turns_call_turn", "call_id", "turn_index", unique=True),
    )

    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(nullable=False)
    speaker: Mapped[str] = mapped_column(String(10), nullable=False)
    text: Mapped[str] = mapped_column(String(8000), nullable=False)
    stt_confidence: Mapped[float | None] = mapped_column(nullable=True)
    spoken_at: Mapped[datetime | None] = mapped_column(nullable=True)

    call: Mapped[Call] = relationship("Call", back_populates="turns")

    def __repr__(self) -> str:
        """Return a PII-light representation (no transcript text).

        Returns:
            str: Debug-safe representation with id, call, speaker, index.
        """
        return (
            f"<TranscriptTurn id={self.id!r} call_id={self.call_id!r} "
            f"speaker={self.speaker!r} idx={self.turn_index}>"
        )