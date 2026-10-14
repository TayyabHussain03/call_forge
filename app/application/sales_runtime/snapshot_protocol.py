"""Protocol for TurnProcessors that support snapshot export/restore.

Processors own the authoritative state and the right to export it. The
persistent runtime never pokes at processor internals. Processors that
do not implement this protocol will fall back to RUNTIME_SNAPSHOT_MISSING
on recovery — fail closed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.application.sales_runtime.snapshot import ConversationRuntimeSnapshot


@runtime_checkable
class SnapshotCapable(Protocol):
    """Marker protocol: processors that can round-trip to a snapshot."""

    def export_snapshot(self, session_revision: int) -> ConversationRuntimeSnapshot:
        """Export the current authoritative runtime state."""
