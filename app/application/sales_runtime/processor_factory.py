"""TurnProcessor reconstruction after restart.

TurnProcessor instances are process-local and intentionally not persisted.
After restart, callers rebuild the correct processor from trusted persisted
configuration via a factory. Factories never pickle runtime objects.

Determinism rule: same persisted (ChatSession, campaign, policy) must
reconstruct a semantically equivalent processor. Provider/model selection
flows from trusted campaign/runtime configuration, never from wall clock
or randomness.
"""

from __future__ import annotations

from typing import Protocol

from app.leads.campaign.contracts import CampaignConfig
from app.precall.contracts import PreCallDiscoveryPolicy
from app.runtime.chat.contracts import ChatSession
from app.runtime.contracts import TurnProcessor


class TurnProcessorFactory(Protocol):
    """Rebuilds a TurnProcessor for a given persisted session."""

    def create_for_session(
        self,
        session: ChatSession,
        campaign: CampaignConfig,
        policy: PreCallDiscoveryPolicy,
    ) -> TurnProcessor:
        """Deterministically construct a TurnProcessor from trusted config."""
