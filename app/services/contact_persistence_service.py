"""Service that persists confirmed contact methods from engine intent.

Ye E2 ka WAHID component hai jo persistence-intent ko repository operations mein
translate karta hai. Engine sirf `ConversationResult.contact_to_persist` banata
hai; ye service use consume karke `ContactMethodRepository` par persist karta hai
(status CONFIRMED).

BOUNDARIES (guide ke mutabiq): ye service natural language parse nahi karta,
references resolve nahi karta, LLM/telephony nahi, confirmation decide nahi,
preferred decide nahi, Business.phone_e164 alter nahi, contact ownership infer
nahi, persist-intent khud nahi banata, malformed values repair nahi.

CONFIRMATION vs PERSISTENCE: conversation-decision succeed hona aur persistence
succeed hona ALAG hain. Ye service ek explicit PersistenceResult deta hai —
repository failure ko fake success mein nahi badalta.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.contracts.pending_contact_method import PendingContactMethod
from app.models.contact import ContactInfoStatus
from app.repositories.contact_method_repository import ContactMethodRepository


class PersistenceOutcome(str, Enum):
    """Outcome of a persistence attempt."""

    PERSISTED = "persisted"          # method persisted as CONFIRMED
    NOOP = "noop"                    # no intent → nothing to do
    INVALID_INTENT = "invalid_intent"  # intent missing required data
    FAILED = "failed"               # repository/infra failure


@dataclass(frozen=True)
class PersistenceResult:
    """Explicit result of a persistence attempt.

    Conversation success se ALAG — ye sirf persistence operation ka natija hai.

    Attributes:
        outcome: The persistence outcome.
        method_id: Persisted method id (PERSISTED par), ya None.
        reason: Machine-readable detail for non-persisted outcomes.
    """

    outcome: PersistenceOutcome
    method_id: str | None = None
    reason: str | None = None


class ContactPersistenceService:
    """Persists confirmed contact methods from engine persist-intent.

    Attributes:
        repo: The ContactMethodRepository to persist through.
    """

    def __init__(self, repository: ContactMethodRepository) -> None:
        """Initialize with a contact method repository.

        Args:
            repository: The repository bound to the active session.
        """
        self._repo = repository

    @staticmethod
    def _is_valid(intent: PendingContactMethod) -> bool:
        """Whether an intent carries all required persistence data.

        Args:
            intent: The persist intent.

        Returns:
            bool: True if contact_id, channel, value, provenance all present and
            channel is not 'unknown'.
        """
        return bool(
            intent.contact_id
            and intent.channel
            and intent.channel != "unknown"
            and intent.value_normalized
            and intent.value_normalized.strip()
            and intent.provenance
        )

    async def persist(
        self, intent: PendingContactMethod | None
    ) -> PersistenceResult:
        """Persist a confirmed contact method from an engine intent.

        Locked rules:
            - intent None → NOOP (no repository write).
            - intent missing required data → INVALID_INTENT (no silent write).
            - valid → repository upsert (canonical identity dedup) + status
              CONFIRMED. Provenance carried as-is (never remapped/invented).
            - repository failure → FAILED (never faked as success).

        Ye Business.phone_e164 ko kabhi touch nahi karta — sirf ContactMethod.

        Args:
            intent: The persist intent from ConversationResult, or None.

        Returns:
            PersistenceResult: Explicit outcome.
        """
        # Case A: no intent → NO-OP.
        if intent is None:
            return PersistenceResult(outcome=PersistenceOutcome.NOOP)

        # Case B: missing required data → no silent write.
        if not self._is_valid(intent):
            return PersistenceResult(
                outcome=PersistenceOutcome.INVALID_INTENT,
                reason="intent missing required contact_id/channel/value/provenance",
            )

        # Case C: persist via repository (dedup by canonical identity), CONFIRMED.
        try:
            method, _created = await self._repo.upsert_method(
                contact_id=intent.contact_id,
                channel=intent.channel,
                value_normalized=intent.value_normalized,
                provenance=intent.provenance,  # carried as-is, never remapped
                value_raw=intent.value_raw,
            )
            await self._repo.update_status(method, ContactInfoStatus.CONFIRMED)
            # Flush (not commit) so the generated id is available for the result.
            # Commit remains the caller's responsibility (transaction boundary).
            await self._repo.session.flush()
        except Exception as exc:  # infra/repository failure — surface, don't fake
            return PersistenceResult(
                outcome=PersistenceOutcome.FAILED,
                reason=type(exc).__name__,
            )

        return PersistenceResult(
            outcome=PersistenceOutcome.PERSISTED, method_id=method.id
        )