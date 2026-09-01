"""Bounded contact clarification engine.

Deterministic control component jo batata hai: contact (email/phone) clear hai,
ya clarification chahiye (repeat → spell/digit), ya attempts exhaust ho gaye.
Ye engine SIRF ye faisla deta hai — na extract karta, na validate karta, na
fallback chalata, na state mutate karta, na LLM/DB/network touch karta.

CANONICAL FLOW:
    Contact Extraction → ClarificationEngine → (CLEAR|RETRY) → ActionValidator
        → StateMachine
    Exhaustion par: ClarificationEngine → EXHAUSTED → caller FallbackEngine.

KEY SEMANTICS (guide ke mutabiq):
    - Attempt tabhi badhta hai jab genuinely unclear extraction ho — clear
      extraction, confirmation, ya unrelated turn par NAHI.
    - Confirmation clarification counter ko chhoti bhi nahi — bilkul alag flow.
    - EXHAUSTED sticky hai: ek session mein attempts>=max ke baad aur unclear
      input na increment kare na reset. Sirf explicit new_session() reset karta.
    - Negative attempts silently normalize NAHI — invalid state, raise.
    - Sab state immutable: evaluate() purana state kabhi mutate nahi karta,
      naya ClarificationState return karta hai.
    - Response keys stable machine-readable — koi natural-language text nahi.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.contracts.contact_info import ContactChannel
from app.conversation.state_machine.states import MachineConfig
from app.core.constants import AgentAction
from app.core.exceptions import ValidationError


class ClarificationStage(str, Enum):
    """Stage of an active clarification sequence for one contact channel."""

    NONE = "none"            # abhi koi clarification nahi hui
    REPEAT = "repeat"        # ek baar unclear -> repeat maango
    SPELL = "spell"          # phir unclear -> spell (email) / digit-by-digit (phone)
    EXHAUSTED = "exhausted"  # attempts>=max -> aur clarification nahi


class ClarificationOutcome(str, Enum):
    """The high-level result of evaluating a contact-clarification situation."""

    CLEAR = "clear"          # usable candidate mil gaya -> confirmation ki taraf
    RETRY = "retry"          # clarification available -> repeat/spell
    EXHAUSTED = "exhausted"  # attempts khatam -> caller fallback chalaye


# Stable machine-readable response keys per channel + stage. Koi natural-language
# text yahan NAHI — wording baad mein LLM/prompt layer bharega.
_RESPONSE_KEYS: dict[ContactChannel, dict[ClarificationStage, str]] = {
    ContactChannel.EMAIL: {
        ClarificationStage.REPEAT: "repeat_email",
        ClarificationStage.SPELL: "spell_email",
    },
    ContactChannel.PHONE: {
        ClarificationStage.REPEAT: "repeat_phone",
        ClarificationStage.SPELL: "digits_phone",
    },
}


@dataclass(frozen=True)
class ClarificationState:
    """Immutable session-scoped state of one channel's clarification sequence.

    Ye per active contact-collection session hai, per-person lifetime nahi.
    Immutable — engine har evaluation par naya state return karta hai.

    Attributes:
        channel: Kaunsa contact channel (email/phone).
        attempts: Ab tak kitni genuinely-unclear clarification koshishein hui.
        stage: Current clarification stage.
        max_attempts: Config-driven maximum for this channel.
    """

    channel: ContactChannel
    attempts: int
    stage: ClarificationStage
    max_attempts: int

    def __post_init__(self) -> None:
        """Validate invariants; reject corrupted state loudly.

        Raises:
            ValidationError: Agar attempts negative ho (silently normalize nahi).
        """
        if self.attempts < 0:
            raise ValidationError(
                f"ClarificationState.attempts cannot be negative: {self.attempts}"
            )
        if self.max_attempts < 1:
            raise ValidationError(
                f"ClarificationState.max_attempts must be >= 1: {self.max_attempts}"
            )


@dataclass(frozen=True)
class ClarificationDecision:
    """The deterministic result of one clarification evaluation.

    Attributes:
        channel: The contact channel evaluated.
        outcome: CLEAR / RETRY / EXHAUSTED.
        stage: The resulting stage.
        action: For RETRY → CLARIFY_CONTACT; for CLEAR → CONFIRM_EMAIL; for
            EXHAUSTED → None (caller invokes FallbackEngine; no synthetic action).
        response_key: Stable identifier for RETRY (repeat_/spell_/digits_); None
            for CLEAR/EXHAUSTED.
        new_state: The updated immutable session state the caller should keep.
    """

    channel: ContactChannel
    outcome: ClarificationOutcome
    stage: ClarificationStage
    action: AgentAction | None
    response_key: str | None
    new_state: ClarificationState


class ClarificationEngine:
    """Deterministically evaluates bounded contact clarification.

    Attributes:
        config: Machine configuration (holds per-channel attempt limits).
    """

    def __init__(self, config: MachineConfig) -> None:
        """Initialize with configuration providing attempt limits.

        Args:
            config: Loaded MachineConfig. Limits `email_clarification_max_attempts`
                aur `phone_clarification_max_attempts` yahan se aate hain.
        """
        self._config = config

    def _max_for(self, channel: ContactChannel) -> int:
        """Return the configured max clarification attempts for a channel.

        Args:
            channel: The contact channel.

        Returns:
            int: Configured maximum (defaults to 3 if unset).

        Raises:
            ValidationError: Agar channel unsupported ho.
        """
        if channel == ContactChannel.EMAIL:
            return int(self._config.limits.get("email_clarification_max_attempts", 3))
        if channel == ContactChannel.PHONE:
            return int(self._config.limits.get("phone_clarification_max_attempts", 3))
        raise ValidationError(f"Unsupported clarification channel: {channel}")

    def new_session(self, channel: ContactChannel) -> ClarificationState:
        """Create a fresh clarification session state for a channel.

        Naya immutable state banata hai (NONE, 0 attempts). Kisi purane state ko
        mutate nahi karta.

        Args:
            channel: The contact channel to start collecting.

        Returns:
            ClarificationState: A fresh state at stage NONE, attempts 0.
        """
        return ClarificationState(
            channel=channel,
            attempts=0,
            stage=ClarificationStage.NONE,
            max_attempts=self._max_for(channel),
        )

    def evaluate(
        self, state: ClarificationState, extraction_clear: bool
    ) -> ClarificationDecision:
        """Evaluate the current clarification situation deterministically.

        `state` READ-ONLY treat hota hai — ye method use mutate nahi karta, ek
        naya ClarificationState `new_state` mein return karta hai.

        Logic:
            - CLEAR: extraction_clear True → outcome CLEAR, stage NONE (clean
              reset), attempts unchanged (increment NAHI), action CONFIRM_EMAIL.
            - Already EXHAUSTED (sticky): outcome EXHAUSTED, no increment/reset.
            - Unclear with attempts+1 < max → RETRY at REPEAT (first) or SPELL.
            - Unclear reaching max → EXHAUSTED.

        Args:
            state: Current immutable session state.
            extraction_clear: True agar is turn ka contact extraction usable/clear
                tha (deterministic validation ka result — engine khud extract
                nahi karta).

        Returns:
            ClarificationDecision: The deterministic decision + new state.
        """
        # 1. CLEAR: koi increment nahi, stage clean NONE par reset.
        if extraction_clear:
            new_state = ClarificationState(
                channel=state.channel,
                attempts=state.attempts,  # unchanged — clear counts nahi karta
                stage=ClarificationStage.NONE,
                max_attempts=state.max_attempts,
            )
            return ClarificationDecision(
                channel=state.channel,
                outcome=ClarificationOutcome.CLEAR,
                stage=ClarificationStage.NONE,
                action=AgentAction.CONFIRM_EMAIL,
                response_key=None,
                new_state=new_state,
            )

        # 2. EXHAUSTED sticky: already exhausted ya attempts>=max → aur kuch nahi.
        if (
            state.stage == ClarificationStage.EXHAUSTED
            or state.attempts >= state.max_attempts
        ):
            new_state = ClarificationState(
                channel=state.channel,
                attempts=state.attempts,  # no increment
                stage=ClarificationStage.EXHAUSTED,
                max_attempts=state.max_attempts,
            )
            return ClarificationDecision(
                channel=state.channel,
                outcome=ClarificationOutcome.EXHAUSTED,
                stage=ClarificationStage.EXHAUSTED,
                action=None,  # synthetic action nahi — caller fallback chalaye
                response_key=None,
                new_state=new_state,
            )

        # 3. Unclear + retry available: increment aur progress karo.
        next_attempts = state.attempts + 1
        if next_attempts >= state.max_attempts:
            next_stage = ClarificationStage.EXHAUSTED
        elif next_attempts == 1:
            next_stage = ClarificationStage.REPEAT
        else:
            next_stage = ClarificationStage.SPELL

        new_state = ClarificationState(
            channel=state.channel,
            attempts=next_attempts,
            stage=next_stage,
            max_attempts=state.max_attempts,
        )

        if next_stage == ClarificationStage.EXHAUSTED:
            return ClarificationDecision(
                channel=state.channel,
                outcome=ClarificationOutcome.EXHAUSTED,
                stage=ClarificationStage.EXHAUSTED,
                action=None,
                response_key=None,
                new_state=new_state,
            )

        return ClarificationDecision(
            channel=state.channel,
            outcome=ClarificationOutcome.RETRY,
            stage=next_stage,
            action=AgentAction.CLARIFY_CONTACT,
            response_key=_RESPONSE_KEYS[state.channel][next_stage],
            new_state=new_state,
        )