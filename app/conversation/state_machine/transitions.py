"""Transition results produced by the state machine.

`ApprovedTransition` woh TRUSTED result hai jo machine ke `apply_transition` se
nikalta hai. Ye tab hi banta hai jab ek action structurally eligible ho — iska
wujood hi is baat ka proof hai ke transition authorized hai.

EXTENSIBLE OUTCOME (guide ka safeguard): `outcome` abhi ek optional string label
hai. Future mein isko call-outcome (NO_ANSWER, VOICEMAIL, DISCONNECTED) aur
lead-status (QUALIFIED, DO_NOT_CALL) mein alag kiya ja sakta hai bina machine ko
rewrite kiye — kyunki callers `outcome` ko ek opaque label ki tarah treat karte
hain, na ke kisi fixed enum ki tarah.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.constants import AgentAction, ConversationState


@dataclass(frozen=True)
class ApprovedTransition:
    """The authoritative result of applying an action to the machine.

    Sirf machine ise banati hai (apply_transition ke through). Callers ise seedha
    construct nahi karte — warna authority boundary toot jaati.

    Attributes:
        action: Jo action apply hua (structurally eligible tha).
        from_state: Transition se pehle ki state.
        to_state: Transition ke baad ki authoritative state. Detour restore ke
            case mein ye resolved previous state hoti hai (sentinel nahi).
        outcome: Optional explicit outcome label (sirf terminal transitions par
            meaningful). Extensible — future mein call-outcome/lead-status split.
        preserved_interest: True jab transition ne lead ka interest barqarar
            rakha (busy→callback), taake lead lost na samjhi jaaye.
        was_detour_return: True jab ye ek detour (HANDLE_QUESTION) se previous
            state par wapasi thi.
    """

    action: AgentAction
    from_state: ConversationState
    to_state: ConversationState
    outcome: str | None = None
    preserved_interest: bool = False
    was_detour_return: bool = False

    @property
    def is_terminal_outcome(self) -> bool:
        """Whether this transition carried an explicit terminal outcome.

        Returns:
            bool: True when an outcome label is present.
        """
        return self.outcome is not None