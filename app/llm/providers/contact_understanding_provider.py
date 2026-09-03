"""ContactUnderstanding provider interface and deterministic mock.

Ye woh boundary hai jahan (future) LLM client utterance ko structured
`ContactUnderstanding` mein badlega. Abhi sirf interface + deterministic mock —
koi Gemini/API/network.

LOCKED RULES:
    - Provider sirf structured proposal produce karta hai, ya exception raise
      karta hai. Failure ko domain result NAHI banata — decision (clarification)
      upper layer (engine/resolver) ka.
    - Provider threshold/confidence check NAHI karta — sirf output deta hai
      (interpretation_confidence included). Threshold deterministic consumer
      enforce karta hai.
    - Client utterance UNTRUSTED data hai — provider ise instruction ki tarah
      treat nahi karta. Output resolver deterministically validate karta hai;
      mock resolver ke rules bypass NAHI karta.

INPUT (bounded — locked): current utterance + trusted state + previous agent
    question + bounded memory + relevant recent turns. Full raw transcript by
    default NAHI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.contracts.contact_info import ContactChannel
from app.contracts.contact_understanding import (
    ContactIntent,
    ContactReference,
    ContactUnderstanding,
)
from app.core.constants import ConversationState


class ContactUnderstandingError(Exception):
    """Raised when a provider cannot produce a valid understanding.

    Provider failure/timeout/malformed-output → ye exception. Provider business
    decision NAHI leta; upper layer deterministic clarification path chunta hai.
    """


@dataclass(frozen=True)
class UnderstandingInput:
    """Bounded, mostly-trusted input for understanding a contact utterance.

    Client utterance UNTRUSTED hai (data, not instruction) — isliye alag field.
    Baaki context trusted. Full transcript NAHI — bounded (locked).

    Attributes:
        client_utterance: The client's spoken text. UNTRUSTED.
        current_state: Trusted current conversational state.
        agent_previous_question: What the agent last asked (trusted context).
        recent_turns: A few relevant recent turns (bounded), not full transcript.
    """

    client_utterance: str
    current_state: ConversationState
    agent_previous_question: str | None = None
    recent_turns: tuple[str, ...] = ()


class ContactUnderstandingProvider(ABC):
    """Interface for turning an utterance into a ContactUnderstanding.

    Real Gemini provider isi interface ko implement karega (later sitting).
    Provider sirf propose karta hai; resolver/engine authority rakhte hain.
    """

    @abstractmethod
    def understand(self, understanding_input: UnderstandingInput) -> ContactUnderstanding:
        """Produce a structured (untrusted) understanding, or raise.

        Args:
            understanding_input: Bounded input (untrusted utterance + trusted
                context).

        Returns:
            ContactUnderstanding: Untrusted structured proposal (resolver
            validates downstream).

        Raises:
            ContactUnderstandingError: On failure/timeout/malformed output.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class _ScriptedUnderstanding:
    """A scripted mock response keyed by utterance substring."""

    match: str
    understanding: ContactUnderstanding


class MockContactUnderstandingProvider(ContactUnderstandingProvider):
    """Deterministic mock provider for testing (no LLM).

    Configurable: scripted responses (utterance substring → understanding), a
    default, ya forced failure. Ye resolver ke rules bypass NAHI karta — sirf
    ContactUnderstanding produce karta hai; validation resolver karta hai.

    Attributes:
        scripts: Substring → understanding mappings (first match wins).
        default: Understanding jab koi script match na kare.
        should_fail: True → ContactUnderstandingError raise.
    """

    def __init__(
        self,
        scripts: tuple[_ScriptedUnderstanding, ...] = (),
        default: ContactUnderstanding | None = None,
        should_fail: bool = False,
    ) -> None:
        """Initialize the mock provider.

        Args:
            scripts: Ordered substring→understanding mappings.
            default: Fallback understanding (None → UNCLEAR/UNKNOWN).
            should_fail: If True, raise on every call.
        """
        self._scripts = scripts
        self._default = default or ContactUnderstanding(
            intent=ContactIntent.UNCLEAR,
            channel=ContactChannel.UNKNOWN,
            interpretation_confidence=0.0,
        )
        self._should_fail = should_fail

    def understand(self, understanding_input: UnderstandingInput) -> ContactUnderstanding:
        """Return a scripted/default understanding, or fail.

        Args:
            understanding_input: The bounded input.

        Returns:
            ContactUnderstanding: The mock understanding.

        Raises:
            ContactUnderstandingError: If configured to fail.
        """
        if self._should_fail:
            raise ContactUnderstandingError("mock provider configured to fail")
        text = understanding_input.client_utterance.lower()
        for script in self._scripts:
            if script.match.lower() in text:
                return script.understanding
        return self._default


def scripted(
    match: str,
    *,
    intent: ContactIntent = ContactIntent.PROVIDE_CONTACT,
    channel: ContactChannel = ContactChannel.EMAIL,
    value: str | None = None,
    reference: ContactReference | None = None,
    confidence: float = 0.9,
) -> _ScriptedUnderstanding:
    """Build a scripted mock understanding (test helper).

    Args:
        match: Utterance substring that triggers this response.
        intent: Contact intent to return.
        channel: Channel to return.
        value: Explicit value, if any.
        reference: Reference, if any.
        confidence: Interpretation confidence.

    Returns:
        _ScriptedUnderstanding: The scripted mapping.
    """
    return _ScriptedUnderstanding(
        match=match,
        understanding=ContactUnderstanding(
            intent=intent,
            channel=channel,
            value=value,
            reference=reference,
            interpretation_confidence=confidence,
        ),
    )