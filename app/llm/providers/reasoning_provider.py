"""Reasoning provider interface and deterministic mock.

Provider-agnostic reasoning boundary: later Gemini/OpenAI/Claude isi interface ke
peeche plug honge bina Brain/engine/catalog/state-machine rewrite ke. Ye sitting
COMPLETELY OFFLINE + DETERMINISTIC — koi SDK/API/network/key.

CORE PRINCIPLE:
    ReasoningProvider PROPOSES. Deterministic system DECIDES. Provider replaceable.
    Mock transparent, scripted, dumb by design — koi hidden intelligence/authority/
    policy.

PROVIDER BOUNDARY:
    - Input: sirf BrainInput (bounded). Koi DB session/catalog/repo/state-machine/
      contact-resolver dependency.
    - Output: sirf BrainProposal (untrusted), ya ReasoningError raise.
    - Provider NAHI karta: business rules, authority, catalog/contact validation,
      state transitions, persistence, pricing, discounts, execution, fallback,
      confidence thresholds.

FAILURE MODEL (locked): failure = EXCEPTION, domain result NAHI. Provider
    BrainProposal(action=ERROR) ya SelectionResult(FAILED) NAHI deta — sirf success
    → BrainProposal ya failure → ReasoningError. Domain decision upper layer ka.

NO THRESHOLDS (locked): provider low action_confidence (0.10) ko AS-IS return karta
    hai — koi min-confidence/gating/accept-reject nahi. Threshold future deterministic
    consumer enforce karega.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.brain.contracts import BrainInput, BrainProposal


class ReasoningError(Exception):
    """Provider-level reasoning failure boundary.

    Provider failure/unavailability/internal-error/malformed-output (future real
    providers) → ye exception. Domain fallback/policy NAHI — upper layer decide
    karti hai. Jaan-boojh kar ek clean boundary; 15 speculative subclasses nahi.
    """


@dataclass(frozen=True)
class ReasoningTrace:
    """Small typed trace metadata a future orchestrator can attach cleanly.

    Ye observability SUBSYSTEM nahi — sirf ek chhota typed structure taake future
    orchestrator provider/correlation/status track kar sake. Koi DB persistence,
    koi secret/PII/API-key. Latency/failure-type future real providers add karenge.

    Attributes:
        provider_name: Which provider produced the result (e.g. "mock").
        correlation_id: Optional caller-supplied id to correlate the call.
        succeeded: Whether the reasoning call succeeded.
    """

    provider_name: str
    correlation_id: str | None = None
    succeeded: bool = True


class ReasoningProvider(ABC):
    """Interface for producing a BrainProposal from a BrainInput.

    Intentionally minimal — application ka reasoning CONTRACT, vendor ki API NAHI.
    Koi Gemini/OpenAI/Claude concept (model names, generation config, SDK objects,
    token/safety fields) yahan NAHI — woh provider implementations mein.

    Ek invocation = exactly ek logical reasoning operation → ek BrainProposal ya
    ek ReasoningError. Koi retry/regeneration/multi-call loop provider mein NAHI.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider's stable name (for traceability).

        Returns:
            str: Provider identifier (e.g. "mock", "gemini").
        """
        raise NotImplementedError

    @abstractmethod
    def reason(self, brain_input: BrainInput) -> BrainProposal:
        """Produce an untrusted BrainProposal, or raise on failure.

        Provider BrainInput ko READ-ONLY treat karta hai — mutate nahi karta.
        Output untrusted — schema-valid hone ka matlab business-correct NAHI.

        Args:
            brain_input: Bounded, orchestrator-assembled input. current_utterance
                untrusted data hai.

        Returns:
            BrainProposal: Untrusted proposal (deterministic layers validate karenge).

        Raises:
            ReasoningError: On provider failure. Domain result NAHI.
        """
        raise NotImplementedError


class MockReasoningProvider(ReasoningProvider):
    """Deterministic, transparent, scripted mock reasoning provider.

    Ye future end-to-end offline simulations enable karta hai bina LLM/network.
    DUMB BY DESIGN — sirf configured proposals REPLAY karta hai. Koi inference,
    koi "if website_exists propose SEO" rule engine, koi keyword logic, koi random/
    time-based behavior. Intelligence future real providers mein.

    Teen deterministic modes:
        - Scripted: proposals ka ek ordered sequence — har call agla proposal
          return karta hai (exact order preserve). Sequence khatam → default (ya
          error agar default na ho).
        - Default: har call ek configured proposal return karta hai.
        - Failure: har call ReasoningError raise karta hai.

    Attributes:
        provider_name: For traceability.
    """

    def __init__(
        self,
        default: BrainProposal | None = None,
        scripted: tuple[BrainProposal, ...] = (),
        should_fail: bool = False,
        provider_name: str = "mock",
    ) -> None:
        """Initialize the deterministic mock.

        Args:
            default: Proposal returned when no scripted item remains. None + no
                scripted → calls raise (nothing to return).
            scripted: Ordered proposals replayed one per call (exact order).
            should_fail: If True, every call raises ReasoningError.
            provider_name: Stable name for traceability.
        """
        self._default = default
        self._scripted = scripted
        self._should_fail = should_fail
        self._provider_name = provider_name
        self._call_count = 0

    @property
    def name(self) -> str:
        """Return the mock provider's name.

        Returns:
            str: The configured provider name.
        """
        return self._provider_name

    @property
    def call_count(self) -> int:
        """Return how many times reason() has been invoked.

        Traceability/test helper — exact invocation count.

        Returns:
            int: Number of reason() calls so far.
        """
        return self._call_count

    def reason(self, brain_input: BrainInput) -> BrainProposal:
        """Replay the next configured proposal, or raise (deterministic).

        Ye NO-inference hai — sirf replay. BrainInput mutate nahi hota (frozen +
        untouched). Same config + same call-index → same output (deterministic).

        Args:
            brain_input: The bounded input (read-only).

        Returns:
            BrainProposal: The next scripted proposal, or the default.

        Raises:
            ReasoningError: If configured to fail, or nothing left to return.
        """
        index = self._call_count
        self._call_count += 1

        if self._should_fail:
            raise ReasoningError(f"mock provider {self._provider_name!r} configured to fail")

        if index < len(self._scripted):
            return self._scripted[index]

        if self._default is not None:
            return self._default

        raise ReasoningError(
            f"mock provider {self._provider_name!r} has no proposal for call {index}"
        )