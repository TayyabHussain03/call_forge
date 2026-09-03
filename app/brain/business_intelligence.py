"""Deterministic business-intelligence model.

Structured, provenance-tagged sales intelligence — NOT a smart layer. Ye koi LLM
call nahi karta, koi decision nahi leta, koi service/contact/pricing/state touch
nahi karta. Sirf facts, inferences, unknowns, provenance, aur history ko safely
representable banata hai.

THREE SEPARATE TYPES (structural, not documented-only):
    - ObservedSignal  → reported evidence + provenance (NO generic confidence)
    - InferredSignal  → reasoned conclusion + inference_confidence + evidence refs
    - UnknownSlot     → a known-unknown field (no value, no confidence)

CRITICAL INVARIANT — INFERENCE ≠ FACT: observed aur inferred ALAG types, ALAG
collections mein. Koi API inferred ko observed mein "promote" nahi karti. Agar
prospect baad mein fact explicitly bataye, to ek NAYA ObservedSignal banta hai —
purani inference history preserve rehti hai.

IMMUTABLE / ADDITIVE HISTORY: signals overwrite nahi hote; naye signals add hote
hain. Current/derived state history se compute hota hai, use replace kiye bina.

NO UNIVERSAL CONFIDENCE: observed → provenance only; inferred → inference_
confidence; unknown → none. Ye interpretation/action/selection/STT confidence se
kabhi mix nahi.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


class SourceKind(str, Enum):
    """How an observed fact entered the system (structured provenance)."""

    CLIENT_STATED = "client_stated"      # prospect ne khud bataya
    AGENT_STATED = "agent_stated"        # agent ne confirm/establish kiya
    LEAD_IMPORT = "lead_import"          # scraped/imported ke saath aaya
    EXTERNAL_DATA = "external_data"      # trusted external source (provenance-carried)


@dataclass(frozen=True)
class Provenance:
    """Structured provenance for an observed fact.

    Free-form string nahi — structured, taake "kahan se aaya / kaunse turn se /
    directly stated ya nahi" answerable ho.

    Attributes:
        source_kind: How this fact entered.
        source_turn: Conversation turn index that produced it (traceability).
        detail: Optional short structured note (not the primary mechanism).
    """

    source_kind: SourceKind
    source_turn: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ObservedSignal:
    """A reported/observed business fact with provenance.

    Evidence, NOT a guess — isliye koi generic confidence NAHI. Immutable.

    Attributes:
        field: The intelligence field (e.g. "has_website", "primary_problem").
        value: The observed value.
        provenance: Structured provenance (source/turn).
    """

    field: str
    value: str
    provenance: Provenance


@dataclass(frozen=True)
class InferredSignal:
    """A reasoned conclusion derived from observed evidence.

    Inference — isliye explicit `inference_confidence` (observed se ALAG, generic
    nahi). Evidence references rakhta hai (kis observed facts par based). Immutable.
    Ye kabhi observed nahi banta.

    Attributes:
        field: The intelligence field.
        value: The inferred value.
        inference_confidence: Confidence in THIS inference (0.0–1.0). Distinct
            semantic — kisi aur confidence se mix nahi.
        basis: Fields of observed signals this inference rests on (evidence refs).
        source_turn: Turn at which this inference was made.
    """

    field: str
    value: str
    inference_confidence: float
    basis: tuple[str, ...] = ()
    source_turn: int | None = None


@dataclass(frozen=True)
class UnknownSlot:
    """A known information slot for which no sufficient info is available.

    Value nahi, confidence nahi — sirf ye record ke ye field abhi unknown hai.

    Attributes:
        field: The unknown field name.
    """

    field: str


@dataclass(frozen=True)
class BusinessIntelligence:
    """Deterministic, additive, provenance-tagged intelligence store.

    Teen ALAG collections — observed/inferred/unknown structurally separate.
    Additive: har "record" method ek NAYA BusinessIntelligence deta hai (immutable);
    purane signals kabhi remove/overwrite nahi hote. History preserve.

    Ye koi decision/LLM/network/service/pricing NAHI karta — sirf data + provenance
    + history.

    Attributes:
        observed: Historical observed signals (additive).
        inferred: Historical inferred signals (additive).
        unknown: Known-unknown slots.
    """

    observed: tuple[ObservedSignal, ...] = ()
    inferred: tuple[InferredSignal, ...] = ()
    unknown: tuple[UnknownSlot, ...] = ()

    def record_observed(
        self, field_name: str, value: str, provenance: Provenance
    ) -> BusinessIntelligence:
        """Return a new BI with an observed signal appended (additive).

        Purane signals unchanged — ye ek nayi immutable copy deta hai. Agar ye
        field pehle unknown tha, to unknown slot resolve ho jaata hai (hata diya),
        lekin observed/inferred history preserve.

        Args:
            field_name: The intelligence field.
            value: The observed value.
            provenance: Structured provenance.

        Returns:
            BusinessIntelligence: New instance with the observation added.
        """
        signal = ObservedSignal(field=field_name, value=value, provenance=provenance)
        new_unknown = tuple(u for u in self.unknown if u.field != field_name)
        return replace(
            self, observed=(*self.observed, signal), unknown=new_unknown
        )

    def record_inferred(
        self,
        field_name: str,
        value: str,
        inference_confidence: float,
        basis: tuple[str, ...] = (),
        source_turn: int | None = None,
    ) -> BusinessIntelligence:
        """Return a new BI with an inferred signal appended (additive).

        Inference kabhi observed nahi banti — ye sirf `inferred` collection mein
        add hoti hai. History preserve.

        Args:
            field_name: The intelligence field.
            value: The inferred value.
            inference_confidence: Confidence in this inference.
            basis: Observed fields this inference rests on.
            source_turn: Turn of inference.

        Returns:
            BusinessIntelligence: New instance with the inference added.
        """
        signal = InferredSignal(
            field=field_name,
            value=value,
            inference_confidence=inference_confidence,
            basis=basis,
            source_turn=source_turn,
        )
        return replace(self, inferred=(*self.inferred, signal))

    def record_unknown(self, field_name: str) -> BusinessIntelligence:
        """Return a new BI marking a field as a known-unknown.

        Agar field pehle se observed hai to unknown add karne ka koi matlab nahi
        (idempotent-ish) — lekin ye method history destroy nahi karta; sirf slot
        add karta hai agar duplicate na ho.

        Args:
            field_name: The unknown field.

        Returns:
            BusinessIntelligence: New instance with the unknown slot (if new).
        """
        if any(u.field == field_name for u in self.unknown):
            return self
        return replace(self, unknown=(*self.unknown, UnknownSlot(field=field_name)))

    def latest_observed(self, field_name: str) -> ObservedSignal | None:
        """Return the most recent observed signal for a field, if any.

        Current/derived state ko history se compute karne ke liye — history
        replace kiye bina. Additive log se "latest" derive hota hai.

        Args:
            field_name: The field to look up.

        Returns:
            ObservedSignal | None: The latest observed signal, or None.
        """
        matches = [s for s in self.observed if s.field == field_name]
        return matches[-1] if matches else None

    def latest_inferred(self, field_name: str) -> InferredSignal | None:
        """Return the most recent inferred signal for a field, if any.

        Args:
            field_name: The field to look up.

        Returns:
            InferredSignal | None: The latest inferred signal, or None.
        """
        matches = [s for s in self.inferred if s.field == field_name]
        return matches[-1] if matches else None

    def is_known(self, field_name: str) -> bool:
        """Whether a field has any observed signal (a genuine fact).

        NOTE: inferred alone does NOT make a field "known" as fact — sirf observed.
        Ye inference≠fact ko surface par bhi enforce karta hai.

        Args:
            field_name: The field to check.

        Returns:
            bool: True only if an observed signal exists for the field.
        """
        return any(s.field == field_name for s in self.observed)