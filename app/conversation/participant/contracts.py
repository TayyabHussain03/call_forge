"""Participant continuity + discussion coverage contracts.

Deterministic typed boundary between untrusted provider interpretation and
authoritative conversation state. Provider proposes evidence; policy decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Participant identity + continuity state
# ---------------------------------------------------------------------------


class ParticipantContinuityState(str, Enum):
    """Deterministic classification of the current speaker's identity."""

    SAME_CONFIRMED = "same_confirmed"
    NEW_CONFIRMED = "new_confirmed"
    LIKELY_NEW = "likely_new"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class ParticipantEvidenceKind(str, Enum):
    """Typed semantic evidence; never raw keywords."""

    NONE = "none"
    EXPLICIT_SELF_IDENTIFICATION = "explicit_self_identification"
    EXPLICIT_NOT_PART_OF_PRIOR_DISCUSSION = "explicit_not_part_of_prior_discussion"
    EXPLICIT_HANDOFF = "explicit_handoff"
    CURRENT_PARTICIPANT_CORRECTION = "current_participant_correction"
    INFERRED = "inferred"


class ParticipantRoleHint(str, Enum):
    """Role claim surface for a participant. Not authority."""

    UNKNOWN = "unknown"
    OWNER = "owner"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    COLLEAGUE = "colleague"
    GATEKEEPER = "gatekeeper"
    ASSISTANT = "assistant"
    CUSTOMER = "customer"


@dataclass(frozen=True)
class ParticipantReference:
    """Session-scoped participant identity. Not an authentication token.

    `reference_id` is a deterministic string like "participant_1". `role_hint`
    is a conversational signal — AuthorityPolicy remains separate truth.
    """

    reference_id: str
    role_hint: ParticipantRoleHint = ParticipantRoleHint.UNKNOWN
    display_label: str | None = None

    def __post_init__(self) -> None:
        if not self.reference_id or len(self.reference_id) > 80:
            raise ValueError("reference_id must be 1-80 chars")
        if self.display_label is not None and len(self.display_label) > 120:
            raise ValueError("display_label exceeds 120 chars")


@dataclass(frozen=True)
class ParticipantContinuityEvidence:
    """Untrusted provider proposal — the semantic signal, not the raw text."""

    kind: ParticipantEvidenceKind
    source_turn_id: str
    claimed_role_hint: ParticipantRoleHint = ParticipantRoleHint.UNKNOWN

    def __post_init__(self) -> None:
        if not self.source_turn_id or len(self.source_turn_id) > 100:
            raise ValueError("source_turn_id must be 1-100 chars")


@dataclass(frozen=True)
class ParticipantContinuityDecision:
    """Policy-authoritative decision made from provider evidence."""

    state: ParticipantContinuityState
    active_participant: ParticipantReference
    evidence_kind: ParticipantEvidenceKind
    archived_prior: ParticipantReference | None = None

    def __post_init__(self) -> None:
        if self.state == ParticipantContinuityState.NEW_CONFIRMED:
            if self.archived_prior is None:
                raise ValueError(
                    "NEW_CONFIRMED must archive the prior participant reference"
                )
            if self.archived_prior.reference_id == self.active_participant.reference_id:
                raise ValueError("archived_prior must differ from active_participant")


# ---------------------------------------------------------------------------
# Participant-scoped conversation state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticipantConversationState:
    """Per-participant facts that must not transfer between people.

    Business/BCI facts stay in BCI. These are strictly individual signals.
    """

    participant: ParticipantReference
    heard_explanation_concepts: frozenset[str] = frozenset()
    role_hint: ParticipantRoleHint = ParticipantRoleHint.UNKNOWN
    interest_observed: bool | None = None
    objection_concepts: frozenset[str] = frozenset()
    language_preference: str | None = None
    busy_observed: bool | None = None

    def __post_init__(self) -> None:
        if len(self.heard_explanation_concepts) > 32:
            raise ValueError("heard_explanation_concepts bounded to 32")
        if len(self.objection_concepts) > 16:
            raise ValueError("objection_concepts bounded to 16")


# ---------------------------------------------------------------------------
# Discussion coverage — participant-aware repetition memory
# ---------------------------------------------------------------------------


class ExplanationDepth(str, Enum):
    """Semantic depth; semantic IDs only, never generated prose."""

    NONE = "none"
    MENTIONED = "mentioned"
    BRIEF = "brief"
    EXPLAINED = "explained"
    DETAILED = "detailed"


@dataclass(frozen=True)
class DiscussionCoverageEntry:
    """One (participant, concept) exposure record."""

    participant_reference_id: str
    concept_id: str
    depth: ExplanationDepth
    source_turn_id: str

    def __post_init__(self) -> None:
        if not self.participant_reference_id or len(self.participant_reference_id) > 80:
            raise ValueError("participant_reference_id must be 1-80 chars")
        if not self.concept_id or len(self.concept_id) > 100:
            raise ValueError("concept_id must be 1-100 chars")
        if not self.source_turn_id or len(self.source_turn_id) > 100:
            raise ValueError("source_turn_id must be 1-100 chars")


@dataclass(frozen=True)
class DiscussionCoverageState:
    """Bounded participant-aware record of what has been discussed."""

    entries: tuple[DiscussionCoverageEntry, ...] = ()
    asked_question_concepts: tuple[tuple[str, str], ...] = ()  # (participant_ref, concept)
    handled_objection_concepts: tuple[tuple[str, str], ...] = ()
    discussed_evidence_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if len(self.entries) > 128:
            raise ValueError("discussion coverage entries bounded to 128")
        if len(self.asked_question_concepts) > 128:
            raise ValueError("asked_question_concepts bounded to 128")
        if len(self.handled_objection_concepts) > 64:
            raise ValueError("handled_objection_concepts bounded to 64")
        if len(self.discussed_evidence_ids) > 128:
            raise ValueError("discussed_evidence_ids bounded to 128")

    def heard_by(self, participant_reference_id: str) -> frozenset[str]:
        return frozenset(
            e.concept_id for e in self.entries
            if e.participant_reference_id == participant_reference_id
        )

    def depth_for(
        self, participant_reference_id: str, concept_id: str,
    ) -> ExplanationDepth:
        best = ExplanationDepth.NONE
        order = {d: i for i, d in enumerate(
            [ExplanationDepth.NONE, ExplanationDepth.MENTIONED,
             ExplanationDepth.BRIEF, ExplanationDepth.EXPLAINED,
             ExplanationDepth.DETAILED]
        )}
        for e in self.entries:
            if (e.participant_reference_id == participant_reference_id
                    and e.concept_id == concept_id
                    and order[e.depth] > order[best]):
                best = e.depth
        return best


# ---------------------------------------------------------------------------
# Re-explanation decision — deterministic policy output
# ---------------------------------------------------------------------------


class ReExplanationStrategy(str, Enum):
    """Deterministic strategy; feeds into existing response planning."""

    SUPPRESS = "suppress"
    CONCISE_RECAP = "concise_recap"
    FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT = "full_re_explain_for_new_participant"
    DEEPEN = "deepen"
    SIMPLIFY = "simplify"
    NO_PRIOR_COVERAGE = "no_prior_coverage"


@dataclass(frozen=True)
class ReExplanationDecision:
    """Which strategy a response planner should use for a concept + participant."""

    strategy: ReExplanationStrategy
    concept_id: str
    participant_reference_id: str
    prior_depth: ExplanationDepth
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or len(self.reason) > 200:
            raise ValueError("reason must be 1-200 chars")
