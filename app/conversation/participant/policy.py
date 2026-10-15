"""Deterministic participant continuity + re-explanation policy.

Provider proposes typed evidence; policy decides. No keywords, no name
inference, no probability thresholds. Just deterministic mapping from
typed evidence kinds to a bounded continuity state and strategy.
"""

from __future__ import annotations

from dataclasses import replace

from app.conversation.participant.contracts import (
    DiscussionCoverageState,
    ExplanationDepth,
    ParticipantContinuityDecision,
    ParticipantContinuityEvidence,
    ParticipantContinuityState,
    ParticipantEvidenceKind,
    ParticipantReference,
    ParticipantRoleHint,
    ReExplanationDecision,
    ReExplanationStrategy,
)


class ParticipantContinuityPolicy:
    """Maps provider evidence → authoritative continuity decision.

    Decision rules:
        EXPLICIT_SELF_IDENTIFICATION with same reference → SAME_CONFIRMED
        EXPLICIT_NOT_PART_OF_PRIOR_DISCUSSION → NEW_CONFIRMED
        EXPLICIT_HANDOFF → NEW_CONFIRMED
        CURRENT_PARTICIPANT_CORRECTION → SAME_CONFIRMED
        INFERRED → LIKELY_NEW (never NEW_CONFIRMED without explicit evidence)
        NONE → UNKNOWN (or SAME_CONFIRMED if a prior active exists; policy
                        preserves the known speaker when no evidence contradicts)

    Ambiguous phrases like "What was this about?" MUST reach this policy as
    `NONE` so the active participant is preserved — the provider must not
    conjure evidence from such phrasing.
    """

    def decide(
        self,
        active: ParticipantReference | None,
        evidence: ParticipantContinuityEvidence,
        new_reference_factory,
    ) -> ParticipantContinuityDecision:
        if evidence.kind == ParticipantEvidenceKind.EXPLICIT_NOT_PART_OF_PRIOR_DISCUSSION:
            if active is None:
                new_ref = new_reference_factory(evidence.claimed_role_hint)
                return ParticipantContinuityDecision(
                    state=ParticipantContinuityState.NEW_CONFIRMED,
                    active_participant=new_ref,
                    evidence_kind=evidence.kind,
                    archived_prior=ParticipantReference(
                        reference_id=f"{new_ref.reference_id}__prior_placeholder",
                        role_hint=ParticipantRoleHint.UNKNOWN,
                    ),
                )
            new_ref = new_reference_factory(evidence.claimed_role_hint)
            return ParticipantContinuityDecision(
                state=ParticipantContinuityState.NEW_CONFIRMED,
                active_participant=new_ref,
                evidence_kind=evidence.kind,
                archived_prior=active,
            )

        if evidence.kind == ParticipantEvidenceKind.EXPLICIT_HANDOFF:
            new_ref = new_reference_factory(evidence.claimed_role_hint)
            if active is None:
                return ParticipantContinuityDecision(
                    state=ParticipantContinuityState.NEW_CONFIRMED,
                    active_participant=new_ref,
                    evidence_kind=evidence.kind,
                    archived_prior=ParticipantReference(
                        reference_id=f"{new_ref.reference_id}__prior_placeholder",
                        role_hint=ParticipantRoleHint.UNKNOWN,
                    ),
                )
            return ParticipantContinuityDecision(
                state=ParticipantContinuityState.NEW_CONFIRMED,
                active_participant=new_ref,
                evidence_kind=evidence.kind,
                archived_prior=active,
            )

        if evidence.kind == ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION:
            if active is None:
                # First-ever participant: no prior to archive. SAME_CONFIRMED
                # is the correct state — "same" because there is no other.
                new_ref = new_reference_factory(evidence.claimed_role_hint)
                return ParticipantContinuityDecision(
                    state=ParticipantContinuityState.SAME_CONFIRMED,
                    active_participant=new_ref,
                    evidence_kind=evidence.kind,
                )
            # Same person re-identifying themselves — role hint may refine
            refined = replace(
                active,
                role_hint=(
                    evidence.claimed_role_hint
                    if evidence.claimed_role_hint != ParticipantRoleHint.UNKNOWN
                    else active.role_hint
                ),
            )
            return ParticipantContinuityDecision(
                state=ParticipantContinuityState.SAME_CONFIRMED,
                active_participant=refined,
                evidence_kind=evidence.kind,
            )

        if evidence.kind == ParticipantEvidenceKind.CURRENT_PARTICIPANT_CORRECTION:
            if active is None:
                new_ref = new_reference_factory(evidence.claimed_role_hint)
                return ParticipantContinuityDecision(
                    state=ParticipantContinuityState.UNCERTAIN,
                    active_participant=new_ref,
                    evidence_kind=evidence.kind,
                )
            return ParticipantContinuityDecision(
                state=ParticipantContinuityState.SAME_CONFIRMED,
                active_participant=active,
                evidence_kind=evidence.kind,
            )

        if evidence.kind == ParticipantEvidenceKind.INFERRED:
            if active is None:
                new_ref = new_reference_factory(evidence.claimed_role_hint)
                return ParticipantContinuityDecision(
                    state=ParticipantContinuityState.LIKELY_NEW,
                    active_participant=new_ref,
                    evidence_kind=evidence.kind,
                )
            # Inferred signal cannot confirm a new participant; stay LIKELY_NEW
            # with the existing active until explicit evidence arrives.
            return ParticipantContinuityDecision(
                state=ParticipantContinuityState.LIKELY_NEW,
                active_participant=active,
                evidence_kind=evidence.kind,
            )

        # NONE: no evidence → preserve active if present, otherwise UNKNOWN
        if active is not None:
            return ParticipantContinuityDecision(
                state=ParticipantContinuityState.SAME_CONFIRMED,
                active_participant=active,
                evidence_kind=ParticipantEvidenceKind.NONE,
            )
        return ParticipantContinuityDecision(
            state=ParticipantContinuityState.UNKNOWN,
            active_participant=new_reference_factory(ParticipantRoleHint.UNKNOWN),
            evidence_kind=ParticipantEvidenceKind.NONE,
        )


# ---------------------------------------------------------------------------
# Re-explanation strategy chooser
# ---------------------------------------------------------------------------


from enum import Enum  # noqa: E402


class ReExplanationIntent(str, Enum):
    """What the current participant seems to want. Provider proposes."""

    NONE = "none"
    RECAP_REQUEST = "recap_request"
    SIMPLIFY_REQUEST = "simplify_request"
    DEEPEN_REQUEST = "deepen_request"
    DIRECT_SERVICE_QUESTION = "direct_service_question"


class ReExplanationPolicy:
    """Participant-aware repetition decision. No scripted answers here.

    Rules:
        (a) prospect requested recap AND same participant → CONCISE_RECAP
        (b) prospect requested simplify → SIMPLIFY
        (c) prospect requested deeper detail → DEEPEN
        (d) new participant AND concept previously explained to someone
            else → FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT
        (e) new participant asked direct service question → FULL_RE_EXPLAIN
        (f) same participant AND concept already DETAILED → SUPPRESS
            (participant has heard full detail; let planner decide to ask
             a question or move on)
        (g) nothing in prior coverage → NO_PRIOR_COVERAGE
    """

    def decide(
        self,
        continuity: ParticipantContinuityState,
        active_reference_id: str,
        concept_id: str,
        coverage: DiscussionCoverageState,
        intent: ReExplanationIntent = ReExplanationIntent.NONE,
    ) -> ReExplanationDecision:
        prior_depth = coverage.depth_for(active_reference_id, concept_id)
        heard_elsewhere = any(
            e.concept_id == concept_id and e.participant_reference_id != active_reference_id
            for e in coverage.entries
        )

        is_new_participant = continuity in (
            ParticipantContinuityState.NEW_CONFIRMED,
            ParticipantContinuityState.LIKELY_NEW,
        )

        if intent == ReExplanationIntent.SIMPLIFY_REQUEST:
            return ReExplanationDecision(
                strategy=ReExplanationStrategy.SIMPLIFY,
                concept_id=concept_id,
                participant_reference_id=active_reference_id,
                prior_depth=prior_depth,
                reason="participant requested simplification",
            )
        if intent == ReExplanationIntent.DEEPEN_REQUEST:
            return ReExplanationDecision(
                strategy=ReExplanationStrategy.DEEPEN,
                concept_id=concept_id,
                participant_reference_id=active_reference_id,
                prior_depth=prior_depth,
                reason="participant requested deeper explanation",
            )
        if intent == ReExplanationIntent.RECAP_REQUEST:
            if is_new_participant:
                return ReExplanationDecision(
                    strategy=ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
                    concept_id=concept_id,
                    participant_reference_id=active_reference_id,
                    prior_depth=prior_depth,
                    reason="new participant asking for context",
                )
            return ReExplanationDecision(
                strategy=ReExplanationStrategy.CONCISE_RECAP,
                concept_id=concept_id,
                participant_reference_id=active_reference_id,
                prior_depth=prior_depth,
                reason="same participant requested recap",
            )
        if intent == ReExplanationIntent.DIRECT_SERVICE_QUESTION and is_new_participant:
            return ReExplanationDecision(
                strategy=ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
                concept_id=concept_id,
                participant_reference_id=active_reference_id,
                prior_depth=prior_depth,
                reason="new participant asked a direct service question",
            )

        # No intent signal
        if prior_depth == ExplanationDepth.NONE and not heard_elsewhere:
            return ReExplanationDecision(
                strategy=ReExplanationStrategy.NO_PRIOR_COVERAGE,
                concept_id=concept_id,
                participant_reference_id=active_reference_id,
                prior_depth=prior_depth,
                reason="concept has not been discussed",
            )
        if is_new_participant and heard_elsewhere and prior_depth == ExplanationDepth.NONE:
            return ReExplanationDecision(
                strategy=ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
                concept_id=concept_id,
                participant_reference_id=active_reference_id,
                prior_depth=prior_depth,
                reason="new participant has not yet heard concept",
            )
        if prior_depth in (ExplanationDepth.EXPLAINED, ExplanationDepth.DETAILED):
            return ReExplanationDecision(
                strategy=ReExplanationStrategy.SUPPRESS,
                concept_id=concept_id,
                participant_reference_id=active_reference_id,
                prior_depth=prior_depth,
                reason="participant already received a full explanation",
            )
        return ReExplanationDecision(
            strategy=ReExplanationStrategy.SUPPRESS,
            concept_id=concept_id,
            participant_reference_id=active_reference_id,
            prior_depth=prior_depth,
            reason="no re-explanation signal",
        )


def deterministic_participant_factory(session_id: str, start: int = 0):
    """Return a reference-factory that mints participant_{start+1}, +2, ...

    `start` lets callers skip IDs already taken by an existing active
    participant so the factory never mints a colliding reference.
    """
    counter = [start]

    def _factory(role_hint: ParticipantRoleHint) -> ParticipantReference:
        counter[0] += 1
        return ParticipantReference(
            reference_id=f"{session_id}:participant_{counter[0]}",
            role_hint=role_hint,
        )

    return _factory
