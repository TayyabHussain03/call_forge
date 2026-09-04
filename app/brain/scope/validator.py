"""Deterministic scope-policy validator.

Answer deta hai: "kya ye proposed conversational move scope mein REHNE ke layak
hai?" — aur kuch nahi. Structured proposal fields (action + topic_category) ko
trusted resolved ScopePolicy ke against check karta hai. RAW utterance/keyword/
regex kabhi nahi.

CORE INVARIANT: Brain reasons. ScopePolicy constrains. ScopePolicy REASON nahi
karti. Deterministic system decides. Execution validator ke BAHAR.

FAIL-CLOSED: agar positive in-scope determination establish na ho — unknown
action/topic, missing category, restricted topic — to reject. Unknown → allowed
KABHI nahi.

DENY BEATS ALLOW: restricted topic hamesha jeetta hai, chahe broad allow ho.

Validator PURE hai: koi mutation, LLM, network, DB, fallback-execution, state-
transition, service/claim/pricing logic. Sirf classification + fallback CATEGORY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.brain.scope.models import ScopePolicy
from app.core.constants import AgentAction, TopicCategory


class ScopeCategory(str, Enum):
    """The scope classification of a proposal."""

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN_SCOPE = "unknown_scope"   # fail-closed: establish nahi ho paaya


@dataclass(frozen=True)
class ScopeCheckInput:
    """Minimal, data-only input for scope validation.

    Koi DB/catalog/repo/state-machine/LLM/raw-transcript. Validator
    `current_utterance` inspect NAHI karta — sirf structured fields.

    Attributes:
        proposal_action: The proposed AgentAction (may be None).
        proposal_topic: The proposed TopicCategory (untrusted, Brain-supplied).
        policy: The resolved, trusted ScopePolicy.
    """

    proposal_action: AgentAction | None
    proposal_topic: TopicCategory
    policy: ScopePolicy


@dataclass(frozen=True)
class ScopeValidationResult:
    """Deterministic result of a scope check (executes nothing).

    Attributes:
        category: IN_SCOPE / OUT_OF_SCOPE / UNKNOWN_SCOPE.
        policy_id: Which policy was evaluated (traceability).
        rejected_reason: Machine-readable reason for non-in-scope, ya None.
        fallback_hint: Machine-readable fallback CATEGORY (not wording, not an
            action to execute), ya None. E.g. "redirect_to_goal".
        requires_escalation: True agar topic in-scope hai lekin escalation-flagged
            (authority/human route). Flag only — authority decision alag layer.
    """

    category: ScopeCategory
    policy_id: str
    rejected_reason: str | None = None
    fallback_hint: str | None = None
    requires_escalation: bool = False

    @property
    def is_in_scope(self) -> bool:
        """Whether the proposal is in scope.

        Returns:
            bool: True only for IN_SCOPE.
        """
        return self.category == ScopeCategory.IN_SCOPE


class ScopePolicyValidator:
    """Validates a proposal's scope against a trusted resolved policy.

    Stateless, deterministic, pure. Same input + same policy → same result.
    """

    def validate(self, check: ScopeCheckInput) -> ScopeValidationResult:
        """Classify a proposal as in/out/unknown scope (fail-closed).

        Order (deterministic):
            1. UNKNOWN topic → UNKNOWN_SCOPE (fail-closed, Brain couldn't classify).
            2. Topic restricted → OUT_OF_SCOPE (deny beats allow).
            3. Action present but not allowed → OUT_OF_SCOPE.
            4. Topic not in allowed set → OUT_OF_SCOPE (fail-closed).
            5. Else IN_SCOPE (+ escalation flag if topic is escalation-flagged).

        Args:
            check: The scope-check input (structured fields + trusted policy).

        Returns:
            ScopeValidationResult: Classification + traceability + fallback hint.
        """
        policy = check.policy
        topic = check.proposal_topic
        action = check.proposal_action

        # 1. Unknown topic → fail-closed. Brain classify nahi kar paaya.
        if topic == TopicCategory.UNKNOWN:
            return ScopeValidationResult(
                category=ScopeCategory.UNKNOWN_SCOPE,
                policy_id=policy.policy_id,
                rejected_reason="topic category unknown",
                fallback_hint="clarify",
            )

        # 2. Restricted topic → out of scope. Deny beats allow (safest).
        if topic in policy.restricted_topics:
            return ScopeValidationResult(
                category=ScopeCategory.OUT_OF_SCOPE,
                policy_id=policy.policy_id,
                rejected_reason=f"topic restricted: {topic.value}",
                fallback_hint="redirect_to_goal",
            )

        # 3. Action present but not allowed → out of scope (fail-closed).
        if action is not None and action not in policy.allowed_actions:
            return ScopeValidationResult(
                category=ScopeCategory.OUT_OF_SCOPE,
                policy_id=policy.policy_id,
                rejected_reason=f"action not in scope: {action.value}",
                fallback_hint="redirect_to_goal",
            )

        # 4. Topic not in allowed set → out of scope (fail-closed).
        if topic not in policy.allowed_topics:
            return ScopeValidationResult(
                category=ScopeCategory.OUT_OF_SCOPE,
                policy_id=policy.policy_id,
                rejected_reason=f"topic not in scope: {topic.value}",
                fallback_hint="redirect_to_goal",
            )

        # 5. In scope. Escalation flag agar topic escalation-list mein (authority
        #    decision alag layer — scope sirf flag deta hai).
        return ScopeValidationResult(
            category=ScopeCategory.IN_SCOPE,
            policy_id=policy.policy_id,
            requires_escalation=topic in policy.escalation_topics,
        )