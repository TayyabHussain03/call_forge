"""Immutable, bounded contracts for the canonical model-context boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.brain.authority.models import AuthorityTier
from app.brain.business_intelligence import SourceKind
from app.conversation.prospect_intelligence.contracts import (
    InferredProspectState,
    ProspectIntelligenceSummary,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
)
from app.conversation.strategy.contracts import (
    ConversationStrategy,
    ConversationStrategyHint,
)
from app.core.constants import ConversationState


@dataclass(frozen=True)
class ContextBounds:
    """Single source for structural model-context limits."""

    max_current_message_chars: int = 2_000
    max_recent_turns: int = 4
    max_recent_turn_chars: int = 300
    max_evidence_items: int = 8
    max_service_ids: int = 12
    max_business_fields: int = 6
    max_discovery_priorities: int = 4

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("context bounds must be positive")


DEFAULT_CONTEXT_BOUNDS = ContextBounds()


class TurnSpeaker(str, Enum):
    USER = "user"
    AGENT = "agent"


@dataclass(frozen=True)
class RecentTurn:
    """One short prior turn; never a full transcript segment."""

    turn_id: str
    speaker: TurnSpeaker
    content: str

    def __post_init__(self) -> None:
        _text(self.turn_id, "turn id", 100)
        if not isinstance(self.speaker, TurnSpeaker):
            raise TypeError("speaker must be TurnSpeaker")
        _text(self.content, "recent turn content", 300)


@dataclass(frozen=True)
class UntrustedUserInput:
    """Raw finalized utterance, explicitly data rather than authority."""

    message: str

    def __post_init__(self) -> None:
        _text(self.message, "current user message", 2_000)


class EvidenceType(str, Enum):
    SERVICE_DESCRIPTION = "service_description"
    APPROVED_CLAIM = "approved_claim"
    CASE_STUDY = "case_study"
    TECHNICAL_FACT = "technical_fact"
    POLICY_FACT = "policy_fact"


class ApprovedEvidenceSourceKind(str, Enum):
    CATALOG_CLAIM = "catalog_claim"
    CURATED_CAMPAIGN = "curated_campaign"
    CURATED_SERVICE = "curated_service"
    TRUSTED_POLICY = "trusted_policy"


class EvidenceScopeKind(str, Enum):
    GLOBAL = "global"
    CAMPAIGN = "campaign"
    SERVICE = "service"


@dataclass(frozen=True)
class EvidenceScope:
    """Typed applicability boundary for one approved statement."""

    kind: EvidenceScopeKind
    scope_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceScopeKind):
            raise TypeError("evidence scope kind is invalid")
        if self.kind == EvidenceScopeKind.GLOBAL and self.scope_id is not None:
            raise ValueError("global evidence cannot have a scope id")
        if self.kind != EvidenceScopeKind.GLOBAL:
            _text(self.scope_id, "evidence scope id", 100)


@dataclass(frozen=True)
class ApprovedEvidenceItem:
    """Short statement admitted by trusted code, with provenance and scope."""

    evidence_id: str
    fact_key: str
    evidence_type: EvidenceType
    statement: str
    source_kind: ApprovedEvidenceSourceKind
    scope: EvidenceScope
    source_reference: str | None = None

    def __post_init__(self) -> None:
        _text(self.evidence_id, "evidence id", 100)
        _text(self.fact_key, "evidence fact key", 100)
        _text(self.statement, "approved evidence statement", 240)
        if not isinstance(self.evidence_type, EvidenceType):
            raise TypeError("evidence type is invalid")
        if not isinstance(self.source_kind, ApprovedEvidenceSourceKind):
            raise TypeError("evidence source kind is invalid")
        if not isinstance(self.scope, EvidenceScope):
            raise TypeError("evidence scope is invalid")
        if self.source_reference is not None:
            _text(self.source_reference, "evidence source reference", 100)


@dataclass(frozen=True)
class PolicyContext:
    """Read-only policy metadata; it grants no action or commercial authority."""

    authority_policy_id: str
    pricing_disclosure_allowed: bool
    default_authority_tier: AuthorityTier

    def __post_init__(self) -> None:
        _text(self.authority_policy_id, "authority policy id", 100)
        if not isinstance(self.default_authority_tier, AuthorityTier):
            raise TypeError("default authority tier is invalid")


@dataclass(frozen=True)
class ObservedBusinessFact:
    field: str
    value: str
    source_kind: SourceKind

    def __post_init__(self) -> None:
        _text(self.field, "business field", 80)
        _text(self.value, "business value", 160)
        if not isinstance(self.source_kind, SourceKind):
            raise TypeError("business source kind is invalid")


@dataclass(frozen=True)
class InferredBusinessSignal:
    field: str
    value: str
    confidence: float
    basis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.field, "business field", 80)
        _text(self.value, "business value", 160)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("business inference confidence must be between 0 and 1")
        basis = tuple(self.basis)
        if len(basis) > 3:
            raise ValueError("business inference basis is limited to three items")
        for item in basis:
            _text(item, "business inference basis", 80)
        object.__setattr__(self, "basis", basis)


@dataclass(frozen=True)
class BusinessContext:
    """Relevant current BI projection, never the additive history store."""

    observed: tuple[ObservedBusinessFact, ...] = ()
    inferred: tuple[InferredBusinessSignal, ...] = ()
    unknown_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.observed) + len(self.inferred) + len(self.unknown_fields) > 6:
            raise ValueError("business context is limited to six fields")


@dataclass(frozen=True)
class CampaignContext:
    campaign_id: str | None = None
    campaign_goal: str | None = None
    discovery_priorities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.campaign_id is not None:
            _text(self.campaign_id, "campaign id", 100)
        if self.campaign_goal is not None:
            _text(self.campaign_goal, "campaign goal", 200)
        if len(self.discovery_priorities) > 4:
            raise ValueError("discovery priorities are limited to four")
        for item in self.discovery_priorities:
            _text(item, "discovery priority", 100)


class ContactContextStatus(str, Enum):
    NONE = "none"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REFERENCE = "reference"


@dataclass(frozen=True)
class ContactContext:
    """Current contact lifecycle only; raw phone/email values are excluded."""

    status: ContactContextStatus = ContactContextStatus.NONE
    channel: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ContactContextStatus):
            raise TypeError("contact status is invalid")
        if self.channel is not None:
            _text(self.channel, "contact channel", 40)
        if self.reference is not None:
            _text(self.reference, "contact reference", 80)


@dataclass(frozen=True)
class SupervisorAdvisoryContext:
    """Validated, fresh supervisor projection with no raw output or reasoning."""

    source_turn_id: str
    inferred_prospect: InferredProspectState | None = None
    strategy_hint: ConversationStrategyHint | None = None

    def __post_init__(self) -> None:
        _text(self.source_turn_id, "supervisor source turn id", 100)
        if self.inferred_prospect is not None and not isinstance(
            self.inferred_prospect, InferredProspectState
        ):
            raise TypeError("supervisor prospect projection is invalid")
        if self.strategy_hint is not None and not isinstance(
            self.strategy_hint, ConversationStrategyHint
        ):
            raise TypeError("supervisor strategy hint is invalid")


@dataclass(frozen=True)
class LeanTurnContext:
    """Canonical deliberately-incomplete model-facing context."""

    untrusted_user_input: UntrustedUserInput
    current_state: ConversationState
    strategy: ConversationStrategy | None = None
    recent_turns: tuple[RecentTurn, ...] = ()
    prospect: ProspectIntelligenceSummary | None = None
    pending_intent: PendingConversationIntent | None = None
    eligible_service_ids: tuple[str, ...] = ()
    offered_service_ids: tuple[str, ...] = ()
    approved_evidence: tuple[ApprovedEvidenceItem, ...] = ()
    policy: PolicyContext | None = None
    business: BusinessContext = BusinessContext()
    campaign: CampaignContext = CampaignContext()
    contact: ContactContext = ContactContext()
    interruption: InterruptionContext = InterruptionContext()
    conversation_category: InterruptionCategory = InterruptionCategory.OTHER
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT
    supervisor_advisory: SupervisorAdvisoryContext | None = None

    def __post_init__(self) -> None:
        bounds = DEFAULT_CONTEXT_BOUNDS
        if len(self.recent_turns) > bounds.max_recent_turns:
            raise ValueError("recent turns exceed the canonical bound")
        if len(self.approved_evidence) > bounds.max_evidence_items:
            raise ValueError("approved evidence exceeds the canonical bound")
        if len(self.eligible_service_ids) > bounds.max_service_ids:
            raise ValueError("eligible services exceed the canonical bound")
        if len(self.offered_service_ids) > bounds.max_service_ids:
            raise ValueError("offered services exceed the canonical bound")
        if tuple(sorted(set(self.eligible_service_ids))) != self.eligible_service_ids:
            raise ValueError("eligible services must be unique and stably ordered")
        if tuple(sorted(set(self.offered_service_ids))) != self.offered_service_ids:
            raise ValueError("offered services must be unique and stably ordered")
        if not isinstance(self.conversation_category, InterruptionCategory):
            raise TypeError("conversation category is invalid")
        if not isinstance(self.addressee_status, AddresseeStatus):
            raise TypeError("addressee status is invalid")


@dataclass(frozen=True)
class BrainContextView:
    """Bounded Brain projection derived from the canonical context."""

    turn: LeanTurnContext


@dataclass(frozen=True)
class SupervisorContextView:
    """Bounded Supervisor projection derived from the same canonical context."""

    turn: LeanTurnContext


def _text(value: str | None, name: str, limit: int) -> None:
    if value is None or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")
