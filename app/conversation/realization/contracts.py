"""Immutable bounded inputs for language-only response realization."""

from __future__ import annotations

from dataclasses import dataclass

from app.conversation.context.contracts import ApprovedEvidenceItem
from app.conversation.response_planning.contracts import ResponsePlan
from app.conversation.understanding.contracts import LanguageProfile


@dataclass(frozen=True)
class HumanConversationPolicy:
    """Deterministic wording constraints; this policy grants no business authority."""

    max_questions: int = 1
    max_fillers: int = 1
    prohibit_lists: bool = True
    require_truthful_identity: bool = True
    identity_statement: str = "I'm an AI assistant calling on behalf of the business."
    known_service_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_questions != 1:
            raise ValueError("human conversation policy permits exactly one question")
        if not 0 <= self.max_fillers <= 1:
            raise ValueError("human conversation policy permits at most one filler")
        if not isinstance(self.prohibit_lists, bool):
            raise TypeError("list prohibition must be boolean")
        if not isinstance(self.require_truthful_identity, bool):
            raise TypeError("identity truthfulness flag must be boolean")
        if not self.require_truthful_identity:
            raise ValueError("truthful AI identity disclosure cannot be disabled")
        if not self.identity_statement.strip() or len(self.identity_statement) > 180:
            raise ValueError("identity statement must be bounded")
        names = tuple(self.known_service_names)
        if len(names) > 12 or any(not name.strip() or len(name) > 120 for name in names):
            raise ValueError("known service names must be bounded")
        if len(set(name.casefold() for name in names)) != len(names):
            raise ValueError("known service names must be unique")
        object.__setattr__(self, "known_service_names", names)


@dataclass(frozen=True)
class LeanContextView:
    """Deliberately small view for wording; never a transcript or domain object."""

    current_user_message: str
    known_problem_summary: str | None = None
    role_label: str | None = None

    def __post_init__(self) -> None:
        if not self.current_user_message.strip() or len(self.current_user_message) > 300:
            raise ValueError("current user message must contain 1-300 characters")
        for value, name in (
            (self.known_problem_summary, "problem summary"),
            (self.role_label, "role label"),
        ):
            if value is not None and (not value.strip() or len(value) > 200):
                raise ValueError(f"{name} must be bounded")


@dataclass(frozen=True)
class RealizationInput:
    """Only language-safe, already-authorized information reaches the provider."""

    plan: ResponsePlan
    policy: HumanConversationPolicy
    lean_context: LeanContextView
    approved_evidence: tuple[ApprovedEvidenceItem, ...] = ()
    language_profile: LanguageProfile | None = None
    allowed_service_names: tuple[str, ...] = ()
    identity_disclosure_required: bool = False

    def __post_init__(self) -> None:
        evidence = tuple(self.approved_evidence)
        names = tuple(self.allowed_service_names)
        if len(evidence) > 8 or any(
            not isinstance(item, ApprovedEvidenceItem) for item in evidence
        ):
            raise ValueError("realization evidence must be typed and bounded")
        if len(names) > 2 or any(not name.strip() or len(name) > 120 for name in names):
            raise ValueError("allowed service names must be bounded")
        if self.language_profile is not None and not isinstance(
            self.language_profile, LanguageProfile
        ):
            raise TypeError("language profile is invalid")
        if not isinstance(self.identity_disclosure_required, bool):
            raise TypeError("identity disclosure flag must be boolean")
        object.__setattr__(self, "approved_evidence", evidence)
        object.__setattr__(self, "allowed_service_names", names)
