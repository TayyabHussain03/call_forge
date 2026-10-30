"""Response renderer wrapper that realizes wording once and fails closed."""

from __future__ import annotations

from dataclasses import replace

from app.conversation.realization.contracts import (
    HumanConversationPolicy,
    LeanContextView,
    RealizationInput,
)
from app.conversation.realization.provider import (
    ConversationRealizationProvider,
    RealizationError,
)
from app.conversation.realization.validator import (
    RealizationValidationError,
    validate_realization,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AuthoritativeResultKind,
)
from app.conversation.response_rendering.contracts import RenderedResponse, ResponseRenderInput
from app.conversation.response_rendering.renderer import (
    ResponseRenderer,
    validate_rendered_text,
)


class GuardedConversationRealizer(ResponseRenderer):
    """Use one wording proposal only when deterministic policy permits it.

    Slice 58B.4 — exposes per-render diagnostic snapshot via
    `last_metrics` for the dev trace sink. Fields are typed, bounded,
    and never carry credentials, raw prompts, or chain-of-thought.
    """

    def __init__(
        self,
        provider: ConversationRealizationProvider,
        fallback_renderer: ResponseRenderer,
        policy: HumanConversationPolicy = HumanConversationPolicy(),
    ) -> None:
        self._provider = provider
        self._fallback = fallback_renderer
        self._policy = policy
        self.last_metrics: dict = {}

    def render(self, render_input: ResponseRenderInput) -> RenderedResponse:
        metrics: dict = {
            "eligible": False,
            "called": False,
            "candidate_present": False,
            "candidate_text": None,
            "validation_status": "not_called",
            "rejection_reason": None,
            "provider_failure_kind": None,
            "deterministic_fallback_used": True,
        }
        self.last_metrics = metrics
        effective_input = replace(
            render_input,
            identity_disclosure_statement=self._policy.identity_statement,
        )
        if not _may_realize(effective_input):
            metrics["rejection_reason"] = "not_eligible"
            return self._fallback.render(effective_input)
        metrics["eligible"] = True
        value = _input(effective_input, self._policy)
        metrics["called"] = True
        try:
            candidate = self._provider.realize(value)
        except RealizationError as exc:
            metrics["provider_failure_kind"] = getattr(exc, "kind", None).__class__.__name__ if getattr(exc, "kind", None) is not None else "RealizationError"
            # exc.kind is RealizationFailureKind enum — stringify safely
            try:
                metrics["provider_failure_kind"] = exc.kind.value
            except Exception:
                metrics["provider_failure_kind"] = "unknown"
            metrics["validation_status"] = "provider_failed"
            return self._fallback.render(effective_input)
        except (TimeoutError, TypeError, ValueError) as exc:
            metrics["provider_failure_kind"] = type(exc).__name__
            metrics["validation_status"] = "provider_failed"
            return self._fallback.render(effective_input)
        if isinstance(candidate, str):
            metrics["candidate_present"] = True
            metrics["candidate_text"] = candidate[:300]
        try:
            text = validate_realization(candidate, value)
            rendered = validate_rendered_text(text, effective_input)
            metrics["validation_status"] = "accepted"
            metrics["deterministic_fallback_used"] = False
            _record_validation(self._provider, True)
            return rendered
        except RealizationValidationError as exc:
            metrics["validation_status"] = "rejected"
            metrics["rejection_reason"] = str(exc)[:200]
            _record_validation(self._provider, False)
            return self._fallback.render(effective_input)
        except (TypeError, ValueError) as exc:
            metrics["validation_status"] = "rejected"
            metrics["rejection_reason"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            _record_validation(self._provider, False)
            return self._fallback.render(effective_input)


def _record_validation(
    provider: ConversationRealizationProvider, accepted: bool
) -> None:
    recorder = getattr(provider, "record_validation", None)
    if callable(recorder):
        recorder(accepted)


def _may_realize(render_input: ResponseRenderInput) -> bool:
    return (
        render_input.authoritative_result == AuthoritativeResultKind.EXECUTED
        and render_input.plan.addressee_status
        == AddresseeStatus.ADDRESSED_TO_AGENT
    )


def _input(
    render_input: ResponseRenderInput, policy: HumanConversationPolicy
) -> RealizationInput:
    plan = render_input.plan
    answer = plan.service_answer_context
    names = tuple(
        name
        for name in (
            answer.service_name if answer is not None else None,
            render_input.trusted_context.service_name,
        )
        if name is not None
    )
    evidence = (
        answer.approved_evidence
        if answer is not None
        else render_input.trusted_context.approved_evidence
    )
    return RealizationInput(
        plan,
        policy,
        render_input.lean_context_view
        or LeanContextView(render_input.current_prospect_message),
        evidence,
        plan.language_profile,
        names,
        render_input.identity_disclosure_required,
    )
