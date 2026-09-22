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
    """Use one wording proposal only when deterministic policy permits it."""

    def __init__(
        self,
        provider: ConversationRealizationProvider,
        fallback_renderer: ResponseRenderer,
        policy: HumanConversationPolicy = HumanConversationPolicy(),
    ) -> None:
        self._provider = provider
        self._fallback = fallback_renderer
        self._policy = policy

    def render(self, render_input: ResponseRenderInput) -> RenderedResponse:
        effective_input = replace(
            render_input,
            identity_disclosure_statement=self._policy.identity_statement,
        )
        if not _may_realize(effective_input):
            return self._fallback.render(effective_input)
        value = _input(effective_input, self._policy)
        try:
            text = validate_realization(self._provider.realize(value), value)
            return validate_rendered_text(text, effective_input)
        except (
            RealizationError,
            RealizationValidationError,
            TimeoutError,
            TypeError,
            ValueError,
        ):
            return self._fallback.render(effective_input)


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
