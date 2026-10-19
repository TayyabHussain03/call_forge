"""Strategy-aware response framing — the GRC consumer bridge.

Wraps any `ResponseRenderer` and adds a bounded, deterministic strategy
framing prefix when the response plan carries a `re_explanation_strategy`.
This is the smallest-bounded way to make Slice 50/51's strategy metadata
visibly influence rendered prose without regressing the many existing
tests that depend on the base renderer's output.

Framings are deterministic strings — they are NOT keyword classification.
They are the *output* of the deterministic policy, applied as a short
natural prefix. Downstream slices can replace this with a richer GRC
template system when needed.
"""

from __future__ import annotations

from dataclasses import replace

from app.conversation.participant.contracts import ReExplanationStrategy
from app.conversation.response_rendering.contracts import (
    RenderedResponse,
    ResponseRenderInput,
)
from app.conversation.response_rendering.renderer import ResponseRenderer


# Deterministic framings. One short natural prefix per strategy.
# These must remain stable because tests assert their presence.
STRATEGY_FRAMINGS: dict[ReExplanationStrategy, str] = {
    ReExplanationStrategy.CONCISE_RECAP: "Quick recap:",
    ReExplanationStrategy.SIMPLIFY: "In simpler terms:",
    ReExplanationStrategy.DEEPEN: "Going deeper:",
    ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT: (
        "To bring you up to speed:"
    ),
    # SUPPRESS, NO_PRIOR_COVERAGE: no framing — base renderer output stands.
}


class StrategyAwareResponseRenderer(ResponseRenderer):
    """Non-breaking wrapper over any ResponseRenderer.

    When `render_input.plan.re_explanation_strategy` is set, the rendered
    text is prepended with a bounded deterministic framing. Base renderer
    is unchanged; existing tests continue to pass.
    """

    def __init__(self, inner: ResponseRenderer) -> None:
        self._inner = inner

    def render(self, render_input: ResponseRenderInput) -> RenderedResponse:
        base = self._inner.render(render_input)
        strategy = getattr(render_input.plan, "re_explanation_strategy", None)
        if not isinstance(strategy, ReExplanationStrategy):
            return base
        framing = STRATEGY_FRAMINGS.get(strategy)
        if not framing:
            return base
        framed_text = f"{framing} {base.text}"
        return replace(base, text=framed_text)


def framing_for(strategy: ReExplanationStrategy) -> str | None:
    """Public access to the strategy → framing map, for test assertions."""
    return STRATEGY_FRAMINGS.get(strategy)
