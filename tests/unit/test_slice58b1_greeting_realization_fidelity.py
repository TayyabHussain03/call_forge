"""Slice 58B.1 — greeting semantic trace + realization-plan fidelity.

User-reported symptom:
    hello → "Could you clarify the specific detail you need so I don't guess?"

Trace result: that exact string is produced ONLY by
`response_rendering/renderer._escalation_text` under the
`ASK_CLARIFYING_QUESTION` branch. The live realizer never runs because
`_may_realize` requires `authoritative_result == EXECUTED` — the plan
reached the deterministic renderer with an `escalation_decision` set.

That means the orchestrator produced ESCALATE/REDIRECT/FALLBACK upstream
of the renderer. The owning layer is the reasoning provider boundary:
the Gemini reasoning adapter sent a RAW `BrainInput` JSON blob with no
role context, so Gemini — while returning schema-valid output — picked
actions that fail scope/authority (producing escalation).

Classification: REASONING / PROVIDER_PROMPT
Owner:           app/llm/gemini_provider.py (and realization prompt)

The fix is semantic, not keyword-matched: a bounded role instruction
preamble tells Gemini its proposal role, the typed enum fields, and
that it must select actions that fit the current state.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from app.brain.contracts import BrainInput
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory
from app.llm.gemini_provider import GeminiReasoningProvider
from app.llm.providers.reasoning_provider import ProviderFailureKind


# ---------------------------------------------------------------------------
# Section A — reasoning prompt now carries a role instruction
# ---------------------------------------------------------------------------


class _Models:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.kwargs: dict = {}

    def generate_content(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _valid_payload() -> dict:
    return {
        "detected_intent": Intent.INTERESTED.value,
        "tone": "neutral",
        "topic_category": TopicCategory.SERVICE_DISCUSSION.value,
        "proposed_action": AgentAction.GREET.value,
        "needs_service_decision": False,
        "involves_contact": False,
        "commercial_request": None,
        "contact_understanding": None,
        "intelligence_updates": [],
        "proposed_goal_update": None,
        "reasoning": "bounded",
        "action_confidence": 0.8,
    }


def _brain_input(utterance: str = "hello") -> BrainInput:
    return BrainInput(
        current_utterance=utterance,
        current_state=ConversationState.NEW_CALL,
        current_goal=None,
        recent_turns=(),
        business_intelligence=None,
        known_signals=frozenset(),
        eligible_service_ids=(),
        offered_service_ids=(),
        campaign_policy_summary=None,
        budget=None,
        resolved_contact_context=None,
        conversation_strategy=None,
        prospect_intelligence=None,
        lean_context=None,
    )


def _call_once(utterance: str = "hello"):
    models = _Models(
        response=SimpleNamespace(text=json.dumps(_valid_payload())),
    )
    provider = GeminiReasoningProvider(
        model="gemini-test",
        client=SimpleNamespace(models=models),
    )
    provider.reason(_brain_input(utterance))
    return models


class TestReasoningPromptHasRoleInstruction:
    def test_contents_now_carries_instruction_and_input(self):
        """§3 Case A — Gemini can no longer receive a bare input dump."""
        models = _call_once("hello")
        raw = models.kwargs["contents"]
        payload = json.loads(raw)
        assert "instruction" in payload
        assert "input" in payload
        assert isinstance(payload["instruction"], str)
        assert isinstance(payload["input"], dict)

    def test_instruction_names_the_proposal_role(self):
        models = _call_once("hello")
        instruction = json.loads(models.kwargs["contents"])["instruction"]
        text = instruction.lower()
        # Role is explicit
        assert "proposal" in text or "propose" in text
        # Does NOT execute
        assert "deterministic" in text
        # Guides the simple-opener case semantically (not keyword matching)
        assert "greet" in text  # typed enum value name is fine; it's semantic

    def test_instruction_never_tells_model_to_bypass_policy(self):
        """§22 — must not instruct Gemini to bypass deterministic policy."""
        models = _call_once("hello")
        instruction = json.loads(models.kwargs["contents"])["instruction"].lower()
        for forbidden in (
            "bypass",
            "ignore the validator",
            "override authority",
            "override scope",
        ):
            assert forbidden not in instruction

    def test_instruction_forbids_inventing_services_or_claims(self):
        models = _call_once("hello")
        instruction = json.loads(models.kwargs["contents"])["instruction"].lower()
        assert "do not invent" in instruction
        assert "services" in instruction
        assert "prices" in instruction

    def test_input_payload_contains_current_utterance(self):
        """The original BrainInput still reaches Gemini; the fix is
        additive, not replacement."""
        models = _call_once("hello there")
        payload = json.loads(models.kwargs["contents"])
        assert payload["input"]["current_utterance"] == "hello there"

    def test_no_keyword_branching_on_utterance_in_adapter(self):
        """§9, §24 — no `if utterance.lower() == 'hello'` style branches."""
        from app.llm import gemini_provider as mod
        src = inspect.getsource(mod)
        for forbidden in (
            'utterance.lower() ==',
            'utterance == "hello"',
            "utterance == 'hello'",
            'if "hello" in',
            "if 'hello' in",
            '.startswith("hello"',
            ".startswith('hello'",
        ):
            assert forbidden not in src, f"keyword hack {forbidden} in adapter"

    def test_schema_mime_afc_disable_still_set(self):
        """Slice 58A/58B invariants must survive the prompt change."""
        models = _call_once("hello")
        cfg = models.kwargs["config"]
        assert cfg["response_mime_type"] == "application/json"
        assert cfg.get("response_json_schema") is not None
        afc = cfg.get("automatic_function_calling")
        disable = afc.get("disable") if isinstance(afc, dict) else getattr(afc, "disable", None)
        assert disable is True

    def test_still_exactly_one_sdk_call(self):
        models = _call_once("hello")
        assert models.calls == 1


# ---------------------------------------------------------------------------
# Section B — realization prompt now binds Gemini to plan semantics
# ---------------------------------------------------------------------------


class TestRealizationPromptPlanFaithfulness:
    def _instruction(self) -> str:
        """Extract the shipped realization-prompt instruction string."""
        from app.conversation.realization.contracts import (
            HumanConversationPolicy, LeanContextView, RealizationInput,
        )
        from app.conversation.realization.prompt import (
            build_llm_realization_prompt, serialize_llm_realization_prompt,
        )
        from app.conversation.response_planning.contracts import (
            AcknowledgementKind, AddresseeStatus, ConversationMove,
            InterruptionHandling, QuestionStrategy, ResponseLength,
            ResponsePlan, Tone,
        )
        plan = ResponsePlan(
            ConversationMove.CONSULTATIVE_DISCOVERY,
            ResponseLength.SHORT,
            Tone.NEUTRAL,
            AcknowledgementKind.NONE,
            QuestionStrategy.NONE,
            False,
            InterruptionHandling.DROP_STALE_POINT,
            False,
            None,
            AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        inp = RealizationInput(
            plan, HumanConversationPolicy(), LeanContextView("hello"),
        )
        raw = serialize_llm_realization_prompt(build_llm_realization_prompt(inp))
        return json.loads(raw)["instruction"]

    def test_instruction_binds_to_plan(self):
        text = self._instruction().lower()
        assert "plan" in text
        assert "communicative_goal" in text

    def test_instruction_forbids_second_decision(self):
        """§6, §19 — realization is wording, not another Brain."""
        text = self._instruction().lower()
        assert "do not reconsider" in text or "do not re-decide" in text or "do not reconsider the action" in text

    def test_instruction_gates_clarification_on_plan_flag(self):
        """§4, §8 — only clarify when plan says so."""
        text = self._instruction().lower()
        # The instruction references the clarification flag by name
        assert "clarification_required" in text
        # And says only / unless / if it's true
        assert "only if" in text or "unless" in text

    def test_instruction_names_language_profile(self):
        """§10, §30-adjacent — language fidelity."""
        text = self._instruction().lower()
        assert "language_profile" in text

    def test_instruction_forbids_inventing_evidence(self):
        """§20 — only approved evidence."""
        text = self._instruction().lower()
        assert "approved_evidence" in text
        assert "do not invent" in text

    def test_realization_adapter_still_disables_afc(self):
        """Slice 58B invariant survives the prompt strengthening."""
        from app.conversation.realization import adapter as mod
        src = inspect.getsource(mod)
        assert '"automatic_function_calling": {"disable": True}' in src


# ---------------------------------------------------------------------------
# Section C — role instruction is semantic, not keyword
# ---------------------------------------------------------------------------


class TestNoKeywordBranching:
    def test_gemini_provider_module_has_no_utterance_keyword_tree(self):
        from app.llm import gemini_provider as mod
        src = inspect.getsource(mod)
        # No branching on raw utterance text
        for pattern in ("utterance ==", "utterance.startswith",
                        "utterance.lower() ==", "utterance.lower().startswith"):
            assert pattern not in src

    def test_realization_prompt_module_has_no_utterance_keyword_tree(self):
        from app.conversation.realization import prompt as mod
        src = inspect.getsource(mod)
        for pattern in ("utterance ==", "utterance.startswith",
                        "utterance.lower() =="):
            assert pattern not in src


# ---------------------------------------------------------------------------
# Section D — Slice 58A/58B invariants still hold
# ---------------------------------------------------------------------------


class TestPriorInvariantsPreserved:
    def test_malformed_response_still_raises_proposal_parsing_error(self):
        from app.llm.structured_output import ProposalParsingError
        models = _Models(response=SimpleNamespace(text="{not json"))
        provider = GeminiReasoningProvider(
            model="gemini-test", client=SimpleNamespace(models=models),
        )
        with pytest.raises(ProposalParsingError):
            provider.reason(_brain_input())

    def test_timeout_still_classified(self):
        from app.llm.providers.reasoning_provider import ReasoningError
        models = _Models(error=TimeoutError("deadline"))
        provider = GeminiReasoningProvider(
            model="gemini-test", client=SimpleNamespace(models=models),
        )
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.TIMEOUT

    def test_realizer_fallback_contract_unchanged(self):
        """Slice 58B: provider failure → deterministic fallback.
        Keep this covered now that the realization prompt changed."""
        from app.conversation.realization.provider import (
            MockConversationRealizationProvider,
        )
        from app.conversation.realization.realizer import (
            GuardedConversationRealizer,
        )
        from app.conversation.response_planning.contracts import (
            AcknowledgementKind, AddresseeStatus, AuthoritativeResultKind,
            ConversationMove, InterruptionHandling, QuestionStrategy,
            ResponseLength, ResponsePlan, Tone,
        )
        from app.conversation.response_rendering.contracts import (
            ResponseRenderInput, ResponseRenderingBudget,
            TrustedRenderingContext,
        )
        from app.conversation.response_rendering.renderer import (
            DeterministicResponseRenderer,
        )
        plan = ResponsePlan(
            ConversationMove.CONSULTATIVE_DISCOVERY,
            ResponseLength.SHORT,
            Tone.NEUTRAL,
            AcknowledgementKind.NONE,
            QuestionStrategy.NONE,
            False,
            InterruptionHandling.DROP_STALE_POINT,
            False,
            None,
            AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        inp = ResponseRenderInput(
            plan,
            authoritative_result=AuthoritativeResultKind.EXECUTED,
            budget=ResponseRenderingBudget(5),
            trusted_context=TrustedRenderingContext(),
            current_prospect_message="hello",
        )
        realizer = GuardedConversationRealizer(
            MockConversationRealizationProvider(should_fail=True),
            DeterministicResponseRenderer(),
        )
        # Does not raise; falls back
        result = realizer.render(inp)
        assert result.text  # non-empty deterministic fallback
