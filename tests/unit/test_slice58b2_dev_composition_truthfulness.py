"""Slice 58B.2 — root-cause investigation of live greeting failures.

User reported live symptoms:
    hello → "How would you like to continue?"
    salam → "Could you clarify that for me?"
    hi    → "Could you clarify that for me?"
    + session shows "completed" after a few turns

Investigation findings (confirmed by inspection + grep):

1. CONFIRMED — `app/main.py` dev bootstrap used `ScriptedIntentTransport`
   with a sequence `[none, catalog_request, direct_service_question,
   none, …]`. For the manual sequence hello/salam/hi:
     - turn 1 hello → NONE (fine)
     - turn 2 salam → CATALOG_REQUEST (semantic contamination)
     - turn 3 hi    → DIRECT_SERVICE_QUESTION (semantic contamination)
   The processor's `_resolve_participant_aware_planner_fields` then
   short-circuits salam into catalog_request=True, driving the planner
   into an unusual code path that renders the generic deterministic
   fallback.
   Classification: DEV_COMPOSITION / INTENT_CLASSIFICATION
   Owner:          app/main.py
   Fix:            `intent_transport = None` — safe no-op; processor's
                   fallback is `ReExplanationIntent.NONE`.

2. CONFIRMED — Slice 58B.1's reasoning instruction told Gemini to
   "Use only the supplied allowed_actions / topic_categories" but
   `BrainInput` has NO such fields. The schema enum literals bound
   valid values; the instruction's reference to nonexistent fields
   was misleading.
   Classification: REASONING_PROMPT
   Owner:          app/llm/gemini_provider.py (_REASONING_INSTRUCTION)
   Fix:            instruction now points to the response schema enum
                   literals + input.current_state + input.current_utterance,
                   and explicitly forbids proposing terminal actions
                   (end_call, mark_dnc) for ordinary conversational turns.

3. Hypothesis — session auto-completion. Trace: `conversation_terminal`
   is True iff `execution.is_terminal` from the state machine; which
   is True iff the executed action transitions to a terminal state
   (END_CALL). Combined with (2) a live Gemini without clear
   terminal-action guidance could reasonably propose end_call on a
   turn it mis-identifies as closing. Fix (2) directly addresses this.

Items NOT YET proven (require live trace):
  - Which exact action Gemini proposes for `hello` on first turn
  - Whether live Gemini has already been producing end_call
  - Whether realization provider is called successfully

These are left for the user's live manual acceptance of this slice.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.brain.contracts import BrainInput
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory
from app.llm.gemini_provider import GeminiReasoningProvider


_MAIN_PY = Path(__file__).resolve().parents[2] / "app" / "main.py"


# ---------------------------------------------------------------------------
# Finding 1 — ScriptedIntentTransport removed from dev bootstrap
# ---------------------------------------------------------------------------


class TestDevBootstrapNoScriptedIntent:
    @pytest.fixture(scope="class")
    def main_source(self) -> str:
        return _MAIN_PY.read_text(encoding="utf-8")

    def test_main_does_not_instantiate_scripted_intent_transport(self, main_source):
        """§19, §20, §21 — turn-index semantic classifier was corrupting
        manual live testing. It is now gone from the dev bootstrap."""
        assert "ScriptedIntentTransport(" not in main_source

    def test_main_does_not_import_scripted_intent_transport(self, main_source):
        """Import removed too — the symbol must not even reach main."""
        for pattern in (
            "from app.conversation.participant.intent_provider import ScriptedIntentTransport",
            "import ScriptedIntentTransport",
        ):
            assert pattern not in main_source

    def test_intent_transport_is_none(self, main_source):
        """The dev bootstrap now passes `None` so the processor
        degrades to safe `ReExplanationIntent.NONE`. Live Gemini
        intent classification is Slice 58C."""
        # The literal assignment appears somewhere in main.py
        assert "intent_transport = None" in main_source

    def test_main_does_not_hardcode_catalog_or_direct_service(self, main_source):
        """§19 — no residual turn-index semantic contamination."""
        # These JSON blobs should no longer appear anywhere in main
        for forbidden in (
            '"intent":"catalog_request"',
            '"intent":"direct_service_question"',
        ):
            assert forbidden not in main_source


# ---------------------------------------------------------------------------
# Finding 2 — reasoning instruction references real schema, not phantom fields
# ---------------------------------------------------------------------------


class _Models:
    def __init__(self, response=None):
        self.response = response
        self.calls = 0
        self.kwargs: dict = {}

    def generate_content(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
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
    models = _Models(response=SimpleNamespace(text=json.dumps(_valid_payload())))
    provider = GeminiReasoningProvider(
        model="gemini-test",
        client=SimpleNamespace(models=models),
    )
    provider.reason(_brain_input(utterance))
    return models


class TestReasoningInstructionIsSchemaAccurate:
    def _instruction(self, utterance: str = "hello") -> str:
        return json.loads(_call_once(utterance).kwargs["contents"])["instruction"]

    def test_instruction_refers_to_response_schema_not_phantom_fields(self):
        """§33 — Slice 58B.1 referenced `allowed_actions`/`topic_categories`
        which don't exist on BrainInput. Correct instruction references
        the actual input fields + the response schema enum literals."""
        instruction = self._instruction()
        # Does NOT claim BrainInput carries allowed_actions/topic_categories
        assert "supplied allowed_actions" not in instruction
        # Does reference the actual field path on `input.`
        assert "input.current_state" in instruction
        assert "input.current_utterance" in instruction
        # Does reference the schema's role
        assert "response schema" in instruction.lower() or "supplied response" in instruction.lower()

    def test_instruction_forbids_terminal_actions_for_ordinary_turns(self):
        """§37 — generic greeting must not propose end_call / mark_dnc."""
        instruction = self._instruction().lower()
        assert "end_call" in instruction or "mark_dnc" in instruction
        assert "terminal action" in instruction
        # Explicit prohibition for ordinary turns
        assert "do not propose a terminal" in instruction or (
            "do not" in instruction and "terminal" in instruction
        )

    def test_instruction_still_names_proposal_role(self):
        instruction = self._instruction()
        assert "proposal" in instruction.lower() or "propose" in instruction.lower()
        assert "deterministic" in instruction.lower()

    def test_instruction_still_forbids_inventing_claims(self):
        instruction = self._instruction().lower()
        assert "do not invent" in instruction

    def test_instruction_still_permits_null_action_when_truly_ambiguous(self):
        instruction = self._instruction()
        assert "action=null" in instruction or "proposed_action=null" in instruction


class TestReasoningInstructionDoesNotKeywordMatch:
    def test_instruction_does_not_whitelist_hello_hi_salam(self):
        """§4 non-negotiable — no utterance keyword tree."""
        instruction = json.loads(_call_once("hello").kwargs["contents"])["instruction"]
        # Instruction says WHAT a conversational opening is (semantic),
        # not which exact strings qualify
        lowered = instruction.lower()
        for forbidden_pattern in (
            '"hello"', "'hello'",
            '"hi"', "'hi'",
            '"salam"', "'salam'",
            '"good morning"', "'good morning'",
            '"hey"', "'hey'",
        ):
            assert forbidden_pattern not in instruction, (
                f"keyword literal {forbidden_pattern} in instruction"
            )
        # But generically describes the semantic category
        assert "opening" in lowered or "greeting" in lowered

    def test_adapter_source_has_no_keyword_branching(self):
        """§4 non-negotiable, scanner-level."""
        from app.llm import gemini_provider as mod
        src = inspect.getsource(mod)
        for pattern in (
            'utterance == "hello"', "utterance == 'hello'",
            'utterance.lower() ==',
            'if "hello" in', "if 'hello' in",
            '"salam"', "'salam'",
            '"hi"', "'hi'",
            '"good morning"', "'good morning'",
        ):
            assert pattern not in src, f"keyword hack {pattern} in adapter"


# ---------------------------------------------------------------------------
# Finding 3 — session completion owners (grep audit)
# ---------------------------------------------------------------------------


class TestSessionCompletionOwnership:
    def test_only_known_writers_mark_session_completed(self):
        """§10, §43 — all paths that set SessionStatus.COMPLETED are
        bounded + reviewed. If a new writer appears, this test fails."""
        import subprocess
        proc = subprocess.run(
            ["python", "-c", (
                "import pathlib,re; "
                "paths = list(pathlib.Path('app').rglob('*.py')); "
                "hits = [(p, i+1, L.strip()) for p in paths "
                "       for i, L in enumerate(p.read_text(encoding='utf-8').splitlines()) "
                "       if 'SessionStatus.COMPLETED' in L and '__pycache__' not in str(p)]; "
                "print('\\n'.join(f'{p}:{i}: {L}' for p, i, L in hits))"
            )],
            capture_output=True, text=True, timeout=10,
        )
        hits = [l for l in proc.stdout.splitlines() if l.strip()]
        # Expected authoritative writers:
        #   persistent_runtime.py (process_turn + finish_session)
        #   runtime/chat/engine.py (lifecycle engine)
        #   runtime/chat/lifecycle/service.py (service layer)
        #   recovery.py (recognizes completed status, doesn't mint it)
        allowed_files = (
            "persistent_runtime.py",
            "engine.py",
            "service.py",
            "recovery.py",
        )
        for hit in hits:
            path = hit.split(":")[0].replace("\\", "/")
            assert any(f in path for f in allowed_files), (
                f"unexpected SessionStatus.COMPLETED writer: {hit}"
            )

    def test_terminal_flag_bound_to_execution_terminal(self):
        """Only `execution.is_terminal` drives `conversation_terminal`
        in the orchestrator. Non-terminal actions on ordinary turns
        cannot flip the session to COMPLETED without an explicit
        terminal state transition (END_CALL etc.)."""
        from app.brain.orchestrator import orchestrator as mod
        src = inspect.getsource(mod)
        # The TERMINAL assignment in Slice 3 (SliceThreeResult) is
        # explicitly gated on execution.is_terminal
        assert "if execution.is_terminal" in src


# ---------------------------------------------------------------------------
# Finding 4 — Slice 58A/58B invariants preserved
# ---------------------------------------------------------------------------


class TestPriorInvariantsPreserved:
    def test_afc_disable_still_in_reasoning_config(self):
        models = _call_once("hello")
        cfg = models.kwargs["config"]
        afc = cfg.get("automatic_function_calling")
        disable = afc.get("disable") if isinstance(afc, dict) else getattr(afc, "disable", None)
        assert disable is True

    def test_schema_still_configured(self):
        models = _call_once("hello")
        cfg = models.kwargs["config"]
        assert cfg["response_mime_type"] == "application/json"
        assert cfg.get("response_json_schema") is not None

    def test_one_sdk_call(self):
        assert _call_once("hello").calls == 1

    def test_input_payload_still_wrapped_with_instruction_and_input(self):
        models = _call_once("hello there")
        payload = json.loads(models.kwargs["contents"])
        assert "instruction" in payload
        assert "input" in payload
        assert payload["input"]["current_utterance"] == "hello there"

    def test_realization_adapter_afc_disable_preserved(self):
        from app.conversation.realization import adapter as mod
        src = inspect.getsource(mod)
        assert '"automatic_function_calling": {"disable": True}' in src

    def test_realization_prompt_plan_fidelity_preserved(self):
        from app.conversation.realization import prompt as mod
        src = inspect.getsource(mod)
        assert "clarification_required" in src
        assert "communicative_goal" in src
        assert "do not reconsider" in src.lower()


# ---------------------------------------------------------------------------
# Finding 5 — dev composition truth test
# ---------------------------------------------------------------------------


class TestDevRuntimeHasNoTurnIndexSemantics:
    def test_dev_bootstrap_does_not_depend_on_turn_order_for_intent(self):
        """§40 — same utterance in fresh sessions produces equivalent
        semantic classification. With `intent_transport=None`, there is
        NO turn-index semantics at all: every utterance maps to
        `ReExplanationIntent.NONE` through the processor's safe fallback.
        """
        src = _MAIN_PY.read_text(encoding="utf-8")
        # The dev bootstrap uses a non-stateful classifier (None),
        # eliminating turn-index dependence by construction
        assert "intent_transport = None" in src
        # And does not INSTANTIATE a sequence-based classifier (comments
        # referencing the type name by way of historical note are fine)
        assert "ScriptedIntentTransport(" not in src
        assert "import ScriptedIntentTransport" not in src
