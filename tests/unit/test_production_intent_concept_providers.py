"""Slice 51E — Production intent + concept providers.

Proves:
    - ReExplanationIntentLLMProvider parses valid typed intents
    - Rejects unknown enum values and malformed JSON safely → NONE
    - Transport exception falls back to NONE (no fabricated intent)
    - Supports English, Roman Urdu, and mixed-language scripted outputs
    - `as_callable()` adapts to processor's two-argument seam
    - BrainConceptProvider reads from live playbook_guidance.selected_service_id
    - Returns None when no guidance / no service selected
    - No keyword trees in production code
"""

from __future__ import annotations

import pytest

from app.conversation.participant.concept_provider import (
    BrainConceptProvider,
    SERVICE_CONCEPT_PREFIX,
    StaticConceptProvider,
)
from app.conversation.participant.intent_provider import (
    IntentClassificationInput,
    IntentClassifierTransport,
    RaisingIntentTransport,
    ReExplanationIntentLLMProvider,
    ScriptedIntentTransport,
)
from app.conversation.participant.policy import ReExplanationIntent


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestIntentClassificationInput:
    def test_minimal_valid(self) -> None:
        inp = IntentClassificationInput(utterance="hi")
        assert inp.utterance == "hi"
        assert inp.primary_language is None

    def test_empty_utterance_rejected(self) -> None:
        with pytest.raises(ValueError):
            IntentClassificationInput(utterance="")

    def test_oversized_utterance_rejected(self) -> None:
        with pytest.raises(ValueError):
            IntentClassificationInput(utterance="x" * 3000)

    def test_prompt_dict_bounded(self) -> None:
        inp = IntentClassificationInput(
            utterance="remind me", primary_language="en",
            active_participant_role_hint="employee",
            concept_in_focus="service:automation",
        )
        d = inp.to_prompt_dict()
        assert "transcript" not in d  # bounded context — no full transcript
        assert "allowed_intents" in d
        assert "recap_request" in d["allowed_intents"]


# ---------------------------------------------------------------------------
# Valid typed outputs
# ---------------------------------------------------------------------------


class TestValidIntentParsing:
    def test_recap(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"recap_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(
            utterance="remind me what you said",
        ))
        assert result == ReExplanationIntent.RECAP_REQUEST

    def test_simplify(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"simplify_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(
            utterance="I still don't understand",
        ))
        assert result == ReExplanationIntent.SIMPLIFY_REQUEST

    def test_deepen(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"deepen_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(
            utterance="tell me more about how it works",
        ))
        assert result == ReExplanationIntent.DEEPEN_REQUEST

    def test_direct_service_question(self) -> None:
        transport = ScriptedIntentTransport(
            ('{"intent":"direct_service_question"}',),
        )
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(
            utterance="what exactly does this service do?",
        ))
        assert result == ReExplanationIntent.DIRECT_SERVICE_QUESTION

    def test_none(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"none"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(
            utterance="we started last year",
        ))
        assert result == ReExplanationIntent.NONE


# ---------------------------------------------------------------------------
# Safe failure modes
# ---------------------------------------------------------------------------


class TestSafeFailureModes:
    def test_transport_exception_returns_none(self) -> None:
        provider = ReExplanationIntentLLMProvider(RaisingIntentTransport())
        result = provider.classify(IntentClassificationInput(utterance="hi"))
        assert result == ReExplanationIntent.NONE

    def test_malformed_json_returns_none(self) -> None:
        transport = ScriptedIntentTransport(('{not json',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(utterance="hi"))
        assert result == ReExplanationIntent.NONE

    def test_unknown_enum_returns_none(self) -> None:
        """Guard against LLM inventing a non-canonical intent string.
        Must never produce a stronger intent than NONE."""
        transport = ScriptedIntentTransport(('{"intent":"invented_type"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(utterance="hi"))
        assert result == ReExplanationIntent.NONE

    def test_missing_intent_field_returns_none(self) -> None:
        transport = ScriptedIntentTransport(('{"other":"recap_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(utterance="hi"))
        assert result == ReExplanationIntent.NONE

    def test_non_dict_output_returns_none(self) -> None:
        transport = ScriptedIntentTransport(('["recap_request"]',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(utterance="hi"))
        assert result == ReExplanationIntent.NONE

    def test_empty_string_output_returns_none(self) -> None:
        transport = ScriptedIntentTransport(("",))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(utterance="hi"))
        assert result == ReExplanationIntent.NONE


# ---------------------------------------------------------------------------
# Mixed-language semantics
# ---------------------------------------------------------------------------


class TestMixedLanguageSupport:
    def test_roman_urdu_simplify(self) -> None:
        """The adapter does not interpret language itself — the transport
        decides. But the adapter must accept and return the typed result
        regardless of input language."""
        transport = ScriptedIntentTransport(('{"intent":"simplify_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(
            utterance="mujhe samajh nahi aya",
            primary_language="ur", secondary_language="en",
        ))
        assert result == ReExplanationIntent.SIMPLIFY_REQUEST

    def test_hinglish_deepen(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"deepen_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        result = provider.classify(IntentClassificationInput(
            utterance="thora detail mein batao kaise kaam karega",
            primary_language="ur", secondary_language="en",
        ))
        assert result == ReExplanationIntent.DEEPEN_REQUEST

    def test_language_fields_passed_in_prompt(self) -> None:
        captured = {}

        class CapturingTransport:
            def classify(self, prompt_input):
                captured.update(prompt_input)
                return '{"intent":"none"}'

        provider = ReExplanationIntentLLMProvider(CapturingTransport())
        provider.classify(IntentClassificationInput(
            utterance="phir se batao",
            primary_language="ur",
            secondary_language="en",
        ))
        assert captured["primary_language"] == "ur"
        assert captured["secondary_language"] == "en"


# ---------------------------------------------------------------------------
# as_callable adapts to the processor's seam
# ---------------------------------------------------------------------------


class TestAsCallableSeam:
    def test_callable_signature(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"recap_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        callable_ = provider.as_callable()

        # Mock the two-argument processor seam (turn, understanding)
        class _T:
            utterance = "remind me"
        class _U:
            class language_profile:
                primary_language = "en"
                secondary_language = None

        result = callable_(_T(), _U())
        assert result == ReExplanationIntent.RECAP_REQUEST

    def test_callable_handles_none_understanding(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"none"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        callable_ = provider.as_callable()

        class _T:
            utterance = "hi"

        assert callable_(_T(), None) == ReExplanationIntent.NONE


# ---------------------------------------------------------------------------
# ConceptProvider — reads from existing trusted structured state
# ---------------------------------------------------------------------------


class TestBrainConceptProvider:
    def test_resolves_service_from_playbook_guidance(self) -> None:
        class FakeGuidance:
            selected_service_id = "automation"

        class FakeProcessor:
            @property
            def playbook_guidance(self):
                return FakeGuidance()

        provider = BrainConceptProvider(FakeProcessor())
        assert provider.resolve() == f"{SERVICE_CONCEPT_PREFIX}automation"

    def test_returns_none_when_no_guidance(self) -> None:
        class FakeProcessor:
            @property
            def playbook_guidance(self):
                return None

        provider = BrainConceptProvider(FakeProcessor())
        assert provider.resolve() is None

    def test_returns_none_when_no_selected_service(self) -> None:
        class FakeGuidance:
            selected_service_id = None

        class FakeProcessor:
            @property
            def playbook_guidance(self):
                return FakeGuidance()

        provider = BrainConceptProvider(FakeProcessor())
        assert provider.resolve() is None

    def test_as_callable_processor_seam(self) -> None:
        class FakeGuidance:
            selected_service_id = "web_development"

        class FakeProcessor:
            @property
            def playbook_guidance(self):
                return FakeGuidance()

        callable_ = BrainConceptProvider(FakeProcessor()).as_callable()
        assert callable_(None, None) == f"{SERVICE_CONCEPT_PREFIX}web_development"

    def test_static_provider_for_tests(self) -> None:
        c = StaticConceptProvider("service:test").as_callable()
        assert c(None, None) == "service:test"
        c_none = StaticConceptProvider(None).as_callable()
        assert c_none(None, None) is None


# ---------------------------------------------------------------------------
# No keyword classification in production code
# ---------------------------------------------------------------------------


class TestNoKeywordTreeInIntentProvider:
    def test_no_phrase_matching_in_intent_module(self) -> None:
        import inspect
        from app.conversation.participant import intent_provider
        source = inspect.getsource(intent_provider)
        # Ensure the module doesn't do its OWN classification by phrase
        # matching — the transport (LLM) does. The deterministic adapter
        # only parses/validates typed JSON output.
        for forbidden in (
            '"remind me"', '"didn\'t understand"', '"tell me more"',
            "'remind me'", "'tell me more'",
        ):
            assert forbidden not in source

    def test_no_industry_hardcoding_in_concept_provider(self) -> None:
        import inspect
        from app.conversation.participant import concept_provider
        source = inspect.getsource(concept_provider)
        for forbidden in (
            '"restaurant"', '"dentist"', '"hvac"',
            'if industry', 'if service ==',
        ):
            assert forbidden not in source
