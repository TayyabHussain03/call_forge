"""Provider-neutral reasoning diagnostic — one call per iteration.

Resolves the configured reasoning provider via the existing
`create_reasoning_provider(settings)` factory. Works for Gemini and
Groq without script changes.

Usage:
    python -m scripts.diag_llm_reasoning                  # use configured
    python -m scripts.diag_llm_reasoning --n 3            # three calls
    python -m scripts.diag_llm_reasoning --provider groq  # override

Output per iteration (stderr-safe; never prints the API key):
    [diag] call N/M: success  proposed_action=... detected_intent=...
                    topic_category=... confidence=...
    [diag] call N/M: failure  kind=... exception_type=... http_status=...
"""

from __future__ import annotations

import argparse
import os
import sys

from app.brain.contracts import BrainInput
from app.config.settings import get_settings
from app.core.constants import ConversationState
from app.llm.provider_factory import create_reasoning_provider
from app.llm.providers.reasoning_provider import ReasoningError


def _bool_present(value: str | None) -> bool:
    return bool(value and value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(prog="diag_llm_reasoning")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--provider", type=str, default=None,
        help="override LLM_PROVIDER for this diagnostic only",
    )
    parser.add_argument("--utterance", type=str, default="hi")
    args = parser.parse_args()

    settings = get_settings()
    if args.provider:
        # Non-persistent override — mutates the Settings instance in
        # this process only; the running server/.env are untouched.
        try:
            object.__setattr__(settings, "llm_provider", args.provider)
        except Exception:
            settings.llm_provider = args.provider

    provider_name = settings.llm_provider
    model_name = settings.llm_model

    # Which key matters?
    provider_norm = provider_name.strip().lower()
    if provider_norm in {"gemini", "llm.gemini"}:
        key_present = _bool_present(settings.gemini_api_key)
    elif provider_norm in {"groq", "llm.groq"}:
        key_present = _bool_present(settings.groq_api_key)
    else:
        key_present = False

    print(
        f"[diag] provider={provider_name} model={model_name} "
        f"api_key_present={key_present}",
        file=sys.stderr, flush=True,
    )
    if not key_present:
        print(
            "[diag] required API key not set — cannot run diagnostic.",
            file=sys.stderr, flush=True,
        )
        return 2

    try:
        provider = create_reasoning_provider(settings)
    except Exception as exc:
        print(
            f"[diag] factory failed: {type(exc).__name__}: {str(exc)[:160]}",
            file=sys.stderr, flush=True,
        )
        return 2

    brain_input = BrainInput(
        current_utterance=args.utterance,
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

    failures = 0
    for i in range(1, args.n + 1):
        try:
            proposal = provider.reason(brain_input)
            print(
                f"[diag] call {i}/{args.n}: success "
                f"proposed_action={getattr(proposal.proposed_action, 'value', None)} "
                f"detected_intent={getattr(proposal.detected_intent, 'value', None)} "
                f"topic_category={getattr(proposal.topic_category, 'value', None)} "
                f"confidence={proposal.action_confidence}",
                file=sys.stderr, flush=True,
            )
        except ReasoningError as exc:
            failures += 1
            print(
                f"[diag] call {i}/{args.n}: failure "
                f"kind={exc.kind.value} "
                f"exception_type={getattr(exc, 'exception_type', None)} "
                f"http_status={getattr(exc, 'http_status', None)}",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            failures += 1
            print(
                f"[diag] call {i}/{args.n}: UNHANDLED "
                f"exception_type={type(exc).__name__}",
                file=sys.stderr, flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
