"""Direct Gemini reasoning diagnostic — one call, sanitized output.

Isolates the GeminiReasoningProvider from WebSocket, PersistentSalesRuntime,
Planner, and Renderer so the operator can see exactly what the provider
is doing in the current environment.

Usage:
    python -m scripts.diag_gemini_reasoning           # one call
    python -m scripts.diag_gemini_reasoning --n 3     # three calls

Output (stderr-safe; never prints the API key):
    [diag] provider=gemini model=<model> api_key_present=True
    [diag] call 1/N: success | failure_kind=... exception_type=... http_status=...
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

from app.brain.contracts import BrainInput
from app.config.settings import get_settings
from app.core.constants import ConversationState
from app.llm.gemini_provider import GeminiReasoningProvider
from app.llm.providers.reasoning_provider import ReasoningError


def _bool_present(value: str | None) -> bool:
    return bool(value and value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(prog="diag_gemini_reasoning")
    parser.add_argument("--n", type=int, default=1, help="how many calls")
    parser.add_argument("--utterance", type=str, default="hi")
    args = parser.parse_args()

    settings = get_settings()
    print(
        f"[diag] provider={settings.llm_provider} "
        f"model={settings.llm_model} "
        f"api_key_present={_bool_present(settings.gemini_api_key)}",
        file=sys.stderr, flush=True,
    )
    if not _bool_present(settings.gemini_api_key):
        print(
            "[diag] GEMINI_API_KEY not set — cannot run live diagnostic.",
            file=sys.stderr, flush=True,
        )
        return 2

    provider = GeminiReasoningProvider(
        model=settings.llm_model,
        api_key=settings.gemini_api_key,
        timeout_seconds=settings.llm_timeout_seconds,
    )

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
                f"[diag] call {i}/{args.n}: UNHANDLED exception_type={type(exc).__name__}",
                file=sys.stderr, flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
