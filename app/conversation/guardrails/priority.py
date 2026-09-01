"""Deterministic priority-action resolution.

Ye layer LLM proposal ke UPAR baithti hai. Kuch client intents itne high-priority
hote hain ke unpe agent ka action deterministic hona chahiye — LLM kya propose
karta hai isse farq nahi padta. Sabse important: DNC.

SEPARATION (guide ke mutabiq):
    - `resolve_priority_action()` = PRIMARY decision: kya koi higher-priority
      deterministic action LLM proposal ko replace kare?
    - ActionValidator = us resulting action ko validate karta hai + defensive
      safety-net deta hai.
    - StateMachine = structural authority + transition mutation.

Ye resolver STATE/CONTEXT MUTATE NAHI karta — sirf padhta hai aur ek action (ya
None) return karta hai.

SCOPE (Sitting 2B): sirf DNC aur NOT_INTERESTED. Signature future-proof hai
(state + context accept karta hai) taake baad mein voicemail, wrong_number,
callback jaise cases bina redesign add ho sakein.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.core.constants import AgentAction, ConversationState, Intent


def resolve_priority_action(
    intent: Intent,
    state: ConversationState,
    context: Mapping[str, Any] | None = None,
) -> AgentAction | None:
    """Return a deterministic action that must override the LLM proposal, if any.

    High-priority intents pe agent ka behaviour deterministic hai:
        - do_not_call    → MARK_DNC   (sabse high priority; koi sales action nahi)
        - not_interested → END_CALL   (polite close; no repeated pitch)
    Baaki sab intents pe None return hota hai — matlab "koi override nahi, normal
    proposed action + validation use karo".

    Signature `state` aur `context` bhi leti hai taake future context/state-
    dependent priorities (voicemail, wrong_number, callback) bina redesign add ho
    sakein — Sitting 2B inhe use nahi karti.

    Args:
        intent: Client ka classified intent (primary signal).
        state: Current conversational state (future use; abhi ignore).
        context: Read-only conversation context (future use; abhi ignore). Ye
            function ise mutate nahi karta.

    Returns:
        AgentAction | None: Forced deterministic action, ya None agar koi
        priority rule apply nahi hoti.
    """
    # DNC: highest priority. Kisi bhi proposed sales action ko override karta hai.
    if intent == Intent.DO_NOT_CALL:
        return AgentAction.MARK_DNC

    # NOT_INTERESTED: deterministic close, no persuasion.
    if intent == Intent.NOT_INTERESTED:
        return AgentAction.END_CALL

    # Koi priority rule nahi — normal flow.
    return None