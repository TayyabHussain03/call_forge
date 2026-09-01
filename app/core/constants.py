"""Shared enumerations used across the whole system.

Yahan sab domain vocabulary ek jagah define hoti hai — call status, intent,
tone, lead status, aur conversation states. Poora system (contracts, models,
state machine, analyzer, dashboard) inhi enums ko use karta hai, taake kahin
bhi raw strings ("interested", "busy") scatter na hon aur typos silent bugs na
banein.

Design note: Intent, Interest, aur Tone jaan-boojh kar ALAG enums hain. Ek
harsh tone wala client bhi high-interest ho sakta hai — inko mila dena tumhare
lead classification ko todega.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum base.

    Members plain strings ki tarah compare/serialize hote hain (JSON, DB,
    Pydantic ke saath seamless), lekin type-safety aur autocomplete milti hai.
    """

    def __str__(self) -> str:
        """Return the raw string value (not 'ClassName.MEMBER').

        Returns:
            str: The member's underlying string value.
        """
        return self.value


class CallStatus(StrEnum):
    """Lifecycle status of a single outbound call attempt."""

    QUEUED = "queued"
    DIALING = "dialing"
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    VOICEMAIL = "voicemail"
    FAILED = "failed"
    COMPLETED = "completed"


class Intent(StrEnum):
    """What the client is trying to convey in the conversation.

    Ye conversation-level meaning hai, lead quality nahi. Ek turn ya poori call
    ka intent classify karne ke liye use hota hai.
    """

    INTERESTED = "interested"
    NOT_INTERESTED = "not_interested"
    BUSY = "busy"
    CALLBACK_REQUESTED = "callback_requested"
    WRONG_PERSON = "wrong_person"
    ASKING_QUESTION = "asking_question"
    DO_NOT_CALL = "do_not_call"
    UNCLEAR = "unclear"


class Interest(StrEnum):
    """Qualification level — kitna strong lead hai."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Tone(StrEnum):
    """Emotional tone of the client during the call.

    Tone lead quality se independent hai — sirf classification/reporting ke
    liye track hota hai.
    """

    CALM = "calm"
    NEUTRAL = "neutral"
    IMPATIENT = "impatient"
    HARSH = "harsh"
    HOSTILE = "hostile"


class LeadStatus(StrEnum):
    """Overall pipeline status of a lead (dashboard lists isi par bante hain)."""

    NEW = "new"
    QUALIFIED = "qualified"
    INTERESTED_FOLLOWUP = "interested_followup"
    CONTACT_OBTAINED = "contact_obtained"
    CALLBACK_SCHEDULED = "callback_scheduled"
    MAYBE = "maybe"
    NOT_INTERESTED = "not_interested"
    DO_NOT_CALL = "do_not_call"


class ConversationState(StrEnum):
    """States of the conversation state machine.

    State machine in states ke beech move karti hai; har state mein LLM ke liye
    sirf kuch allowed actions hote hain (guardrails layer enforce karti hai).
    """

    NEW_CALL = "new_call"
    GREETING = "greeting"
    IDENTIFY_PERSON = "identify_person"
    CHECK_HISTORY = "check_history"
    INTRODUCE = "introduce"
    FOLLOW_UP = "follow_up"
    REASON_FOR_CALL = "reason_for_call"
    LISTEN = "listen"
    HANDLE_QUESTION = "handle_question"
    COLLECT_EMAIL = "collect_email"
    CONFIRM_CONTACT = "confirm_contact"
    SCHEDULE_CALLBACK = "schedule_callback"
    IDENTIFY_DECISION_MAKER = "identify_decision_maker"
    END_CALL = "end_call"


class AgentAction(StrEnum):
    """Concrete actions the agent (LLM) can propose in a turn.

    LLM in actions mein se ek propose karta hai; validator check karta hai ke
    current state mein woh allowed hai ya nahi.
    """

    GREET = "greet"
    ASK_IDENTITY = "ask_identity"
    INTRODUCE_REASON = "introduce_reason"
    ANSWER_QUESTION = "answer_question"
    ASK_EMAIL = "ask_email"
    CONFIRM_EMAIL = "confirm_email"
    CLARIFY_CONTACT = "clarify_contact"
    SCHEDULE_CALLBACK = "schedule_callback"
    MARK_DNC = "mark_dnc"
    END_CALL = "end_call"