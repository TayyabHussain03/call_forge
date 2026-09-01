"""Application-level exception types.

Guide ke mutabiq: generic exception handling har jagah use nahi karni. Ye alag
categories predictable failure semantics deti hain — infrastructure errors ko
saaf application errors mein convert karne ke liye.

Ye exceptions PII-safe hain: messages mein transcript/email/phone jaisa
sensitive data nahi daalna — sirf structural/technical detail.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application-level errors."""


class ConfigurationError(AppError):
    """Raised when configuration is invalid or fails startup validation.

    Fail-fast ke liye: state machine config load hote waqt koi structural
    problem (unknown state, invalid transition, terminal-with-outgoing) is
    exception ko raise karti hai, taake app boot na ho.
    """


class ValidationError(AppError):
    """Raised when data fails application-level (non-Pydantic) validation."""


class StateTransitionError(AppError):
    """Raised when an illegal state transition is attempted.

    Ye tab uthti hai jab koi caller aisa action apply karne ki koshish kare jo
    current state se eligible nahi — machine state ko mutate nahi karti.
    """


class ProviderError(AppError):
    """Raised when an external provider (LLM, voice) fails.

    Placeholder for later stages; abhi state machine isko use nahi karti.
    """