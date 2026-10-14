"""Typed infrastructure failures for persistence operations.

These are infrastructure errors — they must NEVER convert into customer
sales outcomes.
"""

from __future__ import annotations


class PersistenceError(Exception):
    """Base class for all persistence infrastructure failures."""


class DatabaseUnavailableError(PersistenceError):
    """The database file/connection is not accessible."""


class SchemaMismatchError(PersistenceError):
    """The on-disk schema version is not supported by this code."""


class StaleRevisionError(PersistenceError):
    """Optimistic concurrency: expected revision does not match stored."""


class IdempotencyConflictError(PersistenceError):
    """A unique/idempotency constraint was violated."""


class InvalidPersistedPayloadError(PersistenceError):
    """Stored JSON/row payload cannot be deserialized safely."""


class TransactionFailedError(PersistenceError):
    """A multi-step transaction failed and was rolled back."""


class NotFoundError(PersistenceError):
    """Requested entity does not exist."""
