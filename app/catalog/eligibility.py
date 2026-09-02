"""Deterministic eligible-alternative-service computation.

Objection ke baad kaunse alternative services offer kiye ja sakte hain — ye
DETERMINISTIC hai (catalog authority + signals + already-offered). Koi relevance
(LLM ka concern) yahan nahi — sirf eligibility.

FORMULA (guide ke mutabiq):
    eligible   = ScopedCatalog.applicable_services(known_signals)
    remaining  = eligible - already_offered
    offer allowed jab: remaining non-empty AND offers_made < max.

Ye function stateless/pure hai — context values leta hai, remaining ids deta hai.
Engine isse recompute karta hai jab signals/campaign badle (stale boolean nahi).
"""

from __future__ import annotations

from collections.abc import Iterable

from app.catalog.scoped_catalog import ScopedCatalog


def compute_eligible_alternatives(
    scoped_catalog: ScopedCatalog,
    known_signals: frozenset[str],
    offered_service_ids: Iterable[str],
    offers_made: int,
    max_offers: int,
) -> tuple[str, ...]:
    """Return remaining eligible alternative service ids (non-repeating, bounded).

    Args:
        scoped_catalog: Campaign-scoped catalog (authority).
        known_signals: Detected eligibility signals (e.g. {"existing_website"}).
        offered_service_ids: Services already offered this call.
        offers_made: How many offers have been made.
        max_offers: Configured maximum offers.

    Returns:
        tuple[str, ...]: Eligible service ids not yet offered. Empty when the
        offer budget is exhausted or nothing eligible remains — jisse
        has_applicable_alternative False ho jaata hai (polite close).
    """
    if offers_made >= max_offers:
        return ()
    eligible = {svc.id for svc in scoped_catalog.applicable_services(known_signals)}
    remaining = eligible - set(offered_service_ids)
    # Deterministic order (sorted) — reproducible.
    return tuple(sorted(remaining))