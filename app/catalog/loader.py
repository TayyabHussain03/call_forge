"""Service-catalog loading and fail-fast validation.

`service_catalog.yaml` ko typed structures mein load karta hai aur STARTUP par
validate karta hai. Koi structural problem (duplicate ids, campaign referencing
undefined service, allowed/forbidden claim overlap) → ConfigurationError, boot
rukega.

DESIGN: allowed aur forbidden claims OVERLAP nahi kar sakte — ek claim ya allowed
hai ya forbidden, dono nahi. Ye fail-fast catch karta hai.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.catalog.models import Campaign, Service, ServiceCatalog, Subservice
from app.core.exceptions import ConfigurationError


def _parse_subservice(raw: dict[str, Any]) -> Subservice:
    """Parse one subservice, failing fast on claim overlap.

    Args:
        raw: Raw subservice dict.

    Returns:
        Subservice: Parsed subservice.

    Raises:
        ConfigurationError: On missing id/name or allowed/forbidden overlap.
    """
    sid = raw.get("id")
    name = raw.get("name")
    if not sid or not name:
        raise ConfigurationError(f"Subservice needs id and name: {raw!r}")
    allowed = frozenset(raw.get("allowed_claims", []))
    forbidden = frozenset(raw.get("forbidden_claims", []))
    overlap = allowed & forbidden
    if overlap:
        raise ConfigurationError(
            f"Subservice {sid!r} has claims both allowed and forbidden: {overlap}"
        )
    return Subservice(
        id=sid,
        name=name,
        capabilities=tuple(raw.get("capabilities", [])),
        allowed_claims=allowed,
        forbidden_claims=forbidden,
    )


def _parse_service(raw: dict[str, Any]) -> Service:
    """Parse one service (with subservices), failing fast on problems.

    Args:
        raw: Raw service dict.

    Returns:
        Service: Parsed service.

    Raises:
        ConfigurationError: On missing id/name, claim overlap, or duplicate
            subservice ids.
    """
    sid = raw.get("id")
    name = raw.get("name")
    if not sid or not name:
        raise ConfigurationError(f"Service needs id and name: {raw!r}")

    allowed = frozenset(raw.get("allowed_claims", []))
    forbidden = frozenset(raw.get("forbidden_claims", []))
    overlap = allowed & forbidden
    if overlap:
        raise ConfigurationError(
            f"Service {sid!r} has claims both allowed and forbidden: {overlap}"
        )

    subservices: dict[str, Subservice] = {}
    for sub_raw in raw.get("subservices", []):
        sub = _parse_subservice(sub_raw)
        if sub.id in subservices:
            raise ConfigurationError(
                f"Duplicate subservice id {sub.id!r} in service {sid!r}"
            )
        subservices[sub.id] = sub

    return Service(
        id=sid,
        name=name,
        capabilities=tuple(raw.get("capabilities", [])),
        applicable_when=frozenset(raw.get("applicable_when", [])),
        allowed_claims=allowed,
        forbidden_claims=forbidden,
        subservices=subservices,
    )


def load_catalog(path: str | Path) -> ServiceCatalog:
    """Load and validate the service catalog, failing fast on any error.

    Validates: duplicate service ids, duplicate campaign ids, claim overlaps,
    aur har campaign ka service reference (undefined service → fail).

    Args:
        path: Path to the catalog YAML.

    Returns:
        ServiceCatalog: The parsed, validated catalog.

    Raises:
        ConfigurationError: On any structural problem.
    """
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise ConfigurationError(f"Catalog file not found: {catalog_path}")

    raw: dict[str, Any] = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}

    # ── services ──
    services: dict[str, Service] = {}
    for svc_raw in raw.get("services", []):
        svc = _parse_service(svc_raw)
        if svc.id in services:
            raise ConfigurationError(f"Duplicate service id: {svc.id!r}")
        services[svc.id] = svc

    if not services:
        raise ConfigurationError("Catalog has no services defined.")

    # ── campaigns ──
    campaigns: dict[str, Campaign] = {}
    campaigns_raw = raw.get("campaigns", {})
    for camp_id, body in campaigns_raw.items():
        body = body or {}
        service_ids = tuple(body.get("services", []))
        # every referenced service must exist.
        for ref in service_ids:
            if ref not in services:
                raise ConfigurationError(
                    f"Campaign {camp_id!r} references undefined service {ref!r}"
                )
        if camp_id in campaigns:
            raise ConfigurationError(f"Duplicate campaign id: {camp_id!r}")
        campaigns[camp_id] = Campaign(
            id=camp_id,
            name=body.get("name", camp_id),
            service_ids=service_ids,
        )

    return ServiceCatalog(services=services, campaigns=campaigns)