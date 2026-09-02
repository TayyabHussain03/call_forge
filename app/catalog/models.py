"""Typed service-catalog structures.

Catalog authority ka in-memory representation: service → subservice → capability,
allowed/forbidden claims, applicability signals, aur campaign scoping. Ye pure
data structures hain — koi behaviour nahi (loader/validator/matcher alag).

DESIGN: catalog eligibility/authorization deta hai, relevance NAHI. "Client SEO
mein interested hai" catalog ka concern nahi — woh future selector ka.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Subservice:
    """A subservice within a service (e.g. Local SEO under SEO).

    Attributes:
        id: Unique subservice id.
        name: Display name.
        capabilities: Concrete capabilities offered.
        allowed_claims: Claims the agent may make for this subservice.
        forbidden_claims: Claims the agent must never make.
    """

    id: str
    name: str
    capabilities: tuple[str, ...] = ()
    allowed_claims: frozenset[str] = field(default_factory=frozenset)
    forbidden_claims: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Service:
    """A top-level service (e.g. SEO, Website Development).

    Attributes:
        id: Unique service id.
        name: Display name.
        capabilities: Service-level capabilities.
        applicable_when: Generic eligibility signals (e.g. existing_website).
            Eligibility only — NOT relevance.
        allowed_claims: Service-level allowed claims.
        forbidden_claims: Service-level forbidden claims.
        subservices: Nested subservices, keyed by id.
    """

    id: str
    name: str
    capabilities: tuple[str, ...] = ()
    applicable_when: frozenset[str] = field(default_factory=frozenset)
    allowed_claims: frozenset[str] = field(default_factory=frozenset)
    forbidden_claims: frozenset[str] = field(default_factory=frozenset)
    subservices: dict[str, Subservice] = field(default_factory=dict)


@dataclass(frozen=True)
class Campaign:
    """A campaign scoping which services are authorized.

    Attributes:
        id: Unique campaign id.
        name: Display name.
        service_ids: Authorized service ids for this campaign.
    """

    id: str
    name: str
    service_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServiceCatalog:
    """The fully-parsed, validated catalog.

    Attributes:
        services: All defined services, keyed by id.
        campaigns: All campaigns, keyed by id.
    """

    services: dict[str, Service]
    campaigns: dict[str, Campaign]