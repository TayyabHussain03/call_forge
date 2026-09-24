"""Immutable contracts for the derived, call-scoped business memory graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GraphNodeType(str, Enum):
    BUSINESS = "business"
    PERSON = "person"
    DEPARTMENT = "department"
    LOCATION = "location"
    BRANCH = "branch"
    PROCESS = "process"
    WORKFLOW = "workflow"
    SYSTEM = "system"
    TECHNOLOGY = "technology"
    SERVICE_USAGE = "service_usage"
    BUSINESS_GOAL = "business_goal"
    BUSINESS_CONSTRAINT = "business_constraint"
    PROBLEM = "problem"
    QUALIFICATION = "qualification"
    UNKNOWN_ENTITY = "unknown_entity"


class GraphRelationshipType(str, Enum):
    USES = "uses"
    OWNS = "owns"
    MANAGES = "manages"
    WORKS_IN = "works_in"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    CONNECTED_TO = "connected_to"
    BLOCKED_BY = "blocked_by"
    SUPPORTS = "supports"
    LOCATED_AT = "located_at"
    RELATED_TO = "related_to"


class GraphSourceComponent(str, Enum):
    BCI = "bci"
    PROSPECT = "prospect_intelligence"
    PQI = "pqi"
    BDE = "bde"
    BMME = "bmme_structural"


class GraphEvidenceBasis(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"


@dataclass(frozen=True)
class BusinessMemoryScope:
    tenant_id: str
    business_id: str
    campaign_id: str
    contact_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("tenant", self.tenant_id), ("business", self.business_id), ("campaign", self.campaign_id)):
            if not value.strip() or len(value) > 100:
                raise ValueError(f"{name} id must be bounded")
        if self.contact_id is not None and (not self.contact_id.strip() or len(self.contact_id) > 100):
            raise ValueError("contact id must be bounded")


@dataclass(frozen=True)
class GraphLimits:
    max_active_nodes: int = 80
    max_relationships: int = 120
    max_archived_items: int = 80

    def __post_init__(self) -> None:
        if not (1 <= self.max_active_nodes <= 500 and 1 <= self.max_relationships <= 1000 and 1 <= self.max_archived_items <= 500):
            raise ValueError("graph limits are out of bounds")


@dataclass(frozen=True)
class BusinessGraphNode:
    stable_id: str
    identity_key: str
    tenant_id: str
    business_id: str
    campaign_id: str
    contact_id: str | None
    node_type: GraphNodeType
    value: str
    source_component: GraphSourceComponent
    created_turn: str
    last_updated_turn: str
    evidence_ids: tuple[str, ...]
    evidence_basis: GraphEvidenceBasis
    active: bool = True
    archived: bool = False

    def __post_init__(self) -> None:
        if not self.stable_id or not self.identity_key or not self.value or not self.evidence_ids:
            raise ValueError("graph node requires identity, value, and provenance")
        if self.active == self.archived:
            raise ValueError("graph node must be either active or archived")


@dataclass(frozen=True)
class BusinessGraphRelationship:
    stable_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: GraphRelationshipType
    source_component: GraphSourceComponent
    source_turn: str
    evidence_ids: tuple[str, ...]
    evidence_basis: GraphEvidenceBasis
    active: bool = True
    archived: bool = False


@dataclass(frozen=True)
class BusinessMentalSummary:
    business_structure: tuple[str, ...] = ()
    known_processes: tuple[str, ...] = ()
    known_systems: tuple[str, ...] = ()
    known_goals: tuple[str, ...] = ()
    known_constraints: tuple[str, ...] = ()
    known_problems: tuple[str, ...] = ()
    known_relationships: tuple[str, ...] = ()
    known_qualification: tuple[str, ...] = ()
    unknown_areas: tuple[str, ...] = ()


@dataclass(frozen=True)
class BusinessMentalModelSnapshot:
    scope: BusinessMemoryScope
    nodes: tuple[BusinessGraphNode, ...]
    relationships: tuple[BusinessGraphRelationship, ...]
    archived_nodes: tuple[BusinessGraphNode, ...]
    archived_relationships: tuple[BusinessGraphRelationship, ...]
    summary: BusinessMentalSummary

    def __post_init__(self) -> None:
        if len({node.stable_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("active node stable ids must be unique")
        if len({edge.stable_id for edge in self.relationships}) != len(self.relationships):
            raise ValueError("active relationship ids must be unique")
