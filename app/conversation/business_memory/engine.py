"""Deterministic projection of upstream snapshots into a bounded graph."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from app.conversation.business_conversation.contracts import BusinessConversationSnapshot, BusinessFactKind
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
from app.conversation.business_memory.contracts import *
from app.conversation.prospect_intelligence.contracts import ProspectIntelligenceSnapshot
from app.conversation.qualification.contracts import QualificationSnapshot, QualificationValueState


class BusinessMentalModelEngine:
    def project(self, scope: BusinessMemoryScope, conversation: BusinessConversationSnapshot,
                prospect: ProspectIntelligenceSnapshot, qualification: QualificationSnapshot | None,
                diagnostic: BusinessDiagnosticSnapshot, prior: BusinessMentalModelSnapshot | None = None,
                limits: GraphLimits = GraphLimits()) -> BusinessMentalModelSnapshot:
        if prior is not None and prior.scope != scope:
            prior = None
        candidates = [_root(scope), *_conversation_nodes(scope, conversation), *_prospect_nodes(scope, prospect), *_qualification_nodes(scope, qualification), *_diagnostic_nodes(scope, diagnostic)]
        nodes, archived = _reconcile(tuple(candidates), prior, limits)
        relationships = _structural_relationships(nodes, limits)
        _reject_cycles(relationships)
        archived_edges = prior.archived_relationships if prior else ()
        if prior:
            current_ids = {edge.stable_id for edge in relationships}
            archived_edges = (*archived_edges, *(replace(edge, active=False, archived=True) for edge in prior.relationships if edge.stable_id not in current_ids))[-limits.max_archived_items:]
        return BusinessMentalModelSnapshot(scope, nodes, relationships, archived, tuple(archived_edges), _summary(nodes, relationships))


def _id(*parts: str) -> str:
    return sha256("|".join(parts).encode()).hexdigest()[:24]


def _node(scope, key, node_type, value, source, turn, evidence, basis=GraphEvidenceBasis.OBSERVED):
    return BusinessGraphNode(_id(scope.tenant_id, scope.business_id, scope.campaign_id, key), key, scope.tenant_id, scope.business_id, scope.campaign_id, scope.contact_id if node_type == GraphNodeType.PERSON else None, node_type, value, source, turn, turn, tuple(dict.fromkeys(evidence)), basis)


def _root(scope):
    return _node(scope, "business", GraphNodeType.BUSINESS, scope.business_id, GraphSourceComponent.BMME, "scope", ("scope",))


def _conversation_nodes(scope, snapshot):
    mapping = {BusinessFactKind.BRANCHES: GraphNodeType.BRANCH, BusinessFactKind.CURRENT_WORKFLOW: GraphNodeType.WORKFLOW,
               BusinessFactKind.MANUAL_STEP: GraphNodeType.PROCESS, BusinessFactKind.SOFTWARE: GraphNodeType.SYSTEM,
               BusinessFactKind.INTEGRATION: GraphNodeType.TECHNOLOGY, BusinessFactKind.COMMUNICATION: GraphNodeType.SYSTEM,
               BusinessFactKind.WEBSITE: GraphNodeType.TECHNOLOGY, BusinessFactKind.CRM: GraphNodeType.SYSTEM,
               BusinessFactKind.INDUSTRY: GraphNodeType.BUSINESS, BusinessFactKind.BUSINESS_TYPE: GraphNodeType.BUSINESS}
    result=[]
    for fact in snapshot.facts:
        result.append(_node(scope, f"bci:fact:{fact.kind.value}", mapping.get(fact.kind, GraphNodeType.UNKNOWN_ENTITY), fact.value, GraphSourceComponent.BCI, fact.source_turn_id, (fact.source_turn_id,)))
    for problem in snapshot.problems:
        result.append(_node(scope, f"bci:problem:{problem.kind.value}", GraphNodeType.PROBLEM, problem.detail, GraphSourceComponent.BCI, problem.source_turn_id, (problem.source_turn_id,)))
    for goal in snapshot.goals:
        result.append(_node(scope, f"bci:goal:{goal.value}", GraphNodeType.BUSINESS_GOAL, goal.value, GraphSourceComponent.BCI, "current", (goal.value,)))
    for constraint in snapshot.constraints:
        result.append(_node(scope, f"bci:constraint:{constraint.value}", GraphNodeType.BUSINESS_CONSTRAINT, constraint.value, GraphSourceComponent.BCI, "current", (constraint.value,)))
    return result


def _qualification_nodes(scope, snapshot):
    if snapshot is None: return []
    result=[]
    for field in snapshot.fields:
        if field.state == QualificationValueState.UNKNOWN: continue
        basis=GraphEvidenceBasis.OBSERVED if field.state == QualificationValueState.OBSERVED else GraphEvidenceBasis.INFERRED
        result.append(_node(scope, f"pqi:{field.field_id}", GraphNodeType.QUALIFICATION, field.value or field.field_id, GraphSourceComponent.PQI, field.source_turn_id or "current", field.evidence_ids or (field.field_id,), basis))
    return result


def _prospect_nodes(scope, snapshot):
    role = snapshot.observed.explicit_role
    basis = GraphEvidenceBasis.OBSERVED
    if role is None:
        role = snapshot.inferred.likely_role
        basis = GraphEvidenceBasis.INFERRED
    if role is None:
        return []
    turn = role.provenance.source_turn_id
    value = role.value.value
    return [_node(scope, f"prospect:person:{scope.contact_id or 'current'}", GraphNodeType.PERSON, value, GraphSourceComponent.PROSPECT, turn, (turn,), basis)]


def _diagnostic_nodes(scope, snapshot):
    return [_node(scope, f"bde:unknown:{area.value}", GraphNodeType.UNKNOWN_ENTITY, area.value, GraphSourceComponent.BDE, "current", (area.value,)) for area in snapshot.summary.remaining_unknowns]


def _reconcile(candidates, prior, limits):
    current={node.stable_id:node for node in candidates}; archived=list(prior.archived_nodes) if prior else []
    if prior:
        for old in prior.nodes:
            new=current.get(old.stable_id)
            if new is None or new.value != old.value:
                archived.append(replace(old, active=False, archived=True))
                if new is not None: current[old.stable_id]=replace(new, created_turn=old.created_turn)
    nodes=tuple(current.values())
    if len(nodes) > limits.max_active_nodes: raise ValueError("active graph node limit exceeded")
    return nodes, tuple(archived[-limits.max_archived_items:])


def _structural_relationships(nodes, limits):
    root=next(node for node in nodes if node.node_type == GraphNodeType.BUSINESS and node.identity_key == "business")
    edges=[]
    for node in nodes:
        if node.stable_id == root.stable_id: continue
        edges.append(BusinessGraphRelationship(_id(node.stable_id, root.stable_id, "part_of"), node.stable_id, root.stable_id, GraphRelationshipType.PART_OF, GraphSourceComponent.BMME, node.last_updated_turn, node.evidence_ids, node.evidence_basis))
    if len(edges) > limits.max_relationships: raise ValueError("graph relationship limit exceeded")
    return tuple(edges)


def _reject_cycles(edges):
    graph={}
    for edge in edges: graph.setdefault(edge.source_node_id, set()).add(edge.target_node_id)
    def visit(node, path):
        if node in path: raise ValueError("relationship cycle rejected")
        for target in graph.get(node, ()): visit(target, path | {node})
    for node in graph: visit(node, set())


def _summary(nodes, edges):
    values=lambda kinds: tuple(node.value for node in nodes if node.node_type in kinds)
    return BusinessMentalSummary(values({GraphNodeType.BUSINESS,GraphNodeType.BRANCH,GraphNodeType.DEPARTMENT,GraphNodeType.LOCATION}), values({GraphNodeType.PROCESS,GraphNodeType.WORKFLOW}), values({GraphNodeType.SYSTEM,GraphNodeType.TECHNOLOGY}), values({GraphNodeType.BUSINESS_GOAL}), values({GraphNodeType.BUSINESS_CONSTRAINT}), values({GraphNodeType.PROBLEM}), tuple(edge.stable_id for edge in edges), values({GraphNodeType.QUALIFICATION}), values({GraphNodeType.UNKNOWN_ENTITY}))
