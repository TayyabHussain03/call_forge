from dataclasses import FrozenInstanceError, replace
import pytest

from app.conversation.business_conversation.contracts import BusinessConversationSnapshot, BusinessFactKind, ObservedBusinessFact
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticInput
from app.conversation.business_diagnostic.engine import BusinessDiagnosticEngine
from app.conversation.business_memory.contracts import *
from app.conversation.business_memory.engine import BusinessMentalModelEngine, _reject_cycles
from app.conversation.prospect_intelligence.contracts import ProspectIntelligenceSnapshot

def _fact(value="Excel", turn="t1"): return ObservedBusinessFact(BusinessFactKind.SOFTWARE,value,turn)
def _scope(campaign="c", tenant="t", business="b"): return BusinessMemoryScope(tenant,business,campaign,"contact")
def _project(conversation, scope=None, prior=None, limits=GraphLimits()):
    diagnostic=BusinessDiagnosticEngine().diagnose(BusinessDiagnosticInput(conversation))
    return BusinessMentalModelEngine().project(scope or _scope(),conversation,ProspectIntelligenceSnapshot(),None,diagnostic,prior,limits)

def test_node_relationship_creation_identity_and_provenance():
    graph=_project(BusinessConversationSnapshot(facts=(_fact(),)))
    system=next(n for n in graph.nodes if n.node_type==GraphNodeType.SYSTEM)
    assert system.evidence_ids==("t1",) and system.source_component==GraphSourceComponent.BCI
    assert any(e.source_node_id==system.stable_id and e.relationship_type==GraphRelationshipType.PART_OF for e in graph.relationships)
    assert system.stable_id==next(n for n in _project(BusinessConversationSnapshot(facts=(_fact(),))).nodes if n.node_type==GraphNodeType.SYSTEM).stable_id

def test_correction_archives_old_revision_and_preserves_identity():
    first=_project(BusinessConversationSnapshot(facts=(_fact("Excel","t1"),)))
    second=_project(BusinessConversationSnapshot(facts=(_fact("Google Sheets","t2"),)),prior=first)
    current=next(n for n in second.nodes if n.node_type==GraphNodeType.SYSTEM)
    archived=next(n for n in second.archived_nodes if n.node_type==GraphNodeType.SYSTEM)
    assert current.stable_id==archived.stable_id and current.value=="Google Sheets"
    assert archived.value=="Excel" and archived.archived and not archived.active

def test_duplicates_merge_by_stable_identity_and_scope_isolated():
    duplicate=BusinessConversationSnapshot(facts=(_fact("Excel"),_fact("Excel")))
    graph=_project(duplicate)
    assert len([n for n in graph.nodes if n.node_type==GraphNodeType.SYSTEM])==1
    assert _project(duplicate,_scope(tenant="other")).nodes[0].stable_id != graph.nodes[0].stable_id
    assert _project(duplicate,_scope(campaign="other")).scope != graph.scope

def test_disappearing_upstream_fact_archives_without_deletion():
    first=_project(BusinessConversationSnapshot(facts=(_fact(),)))
    second=_project(BusinessConversationSnapshot(),prior=first)
    assert not any(n.node_type==GraphNodeType.SYSTEM for n in second.nodes)
    assert any(n.node_type==GraphNodeType.SYSTEM for n in second.archived_nodes)

def test_bounds_evict_oldest_archived_never_active_and_active_overflow_fails():
    first=_project(BusinessConversationSnapshot(facts=(_fact("one","t1"),)))
    second=_project(BusinessConversationSnapshot(facts=(_fact("two","t2"),)),prior=first,limits=GraphLimits(3,3,1))
    third=_project(BusinessConversationSnapshot(facts=(_fact("three","t3"),)),prior=second,limits=GraphLimits(3,3,1))
    assert len(third.archived_nodes)==1 and third.archived_nodes[0].value=="two"
    with pytest.raises(ValueError,match="active graph node limit"):
        _project(BusinessConversationSnapshot(facts=(_fact(),)),limits=GraphLimits(1,3,1))

def test_cycle_rejection_immutability_replay_and_no_authority_leakage():
    edge1=BusinessGraphRelationship("e1","a","b",GraphRelationshipType.PART_OF,GraphSourceComponent.BMME,"t",("e",),GraphEvidenceBasis.OBSERVED)
    edge2=replace(edge1,stable_id="e2",source_node_id="b",target_node_id="a")
    with pytest.raises(ValueError,match="cycle"): _reject_cycles((edge1,edge2))
    graph=_project(BusinessConversationSnapshot(facts=(_fact(),)))
    assert graph==_project(BusinessConversationSnapshot(facts=(_fact(),)))
    with pytest.raises(FrozenInstanceError): graph.scope=_scope("x")
    forbidden={"action","authority","price","execution","recommend"}
    assert not any(any(word in name for word in forbidden) for name in BusinessMentalModelSnapshot.__dataclass_fields__)
