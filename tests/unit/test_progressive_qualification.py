from app.conversation.qualification.contracts import *
from app.conversation.qualification.engine import ProgressiveQualificationEngine

def _config(campaign="a"):
    return QualificationConfiguration("tenant","business",campaign,(QualificationField("authority","Authority",True,QualificationEvidenceKind.OBSERVED),QualificationField("timeline","Timeline",True,QualificationEvidenceKind.OBSERVED),QualificationField("budget","Budget",False,QualificationEvidenceKind.OBSERVED)))
def _obs(field,value,kind=QualificationEvidenceKind.OBSERVED): return QualificationObservation(field,value,kind,("e1",),"t1")
def test_progressive_generic_updates_and_completeness():
    e=ProgressiveQualificationEngine(); first=e.update(_config(),QualificationEvidence((_obs("authority","owner"),)))
    assert first.completeness==QualificationCompleteness.MOSTLY_KNOWN
    second=e.update(_config(),QualificationEvidence((_obs("timeline","this_month"),)),first)
    assert second.completeness==QualificationCompleteness.SUFFICIENT
    assert {x.field_id for x in second.fields if x.state==QualificationValueState.UNKNOWN}=={"budget"}
def test_observed_beats_inferred_and_correction_archives():
    e=ProgressiveQualificationEngine(); prior=e.update(_config(),QualificationEvidence((_obs("budget","low",QualificationEvidenceKind.INFERRED),)))
    updated=e.update(_config(),QualificationEvidence((_obs("budget","approved"),),frozenset({"budget"})),prior)
    assert next(x for x in updated.fields if x.field_id=="budget").state==QualificationValueState.OBSERVED
    assert updated.corrected==frozenset({"budget"}) and updated.archived
def test_campaign_isolation_replay_and_no_authority_or_questions():
    e=ProgressiveQualificationEngine(); a=e.update(_config(),QualificationEvidence((_obs("authority","owner"),)))
    b=e.update(_config("b"),QualificationEvidence(),a)
    assert all(x.state==QualificationValueState.UNKNOWN for x in b.fields)
    assert e.update(_config(),QualificationEvidence((_obs("authority","owner"),)))==a
    assert not any(word in name for name in QualificationSnapshot.__dataclass_fields__ for word in ("service","authority","question","action","execution"))
