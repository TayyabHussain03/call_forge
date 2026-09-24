"""Pure campaign-isolated progressive qualification reconciliation."""
from __future__ import annotations
from app.conversation.qualification.contracts import *

class ProgressiveQualificationEngine:
    def update(self, configuration: QualificationConfiguration, evidence: QualificationEvidence, prior: QualificationSnapshot | None = None) -> QualificationSnapshot:
        identity=(configuration.tenant_id, configuration.business_id, configuration.campaign_id)
        if prior is not None and prior.configuration_id != identity: prior=None
        prior_values={f.field_id:f for f in prior.fields} if prior else {}
        incoming={o.field_id:o for o in evidence.observations if o.field_id in {f.field_id for f in configuration.fields}}
        archived=list(prior.archived) if prior else []; corrected=set(); newly=set(); replaced=set(); states=[]
        for field in configuration.fields:
            old=prior_values.get(field.field_id); obs=incoming.get(field.field_id)
            if obs and (obs.evidence_kind == QualificationEvidenceKind.OBSERVED or old is None or old.state != QualificationValueState.OBSERVED):
                if old and old.value is not None and old.value != obs.value:
                    archived.append(QualificationObservation(field.field_id, old.value, QualificationEvidenceKind.OBSERVED if old.state == QualificationValueState.OBSERVED else QualificationEvidenceKind.INFERRED, old.evidence_ids, old.source_turn_id or "archived")); replaced.add(field.field_id)
                state=QualificationValueState.OBSERVED if obs.evidence_kind == QualificationEvidenceKind.OBSERVED else QualificationValueState.INFERRED
                states.append(QualificationFieldState(field.field_id,state,obs.value,obs.evidence_ids,obs.source_turn_id)); newly.add(field.field_id)
                if field.field_id in evidence.corrections: corrected.add(field.field_id)
            elif old: states.append(old)
            else: states.append(QualificationFieldState(field.field_id, QualificationValueState.UNKNOWN))
        known=sum(s.state != QualificationValueState.UNKNOWN for s in states); mandatory=sum(f.mandatory for f in configuration.fields); known_mandatory=sum(s.state != QualificationValueState.UNKNOWN and next(f for f in configuration.fields if f.field_id==s.field_id).mandatory for s in states)
        completeness=QualificationCompleteness.SUFFICIENT if mandatory and known_mandatory==mandatory else QualificationCompleteness.MOSTLY_KNOWN if known else QualificationCompleteness.VERY_EARLY
        progress=QualificationProgress.CORRECTED if corrected else QualificationProgress.COMPLETE if completeness==QualificationCompleteness.SUFFICIENT else QualificationProgress.ADVANCING if newly else QualificationProgress.STABLE
        return QualificationSnapshot(identity,tuple(states),completeness,progress,frozenset(newly),frozenset(corrected),frozenset(replaced),tuple(archived)[-30:])
