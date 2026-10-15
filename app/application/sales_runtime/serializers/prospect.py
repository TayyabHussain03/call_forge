"""ProspectIntelligenceSnapshot serializer.

Handles ObservedValue[T] / InferredValue[T] envelopes with evidence
provenance. CurrentSolutionContext and ProspectPain carry fuller
provenance that is also round-tripped.

Observed remains observed. Inferred remains inferred. Unknown remains
unknown.
"""

from __future__ import annotations

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    _coerce_enum,
    build_envelope,
    parse_envelope,
)
from app.conversation.prospect_intelligence.contracts import (
    BuyingStage,
    CurrentSolutionContext,
    DecisionAuthority,
    EvidenceProvenance,
    EvidenceSourceKind,
    InferredProspectState,
    InferredValue,
    InformationLevel,
    ObjectionType,
    ObservedProspectFacts,
    ObservedValue,
    PainCategory,
    PreferredNextStep,
    ProspectIntelligenceSnapshot,
    ProspectPain,
    ProspectRole,
    SolutionSatisfaction,
)
from app.persistence.errors import InvalidPersistedPayloadError


PROSPECT_INTELLIGENCE_KEY = "prospect_intelligence"
PROSPECT_INTELLIGENCE_VERSION = 1


class ProspectIntelligenceSerializer:
    @property
    def key(self) -> str:
        return PROSPECT_INTELLIGENCE_KEY

    @property
    def version(self) -> int:
        return PROSPECT_INTELLIGENCE_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.OPTIONAL_DERIVED

    def serialize(self, snap: ProspectIntelligenceSnapshot) -> str:
        return build_envelope(self.key, self.version, {
            "observed": self._observed_to_dict(snap.observed),
            "inferred": self._inferred_to_dict(snap.inferred),
        })

    def deserialize(self, envelope_json: str) -> ProspectIntelligenceSnapshot:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        try:
            return ProspectIntelligenceSnapshot(
                observed=self._observed_from_dict(payload.get("observed", {})),
                inferred=self._inferred_from_dict(payload.get("inferred", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPersistedPayloadError(
                f"ProspectIntelligence payload invalid: {exc}"
            ) from exc

    # --- Observed ---

    def _observed_to_dict(self, obs: ObservedProspectFacts) -> dict:
        return {
            "explicit_role": self._observed_value_to_dict(
                obs.explicit_role, _enum_value,
            ),
            "explicit_current_solution": (
                self._solution_to_dict(obs.explicit_current_solution)
                if obs.explicit_current_solution else None
            ),
            "explicit_pain_points": [
                self._pain_to_dict(p) for p in obs.explicit_pain_points
            ],
            "explicit_interest_signal": self._observed_value_to_dict(
                obs.explicit_interest_signal, lambda v: v,
            ),
            "explicit_busy_signal": self._observed_value_to_dict(
                obs.explicit_busy_signal, lambda v: v,
            ),
            "explicit_human_request": self._observed_value_to_dict(
                obs.explicit_human_request, lambda v: v,
            ),
            "explicit_next_step_request": self._observed_value_to_dict(
                obs.explicit_next_step_request, _enum_value,
            ),
            "explicit_decision_authority_statement": self._observed_value_to_dict(
                obs.explicit_decision_authority_statement, _enum_value,
            ),
            "explicit_timing_preference": self._observed_value_to_dict(
                obs.explicit_timing_preference, lambda v: v,
            ),
            "explicit_objection": self._observed_value_to_dict(
                obs.explicit_objection, _enum_value,
            ),
        }

    def _observed_from_dict(self, d: dict) -> ObservedProspectFacts:
        return ObservedProspectFacts(
            explicit_role=self._observed_value_from_dict(
                d.get("explicit_role"), lambda v: _coerce_enum(ProspectRole, v, "role"),
            ),
            explicit_current_solution=(
                self._solution_from_dict(d["explicit_current_solution"])
                if d.get("explicit_current_solution") else None
            ),
            explicit_pain_points=tuple(
                self._pain_from_dict(p) for p in d.get("explicit_pain_points", [])
            ),
            explicit_interest_signal=self._observed_value_from_dict(
                d.get("explicit_interest_signal"), bool,
            ),
            explicit_busy_signal=self._observed_value_from_dict(
                d.get("explicit_busy_signal"), bool,
            ),
            explicit_human_request=self._observed_value_from_dict(
                d.get("explicit_human_request"), bool,
            ),
            explicit_next_step_request=self._observed_value_from_dict(
                d.get("explicit_next_step_request"),
                lambda v: _coerce_enum(PreferredNextStep, v, "next_step"),
            ),
            explicit_decision_authority_statement=self._observed_value_from_dict(
                d.get("explicit_decision_authority_statement"),
                lambda v: _coerce_enum(DecisionAuthority, v, "decision_authority"),
            ),
            explicit_timing_preference=self._observed_value_from_dict(
                d.get("explicit_timing_preference"), str,
            ),
            explicit_objection=self._observed_value_from_dict(
                d.get("explicit_objection"),
                lambda v: _coerce_enum(ObjectionType, v, "objection"),
            ),
        )

    # --- Inferred ---

    def _inferred_to_dict(self, inf: InferredProspectState) -> dict:
        return {
            "likely_role": self._inferred_value_to_dict(inf.likely_role, _enum_value),
            "decision_authority": self._inferred_value_to_dict(
                inf.decision_authority, _enum_value,
            ),
            "influence_level": self._inferred_value_to_dict(
                inf.influence_level, _enum_value,
            ),
            "buying_stage": self._inferred_value_to_dict(inf.buying_stage, _enum_value),
            "openness": self._inferred_value_to_dict(inf.openness, _enum_value),
            "urgency": self._inferred_value_to_dict(inf.urgency, _enum_value),
            "objection_type": self._inferred_value_to_dict(
                inf.objection_type, _enum_value,
            ),
            "pain_summary": self._inferred_value_to_dict(inf.pain_summary, lambda v: v),
            "current_solution_satisfaction": self._inferred_value_to_dict(
                inf.current_solution_satisfaction, _enum_value,
            ),
            "preferred_next_step": self._inferred_value_to_dict(
                inf.preferred_next_step, _enum_value,
            ),
        }

    def _inferred_from_dict(self, d: dict) -> InferredProspectState:
        return InferredProspectState(
            likely_role=self._inferred_value_from_dict(
                d.get("likely_role"), lambda v: _coerce_enum(ProspectRole, v, "role"),
            ),
            decision_authority=self._inferred_value_from_dict(
                d.get("decision_authority"),
                lambda v: _coerce_enum(DecisionAuthority, v, "decision_authority"),
            ),
            influence_level=self._inferred_value_from_dict(
                d.get("influence_level"),
                lambda v: _coerce_enum(InformationLevel, v, "influence_level"),
            ),
            buying_stage=self._inferred_value_from_dict(
                d.get("buying_stage"),
                lambda v: _coerce_enum(BuyingStage, v, "buying_stage"),
            ),
            openness=self._inferred_value_from_dict(
                d.get("openness"),
                lambda v: _coerce_enum(InformationLevel, v, "openness"),
            ),
            urgency=self._inferred_value_from_dict(
                d.get("urgency"),
                lambda v: _coerce_enum(InformationLevel, v, "urgency"),
            ),
            objection_type=self._inferred_value_from_dict(
                d.get("objection_type"),
                lambda v: _coerce_enum(ObjectionType, v, "objection"),
            ),
            pain_summary=self._inferred_value_from_dict(
                d.get("pain_summary"), str,
            ),
            current_solution_satisfaction=self._inferred_value_from_dict(
                d.get("current_solution_satisfaction"),
                lambda v: _coerce_enum(
                    SolutionSatisfaction, v, "solution_satisfaction",
                ),
            ),
            preferred_next_step=self._inferred_value_from_dict(
                d.get("preferred_next_step"),
                lambda v: _coerce_enum(PreferredNextStep, v, "next_step"),
            ),
        )

    # --- Envelope helpers ---

    @staticmethod
    def _provenance_to_dict(p: EvidenceProvenance) -> dict:
        return {
            "source_turn_id": p.source_turn_id,
            "source_kind": p.source_kind.value,
        }

    @staticmethod
    def _provenance_from_dict(d: dict) -> EvidenceProvenance:
        return EvidenceProvenance(
            source_turn_id=d["source_turn_id"],
            source_kind=_coerce_enum(
                EvidenceSourceKind, d["source_kind"], "provenance.source_kind",
            ),
        )

    def _observed_value_to_dict(self, v, encode):
        if v is None:
            return None
        return {"value": encode(v.value), "provenance": self._provenance_to_dict(v.provenance)}

    def _observed_value_from_dict(self, d, decode):
        if d is None:
            return None
        return ObservedValue(
            value=decode(d["value"]),
            provenance=self._provenance_from_dict(d["provenance"]),
        )

    def _inferred_value_to_dict(self, v, encode):
        if v is None:
            return None
        return {
            "value": encode(v.value),
            "confidence": v.confidence,
            "provenance": self._provenance_to_dict(v.provenance),
        }

    def _inferred_value_from_dict(self, d, decode):
        if d is None:
            return None
        return InferredValue(
            value=decode(d["value"]),
            confidence=float(d["confidence"]),
            provenance=self._provenance_from_dict(d["provenance"]),
        )

    def _solution_to_dict(self, s: CurrentSolutionContext) -> dict:
        return {
            "name": s.name,
            "category": s.category,
            "satisfaction": s.satisfaction.value,
            "provenance": self._provenance_to_dict(s.provenance),
        }

    def _solution_from_dict(self, d: dict) -> CurrentSolutionContext:
        return CurrentSolutionContext(
            name=d["name"],
            category=d.get("category"),
            satisfaction=_coerce_enum(
                SolutionSatisfaction, d["satisfaction"], "solution.satisfaction",
            ),
            provenance=self._provenance_from_dict(d["provenance"]),
        )

    def _pain_to_dict(self, p: ProspectPain) -> dict:
        return {
            "category": p.category.value,
            "summary": p.summary,
            "provenance": self._provenance_to_dict(p.provenance),
        }

    def _pain_from_dict(self, d: dict) -> ProspectPain:
        return ProspectPain(
            category=_coerce_enum(PainCategory, d["category"], "pain.category"),
            summary=d["summary"],
            provenance=self._provenance_from_dict(d["provenance"]),
        )


def _enum_value(e):
    return e.value if hasattr(e, "value") else e
