"""Deterministic serialization for complex domain objects.

Rules:
    - explicit schema, no pickle
    - enum values stored as stable string values
    - stable collection ordering preserved
    - unknown values fail safely via explicit raises
"""

from __future__ import annotations

import json
from dataclasses import asdict

from app.leads.contracts import FieldKnowledgeState
from app.persistence.errors import InvalidPersistedPayloadError
from app.precall.contracts import (
    AssumptionGuardrail,
    InitialPosture,
    OpeningObjective,
    PreCallConversationPlan,
    PreCallDiscoveryPolicy,
    PreCallFact,
    PreCallPlanStatus,
    RiskFlag,
    ServiceDiscoveryRule,
    ServicePrerequisiteGap,
    VerificationPriority,
    VerificationTarget,
)
from app.runtime.chat.contracts import TranscriptEntry, TranscriptRole


# ---------------------------------------------------------------------------
# TranscriptEntry tuple ⇄ JSON
# ---------------------------------------------------------------------------


def transcript_to_json(entries: tuple[TranscriptEntry, ...]) -> str:
    return json.dumps(
        [
            {"turn_sequence": e.turn_sequence, "role": e.role.value, "text": e.text}
            for e in entries
        ],
        ensure_ascii=False,
    )


def transcript_from_json(payload: str) -> tuple[TranscriptEntry, ...]:
    try:
        raw = json.loads(payload) if payload else []
    except json.JSONDecodeError as exc:
        raise InvalidPersistedPayloadError(f"transcript JSON invalid: {exc}") from exc
    try:
        return tuple(
            TranscriptEntry(
                turn_sequence=int(e["turn_sequence"]),
                role=TranscriptRole(e["role"]),
                text=e["text"],
            )
            for e in raw
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidPersistedPayloadError(f"transcript entry invalid: {exc}") from exc


# ---------------------------------------------------------------------------
# PreCallDiscoveryPolicy ⇄ JSON
# ---------------------------------------------------------------------------


def policy_to_json(policy: PreCallDiscoveryPolicy) -> str:
    return json.dumps(
        {
            "policy_id": policy.policy_id,
            "version": policy.version,
            "service_rules": [
                {
                    "service_id": r.service_id,
                    "required_fact_keys": list(r.required_fact_keys),
                    "optional_fact_keys": list(r.optional_fact_keys),
                    "knowledge_topic_hints": list(r.knowledge_topic_hints),
                }
                for r in policy.service_rules
            ],
            "global_required_facts": list(policy.global_required_facts),
            "global_optional_facts": list(policy.global_optional_facts),
            "priority_overrides": [
                [key, prio.value] for key, prio in policy.priority_overrides
            ],
        },
        ensure_ascii=False,
    )


def policy_from_json(payload: str) -> PreCallDiscoveryPolicy:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InvalidPersistedPayloadError(f"policy JSON invalid: {exc}") from exc
    try:
        rules = tuple(
            ServiceDiscoveryRule(
                service_id=r["service_id"],
                required_fact_keys=tuple(r.get("required_fact_keys", ())),
                optional_fact_keys=tuple(r.get("optional_fact_keys", ())),
                knowledge_topic_hints=tuple(r.get("knowledge_topic_hints", ())),
            )
            for r in raw.get("service_rules", [])
        )
        overrides = tuple(
            (key, VerificationPriority(value))
            for key, value in raw.get("priority_overrides", [])
        )
        return PreCallDiscoveryPolicy(
            policy_id=raw["policy_id"],
            version=raw["version"],
            service_rules=rules,
            global_required_facts=tuple(raw.get("global_required_facts", ())),
            global_optional_facts=tuple(raw.get("global_optional_facts", ())),
            priority_overrides=overrides,
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidPersistedPayloadError(f"policy payload invalid: {exc}") from exc


# ---------------------------------------------------------------------------
# PreCallConversationPlan ⇄ JSON
# ---------------------------------------------------------------------------


def _fact_to_dict(f: PreCallFact) -> dict:
    return {
        "key": f.key,
        "state": f.state.value,
        "value": f.value,
        "sources": list(f.sources),
        "conflict_detail": f.conflict_detail,
    }


def _fact_from_dict(d: dict) -> PreCallFact:
    return PreCallFact(
        key=d["key"],
        state=FieldKnowledgeState(d["state"]),
        value=d.get("value"),
        sources=tuple(d.get("sources", ())),
        conflict_detail=d.get("conflict_detail"),
    )


def _target_to_dict(t: VerificationTarget) -> dict:
    return {
        "fact_key": t.fact_key,
        "priority": t.priority.value,
        "reason": t.reason,
        "tie_breaker_index": t.tie_breaker_index,
    }


def _target_from_dict(d: dict) -> VerificationTarget:
    return VerificationTarget(
        fact_key=d["fact_key"],
        priority=VerificationPriority(d["priority"]),
        reason=d["reason"],
        tie_breaker_index=int(d["tie_breaker_index"]),
    )


def plan_to_json(plan: PreCallConversationPlan) -> str:
    return json.dumps(
        {
            "lead_id": plan.lead_id,
            "campaign_id": plan.campaign_id,
            "tenant_id": plan.tenant_id,
            "policy_id": plan.policy_id,
            "policy_version": plan.policy_version,
            "known_facts": [_fact_to_dict(f) for f in plan.known_facts],
            "unknown_facts": [_fact_to_dict(f) for f in plan.unknown_facts],
            "unverified_facts": [_fact_to_dict(f) for f in plan.unverified_facts],
            "conflicting_facts": [_fact_to_dict(f) for f in plan.conflicting_facts],
            "verification_targets": [_target_to_dict(t) for t in plan.verification_targets],
            "primary_initial_discovery_target": (
                _target_to_dict(plan.primary_initial_discovery_target)
                if plan.primary_initial_discovery_target is not None else None
            ),
            "opening_objective": plan.opening_objective.value,
            "discovery_priorities": list(plan.discovery_priorities),
            "allowed_service_context": list(plan.allowed_service_context),
            "service_prerequisite_gaps": [
                {
                    "service_id": g.service_id,
                    "missing_fact_keys": list(g.missing_fact_keys),
                }
                for g in plan.service_prerequisite_gaps
            ],
            "assumption_guardrails": sorted(
                g.value for g in plan.assumption_guardrails
            ),
            "knowledge_topics_likely_needed": list(plan.knowledge_topics_likely_needed),
            "risk_flags": sorted(r.value for r in plan.risk_flags),
            "initial_posture": plan.initial_posture.value,
            "plan_status": plan.plan_status.value,
        },
        ensure_ascii=False,
    )


def plan_from_json(payload: str | None) -> PreCallConversationPlan | None:
    if payload is None:
        return None
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InvalidPersistedPayloadError(f"plan JSON invalid: {exc}") from exc
    try:
        prim = raw.get("primary_initial_discovery_target")
        return PreCallConversationPlan(
            lead_id=raw["lead_id"],
            campaign_id=raw["campaign_id"],
            tenant_id=raw["tenant_id"],
            policy_id=raw["policy_id"],
            policy_version=raw["policy_version"],
            known_facts=tuple(_fact_from_dict(f) for f in raw.get("known_facts", [])),
            unknown_facts=tuple(_fact_from_dict(f) for f in raw.get("unknown_facts", [])),
            unverified_facts=tuple(_fact_from_dict(f) for f in raw.get("unverified_facts", [])),
            conflicting_facts=tuple(_fact_from_dict(f) for f in raw.get("conflicting_facts", [])),
            verification_targets=tuple(_target_from_dict(t) for t in raw.get("verification_targets", [])),
            primary_initial_discovery_target=(_target_from_dict(prim) if prim else None),
            opening_objective=OpeningObjective(raw["opening_objective"]),
            discovery_priorities=tuple(raw.get("discovery_priorities", ())),
            allowed_service_context=tuple(raw.get("allowed_service_context", ())),
            service_prerequisite_gaps=tuple(
                ServicePrerequisiteGap(
                    service_id=g["service_id"],
                    missing_fact_keys=tuple(g["missing_fact_keys"]),
                )
                for g in raw.get("service_prerequisite_gaps", [])
            ),
            assumption_guardrails=frozenset(
                AssumptionGuardrail(v) for v in raw.get("assumption_guardrails", [])
            ),
            knowledge_topics_likely_needed=tuple(raw.get("knowledge_topics_likely_needed", ())),
            risk_flags=frozenset(RiskFlag(v) for v in raw.get("risk_flags", [])),
            initial_posture=InitialPosture(raw["initial_posture"]),
            plan_status=PreCallPlanStatus(raw["plan_status"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidPersistedPayloadError(f"plan payload invalid: {exc}") from exc
