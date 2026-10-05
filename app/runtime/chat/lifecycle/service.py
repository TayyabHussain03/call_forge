"""Deterministic sales session lifecycle orchestration.

Connects bootstrap → chat runtime → multi-turn conversation → outcome.
No new intelligence. No LLM. No retrieval. No response generation.

Orchestrate, do not re-reason.
"""

from __future__ import annotations

from app.runtime.chat.bootstrap.contracts import BootstrapStatus, SessionBootstrapInput, SessionBootstrapResult
from app.runtime.chat.bootstrap.service import SessionBootstrapService
from app.runtime.chat.contracts import ChatTurnInput, SessionStatus
from app.runtime.chat.engine import ChatRuntime, DuplicateTurnError, SessionTerminatedError
from app.runtime.chat.lifecycle.contracts import (
    LifecycleDiagnostics,
    LifecycleResult,
    LifecycleStatus,
)
from app.runtime.chat.outcome.contracts import (
    ExecutionStatus,
    LeadOutcome,
    OutcomeIntelligenceInput,
    TerminationReason,
)
from app.runtime.chat.outcome.engine import OutcomeIntelligenceEngine
from app.runtime.contracts import TurnProcessor
from app.leads.campaign.contracts import CallEligibility, CampaignConfig, LeadQueueEntry


class SalesSessionLifecycleService:
    """Thin orchestrator: bootstrap → turns → outcome.

    Does not contain conversation logic, state-machine transitions,
    or business decisions. Those live in the existing pipeline layers.
    """

    def __init__(
        self,
        bootstrap_service: SessionBootstrapService,
        chat_runtime: ChatRuntime,
        outcome_engine: OutcomeIntelligenceEngine | None = None,
    ) -> None:
        self._bootstrap = bootstrap_service
        self._runtime = chat_runtime
        self._outcome = outcome_engine or OutcomeIntelligenceEngine()

    def bootstrap_session(
        self,
        bootstrap_input: SessionBootstrapInput,
        campaign: CampaignConfig,
        queue_entry: LeadQueueEntry,
        processor: TurnProcessor,
        eligibility_reason: CallEligibility = CallEligibility.ELIGIBLE,
        queue_transitioner=None,
    ) -> LifecycleResult:
        """Bootstrap a session and register its processor."""
        result = self._bootstrap.bootstrap(
            bootstrap_input, campaign, queue_entry,
            eligibility_reason=eligibility_reason,
            queue_transitioner=queue_transitioner,
        )

        if result.status != BootstrapStatus.CREATED:
            return LifecycleResult(
                status=LifecycleStatus.BOOTSTRAP_REJECTED,
                diagnostics=LifecycleDiagnostics(
                    bootstrap_status=result.status.value,
                    failure_detail=result.rejection_reason,
                ),
            )

        self._runtime.register_processor(result.session.session_id, processor)

        return LifecycleResult(
            status=LifecycleStatus.ACTIVE,
            session=result.session,
            diagnostics=LifecycleDiagnostics(
                session_id=result.session.session_id,
                bootstrap_status=result.status.value,
            ),
        )

    def process_turn(
        self,
        session_id: str,
        message: str,
        turn_sequence: int,
    ) -> LifecycleResult:
        """Process one prospect turn through the existing pipeline."""
        turn_input = ChatTurnInput(
            session_id=session_id,
            message=message,
            turn_sequence=turn_sequence,
        )

        try:
            turn_result = self._runtime.process_turn(turn_input)
        except DuplicateTurnError:
            return LifecycleResult(
                status=LifecycleStatus.TURN_REJECTED,
                diagnostics=LifecycleDiagnostics(
                    session_id=session_id,
                    failure_detail=f"duplicate turn {turn_sequence}",
                ),
            )
        except SessionTerminatedError:
            session = self._runtime.get_session(session_id)
            return LifecycleResult(
                status=LifecycleStatus.TERMINATED,
                session=session,
                diagnostics=LifecycleDiagnostics(
                    session_id=session_id,
                    failure_detail="session already terminated",
                ),
            )
        except Exception as exc:
            return LifecycleResult(
                status=LifecycleStatus.RUNTIME_FAILURE,
                execution_status=ExecutionStatus.FAILED,
                diagnostics=LifecycleDiagnostics(
                    session_id=session_id,
                    failure_detail=str(exc)[:200],
                ),
            )

        session = self._runtime.get_session(session_id)

        if turn_result.conversation_terminal:
            return LifecycleResult(
                status=LifecycleStatus.COMPLETED,
                session=session,
                latest_turn_result=turn_result,
                lead_outcome=self._generate_outcome(session, turn_result),
                execution_status=ExecutionStatus.COMPLETED,
                diagnostics=LifecycleDiagnostics(
                    session_id=session_id,
                    turns_processed=session.turn_count,
                    terminal_turn=turn_sequence,
                ),
            )

        return LifecycleResult(
            status=LifecycleStatus.ACTIVE,
            session=session,
            latest_turn_result=turn_result,
            diagnostics=LifecycleDiagnostics(
                session_id=session_id,
                turns_processed=session.turn_count,
            ),
        )

    def generate_outcome(
        self,
        session_id: str,
        *,
        is_dnc: bool = False,
        is_not_interested: bool = False,
        session_error: bool = False,
        termination_reason: TerminationReason = TerminationReason.NATURAL_COMPLETION,
        prospect_observed: object | None = None,
        prospect_inferred: object | None = None,
        business_conversation: object | None = None,
        playbook_guidance: object | None = None,
        qualification: object | None = None,
    ) -> LeadOutcome:
        """Generate outcome from trusted pipeline state after session ends."""
        session = self._runtime.get_session(session_id)

        inp = OutcomeIntelligenceInput(
            lead_id=session.lead_id,
            campaign_id=session.campaign_id,
            session_id=session.session_id,
            tenant_id=session.tenant_id,
            is_dnc=is_dnc,
            is_not_interested=is_not_interested,
            session_error=session_error,
            termination_reason=termination_reason,
            conversation_terminal=True,
            prospect_observed=prospect_observed,
            prospect_inferred=prospect_inferred,
            business_conversation=business_conversation,
            playbook_guidance=playbook_guidance,
            qualification=qualification,
        )
        return self._outcome.resolve(inp)

    def _generate_outcome(self, session, turn_result) -> LeadOutcome:
        """Generate minimal outcome from terminal turn result."""
        return self.generate_outcome(
            session.session_id,
            termination_reason=TerminationReason.NATURAL_COMPLETION,
        )
