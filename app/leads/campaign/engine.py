"""Deterministic Campaign Queue Engine.

Decides which lead is next, whether a call is allowed, and manages queue
state transitions. All decisions are deterministic. No telephony, no LLM,
no state mutation beyond returned decisions.
"""

from __future__ import annotations

from dataclasses import replace

from app.leads.campaign.contracts import (
    CallEligibility,
    CallEligibilityDecision,
    CampaignConfig,
    CampaignStatus,
    LeadQueueEntry,
    LeadQueueStatus,
    NextLeadDecision,
    NoAnswerPolicy,
    QueueStateSnapshot,
    RetryEligibility,
)


class CampaignQueueEngine:
    """Deterministic campaign queue decision engine."""

    def check_campaign_eligibility(
        self,
        config: CampaignConfig,
        snapshot: QueueStateSnapshot,
        current_hour: int,
        current_minute: int,
    ) -> CallEligibilityDecision:
        """Check whether the campaign allows a call right now."""
        if config.status != CampaignStatus.ACTIVE:
            reason = (
                CallEligibility.CAMPAIGN_PAUSED
                if config.status == CampaignStatus.PAUSED
                else CallEligibility.CAMPAIGN_NOT_ACTIVE
            )
            return CallEligibilityDecision(
                eligible=False,
                reason=reason,
                campaign_id=config.campaign_id,
            )

        if snapshot.calls_today >= config.daily_call_limit:
            return CallEligibilityDecision(
                eligible=False,
                reason=CallEligibility.DAILY_LIMIT_REACHED,
                campaign_id=config.campaign_id,
            )

        if snapshot.active_calls >= config.max_concurrent_calls:
            return CallEligibilityDecision(
                eligible=False,
                reason=CallEligibility.CONCURRENCY_LIMIT,
                campaign_id=config.campaign_id,
            )

        if config.calling_window is not None:
            if not config.calling_window.contains_time(current_hour, current_minute):
                return CallEligibilityDecision(
                    eligible=False,
                    reason=CallEligibility.OUTSIDE_CALLING_WINDOW,
                    campaign_id=config.campaign_id,
                )

        return CallEligibilityDecision(
            eligible=True,
            reason=CallEligibility.ELIGIBLE,
            campaign_id=config.campaign_id,
        )

    def check_gap_elapsed(
        self,
        config: CampaignConfig,
        elapsed_since_last_call_seconds: int | None,
    ) -> bool:
        """Check whether the minimum gap between calls has elapsed."""
        if elapsed_since_last_call_seconds is None:
            return True
        return elapsed_since_last_call_seconds >= config.gap_between_calls_seconds

    def select_next_lead(
        self,
        config: CampaignConfig,
        queue: tuple[LeadQueueEntry, ...],
        snapshot: QueueStateSnapshot,
        current_hour: int,
        current_minute: int,
        elapsed_since_last_call_seconds: int | None = None,
    ) -> NextLeadDecision:
        """Deterministically select the next lead to call."""
        campaign_check = self.check_campaign_eligibility(
            config, snapshot, current_hour, current_minute,
        )
        if not campaign_check.eligible:
            return NextLeadDecision(
                available=False,
                ineligibility_reason=campaign_check.reason,
            )

        if not self.check_gap_elapsed(config, elapsed_since_last_call_seconds):
            return NextLeadDecision(
                available=False,
                ineligibility_reason=CallEligibility.GAP_NOT_ELAPSED,
            )

        eligible_entries = sorted(
            [e for e in queue if self._is_lead_callable(e, config)],
            key=lambda e: e.queue_position,
        )

        if not eligible_entries:
            return NextLeadDecision(
                available=False,
                ineligibility_reason=CallEligibility.NO_LEADS_AVAILABLE,
            )

        return NextLeadDecision(
            available=True,
            lead_entry=eligible_entries[0],
        )

    def check_retry_eligibility(
        self,
        entry: LeadQueueEntry,
        config: CampaignConfig,
    ) -> RetryEligibility:
        """Check whether a lead may be retried."""
        if entry.is_dnc:
            return RetryEligibility.DNC
        if entry.status == LeadQueueStatus.COMPLETED:
            return RetryEligibility.COMPLETED
        if entry.attempt_count >= config.max_retries_per_lead:
            return RetryEligibility.MAX_RETRIES_REACHED
        return RetryEligibility.ELIGIBLE

    def apply_no_answer(
        self,
        entry: LeadQueueEntry,
        config: CampaignConfig,
    ) -> LeadQueueEntry:
        """Return updated entry after a no-answer result."""
        new_count = entry.attempt_count + 1

        if entry.is_dnc:
            return replace(entry, status=LeadQueueStatus.DNC, attempt_count=new_count)

        if new_count >= config.max_retries_per_lead:
            return replace(entry, status=LeadQueueStatus.EXHAUSTED, attempt_count=new_count)

        if config.no_answer_policy == NoAnswerPolicy.SKIP:
            return replace(entry, status=LeadQueueStatus.SKIPPED, attempt_count=new_count)

        if config.no_answer_policy == NoAnswerPolicy.MARK_EXHAUSTED:
            return replace(entry, status=LeadQueueStatus.EXHAUSTED, attempt_count=new_count)

        return replace(entry, status=LeadQueueStatus.RETRY_SCHEDULED, attempt_count=new_count)

    def mark_dnc(self, entry: LeadQueueEntry) -> LeadQueueEntry:
        """Mark a lead as DNC. DNC has highest priority."""
        return replace(entry, is_dnc=True, status=LeadQueueStatus.DNC)

    def mark_completed(self, entry: LeadQueueEntry) -> LeadQueueEntry:
        """Mark a lead as completed after successful conversation."""
        return replace(entry, status=LeadQueueStatus.COMPLETED)

    def pause_campaign(self, config: CampaignConfig) -> CampaignConfig:
        """Pause an active campaign."""
        if config.status != CampaignStatus.ACTIVE:
            return config
        return replace(config, status=CampaignStatus.PAUSED)

    def resume_campaign(self, config: CampaignConfig) -> CampaignConfig:
        """Resume a paused campaign."""
        if config.status != CampaignStatus.PAUSED:
            return config
        return replace(config, status=CampaignStatus.ACTIVE)

    def _is_lead_callable(
        self,
        entry: LeadQueueEntry,
        config: CampaignConfig,
    ) -> bool:
        """Check if a lead entry is callable."""
        if entry.is_dnc:
            return False
        if entry.status in {
            LeadQueueStatus.COMPLETED,
            LeadQueueStatus.DNC,
            LeadQueueStatus.EXHAUSTED,
            LeadQueueStatus.SKIPPED,
            LeadQueueStatus.IN_PROGRESS,
        }:
            return False
        if entry.attempt_count >= config.max_retries_per_lead:
            return False
        return True
