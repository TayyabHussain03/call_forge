"""Comprehensive tests for Campaign & Lead Queue Runtime."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from app.leads.campaign.contracts import (
    CallEligibility,
    CallEligibilityDecision,
    CallingWindow,
    CampaignConfig,
    CampaignStatus,
    LeadQueueEntry,
    LeadQueueStatus,
    NextLeadDecision,
    NoAnswerPolicy,
    QueueStateSnapshot,
    RetryEligibility,
)
from app.leads.campaign.engine import CampaignQueueEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENGINE = CampaignQueueEngine()


def _config(**changes) -> CampaignConfig:
    values = dict(
        campaign_id="campaign_1",
        tenant_id="tenant_1",
        name="Test Campaign",
        status=CampaignStatus.ACTIVE,
        daily_call_limit=50,
        gap_between_calls_seconds=30,
        max_concurrent_calls=1,
        max_retries_per_lead=3,
    )
    values.update(changes)
    return CampaignConfig(**values)


def _entry(
    lead_id: str = "lead_1",
    position: int = 0,
    status: LeadQueueStatus = LeadQueueStatus.READY,
    attempts: int = 0,
    is_dnc: bool = False,
) -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id=lead_id,
        campaign_id="campaign_1",
        tenant_id="tenant_1",
        queue_position=position,
        status=status,
        attempt_count=attempts,
        is_dnc=is_dnc,
    )


def _snapshot(**changes) -> QueueStateSnapshot:
    values = dict(
        campaign_id="campaign_1",
        tenant_id="tenant_1",
        total_leads=10,
        pending=5,
        ready=3,
        in_progress=0,
        completed=1,
        failed=0,
        dnc=1,
        exhausted=0,
        calls_today=0,
        active_calls=0,
    )
    values.update(changes)
    return QueueStateSnapshot(**values)


# ===========================================================================
# Campaign config validation
# ===========================================================================


class TestCampaignConfig:
    def test_valid_config(self):
        c = _config()
        assert c.campaign_id == "campaign_1"

    def test_empty_campaign_id_rejected(self):
        with pytest.raises(ValueError, match="campaign_id"):
            _config(campaign_id="")

    def test_negative_daily_limit_rejected(self):
        with pytest.raises(ValueError, match="daily call limit"):
            _config(daily_call_limit=0)

    def test_frozen(self):
        c = _config()
        with pytest.raises(FrozenInstanceError):
            c.name = "Changed"

    def test_authorized_services(self):
        c = _config(authorized_service_ids=("website", "crm"))
        assert len(c.authorized_service_ids) == 2


# ===========================================================================
# Calling window
# ===========================================================================


class TestCallingWindow:
    def test_within_window(self):
        w = CallingWindow(start_hour=9, start_minute=0, end_hour=17, end_minute=0, timezone="US/Eastern")
        assert w.contains_time(10, 30) is True

    def test_outside_window(self):
        w = CallingWindow(start_hour=9, start_minute=0, end_hour=17, end_minute=0, timezone="US/Eastern")
        assert w.contains_time(18, 0) is False

    def test_boundary_start(self):
        w = CallingWindow(start_hour=9, start_minute=0, end_hour=17, end_minute=0, timezone="US/Eastern")
        assert w.contains_time(9, 0) is True

    def test_boundary_end(self):
        w = CallingWindow(start_hour=9, start_minute=0, end_hour=17, end_minute=0, timezone="US/Eastern")
        assert w.contains_time(17, 0) is False

    def test_invalid_hour_rejected(self):
        with pytest.raises(ValueError, match="start hour"):
            CallingWindow(start_hour=25, start_minute=0, end_hour=17, end_minute=0, timezone="US/Eastern")


# ===========================================================================
# Campaign eligibility
# ===========================================================================


class TestCampaignEligibility:
    def test_active_campaign_eligible(self):
        result = ENGINE.check_campaign_eligibility(_config(), _snapshot(), 10, 0)
        assert result.eligible is True
        assert result.reason == CallEligibility.ELIGIBLE

    def test_paused_campaign_not_eligible(self):
        result = ENGINE.check_campaign_eligibility(
            _config(status=CampaignStatus.PAUSED), _snapshot(), 10, 0,
        )
        assert result.eligible is False
        assert result.reason == CallEligibility.CAMPAIGN_PAUSED

    def test_draft_campaign_not_eligible(self):
        result = ENGINE.check_campaign_eligibility(
            _config(status=CampaignStatus.DRAFT), _snapshot(), 10, 0,
        )
        assert result.eligible is False
        assert result.reason == CallEligibility.CAMPAIGN_NOT_ACTIVE

    def test_daily_limit_reached(self):
        result = ENGINE.check_campaign_eligibility(
            _config(daily_call_limit=10), _snapshot(calls_today=10), 10, 0,
        )
        assert result.eligible is False
        assert result.reason == CallEligibility.DAILY_LIMIT_REACHED

    def test_concurrency_limit(self):
        result = ENGINE.check_campaign_eligibility(
            _config(max_concurrent_calls=1), _snapshot(active_calls=1), 10, 0,
        )
        assert result.eligible is False
        assert result.reason == CallEligibility.CONCURRENCY_LIMIT

    def test_outside_calling_window(self):
        window = CallingWindow(start_hour=9, start_minute=0, end_hour=17, end_minute=0, timezone="US/Eastern")
        result = ENGINE.check_campaign_eligibility(
            _config(calling_window=window), _snapshot(), 20, 0,
        )
        assert result.eligible is False
        assert result.reason == CallEligibility.OUTSIDE_CALLING_WINDOW

    def test_inside_calling_window(self):
        window = CallingWindow(start_hour=9, start_minute=0, end_hour=17, end_minute=0, timezone="US/Eastern")
        result = ENGINE.check_campaign_eligibility(
            _config(calling_window=window), _snapshot(), 10, 0,
        )
        assert result.eligible is True


# ===========================================================================
# Gap between calls
# ===========================================================================


class TestGapBetweenCalls:
    def test_gap_elapsed(self):
        assert ENGINE.check_gap_elapsed(_config(gap_between_calls_seconds=30), 31) is True

    def test_gap_not_elapsed(self):
        assert ENGINE.check_gap_elapsed(_config(gap_between_calls_seconds=30), 10) is False

    def test_no_previous_call(self):
        assert ENGINE.check_gap_elapsed(_config(), None) is True

    def test_exact_gap(self):
        assert ENGINE.check_gap_elapsed(_config(gap_between_calls_seconds=30), 30) is True


# ===========================================================================
# Next lead selection
# ===========================================================================


class TestNextLeadSelection:
    def test_selects_lowest_position(self):
        queue = (
            _entry(lead_id="lead_2", position=2),
            _entry(lead_id="lead_1", position=1),
            _entry(lead_id="lead_3", position=3),
        )
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.available is True
        assert result.lead_entry.lead_id == "lead_1"

    def test_skips_dnc_leads(self):
        queue = (
            _entry(lead_id="lead_1", position=1, is_dnc=True),
            _entry(lead_id="lead_2", position=2),
        )
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.available is True
        assert result.lead_entry.lead_id == "lead_2"

    def test_skips_completed_leads(self):
        queue = (
            _entry(lead_id="lead_1", position=1, status=LeadQueueStatus.COMPLETED),
            _entry(lead_id="lead_2", position=2),
        )
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.available is True
        assert result.lead_entry.lead_id == "lead_2"

    def test_skips_exhausted_leads(self):
        queue = (
            _entry(lead_id="lead_1", position=1, status=LeadQueueStatus.EXHAUSTED),
            _entry(lead_id="lead_2", position=2),
        )
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.lead_entry.lead_id == "lead_2"

    def test_skips_in_progress_leads(self):
        queue = (
            _entry(lead_id="lead_1", position=1, status=LeadQueueStatus.IN_PROGRESS),
            _entry(lead_id="lead_2", position=2),
        )
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.lead_entry.lead_id == "lead_2"

    def test_no_leads_available(self):
        queue = (
            _entry(lead_id="lead_1", position=1, status=LeadQueueStatus.COMPLETED),
        )
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.available is False
        assert result.ineligibility_reason == CallEligibility.NO_LEADS_AVAILABLE

    def test_campaign_not_active_blocks_selection(self):
        result = ENGINE.select_next_lead(
            _config(status=CampaignStatus.PAUSED),
            (_entry(),), _snapshot(), 10, 0,
        )
        assert result.available is False
        assert result.ineligibility_reason == CallEligibility.CAMPAIGN_PAUSED

    def test_gap_not_elapsed_blocks_selection(self):
        result = ENGINE.select_next_lead(
            _config(gap_between_calls_seconds=60),
            (_entry(),), _snapshot(), 10, 0,
            elapsed_since_last_call_seconds=10,
        )
        assert result.available is False
        assert result.ineligibility_reason == CallEligibility.GAP_NOT_ELAPSED

    def test_max_retries_exhausted_skips_lead(self):
        queue = (
            _entry(lead_id="lead_1", position=1, attempts=3),
            _entry(lead_id="lead_2", position=2),
        )
        result = ENGINE.select_next_lead(
            _config(max_retries_per_lead=3), queue, _snapshot(), 10, 0,
        )
        assert result.lead_entry.lead_id == "lead_2"

    def test_retry_scheduled_is_callable(self):
        queue = (_entry(lead_id="lead_1", position=1, status=LeadQueueStatus.RETRY_SCHEDULED, attempts=1),)
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.available is True
        assert result.lead_entry.lead_id == "lead_1"


# ===========================================================================
# Retry eligibility
# ===========================================================================


class TestRetryEligibility:
    def test_eligible_under_limit(self):
        result = ENGINE.check_retry_eligibility(_entry(attempts=1), _config(max_retries_per_lead=3))
        assert result == RetryEligibility.ELIGIBLE

    def test_max_retries_reached(self):
        result = ENGINE.check_retry_eligibility(_entry(attempts=3), _config(max_retries_per_lead=3))
        assert result == RetryEligibility.MAX_RETRIES_REACHED

    def test_dnc_not_retryable(self):
        result = ENGINE.check_retry_eligibility(_entry(is_dnc=True), _config())
        assert result == RetryEligibility.DNC

    def test_completed_not_retryable(self):
        result = ENGINE.check_retry_eligibility(
            _entry(status=LeadQueueStatus.COMPLETED), _config(),
        )
        assert result == RetryEligibility.COMPLETED


# ===========================================================================
# No-answer handling
# ===========================================================================


class TestNoAnswerHandling:
    def test_retry_later_policy(self):
        updated = ENGINE.apply_no_answer(_entry(attempts=0), _config(no_answer_policy=NoAnswerPolicy.RETRY_LATER))
        assert updated.status == LeadQueueStatus.RETRY_SCHEDULED
        assert updated.attempt_count == 1

    def test_skip_policy(self):
        updated = ENGINE.apply_no_answer(_entry(), _config(no_answer_policy=NoAnswerPolicy.SKIP))
        assert updated.status == LeadQueueStatus.SKIPPED

    def test_mark_exhausted_policy(self):
        updated = ENGINE.apply_no_answer(_entry(), _config(no_answer_policy=NoAnswerPolicy.MARK_EXHAUSTED))
        assert updated.status == LeadQueueStatus.EXHAUSTED

    def test_max_retries_marks_exhausted(self):
        updated = ENGINE.apply_no_answer(
            _entry(attempts=2), _config(max_retries_per_lead=3),
        )
        assert updated.status == LeadQueueStatus.EXHAUSTED
        assert updated.attempt_count == 3

    def test_dnc_preserved_on_no_answer(self):
        updated = ENGINE.apply_no_answer(_entry(is_dnc=True), _config())
        assert updated.status == LeadQueueStatus.DNC


# ===========================================================================
# DNC handling
# ===========================================================================


class TestDNCHandling:
    def test_mark_dnc(self):
        entry = _entry()
        updated = ENGINE.mark_dnc(entry)
        assert updated.is_dnc is True
        assert updated.status == LeadQueueStatus.DNC

    def test_dnc_highest_priority(self):
        entry = _entry(status=LeadQueueStatus.READY)
        updated = ENGINE.mark_dnc(entry)
        assert updated.status == LeadQueueStatus.DNC

    def test_dnc_never_selected(self):
        queue = (_entry(lead_id="lead_1", position=1, is_dnc=True),)
        result = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert result.available is False


# ===========================================================================
# Pause / resume
# ===========================================================================


class TestPauseResume:
    def test_pause_active_campaign(self):
        c = _config(status=CampaignStatus.ACTIVE)
        paused = ENGINE.pause_campaign(c)
        assert paused.status == CampaignStatus.PAUSED

    def test_resume_paused_campaign(self):
        c = _config(status=CampaignStatus.PAUSED)
        resumed = ENGINE.resume_campaign(c)
        assert resumed.status == CampaignStatus.ACTIVE

    def test_pause_non_active_is_noop(self):
        c = _config(status=CampaignStatus.DRAFT)
        result = ENGINE.pause_campaign(c)
        assert result.status == CampaignStatus.DRAFT

    def test_resume_non_paused_is_noop(self):
        c = _config(status=CampaignStatus.ACTIVE)
        result = ENGINE.resume_campaign(c)
        assert result.status == CampaignStatus.ACTIVE


# ===========================================================================
# Tenant / campaign isolation
# ===========================================================================


class TestIsolation:
    def test_entry_carries_tenant(self):
        e = _entry()
        assert e.tenant_id == "tenant_1"

    def test_entry_carries_campaign(self):
        e = _entry()
        assert e.campaign_id == "campaign_1"


# ===========================================================================
# Deterministic selection
# ===========================================================================


class TestDeterminism:
    def test_same_queue_same_result(self):
        queue = (
            _entry(lead_id="lead_2", position=2),
            _entry(lead_id="lead_1", position=1),
        )
        r1 = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        r2 = ENGINE.select_next_lead(_config(), queue, _snapshot(), 10, 0)
        assert r1 == r2
        assert r1.lead_entry.lead_id == "lead_1"


# ===========================================================================
# Queue entry DNC auto-correction
# ===========================================================================


class TestDNCAutoCorrection:
    def test_dnc_flag_overrides_status(self):
        e = LeadQueueEntry(
            lead_id="lead_1",
            campaign_id="c1",
            tenant_id="t1",
            queue_position=0,
            status=LeadQueueStatus.READY,
            is_dnc=True,
        )
        assert e.status == LeadQueueStatus.DNC


# ===========================================================================
# No telephony
# ===========================================================================


class TestNoTelephony:
    def test_no_telephony_imports(self):
        from pathlib import Path
        import app.leads.campaign.engine as m
        source = Path(m.__file__).read_text()
        assert "twilio" not in source.lower()
        assert "telnyx" not in source.lower()
        assert "vapi" not in source.lower()
        assert "sip" not in source.lower()

    def test_no_call_placement(self):
        from pathlib import Path
        import app.leads.campaign.engine as m
        source = Path(m.__file__).read_text()
        assert "place_call" not in source
        assert "dial" not in source
        assert "connect_call" not in source


# ===========================================================================
# Crash safety — state is immutable
# ===========================================================================


class TestCrashSafety:
    def test_queue_entry_immutable(self):
        e = _entry()
        with pytest.raises(FrozenInstanceError):
            e.status = LeadQueueStatus.COMPLETED

    def test_config_immutable(self):
        c = _config()
        with pytest.raises(FrozenInstanceError):
            c.status = CampaignStatus.PAUSED

    def test_snapshot_immutable(self):
        s = _snapshot()
        with pytest.raises(FrozenInstanceError):
            s.calls_today = 999
