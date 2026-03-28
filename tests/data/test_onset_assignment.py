"""
Tests for Sepsis-3 onset assignment pipeline.

Covers:
- Straightforward single-candidate onset selection
- Multiple candidate tie-breaking (earliest wins)
- No suspected infection → unusable
- No qualifying SOFA → unusable
- Onset outside ICU boundary → unusable
- OnsetResult mutual exclusivity invariant
- OnsetCandidate immutability
- Audit summary correctness
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from mimic_sepsis_rl.data.onset_models import (
    ONSET_SPEC_VERSION,
    OnsetCandidate,
    OnsetResult,
    UnusableReason,
)
from mimic_sepsis_rl.data.onset import (
    assign_onset_for_stay,
    generate_audit_summary,
    results_to_dataframes,
    load_onset_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2150, 1, 15, 12, 0, 0)

DEFAULT_CFG = {
    "onset": {
        "lookback_hours": 24,
        "lookahead_hours": 24,
        "min_sofa_increase": 2,
        "tie_break": "earliest",
    },
    "icu_boundary": {
        "require_within_icu": True,
        "grace_before_intime_hours": 6.0,
    },
}


def _make_stay(
    stay_id: int = 100,
    subject_id: int = 1,
    hadm_id: int = 10,
    intime: datetime | None = None,
    outtime: datetime | None = None,
) -> dict:
    return {
        "stay_id": stay_id,
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "intime": intime or BASE_TIME - timedelta(hours=2),
        "outtime": outtime or BASE_TIME + timedelta(hours=72),
    }


def _make_sofa_df(
    stay_id: int,
    entries: list[tuple[datetime, int]],
) -> pl.DataFrame:
    """Create a SOFA DataFrame with (charttime, sofa_score) entries."""
    if not entries:
        return pl.DataFrame(schema={
            "stay_id": pl.Int64,
            "charttime": pl.Datetime,
            "sofa_score": pl.Int32,
        })
    return pl.DataFrame({
        "stay_id": [stay_id] * len(entries),
        "charttime": [e[0] for e in entries],
        "sofa_score": [e[1] for e in entries],
    })


# ---------------------------------------------------------------------------
# OnsetResult invariant tests
# ---------------------------------------------------------------------------


class TestOnsetResultInvariants:
    """Validate the mutual exclusivity contract of OnsetResult."""

    def test_usable_result_has_onset_time(self):
        r = OnsetResult(
            stay_id=1, subject_id=1, hadm_id=1,
            onset_time=BASE_TIME,
        )
        assert r.is_usable
        assert r.onset_time == BASE_TIME
        assert r.unusable_reason is None

    def test_unusable_result_has_reason(self):
        r = OnsetResult(
            stay_id=1, subject_id=1, hadm_id=1,
            unusable_reason=UnusableReason.NO_SUSPECTED_INFECTION,
        )
        assert not r.is_usable
        assert r.onset_time is None
        assert r.unusable_reason == UnusableReason.NO_SUSPECTED_INFECTION

    def test_both_set_raises_error(self):
        with pytest.raises(ValueError, match="exactly one"):
            OnsetResult(
                stay_id=1, subject_id=1, hadm_id=1,
                onset_time=BASE_TIME,
                unusable_reason=UnusableReason.NO_SOFA_INCREASE,
            )

    def test_neither_set_raises_error(self):
        with pytest.raises(ValueError, match="exactly one"):
            OnsetResult(stay_id=1, subject_id=1, hadm_id=1)

    def test_onset_result_is_frozen(self):
        r = OnsetResult(stay_id=1, subject_id=1, hadm_id=1, onset_time=BASE_TIME)
        with pytest.raises((AttributeError, TypeError)):
            r.stay_id = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OnsetCandidate tests
# ---------------------------------------------------------------------------


class TestOnsetCandidate:
    def test_candidate_is_frozen(self):
        c = OnsetCandidate(
            stay_id=1,
            suspected_infection_time=BASE_TIME,
            sofa_time=BASE_TIME + timedelta(hours=2),
            sofa_score=3,
            onset_time=BASE_TIME,
        )
        with pytest.raises((AttributeError, TypeError)):
            c.sofa_score = 0  # type: ignore[misc]

    def test_onset_time_is_min_of_infection_and_sofa(self):
        infection = BASE_TIME
        sofa = BASE_TIME + timedelta(hours=5)
        c = OnsetCandidate(
            stay_id=1,
            suspected_infection_time=infection,
            sofa_time=sofa,
            sofa_score=2,
            onset_time=min(infection, sofa),
        )
        assert c.onset_time == infection


# ---------------------------------------------------------------------------
# Onset assignment logic tests
# ---------------------------------------------------------------------------


class TestOnsetAssignment:
    """Test the core onset assignment logic for single stays."""

    def test_straightforward_single_candidate(self):
        """One clear infection + SOFA → usable onset."""
        stay = _make_stay()
        infection_time = BASE_TIME
        sofa = _make_sofa_df(100, [
            (BASE_TIME + timedelta(hours=4), 3),
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert result.is_usable
        assert result.onset_time == BASE_TIME  # min(infection, sofa)
        assert len(result.candidates) == 1

    def test_multiple_candidates_earliest_wins(self):
        """Multiple qualifying SOFAs → earliest onset selected."""
        stay = _make_stay()
        infection_time = BASE_TIME
        sofa = _make_sofa_df(100, [
            (BASE_TIME + timedelta(hours=8), 2),
            (BASE_TIME + timedelta(hours=4), 3),
            (BASE_TIME + timedelta(hours=12), 4),
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert result.is_usable
        assert len(result.candidates) == 3
        # All candidates have onset = min(infection, sofa_time) = infection_time
        # since infection_time is earlier than all sofa_times
        assert result.onset_time == infection_time

    def test_sofa_before_infection_makes_earlier_onset(self):
        """When SOFA time precedes infection, onset = sofa_time."""
        stay = _make_stay()
        infection_time = BASE_TIME + timedelta(hours=6)
        sofa = _make_sofa_df(100, [
            (BASE_TIME + timedelta(hours=2), 3),  # before infection
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert result.is_usable
        # onset = min(infection, sofa) = sofa_time
        assert result.onset_time == BASE_TIME + timedelta(hours=2)

    def test_no_suspected_infection_is_unusable(self):
        """No infection time → unusable with correct reason."""
        stay = _make_stay()
        sofa = _make_sofa_df(100, [(BASE_TIME, 5)])
        result = assign_onset_for_stay(stay, None, sofa, DEFAULT_CFG)
        assert not result.is_usable
        assert result.unusable_reason == UnusableReason.NO_SUSPECTED_INFECTION

    def test_no_qualifying_sofa_is_unusable(self):
        """Infection exists but no SOFA ≥ 2 in window → unusable."""
        stay = _make_stay()
        infection_time = BASE_TIME
        sofa = _make_sofa_df(100, [
            (BASE_TIME + timedelta(hours=4), 1),  # below threshold
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert not result.is_usable
        assert result.unusable_reason == UnusableReason.NO_SOFA_INCREASE

    def test_sofa_outside_window_is_unusable(self):
        """SOFA ≥ 2 but outside the ±24h search window → unusable."""
        stay = _make_stay()
        infection_time = BASE_TIME
        sofa = _make_sofa_df(100, [
            (BASE_TIME + timedelta(hours=30), 5),  # beyond 24h
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert not result.is_usable
        assert result.unusable_reason == UnusableReason.NO_SOFA_INCREASE

    def test_onset_outside_icu_is_unusable(self):
        """Onset before ICU admission (beyond grace) → unusable."""
        # ICU starts at BASE_TIME, onset would be much earlier
        stay = _make_stay(intime=BASE_TIME)
        infection_time = BASE_TIME - timedelta(hours=48)
        sofa = _make_sofa_df(100, [
            (BASE_TIME - timedelta(hours=46), 3),
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert not result.is_usable
        assert result.unusable_reason == UnusableReason.ONSET_OUTSIDE_ICU

    def test_onset_within_grace_period_is_usable(self):
        """Onset slightly before ICU admission (within grace) → usable."""
        stay = _make_stay(intime=BASE_TIME)
        infection_time = BASE_TIME - timedelta(hours=3)  # within 6h grace
        sofa = _make_sofa_df(100, [
            (BASE_TIME - timedelta(hours=1), 2),
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert result.is_usable

    def test_empty_sofa_dataframe(self):
        """No SOFA data at all → unusable."""
        stay = _make_stay()
        infection_time = BASE_TIME
        sofa = _make_sofa_df(100, [])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert not result.is_usable
        assert result.unusable_reason == UnusableReason.NO_SOFA_INCREASE

    def test_no_duplicate_onset_per_stay(self):
        """Regardless of candidates, exactly ONE onset per stay."""
        stay = _make_stay()
        infection_time = BASE_TIME
        sofa = _make_sofa_df(100, [
            (BASE_TIME + timedelta(hours=i), 2 + i)
            for i in range(5)
        ])
        result = assign_onset_for_stay(stay, infection_time, sofa, DEFAULT_CFG)
        assert result.is_usable
        # Only one onset_time despite 5 candidates
        assert isinstance(result.onset_time, datetime)
        assert result.selected_candidate is not None


# ---------------------------------------------------------------------------
# Audit and serialisation tests
# ---------------------------------------------------------------------------


class TestAuditAndSerialisation:
    def test_audit_summary_counts(self):
        results = [
            OnsetResult(stay_id=1, subject_id=1, hadm_id=1, onset_time=BASE_TIME),
            OnsetResult(stay_id=2, subject_id=2, hadm_id=2, onset_time=BASE_TIME),
            OnsetResult(
                stay_id=3, subject_id=3, hadm_id=3,
                unusable_reason=UnusableReason.NO_SUSPECTED_INFECTION,
            ),
        ]
        audit = generate_audit_summary(results)
        assert audit["total_episodes"] == 3
        assert audit["usable"] == 2
        assert audit["unusable"] == 1
        assert audit["usable_pct"] == pytest.approx(66.67, abs=0.01)
        assert audit["unusable_reasons"]["no_suspected_infection_time"] == 1

    def test_results_to_dataframes_shape(self):
        results = [
            OnsetResult(
                stay_id=1, subject_id=1, hadm_id=1,
                onset_time=BASE_TIME,
                candidates=[
                    OnsetCandidate(1, BASE_TIME, BASE_TIME + timedelta(hours=1), 3, BASE_TIME),
                ],
                selected_candidate=OnsetCandidate(1, BASE_TIME, BASE_TIME + timedelta(hours=1), 3, BASE_TIME),
            ),
            OnsetResult(
                stay_id=2, subject_id=2, hadm_id=2,
                unusable_reason=UnusableReason.NO_SOFA_INCREASE,
            ),
        ]
        usable, unusable, candidates = results_to_dataframes(results)
        assert usable.height == 1
        assert unusable.height == 1
        assert candidates.height == 1
        assert "sepsis_onset_time" in usable.columns
        assert "unusable_reason" in unusable.columns

    def test_empty_results_produce_empty_dataframes(self):
        usable, unusable, candidates = results_to_dataframes([])
        assert usable.height == 0
        assert unusable.height == 0
        assert candidates.height == 0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_config_loads_successfully(self, tmp_path):
        cfg_content = """
onset:
  lookback_hours: 24
  lookahead_hours: 24
  min_sofa_increase: 2
  tie_break: earliest
icu_boundary:
  require_within_icu: true
  grace_before_intime_hours: 6.0
"""
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(cfg_content)
        cfg = load_onset_config(cfg_file)
        assert cfg["onset"]["lookback_hours"] == 24

    def test_missing_config_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_onset_config(tmp_path / "nonexistent.yaml")

    def test_incomplete_config_raises_error(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("onset:\n  lookback_hours: 24\n")
        with pytest.raises(ValueError, match="missing required"):
            load_onset_config(cfg_file)


# ---------------------------------------------------------------------------
# Version tests
# ---------------------------------------------------------------------------


class TestOnsetVersion:
    def test_version_is_nonempty(self):
        assert ONSET_SPEC_VERSION
        assert isinstance(ONSET_SPEC_VERSION, str)

    def test_audit_includes_version(self):
        results = [
            OnsetResult(stay_id=1, subject_id=1, hadm_id=1, onset_time=BASE_TIME),
        ]
        audit = generate_audit_summary(results)
        assert audit["spec_version"] == ONSET_SPEC_VERSION
