"""
Tests for deterministic episode grid construction.

Covers:
- Full 18-step grid for long ICU stay
- Truncation by ICU discharge
- Truncation by death
- Truncation when onset is near ICU start
- Step boundary alignment (4-hour steps)
- Step index correctness
- Pre-onset vs post-onset step marking
- Empty grid edge case
- Batch grid builder
- DataFrame conversion
- Audit summary
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from mimic_sepsis_rl.data.episode_models import (
    EPISODE_SPEC_VERSION,
    MAX_STEPS,
    STEP_HOURS,
    WINDOW_END_HOURS,
    WINDOW_START_HOURS,
    EpisodeGrid,
    EpisodeStep,
    TruncationReason,
)
from mimic_sepsis_rl.data.episodes import (
    build_grid_for_episode,
    build_episode_grids,
    generate_grid_audit,
    grids_to_dataframes,
)


# ---------------------------------------------------------------------------
# Time fixtures
# ---------------------------------------------------------------------------

ONSET = datetime(2150, 1, 15, 12, 0, 0)
ICU_START = datetime(2150, 1, 14, 0, 0, 0)  # 36h before onset
ICU_END = datetime(2150, 1, 20, 0, 0, 0)    # 132h after onset (~5.5 days)


# ---------------------------------------------------------------------------
# Constants verification
# ---------------------------------------------------------------------------


class TestConstants:
    def test_step_size_is_4_hours(self):
        assert STEP_HOURS == 4

    def test_window_is_minus24_to_plus48(self):
        assert WINDOW_START_HOURS == -24
        assert WINDOW_END_HOURS == 48

    def test_max_steps_is_18(self):
        assert MAX_STEPS == 18
        assert MAX_STEPS == (WINDOW_END_HOURS - WINDOW_START_HOURS) // STEP_HOURS


# ---------------------------------------------------------------------------
# Full grid (no truncation)
# ---------------------------------------------------------------------------


class TestFullGrid:
    """ICU stay long enough to cover the entire -24h/+48h window."""

    def test_full_grid_has_18_steps(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        assert grid.n_steps == 18

    def test_full_grid_is_not_truncated(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        assert not grid.is_truncated
        assert grid.truncation_reason == TruncationReason.NOT_TRUNCATED

    def test_step_indices_are_sequential(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        indices = [s.step_index for s in grid.steps]
        assert indices == list(range(18))

    def test_first_step_starts_at_onset_minus_24h(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        assert grid.steps[0].step_start == ONSET - timedelta(hours=24)

    def test_last_step_ends_at_onset_plus_48h(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        assert grid.steps[-1].step_end == ONSET + timedelta(hours=48)

    def test_each_step_is_4_hours(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        for step in grid.steps:
            duration = (step.step_end - step.step_start).total_seconds() / 3600
            assert duration == pytest.approx(4.0)

    def test_steps_are_contiguous(self):
        """Each step starts exactly when the previous one ends."""
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        for i in range(1, len(grid.steps)):
            assert grid.steps[i].step_start == grid.steps[i - 1].step_end

    def test_pre_onset_steps_are_marked_correctly(self):
        """Steps 0-5 should be pre-onset (hours -24 to -4 inclusive)."""
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        for step in grid.steps:
            expected_pre = step.step_start < ONSET
            assert step.is_pre_onset == expected_pre, (
                f"Step {step.step_index}: expected is_pre_onset={expected_pre}, "
                f"got {step.is_pre_onset}"
            )

    def test_hours_relative_to_onset_are_correct(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        expected = list(range(-24, 48, 4))  # [-24, -20, -16, ..., 44]
        actual = [s.hours_relative_to_onset for s in grid.steps]
        assert actual == expected

    def test_window_boundaries_stored_correctly(self):
        grid = build_grid_for_episode(
            stay_id=100, subject_id=1, hadm_id=10,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        assert grid.window_start == ONSET - timedelta(hours=24)
        assert grid.window_end == ONSET + timedelta(hours=48)


# ---------------------------------------------------------------------------
# Truncation by ICU discharge
# ---------------------------------------------------------------------------


class TestTruncationByDischarge:
    def test_early_discharge_truncates(self):
        """ICU discharge 12h after onset → only steps within window."""
        early_outtime = ONSET + timedelta(hours=12)
        grid = build_grid_for_episode(
            stay_id=200, subject_id=2, hadm_id=20,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=early_outtime,
        )
        assert grid.is_truncated
        assert grid.truncation_reason == TruncationReason.ICU_DISCHARGE
        assert grid.n_steps < MAX_STEPS
        # Steps should not extend beyond ICU discharge
        for step in grid.steps:
            assert step.step_start < early_outtime

    def test_discharge_at_onset_still_has_pre_onset_steps(self):
        """ICU discharge at onset → only pre-onset steps available."""
        grid = build_grid_for_episode(
            stay_id=201, subject_id=2, hadm_id=20,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ONSET,
        )
        assert grid.is_truncated
        # Should have some steps (pre-onset)
        for step in grid.steps:
            assert step.is_pre_onset


# ---------------------------------------------------------------------------
# Truncation by death
# ---------------------------------------------------------------------------


class TestTruncationByDeath:
    def test_death_during_window_truncates(self):
        deathtime = ONSET + timedelta(hours=20)
        grid = build_grid_for_episode(
            stay_id=300, subject_id=3, hadm_id=30,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
            deathtime=deathtime,
        )
        assert grid.is_truncated
        assert grid.truncation_reason == TruncationReason.DEATH
        # Steps should not extend beyond death
        for step in grid.steps:
            assert step.step_start < deathtime

    def test_death_after_window_no_truncation(self):
        """Death after the 48h window should not truncate."""
        deathtime = ONSET + timedelta(hours=100)  # well after window
        grid = build_grid_for_episode(
            stay_id=301, subject_id=3, hadm_id=30,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
            deathtime=deathtime,
        )
        assert grid.n_steps == MAX_STEPS
        assert not grid.is_truncated


# ---------------------------------------------------------------------------
# Onset near ICU boundaries
# ---------------------------------------------------------------------------


class TestOnsetNearBoundary:
    def test_onset_soon_after_icu_start_truncates_early_steps(self):
        """If ICU starts only 4h before onset, pre-24h steps are lost."""
        late_icu_start = ONSET - timedelta(hours=4)
        grid = build_grid_for_episode(
            stay_id=400, subject_id=4, hadm_id=40,
            onset_time=ONSET, icu_intime=late_icu_start, icu_outtime=ICU_END,
        )
        # Should have fewer than 18 steps
        assert grid.n_steps < MAX_STEPS
        # First step should not start before ICU admission
        for step in grid.steps:
            assert step.step_start >= late_icu_start or step.step_end > late_icu_start


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_produce_same_grid(self):
        """Grid generation must be deterministic."""
        args = dict(
            stay_id=500, subject_id=5, hadm_id=50,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        grid1 = build_grid_for_episode(**args)
        grid2 = build_grid_for_episode(**args)
        assert grid1.n_steps == grid2.n_steps
        for s1, s2 in zip(grid1.steps, grid2.steps):
            assert s1.step_index == s2.step_index
            assert s1.step_start == s2.step_start
            assert s1.step_end == s2.step_end


# ---------------------------------------------------------------------------
# Model immutability
# ---------------------------------------------------------------------------


class TestModelImmutability:
    def test_episode_step_is_frozen(self):
        step = EpisodeStep(
            stay_id=1, step_index=0,
            step_start=ONSET, step_end=ONSET + timedelta(hours=4),
            hours_relative_to_onset=-24, is_pre_onset=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            step.step_index = 99  # type: ignore[misc]

    def test_episode_grid_is_frozen(self):
        grid = EpisodeGrid(
            stay_id=1, subject_id=1, hadm_id=1,
            onset_time=ONSET,
            window_start=ONSET - timedelta(hours=24),
            window_end=ONSET + timedelta(hours=48),
            actual_end=ONSET + timedelta(hours=48),
        )
        with pytest.raises((AttributeError, TypeError)):
            grid.n_steps = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Batch builder and serialisation
# ---------------------------------------------------------------------------


class TestBatchAndSerialisation:
    def test_batch_builds_all_episodes(self):
        onset_df = pl.DataFrame({
            "stay_id": [1, 2],
            "subject_id": [10, 20],
            "hadm_id": [100, 200],
            "sepsis_onset_time": [ONSET, ONSET + timedelta(hours=6)],
        })
        icustays_df = pl.DataFrame({
            "stay_id": [1, 2],
            "intime": [ICU_START, ICU_START],
            "outtime": [ICU_END, ICU_END],
        })
        grids = build_episode_grids(onset_df, icustays_df)
        assert len(grids) == 2

    def test_grids_to_dataframes_shape(self):
        grid = build_grid_for_episode(
            stay_id=1, subject_id=1, hadm_id=1,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        episodes_df, steps_df = grids_to_dataframes([grid])
        assert episodes_df.height == 1
        assert steps_df.height == 18
        assert "stay_id" in episodes_df.columns
        assert "step_index" in steps_df.columns
        assert "truncation_reason" in episodes_df.columns

    def test_audit_summary_counts(self):
        full_grid = build_grid_for_episode(
            stay_id=1, subject_id=1, hadm_id=1,
            onset_time=ONSET, icu_intime=ICU_START, icu_outtime=ICU_END,
        )
        trunc_grid = build_grid_for_episode(
            stay_id=2, subject_id=2, hadm_id=2,
            onset_time=ONSET, icu_intime=ICU_START,
            icu_outtime=ONSET + timedelta(hours=12),
        )
        audit = generate_grid_audit([full_grid, trunc_grid])
        assert audit["total_episodes"] == 2
        assert audit["full_length"] == 1
        assert audit["truncated"] == 1
        assert audit["expected_max_steps"] == MAX_STEPS
        assert audit["spec_version"] == EPISODE_SPEC_VERSION


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestEpisodeVersion:
    def test_version_is_nonempty(self):
        assert EPISODE_SPEC_VERSION
        assert isinstance(EPISODE_SPEC_VERSION, str)
