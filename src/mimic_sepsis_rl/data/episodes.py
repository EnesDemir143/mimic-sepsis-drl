"""
Episode grid construction from onset-anchored episodes.

This module generates deterministic 4-hour analysis windows for each
usable ICU episode, anchored to the assigned ``sepsis_onset_time``.

Window: ``onset - 24h`` to ``onset + 48h``  →  18 steps of 4 hours each.
Steps are truncated when ICU discharge or death ends the window early.

Usage
-----
    from mimic_sepsis_rl.data.episodes import build_episode_grids

    grids = build_episode_grids(onset_df, icustays_df)

Version history
---------------
v1.0.0  2026-03-28  Initial episode grid builder.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import polars as pl

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core grid builder
# ---------------------------------------------------------------------------


def build_grid_for_episode(
    stay_id: int,
    subject_id: int,
    hadm_id: int,
    onset_time: datetime,
    icu_intime: datetime,
    icu_outtime: datetime,
    deathtime: datetime | None = None,
) -> EpisodeGrid:
    """Build a deterministic episode grid for a single ICU stay.

    Parameters
    ----------
    stay_id, subject_id, hadm_id:
        Episode identifiers.
    onset_time:
        Assigned sepsis onset timestamp (from Phase 2 Plan 01).
    icu_intime, icu_outtime:
        ICU stay boundaries.
    deathtime:
        Patient death time (within ICU), None if survived to discharge.

    Returns
    -------
    EpisodeGrid with deterministic steps and truncation metadata.
    """
    # Compute theoretical window boundaries
    window_start = onset_time + timedelta(hours=WINDOW_START_HOURS)
    window_end = onset_time + timedelta(hours=WINDOW_END_HOURS)

    # Determine actual end time
    effective_end = window_end
    truncation_reason = TruncationReason.NOT_TRUNCATED

    if deathtime is not None and deathtime < effective_end:
        effective_end = deathtime
        truncation_reason = TruncationReason.DEATH

    if icu_outtime < effective_end:
        effective_end = icu_outtime
        if truncation_reason == TruncationReason.NOT_TRUNCATED:
            truncation_reason = TruncationReason.ICU_DISCHARGE

    # Adjust window start if onset is near ICU admission
    effective_start = window_start
    if icu_intime > window_start:
        effective_start = icu_intime
        if truncation_reason == TruncationReason.NOT_TRUNCATED:
            truncation_reason = TruncationReason.ONSET_NEAR_ICU_START

    # Generate steps
    steps: list[EpisodeStep] = []
    truncation_step: int | None = None

    for step_idx in range(MAX_STEPS):
        hours_offset = WINDOW_START_HOURS + step_idx * STEP_HOURS
        step_start = onset_time + timedelta(hours=hours_offset)
        step_end = step_start + timedelta(hours=STEP_HOURS)

        # Skip steps that start before effective start
        if step_start < effective_start:
            # Still include the step but mark partial coverage
            if step_end <= effective_start:
                continue

        # Stop if step starts at or after effective end
        if step_start >= effective_end:
            truncation_step = step_idx - 1 if steps else None
            break

        steps.append(EpisodeStep(
            stay_id=stay_id,
            step_index=step_idx,
            step_start=step_start,
            step_end=min(step_end, effective_end),
            hours_relative_to_onset=hours_offset,
            is_pre_onset=step_start < onset_time,
        ))

    n_steps = len(steps)
    is_truncated = n_steps < MAX_STEPS
    if is_truncated and truncation_step is None and steps:
        truncation_step = steps[-1].step_index

    if not is_truncated:
        truncation_reason = TruncationReason.NOT_TRUNCATED

    return EpisodeGrid(
        stay_id=stay_id,
        subject_id=subject_id,
        hadm_id=hadm_id,
        onset_time=onset_time,
        window_start=window_start,
        window_end=window_end,
        actual_end=effective_end,
        steps=steps,
        n_steps=n_steps,
        is_truncated=is_truncated,
        truncation_reason=truncation_reason,
        truncation_step=truncation_step,
    )


# ---------------------------------------------------------------------------
# Batch grid builder
# ---------------------------------------------------------------------------


def build_episode_grids(
    onset_df: pl.DataFrame,
    icustays_df: pl.DataFrame,
    admissions_df: pl.DataFrame | None = None,
) -> list[EpisodeGrid]:
    """Build episode grids for all usable onset assignments.

    Parameters
    ----------
    onset_df:
        DataFrame with columns:
        stay_id, subject_id, hadm_id, sepsis_onset_time.
    icustays_df:
        ICU stays with stay_id, intime, outtime.
    admissions_df:
        Optional admissions table with hadm_id, deathtime.

    Returns
    -------
    List of EpisodeGrid objects.
    """
    # Merge onset with ICU stay boundaries
    merged = onset_df.join(
        icustays_df.select(["stay_id", "intime", "outtime"]),
        on="stay_id",
        how="inner",
    )

    # Optionally add death times
    death_lookup: dict[int, datetime | None] = {}
    if admissions_df is not None and "deathtime" in admissions_df.columns:
        deaths = admissions_df.filter(
            pl.col("deathtime").is_not_null()
        ).select(["hadm_id", "deathtime"])
        for row in deaths.iter_rows(named=True):
            death_lookup[row["hadm_id"]] = row["deathtime"]

    grids: list[EpisodeGrid] = []
    total = merged.height
    for idx, row in enumerate(merged.iter_rows(named=True)):
        if (idx + 1) % 1000 == 0 or idx == 0:
            logger.info(f"Building grid {idx + 1}/{total}...")

        grid = build_grid_for_episode(
            stay_id=row["stay_id"],
            subject_id=row["subject_id"],
            hadm_id=row["hadm_id"],
            onset_time=row["sepsis_onset_time"],
            icu_intime=row["intime"],
            icu_outtime=row["outtime"],
            deathtime=death_lookup.get(row["hadm_id"]),
        )
        grids.append(grid)

    return grids


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


def grids_to_dataframes(
    grids: list[EpisodeGrid],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert episode grids to Polars DataFrames.

    Returns
    -------
    (episodes_df, steps_df)
        - episodes_df: one row per episode with grid metadata
        - steps_df: one row per step with step boundaries
    """
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for g in grids:
        episode_rows.append({
            "stay_id": g.stay_id,
            "subject_id": g.subject_id,
            "hadm_id": g.hadm_id,
            "onset_time": g.onset_time,
            "window_start": g.window_start,
            "window_end": g.window_end,
            "actual_end": g.actual_end,
            "n_steps": g.n_steps,
            "is_truncated": g.is_truncated,
            "truncation_reason": g.truncation_reason.value,
            "truncation_step": g.truncation_step,
        })

        for s in g.steps:
            step_rows.append({
                "stay_id": s.stay_id,
                "step_index": s.step_index,
                "step_start": s.step_start,
                "step_end": s.step_end,
                "hours_relative_to_onset": s.hours_relative_to_onset,
                "is_pre_onset": s.is_pre_onset,
            })

    episodes_df = pl.DataFrame(episode_rows) if episode_rows else pl.DataFrame(schema={
        "stay_id": pl.Int64, "subject_id": pl.Int64, "hadm_id": pl.Int64,
        "onset_time": pl.Datetime, "window_start": pl.Datetime,
        "window_end": pl.Datetime, "actual_end": pl.Datetime,
        "n_steps": pl.Int32, "is_truncated": pl.Boolean,
        "truncation_reason": pl.Utf8, "truncation_step": pl.Int32,
    })
    steps_df = pl.DataFrame(step_rows) if step_rows else pl.DataFrame(schema={
        "stay_id": pl.Int64, "step_index": pl.Int32,
        "step_start": pl.Datetime, "step_end": pl.Datetime,
        "hours_relative_to_onset": pl.Int32, "is_pre_onset": pl.Boolean,
    })

    return episodes_df, steps_df


def generate_grid_audit(grids: list[EpisodeGrid]) -> dict[str, Any]:
    """Generate audit summary for episode grid construction."""
    total = len(grids)
    truncated = sum(1 for g in grids if g.is_truncated)
    full = total - truncated

    reason_counts: dict[str, int] = {}
    step_counts: list[int] = []
    for g in grids:
        step_counts.append(g.n_steps)
        if g.is_truncated:
            key = g.truncation_reason.value
            reason_counts[key] = reason_counts.get(key, 0) + 1

    avg_steps = sum(step_counts) / total if total > 0 else 0

    return {
        "spec_version": EPISODE_SPEC_VERSION,
        "total_episodes": total,
        "full_length": full,
        "truncated": truncated,
        "truncated_pct": round(truncated / total * 100, 2) if total > 0 else 0,
        "truncation_reasons": reason_counts,
        "avg_steps": round(avg_steps, 2),
        "min_steps": min(step_counts) if step_counts else 0,
        "max_steps": max(step_counts) if step_counts else 0,
        "expected_max_steps": MAX_STEPS,
    }
