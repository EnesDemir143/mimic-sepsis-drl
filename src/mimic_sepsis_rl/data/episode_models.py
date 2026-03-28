"""
Typed models for episode grid construction.

Each ICU episode with a usable onset produces a sequence of
``EpisodeStep`` objects that define the deterministic 4-hour
analysis timeline from ``onset - 24h`` to ``onset + 48h``.

Truncated episodes (early ICU discharge or death) keep explicit
metadata about why and where the window was cut short.

Version history
---------------
v1.0.0  2026-03-28  Initial episode grid contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

EPISODE_SPEC_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_HOURS: Final[int] = 4
WINDOW_START_HOURS: Final[int] = -24  # relative to onset
WINDOW_END_HOURS: Final[int] = 48    # relative to onset
MAX_STEPS: Final[int] = (WINDOW_END_HOURS - WINDOW_START_HOURS) // STEP_HOURS  # 18


# ---------------------------------------------------------------------------
# Truncation reasons
# ---------------------------------------------------------------------------


class TruncationReason(str, Enum):
    """Why an episode window was cut short before step 17 (0-indexed)."""

    NOT_TRUNCATED = "not_truncated"
    ICU_DISCHARGE = "icu_discharge"
    DEATH = "death"
    ONSET_NEAR_ICU_END = "onset_near_icu_end"
    ONSET_NEAR_ICU_START = "onset_near_icu_start"


# ---------------------------------------------------------------------------
# Episode step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeStep:
    """A single 4-hour analysis window within an episode.

    Attributes
    ----------
    stay_id:
        ICU stay this step belongs to.
    step_index:
        0-based step index (0 = onset - 24h, 6 = onset, 17 = onset + 44h).
    step_start:
        Absolute start datetime of this 4-hour window.
    step_end:
        Absolute end datetime of this 4-hour window.
    hours_relative_to_onset:
        Hours offset from onset for step_start (e.g. -24, -20, ..., +44).
    is_pre_onset:
        True if step_start is before onset_time.
    """

    stay_id: int
    step_index: int
    step_start: datetime
    step_end: datetime
    hours_relative_to_onset: int
    is_pre_onset: bool


# ---------------------------------------------------------------------------
# Episode grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeGrid:
    """Complete episode grid for one ICU stay.

    Attributes
    ----------
    stay_id:
        ICU stay identifier.
    subject_id:
        Patient identifier.
    hadm_id:
        Hospital admission identifier.
    onset_time:
        Sepsis onset timestamp (anchor for the grid).
    window_start:
        Absolute start of the analysis window (onset - 24h).
    window_end:
        Theoretical end of the analysis window (onset + 48h).
    actual_end:
        Actual end time (may be earlier due to ICU discharge/death).
    steps:
        List of EpisodeStep objects (sorted by step_index).
    n_steps:
        Number of realised steps (≤ MAX_STEPS=18).
    is_truncated:
        Whether the episode was cut short.
    truncation_reason:
        Why the episode was truncated.
    truncation_step:
        Last valid step index if truncated, None if full.
    """

    stay_id: int
    subject_id: int
    hadm_id: int
    onset_time: datetime
    window_start: datetime
    window_end: datetime
    actual_end: datetime
    steps: list[EpisodeStep] = field(default_factory=list)
    n_steps: int = 0
    is_truncated: bool = False
    truncation_reason: TruncationReason = TruncationReason.NOT_TRUNCATED
    truncation_step: int | None = None
