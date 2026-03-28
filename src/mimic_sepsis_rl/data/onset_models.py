"""
Typed models for sepsis onset assignment results.

Every ICU episode produces exactly one ``OnsetResult``:
- **Usable**: a single ``sepsis_onset_time`` was assigned.
- **Unusable**: no valid onset could be determined; reason is captured.

These models are the typed interface between the onset assignment pipeline
(``onset.py``) and downstream consumers (episode grid, audit reporting).

Version history
---------------
v1.0.0  2026-03-28  Initial onset result contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

ONSET_SPEC_VERSION: Final[str] = "1.0.0"


# ---------------------------------------------------------------------------
# Unusable reason codes
# ---------------------------------------------------------------------------


class UnusableReason(str, Enum):
    """Enumerated reasons why an episode has no usable onset."""

    NO_SUSPECTED_INFECTION = "no_suspected_infection_time"
    NO_SOFA_INCREASE = "no_sofa_increase_within_window"
    MULTIPLE_AMBIGUOUS = "multiple_ambiguous_onsets"
    ONSET_OUTSIDE_ICU = "onset_outside_icu_stay"
    MISSING_REQUIRED_DATA = "missing_required_data"


# ---------------------------------------------------------------------------
# Onset candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnsetCandidate:
    """A single candidate onset event before prioritisation.

    Attributes
    ----------
    stay_id:
        ICU stay this candidate belongs to.
    suspected_infection_time:
        Time suspected infection was flagged.
    sofa_time:
        Time of qualifying SOFA increase (≥2 within ±24h of infection).
    sofa_score:
        SOFA score at ``sofa_time``.
    onset_time:
        Derived onset = min(suspected_infection_time, sofa_time).
    """

    stay_id: int
    suspected_infection_time: datetime
    sofa_time: datetime
    sofa_score: int
    onset_time: datetime


# ---------------------------------------------------------------------------
# Onset result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnsetResult:
    """Final onset assignment for a single ICU episode.

    Exactly one of ``onset_time`` or ``unusable_reason`` is non-None.

    Attributes
    ----------
    stay_id:
        ICU stay identifier (foreign key to cohort output).
    subject_id:
        Patient identifier.
    hadm_id:
        Hospital admission identifier.
    onset_time:
        Assigned sepsis onset (earliest qualifying candidate).
        ``None`` if the episode is unusable.
    unusable_reason:
        Reason why no onset was assigned. ``None`` if usable.
    candidates:
        All candidates evaluated for this episode (for audit).
    selected_candidate:
        The candidate that was chosen (``None`` if unusable).
    """

    stay_id: int
    subject_id: int
    hadm_id: int
    onset_time: datetime | None = None
    unusable_reason: UnusableReason | None = None
    candidates: list[OnsetCandidate] = field(default_factory=list)
    selected_candidate: OnsetCandidate | None = None

    def __post_init__(self) -> None:
        """Validate mutual exclusivity."""
        has_onset = self.onset_time is not None
        has_reason = self.unusable_reason is not None
        if has_onset == has_reason:
            raise ValueError(
                f"stay_id={self.stay_id}: exactly one of onset_time or "
                f"unusable_reason must be set (got onset_time={self.onset_time}, "
                f"unusable_reason={self.unusable_reason})"
            )

    @property
    def is_usable(self) -> bool:
        """Whether this episode has a valid onset assignment."""
        return self.onset_time is not None
