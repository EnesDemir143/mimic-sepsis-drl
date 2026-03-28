"""
Typed output models for the cohort extraction pipeline.

Every extraction run produces a ``CohortResult`` containing:
- ``included``: episodes that pass all inclusion/exclusion rules.
- ``excluded``: episodes that failed at least one rule, with a reason column.
- ``audit_summary``: count-level breakdown for reproducibility reporting.

Version history
---------------
v1.0.0  2026-03-28  Initial cohort result models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

import polars as pl


# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

COHORT_MODELS_VERSION: Final[str] = "1.0.0"


# ---------------------------------------------------------------------------
# Exclusion reason codes
# ---------------------------------------------------------------------------


class ExclusionReason(str, Enum):
    """Enumerated reasons why an ICU stay was excluded."""

    AGE_BELOW_MIN = "age_below_minimum"
    AGE_ABOVE_MAX = "age_above_maximum"
    LOS_TOO_SHORT = "los_below_minimum_hours"
    MISSING_DEMOGRAPHICS = "missing_demographics"
    READMISSION = "readmission_not_first_stay"
    NO_SEPSIS_MARKERS = "no_sepsis3_markers"


# ---------------------------------------------------------------------------
# Cohort result
# ---------------------------------------------------------------------------


@dataclass
class CohortResult:
    """Complete output of a single cohort extraction run.

    Attributes
    ----------
    included:
        DataFrame of ICU stays that passed all rules.
    excluded:
        DataFrame of excluded stays with ``exclusion_reason`` column.
    audit_summary:
        Count-level audit dict for JSON serialisation.
    """

    included: pl.DataFrame
    excluded: pl.DataFrame
    audit_summary: dict[str, Any] = field(default_factory=dict)
