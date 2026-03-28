"""
Cohort specification for the MIMIC-IV Sepsis-3 offline RL study.

This module encodes the adult ICU Sepsis-3 eligibility rules as explicit,
typed, and versioned contract definitions.  All downstream extraction,
audit reporting, and documentation must be derived from this single source
of truth rather than from ad-hoc notebook constants.

Version history
---------------
v1.0.0  2026-03-28  Initial Sepsis-3 adult ICU cohort contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

COHORT_SPEC_VERSION: Final[str] = "1.0.0"


# ---------------------------------------------------------------------------
# Canonical column identifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnContract:
    """Stable column names expected from MIMIC-IV source tables.

    These names are used in SQL extraction, Python post-processing, and
    audit output.  Any rename must be propagated here first.
    """

    # patient / admission level
    subject_id: str = "subject_id"
    hadm_id: str = "hadm_id"
    stay_id: str = "stay_id"

    # demographics
    anchor_age: str = "anchor_age"
    gender: str = "gender"

    # ICU stay timing
    intime: str = "intime"
    outtime: str = "outtime"
    los_hours: str = "los_hours"

    # Sepsis-3 criteria columns (from sepsis3 derived table or equivalent)
    sepsis3_flag: str = "sepsis3"
    sofa_score: str = "sofa_score"
    suspected_infection_time: str = "suspected_infection_time"


COLUMNS: Final[ColumnContract] = ColumnContract()


# ---------------------------------------------------------------------------
# Inclusion criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InclusionCriteria:
    """Positive conditions that a stay MUST satisfy to enter the cohort.

    Attributes
    ----------
    min_age_years:
        Minimum anchor age (inclusive) — restricts to adult patients.
    require_icu_stay:
        The episode must be an ICU stay (``icustays`` table row present).
    require_sepsis3:
        The stay must meet Sepsis-3 criteria as derived by MIMIC-IV's
        ``sepsis3`` table or an equivalent query.
    min_los_hours:
        Minimum ICU length-of-stay in hours.  Stays shorter than this
        threshold are considered too brief to have meaningful episode
        windows and are excluded.
    """

    min_age_years: int = 18
    require_icu_stay: bool = True
    require_sepsis3: bool = True
    min_los_hours: float = 4.0


# ---------------------------------------------------------------------------
# Exclusion criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExclusionCriteria:
    """Conditions that cause an otherwise eligible stay to be dropped.

    Attributes
    ----------
    exclude_missing_sepsis_anchor:
        Remove stays that have no usable ``suspected_infection_time`` or
        ``sepsis3`` flag, making onset-time assignment impossible.
    exclude_readmissions:
        Keep only the first ICU stay per ``subject_id`` to avoid
        within-patient leakage and correlated episodes.
    exclude_missing_demographics:
        Remove stays with null ``anchor_age`` or ``gender``.
    max_age_years:
        Exclude stays where ``anchor_age`` exceeds this threshold.
        ``None`` means no upper bound.
    """

    exclude_missing_sepsis_anchor: bool = True
    exclude_readmissions: bool = True
    exclude_missing_demographics: bool = True
    max_age_years: int | None = None


# ---------------------------------------------------------------------------
# Top-level cohort specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """Complete, versioned cohort definition for the study.

    Instantiate with default or overridden sub-specs.  The ``version``
    field ties every artifact produced under this spec to a reproducible
    rule set.

    Examples
    --------
    >>> spec = CohortSpec()
    >>> spec.version
    '1.0.0'
    >>> spec.inclusion.min_age_years
    18
    >>> spec.exclusion.exclude_readmissions
    True
    """

    version: str = field(default_factory=lambda: COHORT_SPEC_VERSION)
    description: str = (
        "Adult ICU Sepsis-3 cohort from MIMIC-IV — first ICU stay per patient, "
        "minimum age 18, Sepsis-3 flag required, onset time must be inferrable."
    )
    adult_only: bool = True  # explicitly signals adult restriction in YAML binding
    inclusion: InclusionCriteria = field(default_factory=InclusionCriteria)
    exclusion: ExclusionCriteria = field(default_factory=ExclusionCriteria)
    columns: ColumnContract = field(default_factory=ColumnContract)

    # Source table names — override if schema differs
    source_tables: dict[str, str] = field(
        default_factory=lambda: {
            "icustays": "mimiciv_icu.icustays",
            "patients": "mimiciv_hosp.patients",
            "admissions": "mimiciv_hosp.admissions",
            "sepsis3": "mimiciv_derived.sepsis3",
        }
    )

    def rule_summary(self) -> dict[str, object]:
        """Return a flat dict of all active rules for logging and audit."""
        return {
            "spec_version": self.version,
            "adult_only": self.adult_only,
            # inclusion
            "min_age_years": self.inclusion.min_age_years,
            "require_icu_stay": self.inclusion.require_icu_stay,
            "require_sepsis3": self.inclusion.require_sepsis3,
            "min_los_hours": self.inclusion.min_los_hours,
            # exclusion
            "exclude_missing_sepsis_anchor": self.exclusion.exclude_missing_sepsis_anchor,
            "exclude_readmissions": self.exclusion.exclude_readmissions,
            "exclude_missing_demographics": self.exclusion.exclude_missing_demographics,
            "max_age_years": self.exclusion.max_age_years,
        }
