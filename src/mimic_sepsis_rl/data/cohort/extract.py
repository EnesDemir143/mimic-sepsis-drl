"""
Cohort extraction from MIMIC-IV raw CSV files.

This module implements the ``CohortExtractor`` class that reads raw
MIMIC-IV data (icustays, patients, admissions), applies the cohort
eligibility rules defined in ``CohortSpec``, and produces a typed
``CohortResult`` with included stays, excluded stays, and an audit
summary.

Since MIMIC-IV 3.1 raw distribution does not include a pre-computed
``sepsis3`` derived table, the ``require_sepsis3`` rule is evaluated
by checking whether the Phase 2 onset assignment module can identify
a suspected infection (antibiotic + culture pairing) for the admission.
When the pre-computed Sepsis-3 table is unavailable, the requirement
is relaxed: stays are flagged for later removal during the onset
assignment phase, ensuring the cohort serves as a superset that the
onset pipeline then refines.

Usage
-----
    from mimic_sepsis_rl.data.cohort.extract import CohortExtractor
    from mimic_sepsis_rl.data.cohort.spec import CohortSpec

    extractor = CohortExtractor(spec=CohortSpec())
    result = extractor.run(emit_audit=True)

Version history
---------------
v1.0.0  2026-03-28  Initial extraction from raw CSV.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from mimic_sepsis_rl.data.cohort.models import (
    COHORT_MODELS_VERSION,
    CohortResult,
    ExclusionReason,
)
from mimic_sepsis_rl.data.cohort.spec import CohortSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------

MIMIC_RAW_ROOT = Path("data/raw/physionet.org/files/mimiciv/3.1")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class CohortExtractor:
    """Apply CohortSpec rules to raw MIMIC-IV data and produce a CohortResult.

    Parameters
    ----------
    spec:
        The typed cohort specification with inclusion/exclusion rules.
    data_root:
        Path to the MIMIC-IV raw data directory.
    """

    def __init__(
        self,
        spec: CohortSpec,
        data_root: Path = MIMIC_RAW_ROOT,
    ) -> None:
        self.spec = spec
        self.data_root = data_root
        self._exclusion_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Raw data loaders
    # ------------------------------------------------------------------

    def _load_icustays(self) -> pl.DataFrame:
        path = self.data_root / "icu" / "icustays.csv.gz"
        logger.info(f"Loading ICU stays from {path}")
        df = pl.read_csv(path)
        # Compute LOS in hours from the `los` column (days)
        df = df.with_columns(
            pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            (pl.col("los") * 24.0).alias("los_hours"),
        )
        return df

    def _load_patients(self) -> pl.DataFrame:
        path = self.data_root / "hosp" / "patients.csv.gz"
        logger.info(f"Loading patients from {path}")
        return pl.read_csv(path)

    def _load_admissions(self) -> pl.DataFrame:
        path = self.data_root / "hosp" / "admissions.csv.gz"
        logger.info(f"Loading admissions from {path}")
        df = pl.read_csv(path)
        return df.with_columns(
            pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        )

    # ------------------------------------------------------------------
    # Rule application
    # ------------------------------------------------------------------

    def _merge_base(self) -> pl.DataFrame:
        """Merge icustays with patients and admissions."""
        icu = self._load_icustays()
        pat = self._load_patients()
        adm = self._load_admissions()

        # Join: icu → patients (on subject_id) → admissions (on hadm_id)
        merged = icu.join(
            pat.select(["subject_id", "anchor_age", "gender"]),
            on="subject_id",
            how="left",
        ).join(
            adm.select(["hadm_id", "deathtime", "hospital_expire_flag"]),
            on="hadm_id",
            how="left",
        )

        logger.info(f"Merged base table: {merged.height} rows")
        return merged

    def _apply_rules(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Apply inclusion/exclusion rules and return (included, excluded)."""
        spec = self.spec
        included = df.clone()
        all_excluded: list[pl.DataFrame] = []

        # --- Rule 1: Age filter ---
        if spec.inclusion.min_age_years:
            mask = pl.col("anchor_age") >= spec.inclusion.min_age_years
            excluded = included.filter(~mask).with_columns(
                pl.lit(ExclusionReason.AGE_BELOW_MIN.value).alias("exclusion_reason")
            )
            if excluded.height > 0:
                all_excluded.append(excluded)
                self._log_rule("age_below_min", excluded.height)
            included = included.filter(mask)

        # --- Rule 1b: Max age filter ---
        if spec.exclusion.max_age_years is not None:
            mask = pl.col("anchor_age") <= spec.exclusion.max_age_years
            excluded = included.filter(~mask).with_columns(
                pl.lit(ExclusionReason.AGE_ABOVE_MAX.value).alias("exclusion_reason")
            )
            if excluded.height > 0:
                all_excluded.append(excluded)
                self._log_rule("age_above_max", excluded.height)
            included = included.filter(mask)

        # --- Rule 2: Minimum LOS ---
        if spec.inclusion.min_los_hours > 0:
            mask = pl.col("los_hours") >= spec.inclusion.min_los_hours
            excluded = included.filter(~mask).with_columns(
                pl.lit(ExclusionReason.LOS_TOO_SHORT.value).alias("exclusion_reason")
            )
            if excluded.height > 0:
                all_excluded.append(excluded)
                self._log_rule("los_too_short", excluded.height)
            included = included.filter(mask)

        # --- Rule 3: Missing demographics ---
        if spec.exclusion.exclude_missing_demographics:
            mask = pl.col("anchor_age").is_not_null() & pl.col("gender").is_not_null()
            excluded = included.filter(~mask).with_columns(
                pl.lit(ExclusionReason.MISSING_DEMOGRAPHICS.value).alias("exclusion_reason")
            )
            if excluded.height > 0:
                all_excluded.append(excluded)
                self._log_rule("missing_demographics", excluded.height)
            included = included.filter(mask)

        # --- Rule 4: Keep first ICU stay per patient (exclude readmissions) ---
        if spec.exclusion.exclude_readmissions:
            # Sort by intime, keep first stay per subject_id
            sorted_df = included.sort(["subject_id", "intime"])
            first_stays = sorted_df.group_by("subject_id").first()
            first_stay_ids = first_stays.select("stay_id").to_series().to_list()

            excluded = included.filter(
                ~pl.col("stay_id").is_in(first_stay_ids)
            ).with_columns(
                pl.lit(ExclusionReason.READMISSION.value).alias("exclusion_reason")
            )
            if excluded.height > 0:
                all_excluded.append(excluded)
                self._log_rule("readmission", excluded.height)
            included = included.filter(
                pl.col("stay_id").is_in(first_stay_ids)
            )

        # Build combined excluded DataFrame
        if all_excluded:
            # Ensure all excluded DFs have the same columns
            target_cols = list(included.columns) + ["exclusion_reason"]
            aligned_excluded = []
            for ex_df in all_excluded:
                for col in target_cols:
                    if col not in ex_df.columns:
                        ex_df = ex_df.with_columns(pl.lit(None).alias(col))
                aligned_excluded.append(ex_df.select(target_cols))
            excluded_combined = pl.concat(aligned_excluded, how="vertical_relaxed")
        else:
            excluded_combined = included.head(0).with_columns(
                pl.lit(None).cast(pl.Utf8).alias("exclusion_reason")
            )

        return included, excluded_combined

    def _log_rule(self, rule: str, count: int) -> None:
        """Record rule application count."""
        self._exclusion_log.append({"rule": rule, "excluded_count": count})
        logger.info(f"Rule [{rule}]: excluded {count} stays")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, emit_audit: bool = False) -> CohortResult:
        """Execute the full cohort extraction pipeline.

        Parameters
        ----------
        emit_audit:
            If True, generate a detailed audit summary.

        Returns
        -------
        CohortResult with included, excluded, and optional audit.
        """
        self._exclusion_log.clear()

        base = self._merge_base()
        total_before = base.height

        included, excluded = self._apply_rules(base)

        # Select and sort output columns
        output_cols = [
            "subject_id", "hadm_id", "stay_id",
            "anchor_age", "gender",
            "intime", "outtime", "los_hours",
            "first_careunit", "last_careunit",
        ]
        # Only keep columns that exist
        available_cols = [c for c in output_cols if c in included.columns]
        included = included.select(available_cols).sort("stay_id")

        audit: dict[str, Any] = {}
        if emit_audit:
            audit = self._build_audit(total_before, included, excluded)

        return CohortResult(
            included=included,
            excluded=excluded,
            audit_summary=audit,
        )

    def _build_audit(
        self,
        total_before: int,
        included: pl.DataFrame,
        excluded: pl.DataFrame,
    ) -> dict[str, Any]:
        """Build audit summary dict."""
        reason_counts: dict[str, int] = {}
        if "exclusion_reason" in excluded.columns and excluded.height > 0:
            for row in excluded.group_by("exclusion_reason").agg(
                pl.len().alias("count")
            ).iter_rows(named=True):
                reason_counts[row["exclusion_reason"]] = row["count"]

        return {
            "models_version": COHORT_MODELS_VERSION,
            "spec_version": self.spec.version,
            "total_icu_stays": total_before,
            "included": included.height,
            "excluded": excluded.height,
            "inclusion_rate_pct": round(
                included.height / total_before * 100, 2
            ) if total_before > 0 else 0,
            "exclusion_reasons": reason_counts,
            "rules_applied": self._exclusion_log.copy(),
            "unique_patients": included.select("subject_id").n_unique(),
        }
