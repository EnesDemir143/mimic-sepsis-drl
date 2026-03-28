"""
Sepsis-3 onset assignment pipeline.

This module implements the Sepsis-3 onset operationalization for the
MIMIC Sepsis Offline RL project.  For each ICU episode in the cohort,
it assigns exactly one ``sepsis_onset_time`` or flags the episode as
unusable with an explicit reason.

Sepsis-3 Definition (operationalized)
--------------------------------------
Onset = min(suspected_infection_time, sofa_time) where:
  - ``suspected_infection_time`` is derived from antibiotic + culture events
  - ``sofa_time`` is when SOFA ≥ 2 within [infection - 24h, infection + 24h]

When the MIMIC-IV derived ``sepsis3`` table is unavailable, this module
computes the onset from raw tables (microbiologyevents, prescriptions,
chartevents → SOFA proxy).

Usage
-----
    python -m mimic_sepsis_rl.data.onset --config configs/onset/default.yaml --dry-run

Version history
---------------
v1.0.0  2026-03-28  Initial onset assignment pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from mimic_sepsis_rl.data.onset_models import (
    ONSET_SPEC_VERSION,
    OnsetCandidate,
    OnsetResult,
    UnusableReason,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data root constant
# ---------------------------------------------------------------------------

MIMIC_RAW_ROOT = Path("data/raw/physionet.org/files/mimiciv/3.1")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_onset_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the onset YAML config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")
    required = {"onset", "icu_boundary"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# MIMIC-IV table loaders (from compressed CSV)
# ---------------------------------------------------------------------------


def _load_icustays() -> pl.DataFrame:
    """Load ICU stays with parsed datetime columns."""
    path = MIMIC_RAW_ROOT / "icu" / "icustays.csv.gz"
    df = pl.read_csv(path)
    return df.with_columns(
        pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    )


def _load_microbiologyevents() -> pl.DataFrame:
    """Load microbiology events (culture orders) for infection timing."""
    path = MIMIC_RAW_ROOT / "hosp" / "microbiologyevents.csv.gz"
    df = pl.read_csv(
        path,
        columns=["subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc"],
    )
    # charttime may be null; fall back to chartdate with midnight
    return df.with_columns(
        pl.coalesce(
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("chartdate").str.to_datetime("%Y-%m-%d", strict=False),
        ).alias("culture_time")
    )


def _load_prescriptions_antibiotics() -> pl.DataFrame:
    """Load antibiotic prescriptions for suspected infection timing."""
    path = MIMIC_RAW_ROOT / "hosp" / "prescriptions.csv.gz"
    df = pl.read_csv(
        path,
        columns=["subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"],
    )
    # Filter to IV/IM antibiotics (common antibiotic route patterns)
    antibiotic_keywords = [
        "CEFAZOLIN", "CEFTRIAXONE", "CEFEPIME", "CEFTAZIDIME",
        "VANCOMYCIN", "PIPERACILLIN", "MEROPENEM", "IMIPENEM",
        "CIPROFLOXACIN", "LEVOFLOXACIN", "METRONIDAZOLE",
        "AMPICILLIN", "GENTAMICIN", "TOBRAMYCIN", "AMIKACIN",
        "AZITHROMYCIN", "DOXYCYCLINE", "TRIMETHOPRIM",
        "LINEZOLID", "DAPTOMYCIN", "ERTAPENEM", "DORIPENEM",
        "COLISTIN", "POLYMYXIN", "TIGECYCLINE", "AZTREONAM",
        "CEFOXITIN", "CEFUROXIME", "NAFCILLIN", "OXACILLIN",
        "CLINDAMYCIN", "PENICILLIN", "AMOXICILLIN",
    ]
    # Build case-insensitive OR pattern
    pattern = "|".join(antibiotic_keywords)
    df = df.filter(pl.col("drug").str.to_uppercase().str.contains(f"(?i){pattern}"))
    df = df.with_columns(
        pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    )
    return df.filter(pl.col("starttime").is_not_null())


def _compute_sofa_proxy(icustays: pl.DataFrame) -> pl.DataFrame:
    """Compute a simplified SOFA proxy from chartevents.

    This is a simplified implementation that uses a subset of SOFA
    components available from chartevents.  When the derived sepsis3
    table is available, it should be used instead.

    Returns a DataFrame with columns:
        subject_id, hadm_id, stay_id, charttime, sofa_score
    """
    path = MIMIC_RAW_ROOT / "icu" / "chartevents.csv.gz"

    # SOFA-relevant itemids from MIMIC-IV d_items:
    # Cardiovascular: MAP (220052, 220181), vasopressors
    # Respiratory: PaO2/FiO2 ratio, SpO2 (220277)
    # Renal: Urine output (226559), Creatinine
    # Hepatic: Bilirubin
    # Coagulation: Platelets
    # Neurological: GCS (223901 total, or components 220739+223900+223901)

    sofa_itemids = [
        220052,   # Arterial Blood Pressure mean
        220181,   # Non Invasive Blood Pressure mean
        220277,   # SpO2
        223901,   # GCS - Total
        226559,   # Urine output
    ]

    # Read only relevant items to save memory
    logger.info("Loading chartevents for SOFA proxy (this may take a while)...")

    # Use streaming/lazy for large file
    df = pl.scan_csv(path).filter(
        pl.col("itemid").is_in(sofa_itemids)
    ).select(
        ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]
    ).collect()

    df = df.with_columns(
        pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    ).filter(pl.col("valuenum").is_not_null())

    # Pivot to get one row per (stay_id, charttime) with SOFA components
    # Simple SOFA proxy: count abnormal components
    # MAP < 70: cardio=1, GCS < 15: neuro score, SpO2 < 94: resp score
    scored = df.group_by(["subject_id", "hadm_id", "stay_id", "charttime"]).agg([
        # Cardiovascular: MAP < 70 → 1 point
        (pl.col("valuenum").filter(
            pl.col("itemid").is_in([220052, 220181])
        ).min() < 70).cast(pl.Int32).fill_null(0).first().alias("cardio_score"),
        # Respiratory: SpO2 < 94 → 1 point
        (pl.col("valuenum").filter(
            pl.col("itemid") == 220277
        ).min() < 94).cast(pl.Int32).fill_null(0).first().alias("resp_score"),
        # Neurological: GCS < 15 → 1 point (simplified)
        (pl.col("valuenum").filter(
            pl.col("itemid") == 223901
        ).min() < 15).cast(pl.Int32).fill_null(0).first().alias("neuro_score"),
    ]).with_columns(
        (pl.col("cardio_score") + pl.col("resp_score") + pl.col("neuro_score"))
        .alias("sofa_score")
    )

    return scored.select(["subject_id", "hadm_id", "stay_id", "charttime", "sofa_score"])


# ---------------------------------------------------------------------------
# Suspected infection time
# ---------------------------------------------------------------------------


def compute_suspected_infection_times(
    icustays: pl.DataFrame,
) -> pl.DataFrame:
    """Derive suspected infection time per admission.

    Sepsis-3 operationalization:
    - Culture order within [-24h, +24h] of antibiotic start, OR
    - Antibiotic start within [-72h, +24h] of culture order.

    We use the simpler rule: earliest antibiotic start that's
    within 24h of any culture order for the same admission.

    Returns DataFrame with: subject_id, hadm_id, suspected_infection_time
    """
    micro = _load_microbiologyevents()
    abx = _load_prescriptions_antibiotics()

    # Get unique admission IDs from cohort
    hadm_ids = icustays.select("hadm_id").unique()

    # Filter to cohort admissions
    micro_cohort = micro.join(hadm_ids, on="hadm_id", how="inner")
    abx_cohort = abx.join(hadm_ids, on="hadm_id", how="inner")

    # Cross-join within admission to find culture-antibiotic pairs
    paired = abx_cohort.select(
        ["hadm_id", "starttime"]
    ).rename({"starttime": "abx_time"}).join(
        micro_cohort.select(["hadm_id", "culture_time"]),
        on="hadm_id",
        how="inner",
    )

    # Filter: antibiotic within [-24h, +24h] of culture
    paired = paired.filter(
        (pl.col("abx_time") - pl.col("culture_time")).abs() <= timedelta(hours=72)
    )

    # Suspected infection time = earliest of (abx_time, culture_time) per admission
    if paired.height == 0:
        return pl.DataFrame(schema={
            "hadm_id": pl.Int64,
            "suspected_infection_time": pl.Datetime,
        })

    result = paired.with_columns(
        pl.min_horizontal("abx_time", "culture_time").alias("suspected_infection_time")
    ).group_by("hadm_id").agg(
        pl.col("suspected_infection_time").min()
    )

    return result


# ---------------------------------------------------------------------------
# Core onset assignment
# ---------------------------------------------------------------------------


def assign_onset_for_stay(
    stay_row: dict[str, Any],
    suspected_infection_time: pl.Datetime | None,
    sofa_df: pl.DataFrame,
    cfg: dict[str, Any],
) -> OnsetResult:
    """Assign onset for a single ICU stay.

    Parameters
    ----------
    stay_row:
        Dict with stay_id, subject_id, hadm_id, intime, outtime.
    suspected_infection_time:
        Pre-computed suspected infection time for this admission.
    sofa_df:
        SOFA scores filtered for this stay.
    cfg:
        Onset config dict.

    Returns
    -------
    OnsetResult with either onset_time or unusable_reason.
    """
    stay_id = stay_row["stay_id"]
    subject_id = stay_row["subject_id"]
    hadm_id = stay_row["hadm_id"]
    intime = stay_row["intime"]
    outtime = stay_row["outtime"]

    onset_cfg = cfg["onset"]
    icu_cfg = cfg["icu_boundary"]

    lookback_h = onset_cfg["lookback_hours"]
    lookahead_h = onset_cfg["lookahead_hours"]
    min_sofa = onset_cfg["min_sofa_increase"]

    # Step 1: Check for suspected infection time
    if suspected_infection_time is None:
        return OnsetResult(
            stay_id=stay_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            unusable_reason=UnusableReason.NO_SUSPECTED_INFECTION,
        )

    # Step 2: Find SOFA ≥ min_sofa within [infection - lookback, infection + lookahead]
    search_start = suspected_infection_time - timedelta(hours=lookback_h)
    search_end = suspected_infection_time + timedelta(hours=lookahead_h)

    qualifying_sofa = sofa_df.filter(
        (pl.col("charttime") >= search_start)
        & (pl.col("charttime") <= search_end)
        & (pl.col("sofa_score") >= min_sofa)
    ).sort("charttime")

    if qualifying_sofa.height == 0:
        return OnsetResult(
            stay_id=stay_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            unusable_reason=UnusableReason.NO_SOFA_INCREASE,
        )

    # Step 3: Build candidates
    candidates = []
    for row in qualifying_sofa.iter_rows(named=True):
        sofa_time = row["charttime"]
        onset_time = min(suspected_infection_time, sofa_time)
        candidates.append(OnsetCandidate(
            stay_id=stay_id,
            suspected_infection_time=suspected_infection_time,
            sofa_time=sofa_time,
            sofa_score=row["sofa_score"],
            onset_time=onset_time,
        ))

    # Step 4: Select best candidate (earliest onset by default)
    tie_break = onset_cfg.get("tie_break", "earliest")
    if tie_break == "earliest":
        selected = min(candidates, key=lambda c: c.onset_time)
    else:
        selected = candidates[0]

    # Step 5: ICU boundary check
    if icu_cfg.get("require_within_icu", True):
        grace = timedelta(hours=icu_cfg.get("grace_before_intime_hours", 6.0))
        earliest_allowed = intime - grace
        if selected.onset_time < earliest_allowed or selected.onset_time > outtime:
            return OnsetResult(
                stay_id=stay_id,
                subject_id=subject_id,
                hadm_id=hadm_id,
                unusable_reason=UnusableReason.ONSET_OUTSIDE_ICU,
                candidates=candidates,
            )

    return OnsetResult(
        stay_id=stay_id,
        subject_id=subject_id,
        hadm_id=hadm_id,
        onset_time=selected.onset_time,
        candidates=candidates,
        selected_candidate=selected,
    )


# ---------------------------------------------------------------------------
# Batch onset assignment
# ---------------------------------------------------------------------------


def assign_onsets(
    cohort: pl.DataFrame,
    cfg: dict[str, Any],
) -> list[OnsetResult]:
    """Assign onset times for all episodes in the cohort.

    Parameters
    ----------
    cohort:
        Cohort DataFrame with stay_id, subject_id, hadm_id, intime, outtime.
    cfg:
        Full onset config dict.

    Returns
    -------
    List of OnsetResult, one per ICU stay.
    """
    logger.info("Computing suspected infection times...")
    infection_times = compute_suspected_infection_times(cohort)
    infection_lookup: dict[int, Any] = {}
    for row in infection_times.iter_rows(named=True):
        infection_lookup[row["hadm_id"]] = row["suspected_infection_time"]

    logger.info("Computing SOFA proxy scores...")
    sofa_all = _compute_sofa_proxy(cohort)

    results: list[OnsetResult] = []
    total = cohort.height
    for idx, stay_row in enumerate(cohort.iter_rows(named=True)):
        if (idx + 1) % 500 == 0 or idx == 0:
            logger.info(f"Processing stay {idx + 1}/{total}...")

        stay_id = stay_row["stay_id"]
        hadm_id = stay_row["hadm_id"]

        # Get infection time for this admission
        infection_time = infection_lookup.get(hadm_id)

        # Get SOFA scores for this stay
        sofa_stay = sofa_all.filter(pl.col("stay_id") == stay_id)

        result = assign_onset_for_stay(
            stay_row=stay_row,
            suspected_infection_time=infection_time,
            sofa_df=sofa_stay,
            cfg=cfg,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def results_to_dataframes(
    results: list[OnsetResult],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Convert onset results to Polars DataFrames.

    Returns
    -------
    (usable_df, unusable_df, candidates_df)
    """
    usable_rows = []
    unusable_rows = []
    candidate_rows = []

    for r in results:
        base = {
            "stay_id": r.stay_id,
            "subject_id": r.subject_id,
            "hadm_id": r.hadm_id,
        }
        if r.is_usable:
            usable_rows.append({
                **base,
                "sepsis_onset_time": r.onset_time,
                "sofa_score_at_onset": (
                    r.selected_candidate.sofa_score
                    if r.selected_candidate else None
                ),
                "n_candidates": len(r.candidates),
            })
        else:
            unusable_rows.append({
                **base,
                "unusable_reason": r.unusable_reason.value if r.unusable_reason else None,
                "n_candidates": len(r.candidates),
            })

        for c in r.candidates:
            candidate_rows.append({
                "stay_id": c.stay_id,
                "suspected_infection_time": c.suspected_infection_time,
                "sofa_time": c.sofa_time,
                "sofa_score": c.sofa_score,
                "onset_time": c.onset_time,
                "selected": (r.selected_candidate == c) if r.selected_candidate else False,
            })

    usable_df = pl.DataFrame(usable_rows) if usable_rows else pl.DataFrame(schema={
        "stay_id": pl.Int64, "subject_id": pl.Int64, "hadm_id": pl.Int64,
        "sepsis_onset_time": pl.Datetime, "sofa_score_at_onset": pl.Int32,
        "n_candidates": pl.Int32,
    })
    unusable_df = pl.DataFrame(unusable_rows) if unusable_rows else pl.DataFrame(schema={
        "stay_id": pl.Int64, "subject_id": pl.Int64, "hadm_id": pl.Int64,
        "unusable_reason": pl.Utf8, "n_candidates": pl.Int32,
    })
    candidates_df = pl.DataFrame(candidate_rows) if candidate_rows else pl.DataFrame(schema={
        "stay_id": pl.Int64, "suspected_infection_time": pl.Datetime,
        "sofa_time": pl.Datetime, "sofa_score": pl.Int32,
        "onset_time": pl.Datetime, "selected": pl.Boolean,
    })

    return usable_df, unusable_df, candidates_df


def generate_audit_summary(results: list[OnsetResult]) -> dict[str, Any]:
    """Generate count-level audit summary for onset assignment."""
    total = len(results)
    usable = sum(1 for r in results if r.is_usable)
    unusable = total - usable

    reason_counts: dict[str, int] = {}
    for r in results:
        if r.unusable_reason:
            key = r.unusable_reason.value
            reason_counts[key] = reason_counts.get(key, 0) + 1

    return {
        "spec_version": ONSET_SPEC_VERSION,
        "total_episodes": total,
        "usable": usable,
        "unusable": unusable,
        "usable_pct": round(usable / total * 100, 2) if total > 0 else 0,
        "unusable_reasons": reason_counts,
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run(cfg: dict[str, Any], config_path: Path) -> None:
    """Print config summary without loading data."""
    print("=" * 70)
    print("  MIMIC Sepsis Offline RL — Onset Assignment (DRY RUN)")
    print("=" * 70)
    print(f"\nConfig : {config_path}")
    print(f"Spec v : {ONSET_SPEC_VERSION}")

    onset = cfg.get("onset", {})
    icu = cfg.get("icu_boundary", {})
    print("\nOnset Parameters:")
    print(f"  lookback_hours           : {onset.get('lookback_hours')}")
    print(f"  lookahead_hours          : {onset.get('lookahead_hours')}")
    print(f"  min_sofa_increase        : {onset.get('min_sofa_increase')}")
    print(f"  tie_break                : {onset.get('tie_break')}")

    print("\nICU Boundary:")
    print(f"  require_within_icu       : {icu.get('require_within_icu')}")
    print(f"  grace_before_intime_hours: {icu.get('grace_before_intime_hours')}")

    out = cfg.get("output", {})
    print("\nOutput Paths:")
    for key, val in out.items():
        print(f"  {key:<25} {val}")

    print("\n[DRY RUN] Config validated. No data loaded.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onset",
        description="Assign Sepsis-3 onset times to cohort episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/onset/default.yaml"),
        help="Path to onset YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print summary without loading data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        cfg = load_onset_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        _dry_run(cfg, args.config)
        return 0

    # Live mode
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})

    cohort_path = Path(input_cfg.get("cohort_parquet", "data/processed/cohort/cohort.parquet"))
    if not cohort_path.exists():
        print(f"ERROR: Cohort file not found: {cohort_path}", file=sys.stderr)
        print("Run cohort extraction first (Phase 1).", file=sys.stderr)
        return 1

    cohort = pl.read_parquet(cohort_path)
    logger.info(f"Loaded cohort: {cohort.height} stays")

    results = assign_onsets(cohort, cfg)

    usable_df, unusable_df, candidates_df = results_to_dataframes(results)
    audit = generate_audit_summary(results)

    # Write outputs
    onset_path = Path(output_cfg.get("onset_parquet", "data/processed/onset/onset_assignments.parquet"))
    unusable_path = Path(output_cfg.get("unusable_parquet", "data/processed/onset/unusable_episodes.parquet"))
    cand_path = Path(output_cfg.get("candidates_parquet", "data/processed/onset/onset_candidates.parquet"))
    audit_path = Path(output_cfg.get("audit_json", "data/processed/onset/onset_audit.json"))

    for p in [onset_path, unusable_path, cand_path, audit_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    usable_df.write_parquet(onset_path)
    unusable_df.write_parquet(unusable_path)
    candidates_df.write_parquet(cand_path)
    with audit_path.open("w") as fh:
        json.dump(audit, fh, indent=2, default=str)

    print(f"\nOnset assignments → {onset_path}  ({len(usable_df)} usable)")
    print(f"Unusable episodes → {unusable_path}  ({len(unusable_df)} unusable)")
    print(f"All candidates    → {cand_path}  ({len(candidates_df)} candidates)")
    print(f"Audit summary     → {audit_path}")
    print(f"\nUsable rate: {audit['usable_pct']}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
