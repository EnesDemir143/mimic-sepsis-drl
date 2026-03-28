"""
build_episode_grid — CLI entrypoint for episode window materialisation.

Usage
-----
    # Dry run — validate onset outputs and print grid summary
    python -m mimic_sepsis_rl.cli.build_episode_grid --dry-run

    # Full grid generation
    python -m mimic_sepsis_rl.cli.build_episode_grid

The CLI performs these steps:
  1. Load onset assignments (Phase 2, Plan 01 output).
  2. Load ICU stays and optionally admissions for death times.
  3. Build deterministic 4-hour episode grids.
  4. Write episodes and steps Parquet files.
  5. Write grid audit summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

from mimic_sepsis_rl.data.episodes import (
    build_episode_grids,
    generate_grid_audit,
    grids_to_dataframes,
)
from mimic_sepsis_rl.data.episode_models import (
    EPISODE_SPEC_VERSION,
    MAX_STEPS,
    STEP_HOURS,
    WINDOW_END_HOURS,
    WINDOW_START_HOURS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

MIMIC_RAW_ROOT = Path("data/raw/physionet.org/files/mimiciv/3.1")

DEFAULT_ONSET_PATH = Path("data/processed/onset/onset_assignments.parquet")
DEFAULT_EPISODES_PATH = Path("data/processed/episodes/episodes.parquet")
DEFAULT_STEPS_PATH = Path("data/processed/episodes/episode_steps.parquet")
DEFAULT_AUDIT_PATH = Path("data/processed/episodes/grid_audit.json")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run() -> None:
    """Print episode grid configuration summary."""
    print("=" * 70)
    print("  MIMIC Sepsis Offline RL — Episode Grid Build (DRY RUN)")
    print("=" * 70)
    print(f"\nSpec version  : {EPISODE_SPEC_VERSION}")
    print(f"Step size     : {STEP_HOURS} hours")
    print(f"Window start  : onset {WINDOW_START_HOURS:+d}h")
    print(f"Window end    : onset +{WINDOW_END_HOURS}h")
    print(f"Max steps     : {MAX_STEPS}")
    print(f"\nOnset input   : {DEFAULT_ONSET_PATH}")
    print(f"Episodes out  : {DEFAULT_EPISODES_PATH}")
    print(f"Steps out     : {DEFAULT_STEPS_PATH}")
    print(f"Audit out     : {DEFAULT_AUDIT_PATH}")

    if DEFAULT_ONSET_PATH.exists():
        onset_df = pl.read_parquet(DEFAULT_ONSET_PATH)
        print(f"\nOnset file found: {onset_df.height} usable episodes")
        print(f"Columns: {onset_df.columns}")
    else:
        print(f"\n⚠ Onset file not found at {DEFAULT_ONSET_PATH}")
        print("  Run onset assignment first (Phase 2, Plan 01).")

    print("\n[DRY RUN] No grid generated.\n")


# ---------------------------------------------------------------------------
# Live build
# ---------------------------------------------------------------------------


def _run_build() -> int:
    """Build episode grids from onset assignments."""
    if not DEFAULT_ONSET_PATH.exists():
        print(f"ERROR: Onset file not found: {DEFAULT_ONSET_PATH}", file=sys.stderr)
        print("Run onset assignment first (Phase 2, Plan 01).", file=sys.stderr)
        return 1

    # Load onset assignments
    onset_df = pl.read_parquet(DEFAULT_ONSET_PATH)
    logger.info(f"Loaded {onset_df.height} onset assignments")

    # Load ICU stays
    icustays_path = MIMIC_RAW_ROOT / "icu" / "icustays.csv.gz"
    if not icustays_path.exists():
        print(f"ERROR: ICU stays not found: {icustays_path}", file=sys.stderr)
        return 1

    icustays = pl.read_csv(icustays_path).with_columns(
        pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    )

    # Load admissions for death times
    admissions_path = MIMIC_RAW_ROOT / "hosp" / "admissions.csv.gz"
    admissions = None
    if admissions_path.exists():
        admissions = pl.read_csv(admissions_path).with_columns(
            pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        )
        logger.info(f"Loaded admissions with death times")

    # Build grids
    logger.info("Building episode grids...")
    grids = build_episode_grids(onset_df, icustays, admissions)
    logger.info(f"Built {len(grids)} episode grids")

    # Convert to DataFrames
    episodes_df, steps_df = grids_to_dataframes(grids)
    audit = generate_grid_audit(grids)

    # Write outputs
    for p in [DEFAULT_EPISODES_PATH, DEFAULT_STEPS_PATH, DEFAULT_AUDIT_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)

    episodes_df.write_parquet(DEFAULT_EPISODES_PATH)
    steps_df.write_parquet(DEFAULT_STEPS_PATH)
    with DEFAULT_AUDIT_PATH.open("w") as fh:
        json.dump(audit, fh, indent=2, default=str)

    print(f"\nEpisodes  → {DEFAULT_EPISODES_PATH}  ({episodes_df.height} episodes)")
    print(f"Steps     → {DEFAULT_STEPS_PATH}  ({steps_df.height} step rows)")
    print(f"Audit     → {DEFAULT_AUDIT_PATH}")
    print(f"\nGrid Summary:")
    print(f"  Full (18 steps)  : {audit['full_length']}")
    print(f"  Truncated        : {audit['truncated']} ({audit['truncated_pct']}%)")
    print(f"  Avg steps        : {audit['avg_steps']}")
    print(f"  Step range       : [{audit['min_steps']}, {audit['max_steps']}]")

    return 0


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_episode_grid",
        description=(
            "Build deterministic 4-hour episode grids from onset assignments.\n"
            "Use --dry-run to inspect configuration without generating grids."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid config and exit without generating data.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dry_run:
        _dry_run()
        return 0

    return _run_build()


if __name__ == "__main__":
    sys.exit(main())
