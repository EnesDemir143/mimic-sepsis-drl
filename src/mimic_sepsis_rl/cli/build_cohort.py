"""
build_cohort — CLI entrypoint for MIMIC-IV Sepsis-3 cohort extraction.

Usage
-----
    # Dry run — validate config and print rule summary, no DB needed
    python -m mimic_sepsis_rl.cli.build_cohort \\
        --config configs/cohort/default.yaml \\
        --dry-run

    # Full extraction with audit output
    python -m mimic_sepsis_rl.cli.build_cohort \\
        --config configs/cohort/default.yaml \\
        --emit-audit

The CLI performs these steps:
  1. Load and validate the YAML config.
  2. Instantiate ``CohortSpec`` from the config values.
  3. In dry-run mode: print rule summary and exit.
  4. In live mode: invoke ``CohortExtractor``, write cohort and audit outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from mimic_sepsis_rl.data.cohort.spec import (
    CohortSpec,
    ExclusionCriteria,
    InclusionCriteria,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> dict:
    """Load and minimally validate the cohort YAML config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")
    # Minimal required keys
    required = {"adult_only", "inclusion", "exclusion"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg


def _spec_from_config(cfg: dict) -> CohortSpec:
    """Instantiate ``CohortSpec`` from a validated config dict."""
    inc = cfg.get("inclusion", {})
    exc = cfg.get("exclusion", {})

    inclusion = InclusionCriteria(
        min_age_years=inc.get("min_age_years", 18),
        require_icu_stay=inc.get("require_icu_stay", True),
        require_sepsis3=inc.get("require_sepsis3", True),
        min_los_hours=float(inc.get("min_los_hours", 4.0)),
    )
    exclusion = ExclusionCriteria(
        exclude_missing_sepsis_anchor=exc.get("exclude_missing_sepsis_anchor", True),
        exclude_readmissions=exc.get("exclude_readmissions", True),
        exclude_missing_demographics=exc.get("exclude_missing_demographics", True),
        max_age_years=exc.get("max_age_years"),
    )

    source_tables = cfg.get("source_tables", {})

    return CohortSpec(
        adult_only=cfg.get("adult_only", True),
        inclusion=inclusion,
        exclusion=exclusion,
        source_tables=source_tables if source_tables else CohortSpec().source_tables,
    )


# ---------------------------------------------------------------------------
# Dry-run handler
# ---------------------------------------------------------------------------


def _dry_run(spec: CohortSpec, config_path: Path) -> None:
    """Print the cohort rule summary and exit without touching the database."""
    print("=" * 70)
    print("  MIMIC Sepsis Offline RL — Cohort Build (DRY RUN)")
    print("=" * 70)
    print(f"\nConfig : {config_path}")
    print(f"Spec v : {spec.version}")
    print(f"\nDescription:\n  {spec.description}")
    print("\nActive Rules:")
    for key, value in spec.rule_summary().items():
        print(f"  {key:<40} {value}")
    print("\nSource Tables:")
    for alias, table in spec.source_tables.items():
        print(f"  {alias:<15} {table}")
    print("\n[DRY RUN] No database connection made. Config validated successfully.\n")


# ---------------------------------------------------------------------------
# Live extraction stub
# ---------------------------------------------------------------------------


def _run_extraction(spec: CohortSpec, cfg: dict, emit_audit: bool) -> None:
    """Orchestrate cohort extraction (live mode).

    In Phase 1 this stub validates that the extraction pipeline can be wired
    together.  Actual SQL execution is completed in Plan 01-02 (extract.py).
    """
    from mimic_sepsis_rl.data.cohort.extract import CohortExtractor

    out_cfg = cfg.get("output", {})
    cohort_path = Path(out_cfg.get("cohort_parquet", "data/processed/cohort/cohort.parquet"))
    audit_path = Path(out_cfg.get("audit_json", "data/processed/cohort/audit.json"))
    excluded_path = Path(out_cfg.get("excluded_parquet", "data/processed/cohort/excluded.parquet"))

    extractor = CohortExtractor(spec=spec)
    result = extractor.run(emit_audit=emit_audit)

    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    result.included.write_parquet(cohort_path)
    result.excluded.write_parquet(excluded_path)

    if emit_audit:
        with audit_path.open("w") as fh:
            json.dump(result.audit_summary, fh, indent=2, default=str)

    print(f"Cohort    → {cohort_path}  ({len(result.included)} stays)")
    print(f"Excluded  → {excluded_path}  ({len(result.excluded)} stays)")
    if emit_audit:
        print(f"Audit     → {audit_path}")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_cohort",
        description=(
            "Build the MIMIC-IV Sepsis-3 adult ICU cohort from a YAML config.\n"
            "Use --dry-run to validate config without a database connection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cohort/default.yaml"),
        help="Path to cohort YAML config (default: configs/cohort/default.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print rule summary without connecting to the database.",
    )
    parser.add_argument(
        "--emit-audit",
        action="store_true",
        help="Write per-stay inclusion and exclusion audit records alongside the cohort.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = _load_config(args.config)
        spec = _spec_from_config(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        _dry_run(spec, args.config)
        return 0

    try:
        _run_extraction(spec, cfg, emit_audit=args.emit_audit)
    except ImportError as exc:
        print(f"ERROR: extraction module not ready — {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
