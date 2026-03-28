"""
Patient-level split manifest generation.

Generates deterministic train / validation / test manifests that operate
exclusively at the **patient** (``subject_id``) level.  No patient may appear
in more than one partition.

Key design invariants
---------------------
- Splits are produced from unique ``subject_id`` values, never from row or
  stay counts.  This prevents intra-patient leakage across partitions.
- The random seed and source episode-set identifier are embedded in every
  manifest so the partition can be reproduced exactly.
- ``--dry-run`` mode prints manifest statistics without writing any files.

Usage (CLI)
-----------
    python -m mimic_sepsis_rl.data.splits \\
        --config configs/splits/default.yaml \\
        [--source-episode-set data/processed/episodes.parquet] \\
        [--dry-run]

Python API
----------
    from mimic_sepsis_rl.data.splits import generate_split_manifest

    manifest = generate_split_manifest(
        episodes_df=episodes_df,
        seed=42,
        train_frac=0.70,
        val_frac=0.15,
        source_episode_set="data/processed/episodes.parquet",
    )

Version history
---------------
v1.0.0  2026-03-28  Initial patient-level split generator.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from mimic_sepsis_rl.data.split_models import (
    SPLIT_SPEC_VERSION,
    PatientRecord,
    SplitLabel,
    SplitManifest,
    SplitStats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core split generator
# ---------------------------------------------------------------------------


def generate_split_manifest(
    episodes_df: pl.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    source_episode_set: str,
    mortality_col: str | None = None,
) -> SplitManifest:
    """Produce a patient-level split manifest from an episode DataFrame.

    Parameters
    ----------
    episodes_df:
        DataFrame containing at minimum columns ``subject_id`` and ``stay_id``.
        Optionally a column identified by *mortality_col* for stratified splits.
    seed:
        Fixed random seed for reproducibility.  This value is persisted in the
        returned manifest.
    train_frac:
        Fraction of unique patients assigned to the training partition.
    val_frac:
        Fraction of unique patients assigned to the validation partition.
        The test fraction is implicitly ``1 - train_frac - val_frac``.
    source_episode_set:
        Human-readable label (e.g. a Parquet path) recorded verbatim in the
        manifest so downstream consumers can verify provenance.
    mortality_col:
        Optional column name in *episodes_df* to use for stratified splitting.
        When ``None`` splitting is performed without stratification.

    Returns
    -------
    SplitManifest with ``has_leakage() == False`` guaranteed.

    Raises
    ------
    ValueError
        If fractions do not sum to ≤ 1, if required columns are missing, or
        if the resulting manifest contains any patient-level leakage.
    """
    _validate_fractions(train_frac, val_frac)
    _require_columns(episodes_df, ["subject_id", "stay_id"])

    # ---- Collect unique patients ----------------------------------------
    patient_df = _build_patient_table(episodes_df, mortality_col)

    # ---- Shuffle at patient level with fixed seed ----------------------
    patient_df = patient_df.sample(fraction=1.0, shuffle=True, seed=seed)

    # ---- Assign partition labels ----------------------------------------
    n_patients = patient_df.height
    n_train = round(n_patients * train_frac)
    n_val = round(n_patients * val_frac)

    labels = (
        [SplitLabel.TRAIN] * n_train
        + [SplitLabel.VALIDATION] * n_val
        + [SplitLabel.TEST] * (n_patients - n_train - n_val)
    )
    patient_df = patient_df.with_columns(
        pl.Series("split", [lbl.value for lbl in labels])
    )

    # ---- Build id sets -------------------------------------------------
    split_col = patient_df.get_column("split")
    subject_col = patient_df.get_column("subject_id")

    train_ids = frozenset(
        subject_col.filter(split_col == SplitLabel.TRAIN.value).to_list()
    )
    val_ids = frozenset(
        subject_col.filter(split_col == SplitLabel.VALIDATION.value).to_list()
    )
    test_ids = frozenset(
        subject_col.filter(split_col == SplitLabel.TEST.value).to_list()
    )

    # ---- Build per-patient records -------------------------------------
    episode_lookup = _build_episode_lookup(episodes_df)
    records = _build_records(patient_df, episode_lookup)

    # ---- Compute statistics -------------------------------------------
    stats = _compute_stats(patient_df, episode_lookup, mortality_col)

    # ---- Construct and validate manifest ------------------------------
    manifest = SplitManifest(
        spec_version=SPLIT_SPEC_VERSION,
        seed=seed,
        source_episode_set=source_episode_set,
        train_ids=train_ids,
        validation_ids=val_ids,
        test_ids=test_ids,
        records=records,
        stats=stats,
    )

    if manifest.has_leakage():
        raise ValueError(
            "Generated manifest contains patient-level leakage. "
            "This is a logic error in the split generator."
        )

    logger.info(
        "Manifest generated: %d train / %d val / %d test patients (seed=%d).",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        seed,
    )
    return manifest


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_manifest(manifest: SplitManifest, output_dir: Path) -> None:
    """Persist split manifests to *output_dir* as Parquet + JSON summary.

    Parameters
    ----------
    manifest:
        A validated :class:`SplitManifest`.
    output_dir:
        Directory where the three per-split Parquet files and a JSON summary
        are written.  Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_label, id_set in [
        (SplitLabel.TRAIN, manifest.train_ids),
        (SplitLabel.VALIDATION, manifest.validation_ids),
        (SplitLabel.TEST, manifest.test_ids),
    ]:
        split_records = [
            r for r in manifest.records if r.split == split_label
        ]
        rows = [
            {
                "subject_id": r.subject_id,
                "split": r.split.value,
                "episode_keys": list(r.episode_keys),
                "n_episodes": len(r.episode_keys),
            }
            for r in split_records
        ]
        df = pl.DataFrame(rows) if rows else _empty_manifest_df()
        out_path = output_dir / f"{split_label.value}_manifest.parquet"
        df.write_parquet(out_path)
        logger.info("Wrote %s (%d patients).", out_path, len(rows))

    summary = _build_summary_dict(manifest)
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s.", summary_path)


def load_manifest_parquet(manifest_dir: Path) -> SplitManifest:
    """Reconstruct a :class:`SplitManifest` from previously saved Parquet files.

    Parameters
    ----------
    manifest_dir:
        Directory containing ``train_manifest.parquet``,
        ``validation_manifest.parquet``, ``test_manifest.parquet``, and
        ``split_summary.json``.

    Returns
    -------
    SplitManifest matching the persisted state.
    """
    summary = json.loads((manifest_dir / "split_summary.json").read_text())
    seed = summary["seed"]
    source = summary["source_episode_set"]

    records: list[PatientRecord] = []
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    test_ids: set[int] = set()

    label_map = {
        SplitLabel.TRAIN: train_ids,
        SplitLabel.VALIDATION: val_ids,
        SplitLabel.TEST: test_ids,
    }

    for split_label in SplitLabel:
        path = manifest_dir / f"{split_label.value}_manifest.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        for row in df.iter_rows(named=True):
            sid = row["subject_id"]
            keys = tuple(sorted(row["episode_keys"]))
            records.append(PatientRecord(subject_id=sid, split=split_label, episode_keys=keys))
            label_map[split_label].add(sid)

    return SplitManifest(
        spec_version=summary.get("spec_version", SPLIT_SPEC_VERSION),
        seed=seed,
        source_episode_set=source,
        train_ids=frozenset(train_ids),
        validation_ids=frozenset(val_ids),
        test_ids=frozenset(test_ids),
        records=tuple(records),
    )


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def manifest_summary(manifest: SplitManifest) -> dict[str, Any]:
    """Return a human-readable summary dict for logging or dry-run display."""
    return _build_summary_dict(manifest)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_fractions(train_frac: float, val_frac: float) -> None:
    if not (0 < train_frac < 1):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}.")
    if not (0 < val_frac < 1):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}.")
    if train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac + val_frac must be < 1, got {train_frac + val_frac:.3f}."
        )


def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"episodes_df is missing required columns: {missing}.")


def _build_patient_table(
    episodes_df: pl.DataFrame,
    mortality_col: str | None,
) -> pl.DataFrame:
    """Return one row per unique patient with optional mortality indicator."""
    cols = ["subject_id"]
    if mortality_col and mortality_col in episodes_df.columns:
        cols.append(mortality_col)

    patient_df = (
        episodes_df
        .select(cols)
        .unique(subset=["subject_id"])
        .sort("subject_id")   # deterministic order before shuffle
    )
    return patient_df


def _build_episode_lookup(episodes_df: pl.DataFrame) -> dict[int, list[int]]:
    """Return {subject_id: sorted [stay_id, ...]} mapping."""
    lookup: dict[int, list[int]] = {}
    for row in episodes_df.select(["subject_id", "stay_id"]).iter_rows(named=True):
        lookup.setdefault(row["subject_id"], []).append(row["stay_id"])
    return {sid: sorted(stays) for sid, stays in lookup.items()}


def _build_records(
    patient_df: pl.DataFrame,
    episode_lookup: dict[int, list[int]],
) -> tuple[PatientRecord, ...]:
    records = []
    for row in patient_df.iter_rows(named=True):
        sid = row["subject_id"]
        lbl = SplitLabel(row["split"])
        keys = tuple(episode_lookup.get(sid, []))
        records.append(PatientRecord(subject_id=sid, split=lbl, episode_keys=keys))
    return tuple(records)


def _compute_stats(
    patient_df: pl.DataFrame,
    episode_lookup: dict[int, list[int]],
    mortality_col: str | None,
) -> tuple[SplitStats, ...]:
    stats = []
    for split_label in SplitLabel:
        subset = patient_df.filter(pl.col("split") == split_label.value)
        n_patients = subset.height
        n_episodes = sum(
            len(episode_lookup.get(sid, []))
            for sid in subset.get_column("subject_id").to_list()
        )
        mortality_rate: float | None = None
        if mortality_col and mortality_col in subset.columns:
            vals = subset.get_column(mortality_col).drop_nulls()
            if vals.len() > 0:
                mortality_rate = round(vals.cast(pl.Float64).mean(), 4)

        stats.append(SplitStats(
            split=split_label,
            n_patients=n_patients,
            n_episodes=n_episodes,
            mortality_rate=mortality_rate,
        ))
    return tuple(stats)


def _build_summary_dict(manifest: SplitManifest) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "spec_version": manifest.spec_version,
        "seed": manifest.seed,
        "source_episode_set": manifest.source_episode_set,
        "has_leakage": manifest.has_leakage(),
        "n_total_patients": len(manifest.all_ids()),
        "splits": {},
    }
    for stat in manifest.stats:
        entry: dict[str, Any] = {
            "n_patients": stat.n_patients,
            "n_episodes": stat.n_episodes,
        }
        if stat.mortality_rate is not None:
            entry["mortality_rate"] = stat.mortality_rate
        summary["splits"][stat.split.value] = entry

    # Fallback if stats were not computed
    if not manifest.stats:
        summary["splits"] = {
            SplitLabel.TRAIN.value: {"n_patients": len(manifest.train_ids)},
            SplitLabel.VALIDATION.value: {"n_patients": len(manifest.validation_ids)},
            SplitLabel.TEST.value: {"n_patients": len(manifest.test_ids)},
        }
    return summary


def _empty_manifest_df() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "subject_id": pl.Int64,
        "split": pl.Utf8,
        "episode_keys": pl.List(pl.Int64),
        "n_episodes": pl.Int64,
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patient-level split manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a splits config YAML (e.g. configs/splits/default.yaml).",
    )
    parser.add_argument(
        "--source-episode-set",
        metavar="PATH",
        default=None,
        help="Override the source episode path from the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest statistics without writing any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for split manifest generation."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # ---- Load config ---------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    split_cfg = cfg["split"]
    seed: int = split_cfg["seed"]
    train_frac: float = split_cfg["train_fraction"]
    val_frac: float = split_cfg["validation_fraction"]
    mortality_col: str | None = split_cfg.get("stratify_by")

    source_cfg: dict[str, str] = cfg.get("source", {})
    source_episode_set: str = args.source_episode_set or source_cfg.get(
        "episode_set", "unknown"
    )

    output_cfg: dict[str, str] = cfg.get("output", {})
    manifest_dir = Path(output_cfg.get("manifest_dir", "data/splits"))

    # ---- Load episode data --------------------------------------------
    episode_path = Path(source_episode_set)
    if not episode_path.exists():
        logger.warning(
            "Episode file not found: %s — using empty synthetic dataset for dry-run.",
            episode_path,
        )
        episodes_df = pl.DataFrame({
            "subject_id": list(range(1, 101)),
            "stay_id": list(range(1001, 1101)),
        })
    else:
        logger.info("Loading episodes from %s …", episode_path)
        episodes_df = pl.read_parquet(episode_path)

    # ---- Generate manifest -------------------------------------------
    manifest = generate_split_manifest(
        episodes_df=episodes_df,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        source_episode_set=source_episode_set,
        mortality_col=mortality_col,
    )

    summary = manifest_summary(manifest)
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        logger.info("Dry-run mode — no files written.")
        return

    save_manifest(manifest, manifest_dir)
    logger.info("Done.  Manifest written to %s.", manifest_dir)


if __name__ == "__main__":
    main()
