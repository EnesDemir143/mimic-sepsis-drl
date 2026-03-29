"""
CLI entrypoint for building transition datasets and replay buffers.

This CLI connects the frozen episode/split artifacts to the MDP contracts:

- state vectors from raw MIMIC-IV tables
- train-only preprocessing artifacts
- train-only action bins
- step rewards
- split-specific transition and replay exports

Usage
-----
    python -m mimic_sepsis_rl.cli.build_transitions --dry-run
    python -m mimic_sepsis_rl.cli.build_transitions

Version history
---------------
v1.1.0  2026-03-29  Add live replay export from processed episodes + raw MIMIC tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl
import yaml

from mimic_sepsis_rl.data.splits import load_manifest_parquet
from mimic_sepsis_rl.datasets.replay_buffer import (
    build_replay_buffer,
    save_replay_buffer,
    validate_replay_buffer,
)
from mimic_sepsis_rl.datasets.transitions import (
    build_transitions,
    save_transitions,
)
from mimic_sepsis_rl.mdp.actions.bins import (
    ACTION_SPEC_VERSION,
    ActionBinner,
    save_action_bin_artifacts,
)
from mimic_sepsis_rl.mdp.actions.fluids import FluidAggregator
from mimic_sepsis_rl.mdp.actions.vasopressors import VasopressorStandardiser
from mimic_sepsis_rl.mdp.features.builder import build_state_table
from mimic_sepsis_rl.mdp.features.dictionary import FeatureSpec, load_feature_registry
from mimic_sepsis_rl.mdp.features.extractors import StepWindowData
from mimic_sepsis_rl.mdp.preprocessing import (
    fit_preprocessing_artifacts,
    fit_train_feature_medians,
    save_preprocessing_artifacts,
    transform_state_table,
)
from mimic_sepsis_rl.mdp.reward_models import RewardConfig, RewardVariant
from mimic_sepsis_rl.mdp.rewards import (
    compute_rewards_batch,
    rewards_to_dataframe,
    save_reward_config,
)

logger = logging.getLogger(__name__)

MIMIC_RAW_ROOT = Path("data/raw/physionet.org/files/mimiciv/3.1")
DEFAULT_COHORT_PATH = Path("data/processed/cohort/cohort.parquet")
DEFAULT_EPISODES_PATH = Path("data/processed/episodes/episodes.parquet")
DEFAULT_STEPS_PATH = Path("data/processed/episodes/episode_steps.parquet")
DEFAULT_SPLIT_MANIFEST_DIR = Path("data/splits")
DEFAULT_FEATURES_CONFIG = Path("configs/features/default.yaml")
DEFAULT_REPLAY_DIR = Path("data/replay")
DEFAULT_ACTION_DIR = Path("data/processed/actions")
DEFAULT_REWARD_DIR = Path("data/processed/rewards")
DEFAULT_PREPROCESSING_ARTIFACT = "preprocessing_artifacts.json"
DEFAULT_RAW_STATE_PATH = "state_table_raw.parquet"
DEFAULT_NORMALIZED_STATE_PATH = "state_table_normalized.parquet"
DEFAULT_STEP_ACTIONS_PATH = "step_actions.parquet"
DEFAULT_STEP_REWARDS_PATH = "step_rewards.parquet"
DEFAULT_ACTION_BINS_PATH = "action_bins.json"
DEFAULT_REWARD_CONFIG_PATH = "reward_config.json"
DEFAULT_SPLITS = ("train", "validation", "test")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.cli.build_transitions",
        description="Build live transition datasets and replay buffers.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic data to verify the pipeline works.",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=MIMIC_RAW_ROOT,
        help="Root directory containing MIMIC-IV raw tables.",
    )
    p.add_argument(
        "--cohort-path",
        type=Path,
        default=DEFAULT_COHORT_PATH,
        help="Processed cohort parquet path.",
    )
    p.add_argument(
        "--episodes-path",
        type=Path,
        default=DEFAULT_EPISODES_PATH,
        help="Episode-level parquet path.",
    )
    p.add_argument(
        "--steps-path",
        type=Path,
        default=DEFAULT_STEPS_PATH,
        help="Episode step parquet path.",
    )
    p.add_argument(
        "--split-manifest-dir",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_DIR,
        help="Directory containing train/validation/test split manifests.",
    )
    p.add_argument(
        "--features-config",
        type=Path,
        default=DEFAULT_FEATURES_CONFIG,
        help="Feature config YAML used to select the state registry.",
    )
    p.add_argument(
        "--reward-variant",
        choices=["sparse", "sofa_shaped", "full_shaped"],
        default="sofa_shaped",
        help="Reward variant used for export.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPLAY_DIR,
        help="Directory where replay and transition artifacts are written.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        choices=["all", "train", "validation", "test"],
        help="Which split exports to generate. Default: all.",
    )
    p.add_argument(
        "--limit-stays",
        type=int,
        default=None,
        help="Optional limit for the number of episode stays to process.",
    )
    return p


def _make_synthetic_data(
    n_episodes: int = 5,
    steps_per_episode: int = 6,
    n_features: int = 8,
) -> tuple[pl.DataFrame, list[str]]:
    """Generate synthetic merged data for dry-run validation."""
    random.seed(42)
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    records: list[dict[str, Any]] = []
    for ep in range(1, n_episodes + 1):
        mortality = random.choice([0, 1])
        for step in range(steps_per_episode):
            row = {
                "stay_id": ep * 100,
                "step_index": step,
                "action_id": random.randint(0, 24),
                "reward_total": random.uniform(-1.0, 1.0),
                "mortality_90d": mortality,
            }
            for feat in feature_cols:
                row[feat] = random.gauss(0, 1)
            records.append(row)

    for ep in range(1, n_episodes + 1):
        idx = ep * steps_per_episode - 1
        mortality = records[idx]["mortality_90d"]
        records[idx]["reward_total"] = 15.0 if mortality == 0 else -15.0

    return pl.DataFrame(records), feature_cols


def _dry_run() -> None:
    """Smoke test with synthetic data."""
    logger.info("Running transition builder dry-run...")

    merged_df, feature_cols = _make_synthetic_data()
    logger.info(
        "Synthetic data: %d rows, %d episodes, %d features",
        merged_df.height,
        merged_df.get_column("stay_id").n_unique(),
        len(feature_cols),
    )

    transitions = build_transitions(
        merged_df,
        feature_columns=feature_cols,
    )
    logger.info("Built %d transitions.", len(transitions))

    buffer = build_replay_buffer(
        transitions,
        feature_columns=feature_cols,
        split_label="train",
        manifest_seed=42,
        action_spec_version=ACTION_SPEC_VERSION,
        reward_spec_version=RewardConfig().version,
    )
    validate_replay_buffer(
        buffer,
        expected_state_dim=len(feature_cols),
        expected_n_actions=25,
    )

    logger.info("Replay buffer: %d episodes, %d transitions", buffer.n_episodes, buffer.n_transitions)
    logger.info("✅ Transition builder dry-run PASSED.")


def _load_feature_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature config not found: {path}")
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Feature config must be a mapping: {path}")
    return payload


def _load_required_parquet(path: Path, label: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pl.read_parquet(path)


def _resolve_requested_splits(raw_splits: Sequence[str]) -> tuple[str, ...]:
    if "all" in raw_splits:
        return DEFAULT_SPLITS
    return tuple(raw_splits)


def _resolve_feature_columns(
    state_df: pl.DataFrame,
    registry: Mapping[str, FeatureSpec],
) -> list[str]:
    feature_columns: list[str] = []
    for feature_id in registry:
        if feature_id in state_df.columns:
            feature_columns.append(feature_id)
        missing_col = f"{feature_id}_missing"
        if missing_col in state_df.columns:
            feature_columns.append(missing_col)
    return feature_columns


def _required_item_ids(
    registry: Mapping[str, FeatureSpec],
    source_table: str,
) -> list[int]:
    item_ids: set[int] = set()
    for spec in registry.values():
        if spec.source_table == source_table:
            item_ids.update(int(item_id) for item_id in spec.item_ids)
    return sorted(item_ids)


def _load_filtered_chartevents(
    raw_root: Path,
    stay_ids: Sequence[int],
    chart_min: Any,
    chart_max: Any,
    item_ids: Sequence[int],
) -> pl.DataFrame:
    path = raw_root / "icu" / "chartevents.csv.gz"
    lazy = (
        pl.scan_csv(
            path,
            schema_overrides={
                "stay_id": pl.Int64,
                "itemid": pl.Int64,
                "valuenum": pl.Float64,
            },
        )
        .filter(pl.col("stay_id").cast(pl.Int64, strict=False).is_in(list(stay_ids)))
        .filter(pl.col("itemid").cast(pl.Int64, strict=False).is_in(list(item_ids)))
        .select(["stay_id", "charttime", "itemid", "valuenum"])
        .with_columns(
            pl.col("stay_id").cast(pl.Int64, strict=False),
            pl.col("itemid").cast(pl.Int64, strict=False),
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("valuenum").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("charttime").is_not_null())
        .filter(pl.col("charttime") >= pl.lit(chart_min))
        .filter(pl.col("charttime") < pl.lit(chart_max))
    )
    return lazy.collect()


def _load_filtered_labevents(
    raw_root: Path,
    hadm_ids: Sequence[int],
    chart_min: Any,
    chart_max: Any,
    item_ids: Sequence[int],
) -> pl.DataFrame:
    path = raw_root / "hosp" / "labevents.csv.gz"
    lazy = (
        pl.scan_csv(
            path,
            schema_overrides={
                "hadm_id": pl.Int64,
                "itemid": pl.Int64,
                "valuenum": pl.Float64,
            },
        )
        .with_columns(
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("itemid").cast(pl.Int64, strict=False),
        )
        .filter(pl.col("hadm_id").is_in(list(hadm_ids)))
        .filter(pl.col("itemid").is_in(list(item_ids)))
        .select(["hadm_id", "charttime", "itemid", "valuenum"])
        .with_columns(
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("valuenum").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("charttime").is_not_null())
        .filter(pl.col("charttime") >= pl.lit(chart_min))
        .filter(pl.col("charttime") < pl.lit(chart_max))
    )
    return lazy.collect()


def _load_filtered_inputevents(
    raw_root: Path,
    stay_ids: Sequence[int],
    chart_min: Any,
    chart_max: Any,
    item_ids: Sequence[int],
) -> pl.DataFrame:
    path = raw_root / "icu" / "inputevents.csv.gz"
    lazy = (
        pl.scan_csv(
            path,
            schema_overrides={
                "stay_id": pl.Int64,
                "itemid": pl.Int64,
                "amount": pl.Float64,
                "rate": pl.Float64,
                "patientweight": pl.Float64,
            },
        )
        .filter(pl.col("stay_id").cast(pl.Int64, strict=False).is_in(list(stay_ids)))
        .filter(pl.col("itemid").cast(pl.Int64, strict=False).is_in(list(item_ids)))
        .select(
            [
                "stay_id",
                "starttime",
                "endtime",
                "itemid",
                "amount",
                "rate",
                "patientweight",
            ]
        )
        .with_columns(
            pl.col("stay_id").cast(pl.Int64, strict=False),
            pl.col("itemid").cast(pl.Int64, strict=False),
            pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("endtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("amount").cast(pl.Float64, strict=False),
            pl.col("rate").cast(pl.Float64, strict=False),
            pl.col("patientweight").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("starttime").is_not_null())
        .filter(pl.col("starttime") < pl.lit(chart_max))
        .filter(pl.col("endtime").is_null() | (pl.col("endtime") >= pl.lit(chart_min)))
        .with_columns(
            pl.coalesce([pl.col("endtime"), pl.col("starttime")]).alias("endtime"),
        )
    )
    return lazy.collect()


def _load_filtered_outputevents(
    raw_root: Path,
    stay_ids: Sequence[int],
    chart_min: Any,
    chart_max: Any,
    item_ids: Sequence[int],
) -> pl.DataFrame:
    path = raw_root / "icu" / "outputevents.csv.gz"
    lazy = (
        pl.scan_csv(
            path,
            schema_overrides={
                "stay_id": pl.Int64,
                "itemid": pl.Int64,
                "value": pl.Float64,
            },
        )
        .filter(pl.col("stay_id").cast(pl.Int64, strict=False).is_in(list(stay_ids)))
        .filter(pl.col("itemid").cast(pl.Int64, strict=False).is_in(list(item_ids)))
        .select(["stay_id", "charttime", "itemid", "value"])
        .with_columns(
            pl.col("stay_id").cast(pl.Int64, strict=False),
            pl.col("itemid").cast(pl.Int64, strict=False),
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("value").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("charttime").is_not_null())
        .filter(pl.col("charttime") >= pl.lit(chart_min))
        .filter(pl.col("charttime") < pl.lit(chart_max))
    )
    return lazy.collect()


def _load_filtered_admissions(
    raw_root: Path,
    hadm_ids: Sequence[int],
) -> pl.DataFrame:
    path = raw_root / "hosp" / "admissions.csv.gz"
    lazy = (
        pl.scan_csv(
            path,
            schema_overrides={
                "hadm_id": pl.Int64,
            },
        )
        .with_columns(pl.col("hadm_id").cast(pl.Int64, strict=False))
        .filter(pl.col("hadm_id").is_in(list(hadm_ids)))
        .select(["hadm_id", "deathtime"])
        .with_columns(
            pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        )
    )
    return lazy.collect()


def _group_table(
    df: pl.DataFrame,
    key: str,
    sort_col: str | None = None,
) -> dict[int, pl.DataFrame]:
    if df.is_empty():
        return {}
    groups: dict[int, pl.DataFrame] = {}
    for values, group_df in df.group_by(key, maintain_order=True):
        group_key = int(values[0] if isinstance(values, tuple) else values)
        if sort_col is not None and sort_col in group_df.columns:
            group_df = group_df.sort(sort_col)
        groups[group_key] = group_df
    return groups


def _slice_window(
    df: pl.DataFrame | None,
    time_col: str,
    start: Any,
    end: Any,
) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()
    return df.filter((pl.col(time_col) >= pl.lit(start)) & (pl.col(time_col) < pl.lit(end)))


def _slice_cumulative_window(
    df: pl.DataFrame | None,
    time_col: str,
    start: Any,
    end: Any,
) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()
    return df.filter((pl.col(time_col) >= pl.lit(start)) & (pl.col(time_col) < pl.lit(end)))


def _compute_stay_weights(inputevents: pl.DataFrame) -> dict[int, float]:
    if inputevents.is_empty() or "patientweight" not in inputevents.columns:
        return {}
    weights = (
        inputevents.filter(pl.col("patientweight").is_not_null())
        .group_by("stay_id")
        .agg(pl.col("patientweight").median().alias("weight_kg"))
    )
    return {
        int(row["stay_id"]): float(row["weight_kg"])
        for row in weights.iter_rows(named=True)
        if row["weight_kg"] is not None
    }


def _build_step_windows(
    step_context_df: pl.DataFrame,
    chartevents: pl.DataFrame,
    labevents: pl.DataFrame,
    inputevents: pl.DataFrame,
    outputevents: pl.DataFrame,
) -> list[StepWindowData]:
    chartevents_by_stay = _group_table(chartevents, "stay_id", sort_col="charttime")
    labevents_by_hadm = _group_table(labevents, "hadm_id", sort_col="charttime")
    inputevents_by_stay = _group_table(inputevents, "stay_id", sort_col="starttime")
    outputevents_by_stay = _group_table(outputevents, "stay_id", sort_col="charttime")
    weights_by_stay = _compute_stay_weights(inputevents)
    episode_start_by_stay = {
        int(row["stay_id"]): row["episode_start"]
        for row in step_context_df.select(["stay_id", "episode_start"]).unique().iter_rows(named=True)
    }

    step_windows: list[StepWindowData] = []
    for row in step_context_df.iter_rows(named=True):
        stay_id = int(row["stay_id"])
        hadm_id = int(row["hadm_id"])
        step_windows.append(
            StepWindowData(
                stay_id=stay_id,
                step_index=int(row["step_index"]),
                hours_relative_to_onset=float(row["hours_relative_to_onset"]),
                chartevents=_slice_window(
                    chartevents_by_stay.get(stay_id),
                    "charttime",
                    row["step_start"],
                    row["step_end"],
                ),
                labevents=_slice_window(
                    labevents_by_hadm.get(hadm_id),
                    "charttime",
                    row["step_start"],
                    row["step_end"],
                ),
                inputevents=_slice_cumulative_window(
                    inputevents_by_stay.get(stay_id),
                    "starttime",
                    episode_start_by_stay[stay_id],
                    row["step_end"],
                ),
                outputevents=_slice_window(
                    outputevents_by_stay.get(stay_id),
                    "charttime",
                    row["step_start"],
                    row["step_end"],
                ),
                age_years=float(row["anchor_age"]) if row["anchor_age"] is not None else None,
                weight_kg=weights_by_stay.get(stay_id),
                subject_id=int(row["subject_id"]),
            )
        )
    return step_windows


def _build_step_context_df(
    episodes_df: pl.DataFrame,
    steps_df: pl.DataFrame,
    cohort_df: pl.DataFrame,
    limit_stays: int | None,
) -> pl.DataFrame:
    episode_ctx = (
        episodes_df.select(["stay_id", "subject_id", "hadm_id", "onset_time", "window_start"])
        .join(cohort_df.select(["stay_id", "anchor_age"]), on="stay_id", how="left")
        .rename({"window_start": "episode_start"})
    )
    if limit_stays is not None:
        selected_stays = episode_ctx.get_column("stay_id").head(limit_stays).to_list()
        episode_ctx = episode_ctx.filter(pl.col("stay_id").is_in(selected_stays))
        steps_df = steps_df.filter(pl.col("stay_id").is_in(selected_stays))

    return (
        steps_df.join(episode_ctx, on="stay_id", how="inner")
        .sort(["stay_id", "step_index"])
    )


def _append_sofa_proxy(raw_state_df: pl.DataFrame) -> pl.DataFrame:
    return raw_state_df.with_columns(
        (
            (pl.col("map") < 70.0).cast(pl.Int32)
            + (pl.col("spo2") < 94.0).cast(pl.Int32)
            + (pl.col("gcs_total") < 15.0).cast(pl.Int32)
        ).alias("sofa_score")
    )


def _build_mortality_labels(
    episodes_df: pl.DataFrame,
    admissions_df: pl.DataFrame,
) -> pl.DataFrame:
    joined = episodes_df.select(["stay_id", "hadm_id", "onset_time"]).join(
        admissions_df,
        on="hadm_id",
        how="left",
    )
    return joined.with_columns(
        (
            pl.col("deathtime").is_not_null()
            & (pl.col("deathtime") <= (pl.col("onset_time") + pl.duration(days=90)))
        ).cast(pl.Int32).alias("mortality_90d")
    ).select(["stay_id", "mortality_90d"])


def _fit_and_apply_action_bins(
    step_context_df: pl.DataFrame,
    inputevents: pl.DataFrame,
    split_manifest_seed: int,
    output_dir: Path,
) -> pl.DataFrame:
    step_boundaries = step_context_df.select(
        ["stay_id", "step_index", "step_start", "step_end", "subject_id", "split"]
    )

    standardiser = VasopressorStandardiser()
    standardised_vaso = standardiser.standardise(inputevents)
    vaso_steps = standardiser.aggregate_per_step(
        standardised_vaso,
        step_boundaries.select(["stay_id", "step_index", "step_start", "step_end"]),
    )
    fluid_steps = FluidAggregator().aggregate_per_step(
        inputevents,
        step_boundaries.select(["stay_id", "step_index", "step_start", "step_end"]),
    )

    treatment_df = (
        step_boundaries.join(vaso_steps, on=["stay_id", "step_index"], how="left")
        .join(fluid_steps, on=["stay_id", "step_index"], how="left")
        .with_columns(
            pl.col("vaso_dose_4h").fill_null(0.0),
            pl.col("fluid_volume_4h").fill_null(0.0),
        )
    )

    train_df = treatment_df.filter(pl.col("split") == "train")
    if train_df.is_empty():
        raise ValueError("Training partition is empty; cannot fit action bins.")
    binner = ActionBinner().fit(train_df, manifest_seed=split_manifest_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_action_bin_artifacts(binner.artifacts, output_dir / DEFAULT_ACTION_BINS_PATH)

    encoded = binner.transform(treatment_df.select(["vaso_dose_4h", "fluid_volume_4h"]))
    result = pl.concat(
        [
            treatment_df.select(["stay_id", "step_index", "vaso_dose_4h", "fluid_volume_4h"]),
            encoded.select(["vaso_bin", "fluid_bin", "action_id"]),
        ],
        how="horizontal",
    )
    result.write_parquet(output_dir / DEFAULT_STEP_ACTIONS_PATH)
    logger.info("Saved step-level action assignments to %s", output_dir / DEFAULT_STEP_ACTIONS_PATH)
    return result


def _save_train_medians(path: Path, medians: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(medians), indent=2))


def _run_live(args: argparse.Namespace) -> int:
    feature_cfg = _load_feature_config(args.features_config)
    registry = load_feature_registry(feature_cfg)
    manifest = load_manifest_parquet(args.split_manifest_dir)

    episodes_df = _load_required_parquet(args.episodes_path, "Episode parquet")
    steps_df = _load_required_parquet(args.steps_path, "Episode steps parquet")
    cohort_df = _load_required_parquet(args.cohort_path, "Cohort parquet")

    step_context_df = _build_step_context_df(
        episodes_df=episodes_df,
        steps_df=steps_df,
        cohort_df=cohort_df,
        limit_stays=args.limit_stays,
    )
    if step_context_df.is_empty():
        raise ValueError("No episode steps available after applying the requested filters.")

    step_context_df = step_context_df.with_columns(
        pl.Series(
            "split",
            [manifest.split_for(int(subject_id)).value for subject_id in step_context_df.get_column("subject_id").to_list()],
        )
    )

    stay_ids = step_context_df.get_column("stay_id").unique().sort().to_list()
    hadm_ids = step_context_df.get_column("hadm_id").unique().sort().to_list()
    chart_min = step_context_df.get_column("episode_start").min()
    chart_max = step_context_df.get_column("step_end").max()

    logger.info(
        "Loading raw tables for %d stays (%d hadm_ids) between %s and %s",
        len(stay_ids),
        len(hadm_ids),
        chart_min,
        chart_max,
    )

    chartevents = _load_filtered_chartevents(
        args.raw_root,
        stay_ids,
        chart_min,
        chart_max,
        _required_item_ids(registry, "chartevents"),
    )
    labevents = _load_filtered_labevents(
        args.raw_root,
        hadm_ids,
        chart_min,
        chart_max,
        _required_item_ids(registry, "labevents"),
    )
    input_item_ids = sorted(
        set(_required_item_ids(registry, "inputevents"))
        | set(FluidAggregator().item_ids)
        | {221906, 221289, 221662, 222315, 222042}
    )
    inputevents = _load_filtered_inputevents(
        args.raw_root,
        stay_ids,
        chart_min,
        chart_max,
        input_item_ids,
    )
    outputevents = _load_filtered_outputevents(
        args.raw_root,
        stay_ids,
        chart_min,
        chart_max,
        _required_item_ids(registry, "outputevents"),
    )
    admissions = _load_filtered_admissions(args.raw_root, hadm_ids)

    step_windows = _build_step_windows(
        step_context_df=step_context_df,
        chartevents=chartevents,
        labevents=labevents,
        inputevents=inputevents,
        outputevents=outputevents,
    )
    logger.info("Built %d step windows.", len(step_windows))

    raw_state_first = build_state_table(
        step_windows,
        registry,
        split_manifest=manifest,
        train_medians={},
    )
    train_medians = fit_train_feature_medians(raw_state_first, registry, manifest)
    raw_state = build_state_table(
        step_windows,
        registry,
        split_manifest=manifest,
        train_medians=train_medians,
    )

    raw_state = _append_sofa_proxy(raw_state)
    mortality_df = _build_mortality_labels(
        episodes_df=step_context_df.select(["stay_id", "hadm_id", "onset_time"]).unique(),
        admissions_df=admissions,
    )
    raw_state = raw_state.join(mortality_df, on="stay_id", how="left").with_columns(
        pl.col("mortality_90d").fill_null(0).cast(pl.Int32)
    )

    preprocessing = fit_preprocessing_artifacts(raw_state, registry, manifest, train_medians)
    normalized_state = transform_state_table(raw_state, preprocessing)

    state_output_dir = Path(
        feature_cfg.get("output", {}).get("state_table_dir", "data/processed/features/state_vectors")
    )
    state_output_dir.mkdir(parents=True, exist_ok=True)
    train_medians_path = Path(
        feature_cfg.get("imputation", {}).get(
            "train_medians_path",
            state_output_dir / "train_medians.json",
        )
    )
    raw_state.write_parquet(state_output_dir / DEFAULT_RAW_STATE_PATH)
    normalized_state.write_parquet(state_output_dir / DEFAULT_NORMALIZED_STATE_PATH)
    _save_train_medians(train_medians_path, train_medians)
    save_preprocessing_artifacts(preprocessing, state_output_dir / DEFAULT_PREPROCESSING_ARTIFACT)

    action_df = _fit_and_apply_action_bins(
        step_context_df=step_context_df,
        inputevents=inputevents,
        split_manifest_seed=manifest.seed,
        output_dir=DEFAULT_ACTION_DIR,
    )

    reward_config = RewardConfig(variant=RewardVariant(args.reward_variant))
    reward_df = rewards_to_dataframe(compute_rewards_batch(raw_state, config=reward_config)).rename(
        {"total": "reward_total"}
    )
    DEFAULT_REWARD_DIR.mkdir(parents=True, exist_ok=True)
    reward_df.write_parquet(DEFAULT_REWARD_DIR / DEFAULT_STEP_REWARDS_PATH)
    save_reward_config(reward_config, DEFAULT_REWARD_DIR / DEFAULT_REWARD_CONFIG_PATH)

    helper_cols = ["stay_id", "step_index", "subject_id", "split", "sofa_score", "mortality_90d"]
    normalized_with_meta = normalized_state.select(
        [col for col in helper_cols if col in normalized_state.columns]
        + [col for col in normalized_state.columns if col not in helper_cols]
    )
    merged_df = (
        normalized_with_meta.join(action_df, on=["stay_id", "step_index"], how="left")
        .join(reward_df.select(["stay_id", "step_index", "reward_total"]), on=["stay_id", "step_index"], how="left")
        .with_columns(pl.col("reward_total").fill_null(0.0))
    )

    feature_columns = _resolve_feature_columns(merged_df, registry)
    if not feature_columns:
        raise ValueError("No feature columns were resolved for transition export.")

    requested_splits = _resolve_requested_splits(args.splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_label in requested_splits:
        split_df = merged_df.filter(pl.col("split") == split_label)
        if split_df.is_empty():
            logger.warning("Skipping split '%s' because it contains no rows.", split_label)
            continue

        transitions = build_transitions(
            split_df,
            feature_columns=feature_columns,
        )
        save_transitions(
            transitions,
            feature_columns=feature_columns,
            output_dir=args.output_dir,
            split_label=split_label,
            manifest_seed=manifest.seed,
            action_spec_version=ACTION_SPEC_VERSION,
            reward_spec_version=reward_config.version,
        )
        replay_buffer = build_replay_buffer(
            transitions,
            feature_columns=feature_columns,
            split_label=split_label,
            manifest_seed=manifest.seed,
            action_spec_version=ACTION_SPEC_VERSION,
            reward_spec_version=reward_config.version,
        )
        validate_replay_buffer(
            replay_buffer,
            expected_state_dim=len(feature_columns),
            expected_n_actions=25,
        )
        save_replay_buffer(replay_buffer, args.output_dir)
        logger.info(
            "Split '%s' complete: %d transitions across %d episodes.",
            split_label,
            replay_buffer.n_transitions,
            replay_buffer.n_episodes,
        )

    print(f"Replay exports written to {args.output_dir}")
    print(f"State tables written to {state_output_dir}")
    print(f"Action artifacts written to {DEFAULT_ACTION_DIR}")
    print(f"Reward artifacts written to {DEFAULT_REWARD_DIR}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        _dry_run()
        return 0

    return _run_live(args)


if __name__ == "__main__":
    sys.exit(main())
