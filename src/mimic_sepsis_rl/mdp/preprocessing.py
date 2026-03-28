"""
Leakage-safe preprocessing artifacts for Phase 4 state vectors.

The builder is responsible for step-level imputation order. This module owns
the learned artifacts that must respect the train-only boundary:

- feature medians used as the fallback imputation prior
- per-feature clipping bounds frozen into an artifact bundle
- normalization statistics fit on train rows only
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from mimic_sepsis_rl.data.split_models import SplitManifest
from mimic_sepsis_rl.mdp.features.dictionary import FeatureSpec, MissingStrategy

PREPROCESSING_SPEC_VERSION = "1.0.0"


@dataclass(frozen=True)
class FeatureTransform:
    """Frozen preprocessing parameters for a single feature column."""

    median: float
    mean: float
    scale: float
    clip_low: float | None
    clip_high: float | None


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Serializable preprocessing bundle fit from the training partition."""

    spec_version: str
    manifest_seed: int
    source_episode_set: str
    feature_order: tuple[str, ...]
    transforms: dict[str, FeatureTransform]

    def to_dict(self) -> dict[str, Any]:
        """Convert artifacts into a JSON-serializable mapping."""
        return {
            "spec_version": self.spec_version,
            "manifest_seed": self.manifest_seed,
            "source_episode_set": self.source_episode_set,
            "feature_order": list(self.feature_order),
            "transforms": {
                feature_id: asdict(transform)
                for feature_id, transform in self.transforms.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreprocessingArtifacts":
        """Reconstruct artifacts from serialized data."""
        transforms = {
            feature_id: FeatureTransform(**transform_payload)
            for feature_id, transform_payload in payload["transforms"].items()
        }
        return cls(
            spec_version=str(payload["spec_version"]),
            manifest_seed=int(payload["manifest_seed"]),
            source_episode_set=str(payload["source_episode_set"]),
            feature_order=tuple(payload["feature_order"]),
            transforms=transforms,
        )


def fit_train_feature_medians(
    state_df: pl.DataFrame,
    registry: Mapping[str, FeatureSpec],
    split_manifest: SplitManifest,
) -> dict[str, float]:
    """Fit fallback medians using only rows from the train manifest."""
    train_df = _select_train_rows(state_df, split_manifest)
    medians: dict[str, float] = {}

    for feature_id, spec in registry.items():
        _require_feature_column(state_df, feature_id)
        observed = train_df.get_column(feature_id).drop_nulls()
        median = observed.median() if not observed.is_empty() else None
        medians[feature_id] = _coerce_default_value(spec, median)

    return medians


def fit_preprocessing_artifacts(
    state_df: pl.DataFrame,
    registry: Mapping[str, FeatureSpec],
    split_manifest: SplitManifest,
    train_medians: Mapping[str, float],
) -> PreprocessingArtifacts:
    """Fit train-only normalization statistics and freeze clip bounds."""
    train_df = _select_train_rows(state_df, split_manifest)
    transforms: dict[str, FeatureTransform] = {}

    for feature_id, spec in registry.items():
        _require_feature_column(state_df, feature_id)
        median = float(train_medians[feature_id])
        prepared = _prepare_feature_values(
            values=train_df.get_column(feature_id).to_list(),
            fill_value=median,
            clip_low=spec.clip_low,
            clip_high=spec.clip_high,
        )
        mean = sum(prepared) / len(prepared) if prepared else median
        variance = (
            sum((value - mean) ** 2 for value in prepared) / len(prepared)
            if prepared
            else 0.0
        )
        scale = math.sqrt(variance)
        if math.isclose(scale, 0.0, abs_tol=1e-12):
            scale = 1.0

        transforms[feature_id] = FeatureTransform(
            median=median,
            mean=mean,
            scale=scale,
            clip_low=spec.clip_low,
            clip_high=spec.clip_high,
        )

    return PreprocessingArtifacts(
        spec_version=PREPROCESSING_SPEC_VERSION,
        manifest_seed=split_manifest.seed,
        source_episode_set=split_manifest.source_episode_set,
        feature_order=tuple(registry.keys()),
        transforms=transforms,
    )


def transform_state_table(
    state_df: pl.DataFrame,
    artifacts: PreprocessingArtifacts,
) -> pl.DataFrame:
    """Apply frozen preprocessing artifacts to a state table."""
    transformed = state_df.clone()

    for feature_id in artifacts.feature_order:
        _require_feature_column(transformed, feature_id)
        transform = artifacts.transforms[feature_id]
        prepared = _prepare_feature_values(
            values=transformed.get_column(feature_id).to_list(),
            fill_value=transform.median,
            clip_low=transform.clip_low,
            clip_high=transform.clip_high,
        )
        normalized = [
            (value - transform.mean) / transform.scale
            for value in prepared
        ]
        transformed = transformed.with_columns(
            pl.Series(feature_id, normalized, dtype=pl.Float64)
        )

    return transformed


def save_preprocessing_artifacts(
    artifacts: PreprocessingArtifacts,
    output_path: Path,
) -> None:
    """Persist preprocessing artifacts as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifacts.to_dict(), indent=2))


def load_preprocessing_artifacts(output_path: Path) -> PreprocessingArtifacts:
    """Load previously persisted preprocessing artifacts."""
    return PreprocessingArtifacts.from_dict(json.loads(output_path.read_text()))


def _select_train_rows(
    state_df: pl.DataFrame,
    split_manifest: SplitManifest,
) -> pl.DataFrame:
    _require_subject_id_column(state_df)
    train_df = state_df.filter(pl.col("subject_id").is_in(sorted(split_manifest.train_ids)))
    if train_df.is_empty():
        raise ValueError("No training rows found for the provided split manifest.")
    return train_df


def _coerce_default_value(
    spec: FeatureSpec,
    value: float | None,
) -> float:
    if value is not None:
        return float(value)
    if spec.missing_strategy == MissingStrategy.ZERO:
        return 0.0
    if spec.normal_value is not None:
        return float(spec.normal_value)
    return 0.0


def _prepare_feature_values(
    values: list[Any],
    fill_value: float,
    clip_low: float | None,
    clip_high: float | None,
) -> list[float]:
    prepared: list[float] = []
    for raw_value in values:
        value = fill_value if raw_value is None else float(raw_value)
        if clip_low is not None and value < clip_low:
            value = clip_low
        if clip_high is not None and value > clip_high:
            value = clip_high
        prepared.append(float(value))
    return prepared


def _require_subject_id_column(state_df: pl.DataFrame) -> None:
    if "subject_id" not in state_df.columns:
        raise ValueError(
            "state_df must contain a subject_id column so train rows can be "
            "selected from the split manifest."
        )


def _require_feature_column(state_df: pl.DataFrame, feature_id: str) -> None:
    if feature_id not in state_df.columns:
        raise ValueError(f"state_df is missing required feature column '{feature_id}'.")


__all__ = [
    "FeatureTransform",
    "PreprocessingArtifacts",
    "fit_train_feature_medians",
    "fit_preprocessing_artifacts",
    "load_preprocessing_artifacts",
    "save_preprocessing_artifacts",
    "transform_state_table",
]
