"""
Standardized comparison artifacts for offline RL algorithm runs.

This module loads per-run checkpoints, manifests, metric curves, and config
provenance into a shared schema so later evaluation phases can compare CQL,
BCQ, and IQL without special-casing artifact layouts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from mimic_sepsis_rl.datasets.transitions import TransitionDatasetMeta
from mimic_sepsis_rl.training.common import CheckpointManager, CheckpointManifest
from mimic_sepsis_rl.training.config import TrainingConfig, load_training_config


@dataclass(frozen=True)
class CurvePoint:
    """One scalar point from a metrics JSONL curve."""

    step: int
    epoch: int
    value: float
    timestamp: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CurvePoint":
        return cls(
            step=int(payload["step"]),
            epoch=int(payload["epoch"]),
            value=float(payload["value"]),
            timestamp=float(payload["timestamp"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricCurve:
    """All points for a single metric name."""

    name: str
    points: tuple[CurvePoint, ...]

    @property
    def final_value(self) -> float | None:
        if not self.points:
            return None
        return self.points[-1].value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": [point.to_dict() for point in self.points],
            "final_value": self.final_value,
        }


@dataclass(frozen=True)
class ComparisonCheckpoint:
    """Checkpoint and manifest metadata for one algorithm run."""

    checkpoint_path: str
    manifest_path: str
    epoch: int
    global_step: int
    metrics: dict[str, float]

    @classmethod
    def from_manifest(
        cls,
        checkpoint_path: Path,
        manifest: CheckpointManifest,
    ) -> "ComparisonCheckpoint":
        return cls(
            checkpoint_path=str(checkpoint_path),
            manifest_path=str(
                checkpoint_path.with_name(f"{checkpoint_path.stem}_manifest.json")
            ),
            epoch=manifest.epoch,
            global_step=manifest.global_step,
            metrics=dict(manifest.metrics),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfigProvenance:
    """Normalized config provenance needed for fair comparisons."""

    config_path: str
    checkpoint_dir: str
    log_dir: str
    experiment_name: str
    dataset_path: str
    dataset_meta_path: str | None
    batch_size: int
    gamma: float
    requested_device: str
    effective_backend: str

    @classmethod
    def from_config(
        cls,
        cfg: TrainingConfig,
        *,
        config_path: Path,
    ) -> "ConfigProvenance":
        return cls(
            config_path=str(config_path),
            checkpoint_dir=str(cfg.checkpoint.checkpoint_dir),
            log_dir=str(cfg.logging.log_dir),
            experiment_name=cfg.logging.experiment_name,
            dataset_path=str(cfg.dataset_path),
            dataset_meta_path=str(cfg.dataset_meta_path)
            if cfg.dataset_meta_path
            else None,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            requested_device=cfg.runtime.requested_device,
            effective_backend=cfg.device_meta.backend,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetContractRecord:
    """Shared replay-contract metadata used to detect comparison drift."""

    spec_version: str
    split_label: str
    n_actions: int
    state_dim: int
    action_spec_version: str
    reward_spec_version: str
    manifest_seed: int
    n_episodes: int
    n_transitions: int
    feature_columns: tuple[str, ...]

    @classmethod
    def from_meta(cls, meta: TransitionDatasetMeta) -> "DatasetContractRecord":
        return cls(
            spec_version=meta.spec_version,
            split_label=meta.split_label,
            n_actions=meta.n_actions,
            state_dim=meta.state_dim,
            action_spec_version=meta.action_spec_version,
            reward_spec_version=meta.reward_spec_version,
            manifest_seed=meta.manifest_seed,
            n_episodes=meta.n_episodes,
            n_transitions=meta.n_transitions,
            feature_columns=meta.feature_columns,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "split_label": self.split_label,
            "n_actions": self.n_actions,
            "state_dim": self.state_dim,
            "action_spec_version": self.action_spec_version,
            "reward_spec_version": self.reward_spec_version,
            "manifest_seed": self.manifest_seed,
            "n_episodes": self.n_episodes,
            "n_transitions": self.n_transitions,
            "feature_columns": list(self.feature_columns),
        }


@dataclass(frozen=True)
class RunArtifact:
    """Algorithm-agnostic artifact bundle for one training run."""

    algorithm: str
    checkpoint: ComparisonCheckpoint | None
    curves: tuple[MetricCurve, ...]
    final_metrics: dict[str, float]
    config_provenance: ConfigProvenance
    dataset_contract: DatasetContractRecord | None

    @property
    def curve_names(self) -> tuple[str, ...]:
        return tuple(curve.name for curve in self.curves)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "curves": [curve.to_dict() for curve in self.curves],
            "curve_names": list(self.curve_names),
            "final_metrics": dict(self.final_metrics),
            "config_provenance": self.config_provenance.to_dict(),
            "dataset_contract": self.dataset_contract.to_dict()
            if self.dataset_contract
            else None,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Multi-algorithm comparison report over normalized run artifacts."""

    algorithms: tuple[str, ...]
    dataset_contract_consistent: bool
    shared_dataset_contract: DatasetContractRecord | None
    runs: tuple[RunArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithms": list(self.algorithms),
            "dataset_contract_consistent": self.dataset_contract_consistent,
            "shared_dataset_contract": self.shared_dataset_contract.to_dict()
            if self.shared_dataset_contract
            else None,
            "runs": [run.to_dict() for run in self.runs],
        }


def resolve_metrics_log_path(cfg: TrainingConfig) -> Path:
    """Return the default metrics JSONL path for a resolved config."""
    return cfg.logging.log_dir / f"{cfg.logging.experiment_name}_metrics.jsonl"


def resolve_latest_checkpoint(cfg: TrainingConfig) -> Path | None:
    """Return the latest checkpoint path matching the shared filename pattern."""
    checkpoint_dir = cfg.checkpoint.checkpoint_dir
    pattern = f"{cfg.algorithm}_epoch*.pt"
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def load_dataset_contract(
    dataset_meta_path: Path | None,
) -> DatasetContractRecord | None:
    """Load the dataset metadata JSON used by the shared experiment runner."""
    if dataset_meta_path is None or not dataset_meta_path.exists():
        return None

    payload = json.loads(dataset_meta_path.read_text())
    meta = TransitionDatasetMeta.from_dict(payload)
    return DatasetContractRecord.from_meta(meta)


def load_metric_curves(metrics_path: Path) -> tuple[MetricCurve, ...]:
    """Load metric JSONL records and group them into named curves."""
    if not metrics_path.exists():
        return tuple()

    grouped: dict[str, list[CurvePoint]] = {}
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        point = CurvePoint.from_dict(payload)
        name = str(payload["name"])
        grouped.setdefault(name, []).append(point)

    return tuple(
        MetricCurve(name=name, points=tuple(grouped[name]))
        for name in sorted(grouped)
    )


def build_run_artifact(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> RunArtifact:
    """Build a normalized artifact bundle for one algorithm config."""
    resolved_config_path = Path(config_path)
    cfg = load_training_config(resolved_config_path)

    resolved_checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else resolve_latest_checkpoint(cfg)
    )
    checkpoint: ComparisonCheckpoint | None = None
    final_metrics: dict[str, float] = {}

    if resolved_checkpoint_path is not None and resolved_checkpoint_path.exists():
        manifest = CheckpointManager.load_manifest(resolved_checkpoint_path)
        checkpoint = ComparisonCheckpoint.from_manifest(
            resolved_checkpoint_path,
            manifest,
        )
        final_metrics = dict(manifest.metrics)

    resolved_metrics_path = (
        Path(metrics_path)
        if metrics_path is not None
        else resolve_metrics_log_path(cfg)
    )
    curves = load_metric_curves(resolved_metrics_path)
    if not final_metrics:
        final_metrics = {
            curve.name: curve.final_value
            for curve in curves
            if curve.final_value is not None
        }

    return RunArtifact(
        algorithm=cfg.algorithm,
        checkpoint=checkpoint,
        curves=curves,
        final_metrics=final_metrics,
        config_provenance=ConfigProvenance.from_config(
            cfg,
            config_path=resolved_config_path,
        ),
        dataset_contract=load_dataset_contract(cfg.dataset_meta_path),
    )


def aggregate_comparison_report(
    run_artifacts: Sequence[RunArtifact],
) -> ComparisonReport:
    """Aggregate normalized runs into one comparison-ready report."""
    runs = tuple(sorted(run_artifacts, key=lambda item: item.algorithm))
    contracts = [run.dataset_contract for run in runs if run.dataset_contract is not None]

    if not contracts:
        shared_contract = None
        consistent = True
    else:
        shared_contract = contracts[0]
        consistent = all(contract == shared_contract for contract in contracts[1:])
        if not consistent:
            shared_contract = None

    return ComparisonReport(
        algorithms=tuple(run.algorithm for run in runs),
        dataset_contract_consistent=consistent,
        shared_dataset_contract=shared_contract,
        runs=runs,
    )


def build_comparison_report(
    config_paths: Iterable[str | Path],
) -> ComparisonReport:
    """Convenience wrapper for building and aggregating runs from config paths."""
    run_artifacts = [build_run_artifact(path) for path in config_paths]
    return aggregate_comparison_report(run_artifacts)


__all__ = [
    "CurvePoint",
    "MetricCurve",
    "ComparisonCheckpoint",
    "ConfigProvenance",
    "DatasetContractRecord",
    "RunArtifact",
    "ComparisonReport",
    "resolve_metrics_log_path",
    "resolve_latest_checkpoint",
    "load_dataset_contract",
    "load_metric_curves",
    "build_run_artifact",
    "aggregate_comparison_report",
    "build_comparison_report",
]
