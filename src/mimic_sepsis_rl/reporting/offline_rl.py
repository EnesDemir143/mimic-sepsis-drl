"""Offline RL experiment reporting utilities.

This module turns raw training metrics, replay-buffer statistics, and optional
evaluation reports into a skill-aligned reporting bundle with JSON summaries,
timestamped log references, and publication-ready plots.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import torch

from mimic_sepsis_rl.datasets.transitions import TransitionDatasetMeta
from mimic_sepsis_rl.evaluation.ope import PolicyOPEReport
from mimic_sepsis_rl.evaluation.safety import (
    FLUID_BIN_LABELS,
    VASO_BIN_LABELS,
    SafetyReviewReport,
)
from mimic_sepsis_rl.mdp.actions.bins import ActionBinner, N_BINS
from mimic_sepsis_rl.training.common import LOG_TIMEZONE_NAME
from mimic_sepsis_rl.training.comparison import MetricCurve, load_metric_curves
from mimic_sepsis_rl.training.config import TrainingConfig

MB = 1024 * 1024
ROLLING_WINDOW = 50


@dataclass(frozen=True)
class ReportArtifactIndex:
    """Index of generated reporting artifacts for one run."""

    artifact_dir: str
    run_manifest_path: str
    training_history_path: str
    metrics_summary_path: str
    runtime_summary_path: str
    artifact_index_path: str
    training_log_path: str | None = None
    runtime_log_path: str | None = None
    plots: dict[str, str] = field(default_factory=dict)
    evaluation_artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _artifact_dir(cfg: TrainingConfig) -> Path:
    return cfg.logging.log_dir / cfg.logging.experiment_name


def _metrics_path(cfg: TrainingConfig) -> Path:
    return cfg.logging.log_dir / f"{cfg.logging.experiment_name}_metrics.jsonl"


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _series_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "final": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "final": values[-1],
    }


def _curve_points(curve: MetricCurve) -> list[dict[str, Any]]:
    return [point.to_dict() for point in curve.points]


def _load_dataset_meta(dataset_meta_path: Path | None) -> TransitionDatasetMeta | None:
    if dataset_meta_path is None or not dataset_meta_path.exists():
        return None
    payload = json.loads(dataset_meta_path.read_text())
    return TransitionDatasetMeta.from_dict(payload)


def _load_replay_frame(dataset_path: Path) -> pl.DataFrame | None:
    if not dataset_path.exists():
        return None
    return pl.read_parquet(dataset_path)


def _episode_returns(frame: pl.DataFrame) -> list[float]:
    episode_key = None
    for candidate in ("episode_id", "stay_id"):
        if candidate in frame.columns:
            episode_key = candidate
            break
    if episode_key is None or "reward" not in frame.columns:
        return []

    ordered = frame.with_row_index("row_index")
    returns = (
        ordered.group_by(episode_key)
        .agg(
            pl.col("reward").sum().alias("episode_return"),
            pl.col("row_index").min().alias("row_index"),
        )
        .sort("row_index")
    )
    return [float(value) for value in returns["episode_return"].to_list()]


def _action_heatmap_counts(frame: pl.DataFrame) -> list[list[int]] | None:
    if "action" not in frame.columns:
        return None

    counts = [[0 for _ in range(N_BINS)] for _ in range(N_BINS)]
    decoder = ActionBinner()
    for action_id in frame["action"].to_list():
        action = int(action_id)
        if not 0 <= action < N_BINS * N_BINS:
            continue
        vaso_bin, fluid_bin = decoder.decode_action(action)
        counts[vaso_bin][fluid_bin] += 1
    return counts


def _plot_curves(
    curves: Sequence[MetricCurve],
    *,
    x_axis: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> bool:
    if not curves:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    for curve in curves:
        if not curve.points:
            continue
        if x_axis == "epoch":
            x_values = [point.epoch for point in curve.points]
        else:
            x_values = [point.step for point in curve.points]
        y_values = [point.value for point in curve.points]
        ax.plot(x_values, y_values, label=curve.name, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel(x_axis.title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(curves) <= 8:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _plot_episode_rewards(episode_returns: Sequence[float], output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    if not episode_returns:
        return paths

    raw_curve_path = output_dir / "episode_reward_curve.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(episode_returns) + 1), episode_returns, linewidth=1.2)
    ax.set_title("Episode Reward Curve")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Episode Return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(raw_curve_path, dpi=180)
    plt.close(fig)
    paths["episode_reward_curve"] = str(raw_curve_path)

    if len(episode_returns) > 1:
        rolling_window = min(ROLLING_WINDOW, len(episode_returns))
        rolling = []
        for idx in range(len(episode_returns)):
            start = max(0, idx - rolling_window + 1)
            window = episode_returns[start : idx + 1]
            rolling.append(statistics.fmean(window))
        rolling_path = output_dir / "episode_reward_rolling_curve.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(rolling) + 1), rolling, linewidth=1.6)
        ax.set_title(f"Episode Reward Rolling Mean ({rolling_window})")
        ax.set_xlabel("Episode Index")
        ax.set_ylabel("Rolling Mean Return")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(rolling_path, dpi=180)
        plt.close(fig)
        paths["episode_reward_rolling_curve"] = str(rolling_path)

    distribution_path = output_dir / "episode_reward_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(episode_returns, bins=min(40, max(10, len(episode_returns) // 4)), alpha=0.8)
    ax.set_title("Episode Reward Distribution")
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)
    paths["episode_reward_distribution"] = str(distribution_path)
    return paths


def _plot_action_heatmap(
    counts: Sequence[Sequence[int]] | None,
    *,
    output_path: Path,
    title: str,
) -> bool:
    if counts is None:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(counts, cmap="YlOrRd")
    ax.set_xticks(range(N_BINS), FLUID_BIN_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(N_BINS), VASO_BIN_LABELS)
    ax.set_title(title)
    ax.set_xlabel("Fluid Bin")
    ax.set_ylabel("Vasopressor Bin")
    for row in range(N_BINS):
        for col in range(N_BINS):
            ax.text(col, row, str(int(counts[row][col])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_metrics_summary(curves: Sequence[MetricCurve]) -> dict[str, Any]:
    metric_summary: dict[str, Any] = {}
    for curve in curves:
        values = [float(point.value) for point in curve.points]
        summary = _series_stats(values)
        summary["first_step"] = curve.points[0].step if curve.points else None
        summary["last_step"] = curve.points[-1].step if curve.points else None
        summary["first_epoch"] = curve.points[0].epoch if curve.points else None
        summary["last_epoch"] = curve.points[-1].epoch if curve.points else None
        metric_summary[curve.name] = summary
    return {
        "metric_count": len(metric_summary),
        "metrics": metric_summary,
    }


def _build_runtime_summary(
    cfg: TrainingConfig,
    *,
    epoch_durations: Sequence[float],
    elapsed_seconds: float,
    total_steps: int,
) -> dict[str, Any]:
    peak_gpu_memory_mb = None
    telemetry_gaps = [
        "cpu_util_percent not collected by the built-in trainer",
        "gpu_power_watts not collected by the built-in trainer",
        "energy_wh_cumulative not collected by the built-in trainer",
    ]

    if cfg.device.type == "cuda" and torch.cuda.is_available():
        peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated(cfg.device) / MB)
    elif cfg.device.type == "mps":
        current_allocated = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current_allocated):
            peak_gpu_memory_mb = float(current_allocated() / MB)

    epoch_durations_list = [float(value) for value in epoch_durations]
    return {
        "run_id": cfg.logging.experiment_name,
        "total_train_seconds": float(elapsed_seconds),
        "total_eval_seconds": 0.0,
        "mean_epoch_seconds": _safe_float(statistics.fmean(epoch_durations_list))
        if epoch_durations_list
        else None,
        "median_epoch_seconds": _safe_float(statistics.median(epoch_durations_list))
        if epoch_durations_list
        else None,
        "peak_cpu_util_percent": None,
        "peak_system_ram_mb": None,
        "peak_gpu_util_percent": None,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "mean_gpu_power_watts": None,
        "peak_gpu_power_watts": None,
        "total_energy_wh": None,
        "power_tracking_available": False,
        "power_tracking_note": "Built-in trainer records wall-clock timing and memory only.",
        "device_backend": cfg.device_meta.backend,
        "total_steps": int(total_steps),
        "epoch_durations_seconds": epoch_durations_list,
        "telemetry_gaps": telemetry_gaps,
    }


def generate_training_report_artifacts(
    cfg: TrainingConfig,
    *,
    algorithm: str,
    state_dim: int,
    n_actions: int,
    total_steps: int,
    elapsed_seconds: float,
    final_metrics: Mapping[str, float],
    checkpoint_path: Path | None,
    epoch_durations: Sequence[float],
    training_log_path: Path | None = None,
    runtime_log_path: Path | None = None,
) -> ReportArtifactIndex:
    """Generate JSON summaries and plots for one training run."""
    artifact_dir = _artifact_dir(cfg)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = _metrics_path(cfg)
    curves = load_metric_curves(metrics_path)
    dataset_meta = _load_dataset_meta(cfg.dataset_meta_path)
    replay_frame = _load_replay_frame(cfg.dataset_path)

    manifest_payload = {
        "run_id": cfg.logging.experiment_name,
        "experiment_group": algorithm,
        "model_name": algorithm,
        "algorithm_name": algorithm,
        "task_family": "offline_rl",
        "task_type": "offline_reinforcement_learning",
        "dataset_path": str(cfg.dataset_path),
        "dataset_version": dataset_meta.spec_version if dataset_meta else cfg.dataset_path.name,
        "split_label": dataset_meta.split_label if dataset_meta else None,
        "reward_version": dataset_meta.reward_spec_version if dataset_meta else None,
        "action_spec_version": dataset_meta.action_spec_version if dataset_meta else None,
        "state_dim": int(state_dim),
        "n_actions": int(n_actions),
        "seed": int(cfg.runtime.seed),
        "deterministic_mode": True,
        "log_timezone": LOG_TIMEZONE_NAME,
        "log_timestamp_format": "ISO8601",
        "batch_size": int(cfg.batch_size),
        "gamma": float(cfg.gamma),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "requested_device": cfg.runtime.requested_device,
        "effective_backend": cfg.device_meta.backend,
        "final_metrics": dict(final_metrics),
        "metrics_log_path": str(metrics_path),
        "training_log_path": str(training_log_path) if training_log_path else None,
        "runtime_log_path": str(runtime_log_path) if runtime_log_path else None,
    }
    run_manifest_path = artifact_dir / "run_manifest.json"
    _write_json(run_manifest_path, manifest_payload)

    training_history_payload = {
        "algorithm": algorithm,
        "experiment_name": cfg.logging.experiment_name,
        "metrics_path": str(metrics_path),
        "curves": {curve.name: _curve_points(curve) for curve in curves},
    }
    training_history_path = artifact_dir / "training_history.json"
    _write_json(training_history_path, training_history_payload)

    metrics_summary = _build_metrics_summary(curves)
    metrics_summary.update(
        {
            "algorithm": algorithm,
            "experiment_name": cfg.logging.experiment_name,
            "final_metrics": dict(final_metrics),
        }
    )
    metrics_summary_path = artifact_dir / "metrics_summary.json"
    _write_json(metrics_summary_path, metrics_summary)

    runtime_summary = _build_runtime_summary(
        cfg,
        epoch_durations=epoch_durations,
        elapsed_seconds=elapsed_seconds,
        total_steps=total_steps,
    )
    runtime_summary_path = artifact_dir / "runtime_summary.json"
    _write_json(runtime_summary_path, runtime_summary)

    plots: dict[str, str] = {}
    step_curves = tuple(
        curve
        for curve in curves
        if not curve.name.endswith("_mean")
        and not curve.name.endswith("_min")
        and not curve.name.endswith("_max")
    )
    if _plot_curves(
        step_curves,
        x_axis="step",
        output_path=artifact_dir / "step_metrics.png",
        title=f"{algorithm.upper()} Step Metrics",
        ylabel="Metric Value",
    ):
        plots["step_metrics"] = str(artifact_dir / "step_metrics.png")

    epoch_curves = tuple(
        curve
        for curve in curves
        if curve.name.endswith("_mean")
    )
    if _plot_curves(
        epoch_curves,
        x_axis="epoch",
        output_path=artifact_dir / "epoch_metrics.png",
        title=f"{algorithm.upper()} Epoch Metrics",
        ylabel="Metric Value",
    ):
        plots["epoch_metrics"] = str(artifact_dir / "epoch_metrics.png")

    q_curves = tuple(
        curve
        for curve in curves
        if "q" in curve.name.lower() or "advantage" in curve.name.lower()
    )
    if _plot_curves(
        q_curves,
        x_axis="step",
        output_path=artifact_dir / "q_diagnostics.png",
        title=f"{algorithm.upper()} Value Diagnostics",
        ylabel="Diagnostic Value",
    ):
        plots["q_diagnostics"] = str(artifact_dir / "q_diagnostics.png")

    if replay_frame is not None:
        counts = _action_heatmap_counts(replay_frame)
        if _plot_action_heatmap(
            counts,
            output_path=artifact_dir / "dataset_action_heatmap.png",
            title="Replay Action Heatmap",
        ):
            plots["dataset_action_heatmap"] = str(artifact_dir / "dataset_action_heatmap.png")

        episode_returns = _episode_returns(replay_frame)
        plots.update(_plot_episode_rewards(episode_returns, artifact_dir))

    artifact_index = ReportArtifactIndex(
        artifact_dir=str(artifact_dir),
        run_manifest_path=str(run_manifest_path),
        training_history_path=str(training_history_path),
        metrics_summary_path=str(metrics_summary_path),
        runtime_summary_path=str(runtime_summary_path),
        artifact_index_path=str(artifact_dir / "artifact_index.json"),
        training_log_path=str(training_log_path) if training_log_path else None,
        runtime_log_path=str(runtime_log_path) if runtime_log_path else None,
        plots=plots,
    )
    _write_json(Path(artifact_index.artifact_index_path), artifact_index.to_dict())
    return artifact_index


def _plot_ope_summary(report: PolicyOPEReport, output_path: Path) -> bool:
    metrics = report.metrics.to_dict()
    values = {
        "wis": float(metrics["wis"]),
        "fqe": float(metrics["fqe"]),
        "ess": float(metrics["ess"]),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(values.keys()), list(values.values()), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title(f"{report.algorithm.upper()} OPE Summary")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _plot_safety_heatmaps(report: SafetyReviewReport, output_path: Path) -> bool:
    heatmaps = (
        ("Clinician", report.clinician_heatmap.counts),
        ("Policy", report.policy_heatmap.counts),
        ("Delta", report.delta_heatmap.counts),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (title, counts) in zip(axes, heatmaps, strict=True):
        image = axis.imshow(counts, cmap="YlOrRd")
        axis.set_title(title)
        axis.set_xticks(range(N_BINS), FLUID_BIN_LABELS, rotation=30, ha="right")
        axis.set_yticks(range(N_BINS), VASO_BIN_LABELS)
        for row in range(N_BINS):
            for col in range(N_BINS):
                axis.text(col, row, str(int(counts[row][col])), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def write_evaluation_report_artifacts(
    output_dir: Path,
    *,
    ope_report: PolicyOPEReport | None = None,
    safety_report: SafetyReviewReport | None = None,
) -> dict[str, str]:
    """Persist evaluation-side JSON summaries and optional plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    if ope_report is not None:
        ope_summary_path = output_dir / "ope_summary.json"
        _write_json(ope_summary_path, ope_report.to_dict())
        artifacts["ope_summary"] = str(ope_summary_path)
        if _plot_ope_summary(ope_report, output_dir / "ope_summary.png"):
            artifacts["ope_summary_plot"] = str(output_dir / "ope_summary.png")

    if safety_report is not None:
        policy_diag_path = output_dir / "policy_diagnostics.json"
        _write_json(policy_diag_path, safety_report.to_dict())
        artifacts["policy_diagnostics"] = str(policy_diag_path)
        if _plot_safety_heatmaps(safety_report, output_dir / "policy_heatmaps.png"):
            artifacts["policy_heatmaps"] = str(output_dir / "policy_heatmaps.png")

    return artifacts


__all__ = [
    "ReportArtifactIndex",
    "generate_training_report_artifacts",
    "write_evaluation_report_artifacts",
]
