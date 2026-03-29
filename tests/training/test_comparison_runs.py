"""Regression tests for standardized multi-algorithm comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from mimic_sepsis_rl.datasets.transitions import TransitionDatasetMeta
from mimic_sepsis_rl.training.common import CheckpointManager, MetricLogger
from mimic_sepsis_rl.training.comparison import (
    aggregate_comparison_report,
    build_comparison_report,
    build_run_artifact,
)
from mimic_sepsis_rl.training.config import load_training_config

COMMON_TOP_LEVEL_KEYS = {
    "algorithm",
    "checkpoint",
    "curves",
    "curve_names",
    "final_metrics",
    "config_provenance",
    "dataset_contract",
}


def _write_dataset_meta(
    tmp_path: Path,
    *,
    n_actions: int = 25,
    state_dim: int = 33,
    manifest_seed: int = 42,
    reward_spec_version: str = "1.0.0",
) -> Path:
    meta = TransitionDatasetMeta(
        spec_version="1.0.0",
        n_episodes=8,
        n_transitions=64,
        state_dim=state_dim,
        n_actions=n_actions,
        split_label="train",
        manifest_seed=manifest_seed,
        action_spec_version="1.0.0",
        reward_spec_version=reward_spec_version,
        feature_columns=("sofa", "lactate", "map"),
    )
    path = tmp_path / f"replay_train_meta_{manifest_seed}_{n_actions}.json"
    path.write_text(json.dumps(meta.to_dict(), indent=2))
    return path


def _write_config(
    tmp_path: Path,
    *,
    algorithm: str,
    dataset_meta_path: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "algorithm": algorithm,
        "schema_version": "1.0.0",
        "runtime": {"device": "cpu", "seed": 123, "num_workers": 0},
        "dataset_path": "data/replay/replay_train.parquet",
        "dataset_meta_path": str(dataset_meta_path),
        "n_epochs": 3,
        "batch_size": 32,
        "gamma": 0.99,
        "checkpoint": {
            "checkpoint_dir": str(tmp_path / f"{algorithm}_checkpoints"),
            "save_every_n_epochs": 1,
            "keep_last_n": 2,
        },
        "logging": {
            "log_dir": str(tmp_path / f"{algorithm}_runs"),
            "experiment_name": f"{algorithm}_comparison",
            "log_every_n_steps": 5,
        },
    }
    if extra:
        payload.update(extra)

    path = tmp_path / f"{algorithm}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _emit_run_artifacts(
    config_path: Path,
    *,
    final_metrics: dict[str, float],
    step_metrics: dict[str, float],
) -> None:
    cfg = load_training_config(config_path)

    metric_logger = MetricLogger.from_config(cfg)
    for step, (name, value) in enumerate(step_metrics.items(), start=1):
        metric_logger.log_scalar(name, value, step=step, epoch=1)
    metric_logger.log_epoch_summary(1, len(step_metrics), final_metrics)

    checkpoint_manager = CheckpointManager(
        cfg.checkpoint.checkpoint_dir,
        algorithm=cfg.algorithm,
        keep_last_n=cfg.checkpoint.keep_last_n,
    )
    checkpoint_manager.save(
        {"weights": torch.ones(1)},
        epoch=1,
        global_step=12,
        metrics=final_metrics,
        cfg=cfg,
        optimizer_state_dict={"optimizer": {"lr": 1e-3}},
    )


def _build_configured_run(
    tmp_path: Path,
    *,
    algorithm: str,
    dataset_meta_path: Path,
) -> Path:
    metric_sets = {
        "cql": (
            {
                "td_loss_mean": 0.5,
                "cql_loss_mean": 0.3,
                "total_loss_mean": 0.8,
            },
            {"td_loss": 0.55, "cql_loss": 0.35},
        ),
        "bcq": (
            {
                "td_loss_mean": 0.4,
                "imitation_loss_mean": 0.2,
                "total_loss_mean": 0.6,
            },
            {"td_loss": 0.45, "imitation_loss": 0.25},
        ),
        "iql": (
            {
                "critic_loss_mean": 0.7,
                "value_loss_mean": 0.25,
                "actor_loss_mean": 0.15,
                "total_loss_mean": 1.1,
            },
            {"critic_loss": 0.75, "value_loss": 0.3, "actor_loss": 0.2},
        ),
    }

    config_path = _write_config(
        tmp_path,
        algorithm=algorithm,
        dataset_meta_path=dataset_meta_path,
    )
    final_metrics, step_metrics = metric_sets[algorithm]
    _emit_run_artifacts(
        config_path,
        final_metrics=final_metrics,
        step_metrics=step_metrics,
    )
    return config_path


def test_build_run_artifact_normalizes_shared_schema_across_algorithms(
    tmp_path: Path,
) -> None:
    dataset_meta_path = _write_dataset_meta(tmp_path)
    artifacts = [
        build_run_artifact(
            _build_configured_run(
                tmp_path,
                algorithm=algorithm,
                dataset_meta_path=dataset_meta_path,
            )
        )
        for algorithm in ("cql", "bcq", "iql")
    ]

    top_level_keys = {frozenset(artifact.to_dict().keys()) for artifact in artifacts}
    assert top_level_keys == {frozenset(COMMON_TOP_LEVEL_KEYS)}

    for artifact in artifacts:
        payload = artifact.to_dict()
        assert set(payload) == COMMON_TOP_LEVEL_KEYS
        assert artifact.checkpoint is not None
        assert artifact.dataset_contract is not None
        assert artifact.dataset_contract.n_actions == 25
        assert artifact.config_provenance.batch_size == 32
        assert artifact.config_provenance.gamma == 0.99
        assert "total_loss_mean" in artifact.final_metrics
        assert artifact.curve_names


def test_aggregate_comparison_report_confirms_shared_dataset_contract(
    tmp_path: Path,
) -> None:
    dataset_meta_path = _write_dataset_meta(tmp_path)
    config_paths = [
        _build_configured_run(
            tmp_path,
            algorithm=algorithm,
            dataset_meta_path=dataset_meta_path,
        )
        for algorithm in ("cql", "bcq", "iql")
    ]

    report = build_comparison_report(config_paths)

    assert report.algorithms == ("bcq", "cql", "iql")
    assert report.dataset_contract_consistent is True
    assert report.shared_dataset_contract is not None
    assert report.shared_dataset_contract.reward_spec_version == "1.0.0"
    assert len(report.runs) == 3


def test_aggregate_comparison_report_flags_dataset_contract_drift(
    tmp_path: Path,
) -> None:
    cql_meta = _write_dataset_meta(tmp_path, n_actions=25, manifest_seed=42)
    bcq_meta = _write_dataset_meta(tmp_path, n_actions=37, manifest_seed=42)
    iql_meta = _write_dataset_meta(
        tmp_path,
        n_actions=25,
        manifest_seed=42,
        reward_spec_version="2.0.0",
    )

    artifacts = [
        build_run_artifact(
            _build_configured_run(
                tmp_path,
                algorithm="cql",
                dataset_meta_path=cql_meta,
            )
        ),
        build_run_artifact(
            _build_configured_run(
                tmp_path,
                algorithm="bcq",
                dataset_meta_path=bcq_meta,
            )
        ),
        build_run_artifact(
            _build_configured_run(
                tmp_path,
                algorithm="iql",
                dataset_meta_path=iql_meta,
            )
        ),
    ]

    report = aggregate_comparison_report(artifacts)

    assert report.dataset_contract_consistent is False
    assert report.shared_dataset_contract is None
