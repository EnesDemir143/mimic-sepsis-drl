"""Regression tests for offline RL reporting artifacts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mimic_sepsis_rl.evaluation.ope import (
    FrozenFQEOutputs,
    HeldOutEpisode,
    HeldOutStep,
    evaluate_policy_run,
)
from mimic_sepsis_rl.evaluation.safety import (
    ActionSupport,
    build_safety_review,
    build_safety_review_rows,
)
from mimic_sepsis_rl.mdp.actions.bins import ActionBinArtifacts
from mimic_sepsis_rl.reporting.offline_rl import (
    generate_training_report_artifacts,
    write_evaluation_report_artifacts,
)
from mimic_sepsis_rl.training.common import EventLogger, MetricLogger
from mimic_sepsis_rl.training.comparison import (
    ConfigProvenance,
    DatasetContractRecord,
    RunArtifact,
)
from mimic_sepsis_rl.training.config import build_training_config

STATE_DIM = 4
N_ACTIONS = 25


def _make_replay_parquet(tmp_path: Path) -> Path:
    rng = random.Random(42)
    rows: list[dict[str, float | int | bool]] = []
    for stay_id in range(4):
        for step_index in range(3):
            done = step_index == 2
            row: dict[str, float | int | bool] = {
                "stay_id": stay_id,
                "step_index": step_index,
                "action": (stay_id * 5 + step_index) % N_ACTIONS,
                "reward": 10.0 if done and stay_id % 2 == 0 else -10.0 if done else rng.uniform(-0.5, 0.5),
                "done": done,
            }
            for feature_idx in range(STATE_DIM):
                row[f"s_feat_{feature_idx}"] = rng.uniform(-1.0, 1.0)
                row[f"ns_feat_{feature_idx}"] = rng.uniform(-1.0, 1.0)
            rows.append(row)

    path = tmp_path / "replay_train.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _build_cfg(tmp_path: Path):
    return build_training_config(
        algorithm="cql",
        device="cpu",
        dataset_path=_make_replay_parquet(tmp_path),
        n_epochs=2,
        batch_size=8,
        gamma=0.99,
        seed=42,
        log_dir=tmp_path / "runs",
        checkpoint_dir=tmp_path / "checkpoints",
        experiment_name="report_test",
    )


def test_generate_training_report_artifacts_writes_json_logs_and_plots(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    metric_logger = MetricLogger.from_config(cfg)
    metric_logger.log_scalar("td_loss", 0.9, step=1, epoch=1)
    metric_logger.log_scalar("cql_loss", 0.4, step=1, epoch=1)
    metric_logger.log_scalar("mean_q_dataset", 0.2, step=1, epoch=1)
    metric_logger.log_scalar("td_loss", 0.6, step=2, epoch=2)
    metric_logger.log_scalar("cql_loss", 0.3, step=2, epoch=2)
    metric_logger.log_scalar("mean_q_dataset", 0.5, step=2, epoch=2)
    metric_logger.log_epoch_summary(
        2,
        2,
        {
            "td_loss_mean": 0.75,
            "cql_loss_mean": 0.35,
            "total_loss_mean": 1.10,
        },
    )

    training_logger = EventLogger.from_config(cfg, filename="training.log")
    runtime_logger = EventLogger.from_config(cfg, filename="runtime.log")
    training_logger.log_event(
        level="INFO",
        component="trainer",
        event="run_start",
        payload={"algorithm": "cql"},
    )
    runtime_logger.log_event(
        level="INFO",
        component="runtime",
        event="epoch_runtime",
        payload={"epoch": 1, "epoch_elapsed_seconds": 1.25},
    )

    artifacts = generate_training_report_artifacts(
        cfg,
        algorithm="cql",
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        total_steps=2,
        elapsed_seconds=2.5,
        final_metrics={
            "td_loss_mean": 0.75,
            "cql_loss_mean": 0.35,
            "total_loss_mean": 1.10,
        },
        checkpoint_path=None,
        epoch_durations=[1.2, 1.3],
        training_log_path=training_logger.log_path,
        runtime_log_path=runtime_logger.log_path,
    )

    artifact_dir = Path(artifacts.artifact_dir)
    assert artifact_dir.exists()
    assert Path(artifacts.run_manifest_path).exists()
    assert Path(artifacts.training_history_path).exists()
    assert Path(artifacts.metrics_summary_path).exists()
    assert Path(artifacts.runtime_summary_path).exists()
    assert Path(artifacts.artifact_index_path).exists()
    assert Path(artifacts.training_log_path).exists()
    assert Path(artifacts.runtime_log_path).exists()
    assert Path(artifacts.plots["step_metrics"]).exists()
    assert Path(artifacts.plots["epoch_metrics"]).exists()
    assert Path(artifacts.plots["q_diagnostics"]).exists()
    assert Path(artifacts.plots["dataset_action_heatmap"]).exists()
    assert Path(artifacts.plots["episode_reward_distribution"]).exists()

    manifest = json.loads(Path(artifacts.run_manifest_path).read_text())
    assert manifest["log_timezone"] == "Europe/Istanbul"
    assert manifest["n_actions"] == N_ACTIONS
    assert manifest["algorithm_name"] == "cql"


@dataclass
class MappingPolicy:
    """Deterministic policy keyed by exact held-out state tuples."""

    actions: dict[tuple[float, ...], int]

    def select_action(self, state: tuple[float, ...]) -> int:
        return self.actions[tuple(float(value) for value in state)]


def _q_values(best_action: int, *, best_value: float) -> tuple[float, ...]:
    values = [0.0] * N_ACTIONS
    values[best_action] = best_value
    return tuple(values)


def _run_artifact() -> RunArtifact:
    return RunArtifact(
        algorithm="cql",
        checkpoint=None,
        curves=tuple(),
        final_metrics={"total_loss_mean": 0.5},
        config_provenance=ConfigProvenance(
            config_path="configs/training/cql.yaml",
            checkpoint_dir="checkpoints/cql",
            log_dir="runs/cql",
            experiment_name="cql_reference",
            dataset_path="data/replay/replay_train.parquet",
            dataset_meta_path="data/replay/replay_train_meta.json",
            batch_size=256,
            gamma=1.0,
            requested_device="cpu",
            effective_backend="cpu",
        ),
        dataset_contract=DatasetContractRecord(
            spec_version="1.0.0",
            split_label="train",
            n_actions=N_ACTIONS,
            state_dim=2,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
            manifest_seed=42,
            n_episodes=10,
            n_transitions=40,
            feature_columns=("sofa", "lactate"),
        ),
    )


def _held_out_episodes() -> tuple[HeldOutEpisode, ...]:
    return (
        HeldOutEpisode(
            episode_id="ep-1",
            steps=(
                HeldOutStep(
                    episode_id="ep-1",
                    step_index=0,
                    state=(0.0, 1.0),
                    action=0,
                    reward=1.0,
                    done=False,
                    behavior_action_prob=0.5,
                ),
                HeldOutStep(
                    episode_id="ep-1",
                    step_index=1,
                    state=(0.1, 1.1),
                    action=1,
                    reward=2.0,
                    done=True,
                    behavior_action_prob=0.5,
                ),
            ),
        ),
        HeldOutEpisode(
            episode_id="ep-2",
            steps=(
                HeldOutStep(
                    episode_id="ep-2",
                    step_index=0,
                    state=(2.0, 0.0),
                    action=1,
                    reward=0.5,
                    done=False,
                    behavior_action_prob=0.25,
                ),
                HeldOutStep(
                    episode_id="ep-2",
                    step_index=1,
                    state=(2.1, 0.1),
                    action=0,
                    reward=0.5,
                    done=True,
                    behavior_action_prob=0.5,
                ),
            ),
        ),
    )


def _held_out_contract() -> DatasetContractRecord:
    return DatasetContractRecord(
        spec_version="1.0.0",
        split_label="test",
        n_actions=N_ACTIONS,
        state_dim=2,
        action_spec_version="1.0.0",
        reward_spec_version="1.0.0",
        manifest_seed=42,
        n_episodes=2,
        n_transitions=4,
        feature_columns=("sofa", "lactate"),
    )


def _action_bins() -> ActionBinArtifacts:
    return ActionBinArtifacts(
        spec_version="1.0.0",
        manifest_seed=42,
        vaso_edges=(0.1, 0.2, 0.3),
        fluid_edges=(100.0, 200.0, 300.0),
        n_train_vaso_nonzero=100,
        n_train_fluid_nonzero=100,
    )


def test_write_evaluation_report_artifacts_writes_json_and_plots(tmp_path: Path) -> None:
    policy = MappingPolicy(
        actions={
            (0.0, 1.0): 0,
            (0.1, 1.1): 1,
            (2.0, 0.0): 1,
            (2.1, 0.1): 2,
        }
    )
    ope_report = evaluate_policy_run(
        _run_artifact(),
        _held_out_episodes(),
        policy,
        FrozenFQEOutputs(
            fitted_split="train",
            initial_state_action_values={
                "ep-1": _q_values(0, best_value=2.5),
                "ep-2": _q_values(1, best_value=1.5),
            },
            artifact_label="fqe_train_fold0",
        ),
        held_out_contract=_held_out_contract(),
    )

    support_map = {
        0: ActionSupport(behavior_prob=0.30, count=25),
        1: ActionSupport(behavior_prob=0.25, count=20),
        2: ActionSupport(behavior_prob=0.01, count=2),
    }
    safety_rows = build_safety_review_rows(
        policy,
        _held_out_episodes(),
        lambda state, action: support_map[action],
        subgroup_lookup=lambda step: "early" if step.step_index == 0 else "late",
    )
    safety_report = build_safety_review(
        safety_rows,
        _action_bins(),
        min_behavior_prob=0.05,
        min_count=10,
        sanity_case_limit=2,
    )

    artifacts = write_evaluation_report_artifacts(
        tmp_path / "evaluation",
        ope_report=ope_report,
        safety_report=safety_report,
    )

    assert Path(artifacts["ope_summary"]).exists()
    assert Path(artifacts["ope_summary_plot"]).exists()
    assert Path(artifacts["policy_diagnostics"]).exists()
    assert Path(artifacts["policy_heatmaps"]).exists()
