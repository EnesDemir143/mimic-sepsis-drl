"""Regression tests for held-out OPE evaluation wiring."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mimic_sepsis_rl.evaluation.ope import (
    FrozenFQEOutputs,
    HeldOutEpisode,
    HeldOutStep,
    evaluate_policy_run,
)
from mimic_sepsis_rl.training.comparison import (
    ConfigProvenance,
    DatasetContractRecord,
    RunArtifact,
)


@dataclass
class MappingPolicy:
    """Deterministic policy keyed by exact held-out state tuples."""

    actions: dict[tuple[float, ...], int]

    def select_action(self, state: tuple[float, ...]) -> int:
        return self.actions[tuple(float(value) for value in state)]


def _q_values(best_action: int, *, best_value: float) -> tuple[float, ...]:
    values = [0.0] * 25
    values[best_action] = best_value
    return tuple(values)


def _run_artifact() -> RunArtifact:
    config_provenance = ConfigProvenance(
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
    )
    dataset_contract = DatasetContractRecord(
        spec_version="1.0.0",
        split_label="train",
        n_actions=25,
        state_dim=2,
        action_spec_version="1.0.0",
        reward_spec_version="1.0.0",
        manifest_seed=42,
        n_episodes=10,
        n_transitions=40,
        feature_columns=("sofa", "lactate"),
    )
    return RunArtifact(
        algorithm="cql",
        checkpoint=None,
        curves=tuple(),
        final_metrics={"total_loss_mean": 0.25},
        config_provenance=config_provenance,
        dataset_contract=dataset_contract,
    )


def _held_out_contract() -> DatasetContractRecord:
    return DatasetContractRecord(
        spec_version="1.0.0",
        split_label="test",
        n_actions=25,
        state_dim=2,
        action_spec_version="1.0.0",
        reward_spec_version="1.0.0",
        manifest_seed=42,
        n_episodes=2,
        n_transitions=4,
        feature_columns=("sofa", "lactate"),
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


def test_evaluate_policy_run_reports_wis_ess_and_fqe_from_held_out_artifacts() -> None:
    policy = MappingPolicy(
        actions={
            (0.0, 1.0): 0,
            (0.1, 1.1): 1,
            (2.0, 0.0): 1,
            (2.1, 0.1): 2,
        }
    )
    frozen_fqe = FrozenFQEOutputs(
        fitted_split="train",
        initial_state_action_values={
            "ep-1": _q_values(0, best_value=2.5),
            "ep-2": _q_values(1, best_value=1.5),
        },
        artifact_label="fqe_train_fold0",
    )

    report = evaluate_policy_run(
        _run_artifact(),
        _held_out_episodes(),
        policy,
        frozen_fqe,
        held_out_contract=_held_out_contract(),
    )

    assert report.algorithm == "cql"
    assert report.contract_check.is_consistent is True
    assert report.metrics.wis == pytest.approx(3.0)
    assert report.metrics.ess == pytest.approx(1.0)
    assert report.metrics.fqe == pytest.approx(2.0)
    assert report.metrics.mean_behavior_return == pytest.approx(2.0)
    assert report.metrics.wis_nonzero_episodes == 1
    assert report.per_episode[0].importance_weight == pytest.approx(4.0)
    assert report.per_episode[1].importance_weight == pytest.approx(0.0)
    assert report.per_episode[1].matched_step_fraction == pytest.approx(0.5)


def test_evaluate_policy_run_requires_logged_behavior_policy_probabilities() -> None:
    invalid_episode = HeldOutEpisode(
        episode_id="bad-ep",
        steps=(
            HeldOutStep(
                episode_id="bad-ep",
                step_index=0,
                state=(0.0, 0.0),
                action=0,
                reward=1.0,
                done=True,
                behavior_action_prob=0.0,
            ),
        ),
    )
    policy = MappingPolicy(actions={(0.0, 0.0): 0})
    frozen_fqe = FrozenFQEOutputs(
        fitted_split="train",
        initial_state_action_values={"bad-ep": _q_values(0, best_value=1.0)},
    )

    with pytest.raises(ValueError, match="behavior_action_prob"):
        evaluate_policy_run(
            _run_artifact(),
            (invalid_episode,),
            policy,
            frozen_fqe,
            held_out_contract=_held_out_contract(),
        )


def test_evaluate_policy_run_rejects_fqe_fit_on_held_out_split() -> None:
    policy = MappingPolicy(
        actions={
            (0.0, 1.0): 0,
            (0.1, 1.1): 1,
            (2.0, 0.0): 1,
            (2.1, 0.1): 0,
        }
    )
    frozen_fqe = FrozenFQEOutputs(
        fitted_split="test",
        initial_state_action_values={
            "ep-1": _q_values(0, best_value=2.0),
            "ep-2": _q_values(1, best_value=1.0),
        },
    )

    with pytest.raises(ValueError, match="never on the held-out evaluation split"):
        evaluate_policy_run(
            _run_artifact(),
            _held_out_episodes(),
            policy,
            frozen_fqe,
            held_out_contract=_held_out_contract(),
        )
