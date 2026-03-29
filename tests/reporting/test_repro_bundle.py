"""Regression tests for the reproducible reporting bundle."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mimic_sepsis_rl.evaluation.ablations import (
    AblationDefaults,
    AblationDimension,
    AblationExperimentMetadata,
    build_default_ablation_registry,
)
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
from mimic_sepsis_rl.mdp.reward_models import RewardConfig, RewardVariant
from mimic_sepsis_rl.reporting.package import (
    BackendMetadataRecord,
    ReproducibilityBundleInputs,
    RunBundleRecord,
    assemble_reproducibility_bundle,
    build_action_bin_record,
    build_cohort_record,
    build_feature_dictionary_record,
    build_reward_config_record,
    build_split_manifest_record,
)
from mimic_sepsis_rl.training.comparison import (
    ComparisonCheckpoint,
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
    return RunArtifact(
        algorithm="cql",
        checkpoint=ComparisonCheckpoint(
            checkpoint_path="checkpoints/cql/cql_epoch0003_step0000012.pt",
            manifest_path="checkpoints/cql/cql_epoch0003_step0000012_manifest.json",
            epoch=3,
            global_step=12,
            metrics={"total_loss_mean": 0.6},
        ),
        curves=tuple(),
        final_metrics={"total_loss_mean": 0.6},
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
            n_actions=25,
            state_dim=2,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
            manifest_seed=42,
            n_episodes=10,
            n_transitions=40,
            feature_columns=("sofa", "lactate"),
        ),
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


def _action_bins() -> ActionBinArtifacts:
    return ActionBinArtifacts(
        spec_version="1.0.0",
        manifest_seed=42,
        vaso_edges=(0.1, 0.2, 0.3),
        fluid_edges=(100.0, 200.0, 300.0),
        n_train_vaso_nonzero=100,
        n_train_fluid_nonzero=100,
    )


def _ope_and_safety_reports() -> tuple[object, object]:
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
    return ope_report, safety_report


def _backend_metadata() -> BackendMetadataRecord:
    return BackendMetadataRecord.from_mapping(
        {
            "backend": "cpu",
            "requested_backend": "cpu",
            "torch_device_str": "cpu",
            "torch_version": "2.11.0",
            "cuda_available": False,
            "mps_available": False,
            "mps_built": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "python_version": "3.12.0",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "device_module_version": "1.0.0",
            "fallback_applied": False,
            "mps_fallback_ops_enabled": False,
        },
        checkpoint_manifest_path="checkpoints/cql/cql_epoch0003_step0000012_manifest.json",
    )


def _ablation_report():
    defaults = AblationDefaults(benchmark_version="phase9-benchmark-v1")
    base_metadata = AblationExperimentMetadata.from_run_artifact(_run_artifact(), defaults)
    registry = build_default_ablation_registry(base_metadata)
    return registry.build_report(
        AblationDimension.REWARD_SHAPING,
        {
            "reward_sofa_shaped": {"wis": 1.0, "ess": 20.0},
            "reward_sparse": {"wis": 0.8, "ess": 18.0},
            "reward_full_shaped": {"wis": 1.1, "ess": 17.0},
        },
    )


def test_assemble_reproducibility_bundle_collects_required_metadata() -> None:
    ope_report, safety_report = _ope_and_safety_reports()
    run_record = RunBundleRecord.from_run_artifact(
        _run_artifact(),
        backend_metadata=_backend_metadata(),
        ope_report=ope_report,
        safety_report=safety_report,
    )

    bundle = assemble_reproducibility_bundle(
        ReproducibilityBundleInputs(
            benchmark_version="phase9-benchmark-v1",
            cohort=build_cohort_record(
                {"spec_version": "1.0.0", "adult_only": True, "min_age_years": 18},
                spec_path="configs/cohort/default.yaml",
                source_tables={"icustays": "mimiciv_icu.icustays"},
            ),
            feature_dictionary=build_feature_dictionary_record(
                {
                    "spec_version": "1.0.0",
                    "total_features": 33,
                    "total_missingness_flag_features": 5,
                },
                spec_path="src/mimic_sepsis_rl/mdp/features/dictionary.py",
            ),
            split_manifest=build_split_manifest_record(
                {
                    "spec_version": "1.0.0",
                    "seed": 42,
                    "source_episode_set": "episodes_v1",
                    "has_leakage": False,
                },
                manifest_dir="data/splits",
            ),
            action_bins=build_action_bin_record(
                _action_bins(),
                artifact_path="artifacts/actions/action_bins.json",
            ),
            reward_config=build_reward_config_record(
                RewardConfig(variant=RewardVariant.SOFA_SHAPED),
                config_path="artifacts/rewards/reward_config.json",
            ),
            run_records=(run_record,),
            ablation_reports=(_ablation_report(),),
        )
    )

    payload = bundle.to_dict()

    assert payload["benchmark_version"] == "phase9-benchmark-v1"
    assert payload["cohort"]["metadata"]["spec_version"] == "1.0.0"
    assert payload["feature_dictionary"]["metadata"]["total_features"] == 33
    assert payload["split_manifest"]["metadata"]["seed"] == 42
    assert payload["action_bins"]["metadata"]["manifest_seed"] == 42
    assert payload["reward_config"]["metadata"]["variant"] == "sofa_shaped"
    assert payload["run_records"][0]["backend_metadata"]["device_meta"]["backend"] == "cpu"
    assert payload["run_records"][0]["evaluation"]["ope_metrics"]["wis"] == pytest.approx(3.0)
    assert payload["ablation_reports"][0]["dimension"] == "reward_shaping"
    assert payload["rerun_instructions"]


def test_bundle_requires_recorded_backend_metadata_fields() -> None:
    ope_report, _ = _ope_and_safety_reports()

    with pytest.raises(ValueError, match="Recorded backend metadata must come from a checkpoint manifest"):
        run_record = RunBundleRecord.from_run_artifact(
            _run_artifact(),
            backend_metadata=BackendMetadataRecord.from_mapping(
                {"backend": "cpu", "requested_backend": "cpu"},
                checkpoint_manifest_path="checkpoints/cql/cql_epoch0003_step0000012_manifest.json",
            ),
            ope_report=ope_report,
        )
        run_record.validate()
