"""Regression tests for the Phase 9 ablation registry."""

from __future__ import annotations

import pytest

from mimic_sepsis_rl.evaluation.ablations import (
    AblationComparisonReport,
    AblationDefaults,
    AblationDimension,
    AblationExperimentMetadata,
    REQUIRED_ABLATION_DIMENSIONS,
    build_default_ablation_registry,
)
from mimic_sepsis_rl.training.comparison import (
    ComparisonCheckpoint,
    ConfigProvenance,
    DatasetContractRecord,
    RunArtifact,
)


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
            gamma=0.99,
            requested_device="mps",
            effective_backend="mps",
        ),
        dataset_contract=DatasetContractRecord(
            spec_version="1.0.0",
            split_label="train",
            n_actions=25,
            state_dim=33,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
            manifest_seed=42,
            n_episodes=128,
            n_transitions=4096,
            feature_columns=("sofa", "lactate", "map"),
        ),
    )


def _base_metadata() -> AblationExperimentMetadata:
    defaults = AblationDefaults(benchmark_version="phase9-benchmark-v1")
    return AblationExperimentMetadata.from_run_artifact(_run_artifact(), defaults)


def test_default_ablation_registry_covers_all_required_dimensions() -> None:
    registry = build_default_ablation_registry(_base_metadata())

    seen_dimensions = {plan.variant.dimension for plan in registry.plans()}

    assert seen_dimensions == set(REQUIRED_ABLATION_DIMENSIONS)
    assert registry.base_metadata.benchmark_version == "phase9-benchmark-v1"


def test_ablation_plans_share_one_metadata_schema_and_benchmark_version() -> None:
    registry = build_default_ablation_registry(_base_metadata())
    plans = registry.plans()

    schemas = {
        tuple(sorted(plan.experiment_metadata.to_dict().keys()))
        for plan in plans
    }

    assert len(plans) >= len(REQUIRED_ABLATION_DIMENSIONS)
    assert schemas == {
        tuple(sorted(_base_metadata().to_dict().keys()))
    }
    assert all(
        plan.experiment_metadata.benchmark_version == "phase9-benchmark-v1"
        for plan in plans
    )
    assert {
        plan.experiment_metadata.reward_variant
        for plan in registry.plans_for_dimension(AblationDimension.REWARD_SHAPING)
    } == {"sofa_shaped", "sparse", "full_shaped"}


def test_build_report_requires_metrics_for_every_registered_variant() -> None:
    registry = build_default_ablation_registry(_base_metadata())

    with pytest.raises(ValueError, match="Controlled ablation comparisons require metrics"):
        registry.build_report(
            AblationDimension.REWARD_SHAPING,
            {
                "reward_sofa_shaped": {"wis": 1.0},
                "reward_sparse": {"wis": 0.8},
            },
        )


def test_build_report_preserves_metadata_schema_for_dimension() -> None:
    registry = build_default_ablation_registry(_base_metadata())

    report = registry.build_report(
        AblationDimension.REWARD_SHAPING,
        {
            "reward_sofa_shaped": {"wis": 1.0, "ess": 20.0},
            "reward_sparse": {"wis": 0.8, "ess": 18.0},
            "reward_full_shaped": {"wis": 1.1, "ess": 17.0},
        },
    )

    assert isinstance(report, AblationComparisonReport)
    assert report.dimension == AblationDimension.REWARD_SHAPING
    assert report.baseline_variant_id == "reward_sofa_shaped"
    assert report.metric_names == ("ess", "wis")
    assert all(
        result.plan.experiment_metadata.benchmark_version == "phase9-benchmark-v1"
        for result in report.results
    )
    report.validate()
