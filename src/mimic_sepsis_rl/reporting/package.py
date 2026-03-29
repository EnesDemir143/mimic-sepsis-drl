"""
Reproducible reporting bundle assembly for the Phase 9 benchmark.

The bundle collects frozen upstream specifications, model artifacts,
evaluation outputs, ablation reports, and recorded accelerator metadata so
researchers can rerun or audit benchmark claims without rediscovering file
contracts by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from mimic_sepsis_rl.evaluation.ablations import AblationComparisonReport
from mimic_sepsis_rl.evaluation.ope import PolicyOPEReport
from mimic_sepsis_rl.evaluation.safety import SafetyReviewReport
from mimic_sepsis_rl.mdp.actions.bins import ActionBinArtifacts
from mimic_sepsis_rl.mdp.reward_models import RewardConfig
from mimic_sepsis_rl.training.common import CheckpointManifest
from mimic_sepsis_rl.training.comparison import RunArtifact

REPRO_BUNDLE_SCHEMA_VERSION = "1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _copy_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items()}


@dataclass(frozen=True)
class BundleArtifactRecord:
    """Reference to one frozen upstream artifact and its metadata."""

    label: str
    source_path: str | None
    metadata: dict[str, Any]

    def validate(self) -> None:
        if not self.label.strip():
            raise ValueError("Bundle artifact label must be non-empty.")
        if not self.metadata:
            raise ValueError(f"{self.label} metadata must not be empty.")

    def require_keys(self, *keys: str) -> None:
        self.validate()
        missing = [key for key in keys if key not in self.metadata]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{self.label} metadata is missing required keys: {joined}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_path": self.source_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BackendMetadataRecord:
    """Recorded accelerator metadata captured from a checkpoint manifest."""

    requested_backend: str
    effective_backend: str
    checkpoint_manifest_path: str | None
    device_meta: dict[str, Any]

    @classmethod
    def from_checkpoint_manifest(
        cls,
        manifest: CheckpointManifest,
        *,
        checkpoint_manifest_path: str | None = None,
    ) -> "BackendMetadataRecord":
        device_meta = _copy_mapping(manifest.device_meta)
        return cls(
            requested_backend=str(device_meta.get("requested_backend", "")),
            effective_backend=str(device_meta.get("backend", "")),
            checkpoint_manifest_path=checkpoint_manifest_path,
            device_meta=device_meta,
        )

    @classmethod
    def from_mapping(
        cls,
        device_meta: Mapping[str, Any],
        *,
        checkpoint_manifest_path: str | None = None,
    ) -> "BackendMetadataRecord":
        copied = _copy_mapping(device_meta)
        return cls(
            requested_backend=str(copied.get("requested_backend", "")),
            effective_backend=str(copied.get("backend", "")),
            checkpoint_manifest_path=checkpoint_manifest_path,
            device_meta=copied,
        )

    def validate(self) -> None:
        required_device_keys = ("backend", "requested_backend", "torch_device_str", "torch_version")
        missing = [key for key in required_device_keys if key not in self.device_meta]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "Recorded backend metadata must come from a checkpoint manifest; "
                f"missing {joined}."
            )
        if self.effective_backend != str(self.device_meta["backend"]):
            raise ValueError("effective_backend does not match device_meta.backend.")
        if self.requested_backend != str(self.device_meta["requested_backend"]):
            raise ValueError("requested_backend does not match device_meta.requested_backend.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "requested_backend": self.requested_backend,
            "effective_backend": self.effective_backend,
            "checkpoint_manifest_path": self.checkpoint_manifest_path,
            "device_meta": dict(self.device_meta),
        }


@dataclass(frozen=True)
class EvaluationSummaryRecord:
    """Evaluation and safety outputs attached to one run artifact."""

    algorithm: str
    ope_metrics: dict[str, Any]
    contract_check: dict[str, Any]
    held_out_contract: dict[str, Any] | None
    safety_summary: dict[str, Any] | None
    support_warning_count: int
    sanity_case_count: int

    @classmethod
    def from_reports(
        cls,
        ope_report: PolicyOPEReport,
        *,
        safety_report: SafetyReviewReport | None = None,
    ) -> "EvaluationSummaryRecord":
        return cls(
            algorithm=ope_report.algorithm,
            ope_metrics=ope_report.metrics.to_dict(),
            contract_check=ope_report.contract_check.to_dict(),
            held_out_contract=ope_report.held_out_contract.to_dict()
            if ope_report.held_out_contract
            else None,
            safety_summary=safety_report.agreement_summary.to_dict() if safety_report else None,
            support_warning_count=len(safety_report.support_warnings) if safety_report else 0,
            sanity_case_count=len(safety_report.sanity_cases) if safety_report else 0,
        )

    def validate(self) -> None:
        required_metrics = ("wis", "ess", "fqe", "n_episodes")
        missing = [key for key in required_metrics if key not in self.ope_metrics]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Evaluation summary is missing OPE metrics: {joined}.")
        if not self.algorithm.strip():
            raise ValueError("Evaluation summary algorithm must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "algorithm": self.algorithm,
            "ope_metrics": dict(self.ope_metrics),
            "contract_check": dict(self.contract_check),
            "held_out_contract": dict(self.held_out_contract)
            if self.held_out_contract
            else None,
            "safety_summary": dict(self.safety_summary) if self.safety_summary else None,
            "support_warning_count": self.support_warning_count,
            "sanity_case_count": self.sanity_case_count,
        }


@dataclass(frozen=True)
class RunBundleRecord:
    """One algorithm run packaged for reproducibility review."""

    algorithm: str
    experiment_name: str
    config_path: str
    dataset_path: str
    dataset_meta_path: str | None
    requested_device: str
    effective_backend: str
    checkpoint_path: str
    checkpoint_manifest_path: str
    final_metrics: dict[str, float]
    dataset_contract: dict[str, Any] | None
    backend_metadata: BackendMetadataRecord
    evaluation: EvaluationSummaryRecord

    @classmethod
    def from_run_artifact(
        cls,
        run_artifact: RunArtifact,
        *,
        backend_metadata: BackendMetadataRecord,
        ope_report: PolicyOPEReport,
        safety_report: SafetyReviewReport | None = None,
    ) -> "RunBundleRecord":
        checkpoint = run_artifact.checkpoint
        if checkpoint is None:
            raise ValueError("Run artifact must include a checkpoint for reproducibility bundling.")

        evaluation = EvaluationSummaryRecord.from_reports(
            ope_report,
            safety_report=safety_report,
        )
        config = run_artifact.config_provenance
        return cls(
            algorithm=run_artifact.algorithm,
            experiment_name=config.experiment_name,
            config_path=config.config_path,
            dataset_path=config.dataset_path,
            dataset_meta_path=config.dataset_meta_path,
            requested_device=config.requested_device,
            effective_backend=config.effective_backend,
            checkpoint_path=checkpoint.checkpoint_path,
            checkpoint_manifest_path=checkpoint.manifest_path,
            final_metrics=dict(run_artifact.final_metrics),
            dataset_contract=run_artifact.dataset_contract.to_dict()
            if run_artifact.dataset_contract
            else None,
            backend_metadata=backend_metadata,
            evaluation=evaluation,
        )

    def validate(self) -> None:
        if not self.algorithm.strip():
            raise ValueError("Run bundle record algorithm must be non-empty.")
        if not self.config_path.strip():
            raise ValueError("Run bundle record config_path must be non-empty.")
        if not self.dataset_path.strip():
            raise ValueError("Run bundle record dataset_path must be non-empty.")
        if not self.requested_device.strip():
            raise ValueError("Run bundle record requested_device must be non-empty.")
        if not self.effective_backend.strip():
            raise ValueError("Run bundle record effective_backend must be non-empty.")
        if not self.checkpoint_path.strip():
            raise ValueError("Run bundle record checkpoint_path must be non-empty.")
        if not self.checkpoint_manifest_path.strip():
            raise ValueError("Run bundle record checkpoint_manifest_path must be non-empty.")
        self.backend_metadata.validate()
        self.evaluation.validate()
        if self.backend_metadata.requested_backend != self.requested_device:
            raise ValueError(
                "Checkpoint manifest requested_backend does not match the saved run config."
            )
        if self.backend_metadata.effective_backend != self.effective_backend:
            raise ValueError(
                "Backend metadata must match the effective backend implied by the run config."
            )
        if self.evaluation.algorithm != self.algorithm:
            raise ValueError("Evaluation summary algorithm does not match run record algorithm.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "algorithm": self.algorithm,
            "experiment_name": self.experiment_name,
            "config_path": self.config_path,
            "dataset_path": self.dataset_path,
            "dataset_meta_path": self.dataset_meta_path,
            "requested_device": self.requested_device,
            "effective_backend": self.effective_backend,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_manifest_path": self.checkpoint_manifest_path,
            "final_metrics": dict(self.final_metrics),
            "dataset_contract": dict(self.dataset_contract) if self.dataset_contract else None,
            "backend_metadata": self.backend_metadata.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True)
class ReproducibilityBundleInputs:
    """Inputs required to assemble one reproducible benchmark bundle."""

    benchmark_version: str
    cohort: BundleArtifactRecord
    feature_dictionary: BundleArtifactRecord
    split_manifest: BundleArtifactRecord
    action_bins: BundleArtifactRecord
    reward_config: BundleArtifactRecord
    run_records: tuple[RunBundleRecord, ...]
    ablation_reports: tuple[AblationComparisonReport, ...] = field(default_factory=tuple)
    rerun_instructions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReproducibilityBundle:
    """Research-facing reproducibility package for the full benchmark."""

    benchmark_version: str
    cohort: BundleArtifactRecord
    feature_dictionary: BundleArtifactRecord
    split_manifest: BundleArtifactRecord
    action_bins: BundleArtifactRecord
    reward_config: BundleArtifactRecord
    run_records: tuple[RunBundleRecord, ...]
    ablation_reports: tuple[AblationComparisonReport, ...]
    rerun_instructions: tuple[str, ...]
    generated_at: str
    schema_version: str = REPRO_BUNDLE_SCHEMA_VERSION

    def validate(self) -> None:
        if not self.benchmark_version.strip():
            raise ValueError("benchmark_version must be non-empty.")

        self.cohort.require_keys("spec_version")
        self.feature_dictionary.require_keys("spec_version", "total_features")
        self.split_manifest.require_keys("spec_version", "seed")
        self.action_bins.require_keys("spec_version", "manifest_seed")
        self.reward_config.require_keys("version", "variant")

        if not self.run_records:
            raise ValueError("At least one run record is required in the reproducibility bundle.")

        for run_record in self.run_records:
            run_record.validate()

        for report in self.ablation_reports:
            report.validate()
            if report.benchmark_version != self.benchmark_version:
                raise ValueError(
                    "Ablation report benchmark_version does not match the bundle version."
                )

        if not self.rerun_instructions:
            raise ValueError("Reproducibility bundle must include rerun instructions.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "benchmark_version": self.benchmark_version,
            "generated_at": self.generated_at,
            "cohort": self.cohort.to_dict(),
            "feature_dictionary": self.feature_dictionary.to_dict(),
            "split_manifest": self.split_manifest.to_dict(),
            "action_bins": self.action_bins.to_dict(),
            "reward_config": self.reward_config.to_dict(),
            "run_records": [run_record.to_dict() for run_record in self.run_records],
            "ablation_reports": [report.to_dict() for report in self.ablation_reports],
            "rerun_instructions": list(self.rerun_instructions),
        }


def build_cohort_record(
    summary: Mapping[str, Any],
    *,
    spec_path: str | None = None,
    source_tables: Mapping[str, str] | None = None,
) -> BundleArtifactRecord:
    metadata = _copy_mapping(summary)
    if source_tables:
        metadata["source_tables"] = dict(source_tables)
    return BundleArtifactRecord("cohort_spec", spec_path, metadata)


def build_feature_dictionary_record(
    summary: Mapping[str, Any],
    *,
    spec_path: str | None = None,
) -> BundleArtifactRecord:
    return BundleArtifactRecord("feature_dictionary", spec_path, _copy_mapping(summary))


def build_split_manifest_record(
    summary: Mapping[str, Any],
    *,
    manifest_dir: str | None = None,
) -> BundleArtifactRecord:
    return BundleArtifactRecord("split_manifest", manifest_dir, _copy_mapping(summary))


def build_action_bin_record(
    artifacts: ActionBinArtifacts,
    *,
    artifact_path: str | None = None,
) -> BundleArtifactRecord:
    return BundleArtifactRecord("action_bins", artifact_path, artifacts.to_dict())


def build_reward_config_record(
    config: RewardConfig,
    *,
    config_path: str | None = None,
) -> BundleArtifactRecord:
    metadata = {
        "version": config.version,
        "variant": config.variant.value,
        "terminal_reward_survived": config.terminal_reward_survived,
        "terminal_reward_died": config.terminal_reward_died,
        "sofa_delta_weight": config.sofa_delta_weight,
        "lactate_clearance_weight": config.lactate_clearance_weight,
        "map_stability_weight": config.map_stability_weight,
        "map_threshold": config.map_threshold,
    }
    return BundleArtifactRecord("reward_config", config_path, metadata)


def default_rerun_instructions(run_records: Sequence[RunBundleRecord]) -> tuple[str, ...]:
    instructions = [
        (
            "python -m mimic_sepsis_rl.training.experiment_runner "
            f"--algorithm {run_record.algorithm} "
            f"--config {run_record.config_path} "
            f"--device {run_record.backend_metadata.requested_backend}"
        )
        for run_record in run_records
    ]
    instructions.append(
        "Audit checkpoint manifests, dataset metadata, and evaluation artifacts before "
        "interpreting retrospective policy results."
    )
    return tuple(instructions)


def assemble_reproducibility_bundle(
    inputs: ReproducibilityBundleInputs,
) -> ReproducibilityBundle:
    """Assemble and validate the research-facing reproducibility bundle."""
    bundle = ReproducibilityBundle(
        benchmark_version=inputs.benchmark_version,
        cohort=inputs.cohort,
        feature_dictionary=inputs.feature_dictionary,
        split_manifest=inputs.split_manifest,
        action_bins=inputs.action_bins,
        reward_config=inputs.reward_config,
        run_records=inputs.run_records,
        ablation_reports=inputs.ablation_reports,
        rerun_instructions=inputs.rerun_instructions
        or default_rerun_instructions(inputs.run_records),
        generated_at=_utc_now(),
    )
    bundle.validate()
    return bundle


__all__ = [
    "BackendMetadataRecord",
    "BundleArtifactRecord",
    "EvaluationSummaryRecord",
    "REPRO_BUNDLE_SCHEMA_VERSION",
    "ReproducibilityBundle",
    "ReproducibilityBundleInputs",
    "RunBundleRecord",
    "assemble_reproducibility_bundle",
    "build_action_bin_record",
    "build_cohort_record",
    "build_feature_dictionary_record",
    "build_reward_config_record",
    "build_split_manifest_record",
    "default_rerun_instructions",
]
