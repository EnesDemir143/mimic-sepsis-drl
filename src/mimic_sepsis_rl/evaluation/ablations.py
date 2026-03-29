"""
Ablation registry and comparison helpers for benchmark robustness checks.

Phase 9 uses this module to keep reward-shaping, action-granularity,
timestep, missingness-flag, and feature-subset experiments attributable to
one stable benchmark version and one shared metadata schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Final, Mapping

from mimic_sepsis_rl.mdp.features.dictionary import FEATURE_SPEC_VERSION
from mimic_sepsis_rl.mdp.reward_models import RewardVariant
from mimic_sepsis_rl.training.comparison import RunArtifact

ABLATION_SCHEMA_VERSION: Final[str] = "1.0.0"
DEFAULT_ACTION_GRANULARITY: Final[str] = "5x5_bins"
DEFAULT_FEATURE_SUBSET: Final[str] = "full_state"
DEFAULT_TIMESTEP_HOURS: Final[int] = 4


class AblationDimension(str, Enum):
    """Canonical ablation axes required by the Phase 9 benchmark."""

    REWARD_SHAPING = "reward_shaping"
    ACTION_GRANULARITY = "action_granularity"
    TIMESTEP_CHOICE = "timestep_choice"
    MISSINGNESS_FLAGS = "missingness_flags"
    FEATURE_SUBSET = "feature_subset"


REQUIRED_ABLATION_DIMENSIONS: Final[tuple[AblationDimension, ...]] = (
    AblationDimension.REWARD_SHAPING,
    AblationDimension.ACTION_GRANULARITY,
    AblationDimension.TIMESTEP_CHOICE,
    AblationDimension.MISSINGNESS_FLAGS,
    AblationDimension.FEATURE_SUBSET,
)


@dataclass(frozen=True)
class AblationDefaults:
    """Default benchmark settings inherited by every ablation variant."""

    benchmark_version: str
    feature_spec_version: str = FEATURE_SPEC_VERSION
    reward_variant: str = RewardVariant.SOFA_SHAPED.value
    action_granularity: str = DEFAULT_ACTION_GRANULARITY
    timestep_hours: int = DEFAULT_TIMESTEP_HOURS
    include_missingness_flags: bool = True
    feature_subset: str = DEFAULT_FEATURE_SUBSET


@dataclass(frozen=True)
class AblationExperimentMetadata:
    """Stable experiment metadata copied into every ablation variant."""

    benchmark_version: str
    algorithm: str
    experiment_name: str
    config_path: str
    dataset_path: str
    dataset_meta_path: str | None
    dataset_split_label: str | None
    manifest_seed: int | None
    action_spec_version: str | None
    reward_spec_version: str | None
    feature_spec_version: str
    requested_device: str
    effective_backend: str
    source_checkpoint_path: str | None
    reward_variant: str
    action_granularity: str
    timestep_hours: int
    include_missingness_flags: bool
    feature_subset: str
    schema_version: str = ABLATION_SCHEMA_VERSION

    @classmethod
    def from_run_artifact(
        cls,
        run_artifact: RunArtifact,
        defaults: AblationDefaults,
    ) -> "AblationExperimentMetadata":
        """Build the shared ablation metadata surface from a Phase 8 run."""
        dataset_contract = run_artifact.dataset_contract
        checkpoint = run_artifact.checkpoint
        config = run_artifact.config_provenance
        metadata = cls(
            benchmark_version=defaults.benchmark_version,
            algorithm=run_artifact.algorithm,
            experiment_name=config.experiment_name,
            config_path=config.config_path,
            dataset_path=config.dataset_path,
            dataset_meta_path=config.dataset_meta_path,
            dataset_split_label=dataset_contract.split_label if dataset_contract else None,
            manifest_seed=dataset_contract.manifest_seed if dataset_contract else None,
            action_spec_version=dataset_contract.action_spec_version if dataset_contract else None,
            reward_spec_version=dataset_contract.reward_spec_version if dataset_contract else None,
            feature_spec_version=defaults.feature_spec_version,
            requested_device=config.requested_device,
            effective_backend=config.effective_backend,
            source_checkpoint_path=checkpoint.checkpoint_path if checkpoint else None,
            reward_variant=defaults.reward_variant,
            action_granularity=defaults.action_granularity,
            timestep_hours=defaults.timestep_hours,
            include_missingness_flags=defaults.include_missingness_flags,
            feature_subset=defaults.feature_subset,
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if not self.benchmark_version.strip():
            raise ValueError("benchmark_version must be a non-empty string.")
        if not self.algorithm.strip():
            raise ValueError("algorithm must be a non-empty string.")
        if not self.config_path.strip():
            raise ValueError("config_path must be a non-empty string.")
        if self.timestep_hours <= 0:
            raise ValueError("timestep_hours must be a positive integer.")
        if not self.reward_variant.strip():
            raise ValueError("reward_variant must be a non-empty string.")
        if not self.action_granularity.strip():
            raise ValueError("action_granularity must be a non-empty string.")
        if not self.feature_subset.strip():
            raise ValueError("feature_subset must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AblationVariant:
    """One benchmark variant along a single ablation dimension."""

    variant_id: str
    dimension: AblationDimension
    value: str | int | bool
    description: str

    def apply(self, base_metadata: AblationExperimentMetadata) -> AblationExperimentMetadata:
        """Return ablation metadata with this variant applied."""
        if not self.variant_id.strip():
            raise ValueError("variant_id must be a non-empty string.")

        if self.dimension == AblationDimension.REWARD_SHAPING:
            return replace(base_metadata, reward_variant=str(self.value))
        if self.dimension == AblationDimension.ACTION_GRANULARITY:
            return replace(base_metadata, action_granularity=str(self.value))
        if self.dimension == AblationDimension.TIMESTEP_CHOICE:
            timestep_hours = int(self.value)
            if timestep_hours <= 0:
                raise ValueError("timestep ablation values must be positive.")
            return replace(base_metadata, timestep_hours=timestep_hours)
        if self.dimension == AblationDimension.MISSINGNESS_FLAGS:
            if not isinstance(self.value, bool):
                raise TypeError("missingness flag ablations must use boolean values.")
            return replace(base_metadata, include_missingness_flags=self.value)
        if self.dimension == AblationDimension.FEATURE_SUBSET:
            return replace(base_metadata, feature_subset=str(self.value))
        raise ValueError(f"Unsupported ablation dimension: {self.dimension!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "dimension": self.dimension.value,
            "value": self.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class AblationPlan:
    """One launchable ablation plan under the shared benchmark schema."""

    variant: AblationVariant
    experiment_metadata: AblationExperimentMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.to_dict(),
            "experiment_metadata": self.experiment_metadata.to_dict(),
        }


@dataclass(frozen=True)
class AblationResult:
    """Recorded metrics for one executed ablation plan."""

    plan: AblationPlan
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class AblationComparisonReport:
    """Comparable results for one ablation dimension."""

    dimension: AblationDimension
    benchmark_version: str
    baseline_variant_id: str
    results: tuple[AblationResult, ...]

    @property
    def metric_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for result in self.results:
            names.update(result.metrics)
        return tuple(sorted(names))

    def validate(self) -> None:
        if not self.results:
            raise ValueError("Ablation comparison requires at least one result.")

        expected_keys = tuple(self.results[0].plan.experiment_metadata.to_dict().keys())
        for result in self.results:
            if result.plan.variant.dimension != self.dimension:
                raise ValueError("Comparison report mixes multiple ablation dimensions.")
            metadata = result.plan.experiment_metadata
            metadata.validate()
            if metadata.benchmark_version != self.benchmark_version:
                raise ValueError("Comparison report benchmark_version does not match results.")
            current_keys = tuple(metadata.to_dict().keys())
            if current_keys != expected_keys:
                raise ValueError("Ablation results do not share one experiment metadata schema.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "dimension": self.dimension.value,
            "benchmark_version": self.benchmark_version,
            "baseline_variant_id": self.baseline_variant_id,
            "metric_names": list(self.metric_names),
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class AblationRegistry:
    """Registry of required benchmark ablations under one base metadata surface."""

    base_metadata: AblationExperimentMetadata
    variants: tuple[AblationVariant, ...]

    def validate(self) -> None:
        self.base_metadata.validate()
        seen_variant_ids: set[str] = set()
        seen_dimensions = {variant.dimension for variant in self.variants}

        missing_dimensions = [
            dimension.value
            for dimension in REQUIRED_ABLATION_DIMENSIONS
            if dimension not in seen_dimensions
        ]
        if missing_dimensions:
            joined = ", ".join(missing_dimensions)
            raise ValueError(f"Missing required ablation dimensions: {joined}.")

        for variant in self.variants:
            if variant.variant_id in seen_variant_ids:
                raise ValueError(f"Duplicate ablation variant_id: {variant.variant_id!r}.")
            seen_variant_ids.add(variant.variant_id)
            variant.apply(self.base_metadata).validate()

    def variants_for_dimension(
        self,
        dimension: AblationDimension,
    ) -> tuple[AblationVariant, ...]:
        self.validate()
        return tuple(variant for variant in self.variants if variant.dimension == dimension)

    def plans(self) -> tuple[AblationPlan, ...]:
        self.validate()
        plans: list[AblationPlan] = []
        for dimension in REQUIRED_ABLATION_DIMENSIONS:
            for variant in self.variants_for_dimension(dimension):
                plans.append(
                    AblationPlan(
                        variant=variant,
                        experiment_metadata=variant.apply(self.base_metadata),
                    )
                )
        return tuple(plans)

    def plans_for_dimension(
        self,
        dimension: AblationDimension,
    ) -> tuple[AblationPlan, ...]:
        return tuple(plan for plan in self.plans() if plan.variant.dimension == dimension)

    def build_report(
        self,
        dimension: AblationDimension,
        metrics_by_variant_id: Mapping[str, Mapping[str, float]],
    ) -> AblationComparisonReport:
        """Attach metric rows to one dimension under the shared metadata schema."""
        plans = self.plans_for_dimension(dimension)
        missing_variant_ids = [
            plan.variant.variant_id
            for plan in plans
            if plan.variant.variant_id not in metrics_by_variant_id
        ]
        if missing_variant_ids:
            joined = ", ".join(missing_variant_ids)
            raise ValueError(
                "Controlled ablation comparisons require metrics for every "
                f"registered variant in {dimension.value}: missing {joined}."
            )

        results = tuple(
            AblationResult(
                plan=plan,
                metrics={
                    metric_name: float(metric_value)
                    for metric_name, metric_value in metrics_by_variant_id[
                        plan.variant.variant_id
                    ].items()
                },
            )
            for plan in plans
        )
        report = AblationComparisonReport(
            dimension=dimension,
            benchmark_version=self.base_metadata.benchmark_version,
            baseline_variant_id=plans[0].variant.variant_id,
            results=results,
        )
        report.validate()
        return report


DEFAULT_ABLATION_VARIANTS: Final[tuple[AblationVariant, ...]] = (
    AblationVariant(
        variant_id="reward_sofa_shaped",
        dimension=AblationDimension.REWARD_SHAPING,
        value=RewardVariant.SOFA_SHAPED.value,
        description="Phase 5 default sparse-plus-SOFA reward shaping.",
    ),
    AblationVariant(
        variant_id="reward_sparse",
        dimension=AblationDimension.REWARD_SHAPING,
        value=RewardVariant.SPARSE.value,
        description="Terminal-only reward without intermediate shaping.",
    ),
    AblationVariant(
        variant_id="reward_full_shaped",
        dimension=AblationDimension.REWARD_SHAPING,
        value=RewardVariant.FULL_SHAPED.value,
        description="Full reward shaping with SOFA, lactate, and MAP terms.",
    ),
    AblationVariant(
        variant_id="action_5x5_bins",
        dimension=AblationDimension.ACTION_GRANULARITY,
        value="5x5_bins",
        description="Phase 5 default 25-action vasopressor x fluid grid.",
    ),
    AblationVariant(
        variant_id="action_3x3_bins",
        dimension=AblationDimension.ACTION_GRANULARITY,
        value="3x3_bins",
        description="Coarser treatment bins for sensitivity analysis.",
    ),
    AblationVariant(
        variant_id="action_7x7_bins",
        dimension=AblationDimension.ACTION_GRANULARITY,
        value="7x7_bins",
        description="Finer treatment bins to stress action discretization.",
    ),
    AblationVariant(
        variant_id="timestep_4h",
        dimension=AblationDimension.TIMESTEP_CHOICE,
        value=4,
        description="Phase 2 and Phase 4 default four-hour decision interval.",
    ),
    AblationVariant(
        variant_id="timestep_2h",
        dimension=AblationDimension.TIMESTEP_CHOICE,
        value=2,
        description="Finer two-hour timestep for higher decision frequency.",
    ),
    AblationVariant(
        variant_id="timestep_6h",
        dimension=AblationDimension.TIMESTEP_CHOICE,
        value=6,
        description="Coarser six-hour timestep for robustness checks.",
    ),
    AblationVariant(
        variant_id="missingness_flags_on",
        dimension=AblationDimension.MISSINGNESS_FLAGS,
        value=True,
        description="Keep paired missingness indicators in the state vector.",
    ),
    AblationVariant(
        variant_id="missingness_flags_off",
        dimension=AblationDimension.MISSINGNESS_FLAGS,
        value=False,
        description="Drop paired missingness indicators from the state vector.",
    ),
    AblationVariant(
        variant_id="feature_subset_full_state",
        dimension=AblationDimension.FEATURE_SUBSET,
        value="full_state",
        description="All Phase 4 features plus configured missingness flags.",
    ),
    AblationVariant(
        variant_id="feature_subset_clinical_core",
        dimension=AblationDimension.FEATURE_SUBSET,
        value="clinical_core",
        description="Compact state using only the main bedside and lab covariates.",
    ),
    AblationVariant(
        variant_id="feature_subset_hemodynamic_core",
        dimension=AblationDimension.FEATURE_SUBSET,
        value="hemodynamic_core",
        description="Focused state centered on vasopressor and fluid response drivers.",
    ),
)


def build_default_ablation_registry(
    base_metadata: AblationExperimentMetadata,
) -> AblationRegistry:
    """Return the project default ablation registry for Phase 9."""
    registry = AblationRegistry(
        base_metadata=base_metadata,
        variants=DEFAULT_ABLATION_VARIANTS,
    )
    registry.validate()
    return registry


__all__ = [
    "ABLATION_SCHEMA_VERSION",
    "AblationComparisonReport",
    "AblationDefaults",
    "AblationDimension",
    "AblationExperimentMetadata",
    "AblationPlan",
    "AblationRegistry",
    "AblationResult",
    "AblationVariant",
    "DEFAULT_ABLATION_VARIANTS",
    "REQUIRED_ABLATION_DIMENSIONS",
    "build_default_ablation_registry",
]
