"""
Offline policy evaluation helpers for held-out retrospective assessment.

Consumes the standardized Phase 8 run artifacts together with held-out
transition summaries so researchers can report WIS, ESS, and frozen FQE
estimates without refitting anything on the held-out split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

from mimic_sepsis_rl.training.comparison import DatasetContractRecord, RunArtifact

EpisodeId: TypeAlias = str | int
_HELD_OUT_SPLIT_ALIASES = frozenset({"eval", "evaluation", "heldout", "holdout", "test"})


@runtime_checkable
class ActionSelectionPolicy(Protocol):
    """Minimal policy surface required for deterministic held-out OPE."""

    def select_action(self, state: Sequence[float]) -> int:
        """Return the policy action for one state vector."""


@runtime_checkable
class ActionProbabilityPolicy(ActionSelectionPolicy, Protocol):
    """Optional extension for stochastic target policies."""

    def action_probability(self, state: Sequence[float], action: int) -> float:
        """Return π(a | s) for the supplied action."""


@dataclass(frozen=True)
class HeldOutStep:
    """One held-out transition summary needed for OPE.

    ``behavior_action_prob`` is the logged clinician policy probability for the
    *observed* action on this step.
    """

    episode_id: EpisodeId
    step_index: int
    state: tuple[float, ...]
    action: int
    reward: float
    done: bool
    behavior_action_prob: float

    def validate(self, *, expected_state_dim: int | None = None) -> None:
        if expected_state_dim is not None and len(self.state) != expected_state_dim:
            raise ValueError(
                f"State dim mismatch for episode {self.episode_id}, step {self.step_index}: "
                f"expected {expected_state_dim}, got {len(self.state)}."
            )
        if not 0.0 < self.behavior_action_prob <= 1.0:
            raise ValueError(
                "behavior_action_prob must be in (0, 1] for held-out OPE; "
                f"got {self.behavior_action_prob!r} at episode {self.episode_id}, "
                f"step {self.step_index}."
            )


@dataclass(frozen=True)
class HeldOutEpisode:
    """Ordered held-out trajectory used by WIS/ESS evaluation."""

    episode_id: EpisodeId
    steps: tuple[HeldOutStep, ...]

    def validate(self, *, expected_state_dim: int | None = None) -> None:
        if not self.steps:
            raise ValueError(f"Held-out episode {self.episode_id} has no steps.")

        previous_step_index = -1
        for index, step in enumerate(self.steps):
            if step.episode_id != self.episode_id:
                raise ValueError(
                    f"Episode {self.episode_id} contains a step from episode {step.episode_id}."
                )
            step.validate(expected_state_dim=expected_state_dim)
            if step.step_index <= previous_step_index:
                raise ValueError(
                    f"Episode {self.episode_id} step indices must be strictly increasing."
                )
            previous_step_index = step.step_index
            if index < len(self.steps) - 1 and step.done:
                raise ValueError(
                    f"Episode {self.episode_id} marks step {step.step_index} done before the end."
                )

        if not self.steps[-1].done:
            raise ValueError(
                f"Episode {self.episode_id} must end with done=True on the final step."
            )

    def discounted_return(self, gamma: float) -> float:
        return sum((gamma**step.step_index) * step.reward for step in self.steps)


@dataclass(frozen=True)
class FrozenFQEOutputs:
    """Frozen FQE action-value estimates for held-out initial states.

    The estimates must come from a non-held-out split. Phase 9 uses them only
    for scoring held-out initial states, never for fitting.
    """

    fitted_split: str
    initial_state_action_values: Mapping[EpisodeId, tuple[float, ...]]
    artifact_label: str | None = None

    def validate(self, *, n_actions: int) -> None:
        fitted_split = self.fitted_split.strip().lower()
        if fitted_split in _HELD_OUT_SPLIT_ALIASES:
            raise ValueError(
                "Frozen FQE outputs must be fitted on train/validation data, "
                "never on the held-out evaluation split."
            )
        if not self.initial_state_action_values:
            raise ValueError("Frozen FQE outputs must include at least one episode value.")

        for episode_id, q_values in self.initial_state_action_values.items():
            if len(q_values) != n_actions:
                raise ValueError(
                    f"FQE action-value width mismatch for episode {episode_id}: "
                    f"expected {n_actions}, got {len(q_values)}."
                )

    def estimate_policy_value(
        self,
        policy: ActionSelectionPolicy,
        episodes: Sequence[HeldOutEpisode],
        *,
        n_actions: int,
    ) -> float:
        self.validate(n_actions=n_actions)
        values: list[float] = []

        for episode in episodes:
            if episode.episode_id not in self.initial_state_action_values:
                raise KeyError(
                    f"Missing frozen FQE values for held-out episode {episode.episode_id}."
                )
            q_values = self.initial_state_action_values[episode.episode_id]
            initial_state = episode.steps[0].state

            if isinstance(policy, ActionProbabilityPolicy):
                action_probs = tuple(
                    float(policy.action_probability(initial_state, action))
                    for action in range(n_actions)
                )
                total_prob = sum(action_probs)
                if total_prob <= 0.0:
                    raise ValueError(
                        "Target policy returned zero total probability for the initial state."
                    )
                value = sum(
                    (prob / total_prob) * q_value
                    for prob, q_value in zip(action_probs, q_values)
                )
            else:
                chosen_action = int(policy.select_action(initial_state))
                if not 0 <= chosen_action < n_actions:
                    raise ValueError(
                        f"Policy chose invalid action {chosen_action}; expected [0, {n_actions})."
                    )
                value = q_values[chosen_action]

            values.append(float(value))

        return sum(values) / len(values)


@dataclass(frozen=True)
class EpisodeOPEEstimate:
    """Per-episode OPE diagnostics used to derive WIS and ESS."""

    episode_id: EpisodeId
    discounted_return: float
    importance_weight: float
    matched_steps: int
    n_steps: int

    @property
    def matched_step_fraction(self) -> float:
        if self.n_steps == 0:
            return 0.0
        return self.matched_steps / self.n_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OPEMetrics:
    """Top-line OPE metrics for one evaluated policy."""

    wis: float
    ess: float
    fqe: float
    n_episodes: int
    wis_weight_sum: float
    wis_nonzero_episodes: int
    mean_behavior_return: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetContractCheck:
    """Compatibility check between training and held-out dataset artifacts."""

    is_consistent: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"is_consistent": self.is_consistent, "issues": list(self.issues)}


@dataclass(frozen=True)
class PolicyOPEReport:
    """Normalized held-out OPE report for one run artifact."""

    algorithm: str
    run_artifact: RunArtifact
    held_out_contract: DatasetContractRecord | None
    contract_check: DatasetContractCheck
    metrics: OPEMetrics
    per_episode: tuple[EpisodeOPEEstimate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "run_artifact": self.run_artifact.to_dict(),
            "held_out_contract": self.held_out_contract.to_dict()
            if self.held_out_contract
            else None,
            "contract_check": self.contract_check.to_dict(),
            "metrics": self.metrics.to_dict(),
            "per_episode": [episode.to_dict() for episode in self.per_episode],
        }


def validate_held_out_episodes(
    episodes: Sequence[HeldOutEpisode],
    *,
    expected_state_dim: int | None = None,
) -> tuple[HeldOutEpisode, ...]:
    """Validate and freeze held-out trajectories before scoring."""
    if not episodes:
        raise ValueError("At least one held-out episode is required for OPE.")

    validated = tuple(episodes)
    for episode in validated:
        episode.validate(expected_state_dim=expected_state_dim)
    return validated


def compare_dataset_contracts(
    train_contract: DatasetContractRecord | None,
    held_out_contract: DatasetContractRecord | None,
) -> DatasetContractCheck:
    """Check that train and held-out artifacts share the same frozen contract."""
    if train_contract is None or held_out_contract is None:
        return DatasetContractCheck(
            is_consistent=True,
            issues=tuple(),
        )

    issues: list[str] = []
    shared_fields = (
        ("spec_version", train_contract.spec_version, held_out_contract.spec_version),
        ("n_actions", train_contract.n_actions, held_out_contract.n_actions),
        ("state_dim", train_contract.state_dim, held_out_contract.state_dim),
        (
            "action_spec_version",
            train_contract.action_spec_version,
            held_out_contract.action_spec_version,
        ),
        (
            "reward_spec_version",
            train_contract.reward_spec_version,
            held_out_contract.reward_spec_version,
        ),
        ("manifest_seed", train_contract.manifest_seed, held_out_contract.manifest_seed),
        (
            "feature_columns",
            train_contract.feature_columns,
            held_out_contract.feature_columns,
        ),
    )

    for name, train_value, held_out_value in shared_fields:
        if train_value != held_out_value:
            issues.append(
                f"Dataset contract drift in {name}: train={train_value!r}, "
                f"held_out={held_out_value!r}."
            )

    held_out_split = held_out_contract.split_label.strip().lower()
    if held_out_split not in _HELD_OUT_SPLIT_ALIASES:
        issues.append(
            "Held-out dataset contract should identify a test/eval split; "
            f"got split_label={held_out_contract.split_label!r}."
        )

    return DatasetContractCheck(is_consistent=not issues, issues=tuple(issues))


def _policy_action_probability(
    policy: ActionSelectionPolicy,
    state: Sequence[float],
    action: int,
) -> tuple[float, bool]:
    """Return π(a | s) and whether the policy matched the logged action."""
    if isinstance(policy, ActionProbabilityPolicy):
        probability = float(policy.action_probability(state, action))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Policy probability must be in [0, 1]; got {probability!r}."
            )
        return probability, probability > 0.0

    chosen_action = int(policy.select_action(state))
    if chosen_action < 0:
        raise ValueError(f"Policy chose an invalid negative action {chosen_action}.")
    return (1.0, True) if chosen_action == action else (0.0, False)


def _evaluate_episode(
    episode: HeldOutEpisode,
    policy: ActionSelectionPolicy,
    *,
    gamma: float,
    max_importance_ratio: float | None,
) -> EpisodeOPEEstimate:
    weight = 1.0
    matched_steps = 0

    for step in episode.steps:
        policy_prob, matched = _policy_action_probability(policy, step.state, step.action)
        if matched:
            matched_steps += 1
        ratio = policy_prob / step.behavior_action_prob
        if max_importance_ratio is not None:
            ratio = min(ratio, max_importance_ratio)
        weight *= ratio

    return EpisodeOPEEstimate(
        episode_id=episode.episode_id,
        discounted_return=episode.discounted_return(gamma),
        importance_weight=weight,
        matched_steps=matched_steps,
        n_steps=len(episode.steps),
    )


def compute_wis_and_ess(
    episodes: Sequence[HeldOutEpisode],
    policy: ActionSelectionPolicy,
    *,
    gamma: float = 1.0,
    max_importance_ratio: float | None = None,
) -> tuple[OPEMetrics, tuple[EpisodeOPEEstimate, ...]]:
    """Compute WIS/ESS from held-out trajectories and a target policy."""
    validated_episodes = validate_held_out_episodes(episodes)
    per_episode = tuple(
        _evaluate_episode(
            episode,
            policy,
            gamma=gamma,
            max_importance_ratio=max_importance_ratio,
        )
        for episode in validated_episodes
    )

    weights = [episode.importance_weight for episode in per_episode]
    returns = [episode.discounted_return for episode in per_episode]
    weight_sum = sum(weights)
    squared_weight_sum = sum(weight * weight for weight in weights)

    wis = 0.0
    if weight_sum > 0.0:
        wis = sum(weight * ret for weight, ret in zip(weights, returns)) / weight_sum

    ess = 0.0
    if weight_sum > 0.0 and squared_weight_sum > 0.0:
        ess = (weight_sum * weight_sum) / squared_weight_sum

    metrics = OPEMetrics(
        wis=wis,
        ess=ess,
        fqe=0.0,
        n_episodes=len(per_episode),
        wis_weight_sum=weight_sum,
        wis_nonzero_episodes=sum(1 for weight in weights if weight > 0.0),
        mean_behavior_return=sum(returns) / len(returns),
    )
    return metrics, per_episode


def evaluate_policy_run(
    run_artifact: RunArtifact,
    held_out_episodes: Sequence[HeldOutEpisode],
    policy: ActionSelectionPolicy,
    frozen_fqe_outputs: FrozenFQEOutputs,
    *,
    held_out_contract: DatasetContractRecord | None = None,
    gamma: float | None = None,
    max_importance_ratio: float | None = None,
) -> PolicyOPEReport:
    """Evaluate one run artifact on held-out trajectories without refitting."""
    expected_state_dim: int | None = None
    expected_n_actions: int | None = None

    if held_out_contract is not None:
        expected_state_dim = held_out_contract.state_dim
        expected_n_actions = held_out_contract.n_actions
    elif run_artifact.dataset_contract is not None:
        expected_state_dim = run_artifact.dataset_contract.state_dim
        expected_n_actions = run_artifact.dataset_contract.n_actions

    validated_episodes = validate_held_out_episodes(
        held_out_episodes,
        expected_state_dim=expected_state_dim,
    )
    resolved_gamma = (
        float(gamma)
        if gamma is not None
        else float(run_artifact.config_provenance.gamma)
    )

    metrics, per_episode = compute_wis_and_ess(
        validated_episodes,
        policy,
        gamma=resolved_gamma,
        max_importance_ratio=max_importance_ratio,
    )

    resolved_n_actions = expected_n_actions or 25
    fqe_value = frozen_fqe_outputs.estimate_policy_value(
        policy,
        validated_episodes,
        n_actions=resolved_n_actions,
    )
    metrics = OPEMetrics(
        wis=metrics.wis,
        ess=metrics.ess,
        fqe=fqe_value,
        n_episodes=metrics.n_episodes,
        wis_weight_sum=metrics.wis_weight_sum,
        wis_nonzero_episodes=metrics.wis_nonzero_episodes,
        mean_behavior_return=metrics.mean_behavior_return,
    )

    contract_check = compare_dataset_contracts(
        run_artifact.dataset_contract,
        held_out_contract,
    )

    return PolicyOPEReport(
        algorithm=run_artifact.algorithm,
        run_artifact=run_artifact,
        held_out_contract=held_out_contract,
        contract_check=contract_check,
        metrics=metrics,
        per_episode=per_episode,
    )


__all__ = [
    "ActionProbabilityPolicy",
    "ActionSelectionPolicy",
    "DatasetContractCheck",
    "EpisodeOPEEstimate",
    "FrozenFQEOutputs",
    "HeldOutEpisode",
    "HeldOutStep",
    "OPEMetrics",
    "PolicyOPEReport",
    "compare_dataset_contracts",
    "compute_wis_and_ess",
    "evaluate_policy_run",
    "validate_held_out_episodes",
]

