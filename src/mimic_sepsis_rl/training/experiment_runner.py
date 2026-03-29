"""
Shared experiment runner for offline RL comparisons.

The runner resolves algorithms through :mod:`mimic_sepsis_rl.training.registry`,
loads a single training config surface, and exposes one CLI for listing,
describing, and launching experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimic_sepsis_rl.datasets.transitions import TransitionDatasetMeta
from mimic_sepsis_rl.training.config import TrainingConfig
from mimic_sepsis_rl.training.registry import (
    DEFAULT_ACTION_COUNT,
    AlgorithmDefinition,
    AlgorithmRunRequest,
    get_default_registry,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetContractSummary:
    """Minimal dataset-contract metadata needed for fair comparisons."""

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
    def from_meta(cls, meta: TransitionDatasetMeta) -> "DatasetContractSummary":
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
class ResolvedExperiment:
    """Resolved experiment launch request."""

    definition: AlgorithmDefinition
    config_path: Path
    config: TrainingConfig
    dataset_contract: DatasetContractSummary | None

    def resolve_n_actions(self, requested_n_actions: int | None = None) -> int:
        if requested_n_actions is not None:
            return requested_n_actions
        if self.dataset_contract is not None:
            return self.dataset_contract.n_actions
        return DEFAULT_ACTION_COUNT

    def execute(
        self,
        *,
        dry_run: bool = False,
        n_actions: int | None = None,
    ) -> dict[str, Any]:
        registry = get_default_registry()
        request = AlgorithmRunRequest(
            dry_run=dry_run,
            n_actions=self.resolve_n_actions(n_actions),
        )
        return registry.execute(self.definition.name, self.config, request)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.definition.name,
            "description": self.definition.description,
            "status": "ready" if self.definition.is_available else "planned",
            "config_path": str(self.config_path),
            "module_path": self.definition.module_path,
            "config": self.config.to_dict(),
            "dataset_contract": self.dataset_contract.to_dict()
            if self.dataset_contract is not None
            else None,
        }


def _load_dataset_contract(
    dataset_meta_path: Path | None,
) -> DatasetContractSummary | None:
    if dataset_meta_path is None or not dataset_meta_path.exists():
        return None

    payload = json.loads(dataset_meta_path.read_text())
    meta = TransitionDatasetMeta.from_dict(payload)
    return DatasetContractSummary.from_meta(meta)


def resolve_experiment(
    algorithm: str,
    *,
    config_path: str | Path | None = None,
    device: str | None = None,
) -> ResolvedExperiment:
    """Resolve config and dataset-contract metadata for one experiment."""
    registry = get_default_registry()
    definition = registry.require(algorithm)
    resolved_config_path = registry.resolve_config_path(
        algorithm,
        config_path=config_path,
    )
    config = registry.load_config(
        algorithm,
        config_path=resolved_config_path,
        device=device,
    )
    if config.algorithm != definition.name:
        raise ValueError(
            f"Config algorithm mismatch: registry expected '{definition.name}' "
            f"but loaded '{config.algorithm}' from {resolved_config_path}."
        )

    dataset_contract = _load_dataset_contract(config.dataset_meta_path)
    return ResolvedExperiment(
        definition=definition,
        config_path=resolved_config_path,
        config=config,
        dataset_contract=dataset_contract,
    )


def format_algorithm_listing() -> str:
    """Render a human-readable algorithm catalogue."""
    lines = ["Available offline RL algorithms:"]
    for definition in get_default_registry().list_algorithms():
        status = "ready" if definition.is_available else "planned"
        lines.append(
            f"- {definition.name} [{status}] "
            f"config={definition.default_config_path} "
            f"module={definition.module_path}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for shared offline RL experiment launches."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.training.experiment_runner",
        description="List, describe, and launch offline RL experiments.",
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="Print the shared algorithm registry and exit.",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Algorithm identifier to resolve or run (cql, bcq, iql).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to an explicit training YAML config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override runtime.device in the resolved config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute the algorithm adapter in dry-run mode.",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=None,
        help="Override the discrete action count; defaults to dataset metadata.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Resolve config and dataset contract, print JSON, then exit.",
    )
    args = parser.parse_args(argv)

    if args.list_algorithms:
        print(format_algorithm_listing())
        return

    if args.algorithm is None:
        parser.error("--algorithm is required unless --list-algorithms is used.")

    resolved = resolve_experiment(
        args.algorithm,
        config_path=args.config,
        device=args.device,
    )

    if args.describe:
        print(json.dumps(resolved.to_dict(), indent=2))
        return

    result = resolved.execute(dry_run=args.dry_run, n_actions=args.n_actions)
    print(
        json.dumps(
            {
                "experiment": resolved.to_dict(),
                "result": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DatasetContractSummary",
    "ResolvedExperiment",
    "resolve_experiment",
    "format_algorithm_listing",
    "main",
]
