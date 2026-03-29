"""
Shared algorithm registry for offline RL experiment launches.

Centralises the supported algorithm catalogue so the experiment runner can
resolve config paths and execution adapters without scattering per-algorithm
branches across the CLI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from mimic_sepsis_rl.training.common import load_replay_dataset
from mimic_sepsis_rl.training.config import TrainingConfig, load_training_config

DEFAULT_ACTION_COUNT: int = 25

AlgorithmHandler = Callable[
    ["TrainingConfig", "AlgorithmRunRequest"],
    dict[str, Any],
]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML document and require a mapping root."""
    with path.open("r") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at root level in {path}.")
    return payload


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-merged copy of *base* updated with *updates*."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class AlgorithmRunRequest:
    """Standard execution request passed to every algorithm adapter."""

    dry_run: bool = False
    n_actions: int | None = None


@dataclass(frozen=True)
class AlgorithmDefinition:
    """Registry entry describing one offline RL algorithm."""

    name: str
    description: str
    default_config_path: Path
    module_path: str
    handler: AlgorithmHandler
    is_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "default_config_path": str(self.default_config_path),
            "module_path": self.module_path,
            "status": "ready" if self.is_available else "planned",
        }


class AlgorithmRegistry:
    """In-memory registry for supported offline RL algorithms."""

    def __init__(
        self,
        definitions: Iterable[AlgorithmDefinition] | None = None,
    ) -> None:
        self._definitions: dict[str, AlgorithmDefinition] = {}
        if definitions is None:
            return
        for definition in definitions:
            self.register(definition)

    def register(self, definition: AlgorithmDefinition) -> None:
        key = definition.name.lower()
        if key in self._definitions:
            raise ValueError(f"Algorithm '{definition.name}' is already registered.")
        self._definitions[key] = definition

    def list_algorithms(self) -> tuple[AlgorithmDefinition, ...]:
        return tuple(self._definitions[name] for name in sorted(self._definitions))

    def names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.list_algorithms())

    def get(self, name: str) -> AlgorithmDefinition | None:
        return self._definitions.get(name.lower())

    def require(self, name: str) -> AlgorithmDefinition:
        definition = self.get(name)
        if definition is None:
            supported = ", ".join(self.names())
            raise ValueError(f"Unknown algorithm '{name}'. Supported: {supported}.")
        return definition

    def resolve_config_path(
        self,
        name: str,
        *,
        config_path: str | Path | None = None,
    ) -> Path:
        definition = self.require(name)
        resolved = Path(config_path) if config_path is not None else definition.default_config_path
        if not resolved.exists():
            raise FileNotFoundError(f"Training config not found: {resolved}")
        return resolved

    def load_config(
        self,
        name: str,
        *,
        config_path: str | Path | None = None,
        device: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> TrainingConfig:
        resolved_path = self.resolve_config_path(name, config_path=config_path)

        merged_overrides = dict(overrides or {})
        if device is not None:
            merged_overrides = _deep_merge(
                merged_overrides,
                {"runtime": {"device": device}},
            )

        if not merged_overrides:
            return load_training_config(resolved_path)

        full_config = _deep_merge(_load_yaml_mapping(resolved_path), merged_overrides)
        return load_training_config(resolved_path, overrides=full_config)

    def execute(
        self,
        name: str,
        cfg: TrainingConfig,
        request: AlgorithmRunRequest,
    ) -> dict[str, Any]:
        definition = self.require(name)
        return definition.handler(cfg, request)


def _run_cql_experiment(
    cfg: TrainingConfig,
    request: AlgorithmRunRequest,
) -> dict[str, Any]:
    """Execute the CQL adapter behind the shared registry surface."""
    from mimic_sepsis_rl.training.cql import CQLTrainer, _dry_run

    n_actions = request.n_actions or DEFAULT_ACTION_COUNT
    if request.dry_run:
        _dry_run(cfg, n_actions=n_actions)
        return {
            "algorithm": "cql",
            "mode": "dry_run",
            "device_backend": cfg.device_meta.backend,
            "n_actions": n_actions,
        }

    dataset = load_replay_dataset(cfg)
    trainer = CQLTrainer(cfg, dataset, n_actions=n_actions)
    return trainer.train().to_dict()


def _make_pending_handler(name: str) -> AlgorithmHandler:
    """Create a placeholder handler for future algorithm modules."""

    def _pending_handler(
        cfg: TrainingConfig,
        request: AlgorithmRunRequest,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"Algorithm '{name}' is registered on the shared experiment surface, "
            "but its trainer module is scheduled for Phase 08-02."
        )

    return _pending_handler


def build_default_registry() -> AlgorithmRegistry:
    """Construct the project default algorithm registry."""
    config_dir = Path("configs/training")

    return AlgorithmRegistry(
        definitions=(
            AlgorithmDefinition(
                name="bcq",
                description="Batch-Constrained Q-learning on the shared replay contract.",
                default_config_path=config_dir / "bcq.yaml",
                module_path="mimic_sepsis_rl.training.bcq",
                handler=_make_pending_handler("bcq"),
                is_available=False,
            ),
            AlgorithmDefinition(
                name="cql",
                description="Discrete Conservative Q-Learning reference trainer.",
                default_config_path=config_dir / "cql.yaml",
                module_path="mimic_sepsis_rl.training.cql",
                handler=_run_cql_experiment,
            ),
            AlgorithmDefinition(
                name="iql",
                description="Implicit Q-Learning on the shared replay contract.",
                default_config_path=config_dir / "iql.yaml",
                module_path="mimic_sepsis_rl.training.iql",
                handler=_make_pending_handler("iql"),
                is_available=False,
            ),
        )
    )


_DEFAULT_REGISTRY = build_default_registry()


def get_default_registry() -> AlgorithmRegistry:
    """Return the process-wide default algorithm registry."""
    return _DEFAULT_REGISTRY


__all__ = [
    "DEFAULT_ACTION_COUNT",
    "AlgorithmRunRequest",
    "AlgorithmDefinition",
    "AlgorithmRegistry",
    "build_default_registry",
    "get_default_registry",
]
