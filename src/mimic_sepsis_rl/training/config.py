"""
Training configuration resolution for the MIMIC Sepsis Offline RL stack.

Loads YAML training configs and resolves the runtime device through the
shared :mod:`mimic_sepsis_rl.training.device` abstraction so no algorithm
code hard-codes device selection.

Responsibilities
----------------
- Parse and validate training config YAML files.
- Resolve ``device`` fields through :func:`~mimic_sepsis_rl.training.device.resolve_device`.
- Expose a frozen :class:`TrainingConfig` dataclass so algorithm trainers
  receive a single, validated config object.
- Record backend metadata in the resolved config for checkpoint manifests.

Usage
-----
    from mimic_sepsis_rl.training.config import load_training_config

    cfg = load_training_config("configs/training/cql.yaml")
    print(cfg.device)          # torch.device("mps") / cuda / cpu
    print(cfg.device_meta)     # DeviceMetadata snapshot

Version history
---------------
v1.0.0  2026-03-29  Initial training config layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from mimic_sepsis_rl.training.device import (
    Backend,
    DeviceMetadata,
    resolve_device,
)

logger = logging.getLogger(__name__)

CONFIG_SCHEMA_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Raw YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file, returning a plain dict."""
    with path.open("r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at root level in {path}.")
    return data


# ---------------------------------------------------------------------------
# Resolved training configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime/device sub-config.

    Attributes
    ----------
    requested_device : str
        The device string from the YAML file (e.g. ``"auto"``).
    device : torch.device
        Resolved ``torch.device`` ready for tensor allocation.
    device_meta : DeviceMetadata
        Full backend snapshot for reproducibility records.
    seed : int
        Global random seed for reproducible training runs.
    num_workers : int
        DataLoader / prefetch worker count (0 = main process only).
    """

    requested_device: str
    device: torch.device
    device_meta: DeviceMetadata
    seed: int
    num_workers: int


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint persistence sub-config.

    Attributes
    ----------
    checkpoint_dir : Path
        Directory where model checkpoints are written.
    save_every_n_epochs : int
        Save a checkpoint every N epochs (0 = only at end).
    keep_last_n : int
        Number of recent checkpoints to retain (0 = keep all).
    """

    checkpoint_dir: Path
    save_every_n_epochs: int
    keep_last_n: int


@dataclass(frozen=True)
class LoggingConfig:
    """Experiment logging sub-config.

    Attributes
    ----------
    log_dir : Path
        Root directory for run logs and metric curves.
    experiment_name : str
        Human-readable experiment identifier (used in log paths).
    log_every_n_steps : int
        Emit scalar metrics every N gradient steps.
    """

    log_dir: Path
    experiment_name: str
    log_every_n_steps: int


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level resolved training configuration.

    Combines device runtime, checkpointing, logging, and algorithm-specific
    hyper-parameters into one frozen object passed to every trainer.

    Attributes
    ----------
    algorithm : str
        Algorithm identifier string (e.g. ``"cql"``).
    schema_version : str
        Config schema version for forward-compatibility checks.
    runtime : RuntimeConfig
        Resolved device and seed settings.
    checkpoint : CheckpointConfig
        Checkpoint persistence settings.
    logging : LoggingConfig
        Metric logging settings.
    dataset_path : Path
        Path to the replay-buffer parquet file consumed by the trainer.
    dataset_meta_path : Path | None
        Optional path to the replay-buffer JSON meta file.
    n_epochs : int
        Total number of training epochs.
    batch_size : int
        Mini-batch size for gradient updates.
    gamma : float
        MDP discount factor (γ).
    extra : dict[str, Any]
        Algorithm-specific hyper-parameters not captured by the base fields
        (e.g. CQL alpha, BCQ threshold).  Passed through without validation.
    """

    algorithm: str
    schema_version: str
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    dataset_path: Path
    dataset_meta_path: Path | None
    n_epochs: int
    batch_size: int
    gamma: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        """Shortcut to the resolved :attr:`RuntimeConfig.device`."""
        return self.runtime.device

    @property
    def device_meta(self) -> DeviceMetadata:
        """Shortcut to the resolved :attr:`RuntimeConfig.device_meta`."""
        return self.runtime.device_meta

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dict for checkpoint manifests."""
        return {
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "runtime": {
                "requested_device": self.runtime.requested_device,
                "effective_device": self.runtime.device_meta.backend,
                "torch_device_str": self.runtime.device_meta.torch_device_str,
                "seed": self.runtime.seed,
                "num_workers": self.runtime.num_workers,
                "device_meta": self.runtime.device_meta.to_dict(),
            },
            "checkpoint": {
                "checkpoint_dir": str(self.checkpoint.checkpoint_dir),
                "save_every_n_epochs": self.checkpoint.save_every_n_epochs,
                "keep_last_n": self.checkpoint.keep_last_n,
            },
            "logging": {
                "log_dir": str(self.logging.log_dir),
                "experiment_name": self.logging.experiment_name,
                "log_every_n_steps": self.logging.log_every_n_steps,
            },
            "dataset_path": str(self.dataset_path),
            "dataset_meta_path": str(self.dataset_meta_path)
            if self.dataset_meta_path
            else None,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    """Parse the ``runtime`` block from raw YAML."""
    requested = str(raw.get("device", "auto"))
    device, meta = resolve_device(Backend.from_str(requested))

    return RuntimeConfig(
        requested_device=requested,
        device=device,
        device_meta=meta,
        seed=int(raw.get("seed", 42)),
        num_workers=int(raw.get("num_workers", 0)),
    )


def _parse_checkpoint(raw: dict[str, Any]) -> CheckpointConfig:
    """Parse the ``checkpoint`` block from raw YAML."""
    return CheckpointConfig(
        checkpoint_dir=Path(raw.get("checkpoint_dir", "checkpoints")),
        save_every_n_epochs=int(raw.get("save_every_n_epochs", 10)),
        keep_last_n=int(raw.get("keep_last_n", 3)),
    )


def _parse_logging_cfg(raw: dict[str, Any]) -> LoggingConfig:
    """Parse the ``logging`` block from raw YAML."""
    return LoggingConfig(
        log_dir=Path(raw.get("log_dir", "runs")),
        experiment_name=str(raw.get("experiment_name", "mimic_rl")),
        log_every_n_steps=int(raw.get("log_every_n_steps", 50)),
    )


def _known_base_keys() -> frozenset[str]:
    return frozenset(
        {
            "algorithm",
            "schema_version",
            "runtime",
            "checkpoint",
            "logging",
            "dataset_path",
            "dataset_meta_path",
            "n_epochs",
            "batch_size",
            "gamma",
        }
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_training_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Load and resolve a training config YAML file.

    Parameters
    ----------
    path:
        Path to the YAML config file.
    overrides:
        Optional flat dict of top-level key overrides applied *after*
        parsing the file (useful for CLI ``--set key=value`` flags).

    Returns
    -------
    TrainingConfig
        Fully resolved, frozen training configuration with device ready.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required fields are missing or values are invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    raw = _load_yaml(config_path)

    if overrides:
        raw.update(overrides)
        logger.debug("Applied %d config overrides.", len(overrides))

    algorithm = str(raw.get("algorithm", "unknown"))
    schema_version = str(raw.get("schema_version", CONFIG_SCHEMA_VERSION))

    runtime = _parse_runtime(raw.get("runtime", {}))
    checkpoint = _parse_checkpoint(raw.get("checkpoint", {}))
    log_cfg = _parse_logging_cfg(raw.get("logging", {}))

    dataset_path_raw = raw.get("dataset_path")
    if dataset_path_raw is None:
        raise ValueError(
            f"'dataset_path' is required in training config {config_path}."
        )
    dataset_path = Path(dataset_path_raw)

    meta_raw = raw.get("dataset_meta_path")
    dataset_meta_path = Path(meta_raw) if meta_raw else None

    n_epochs = int(raw.get("n_epochs", 100))
    batch_size = int(raw.get("batch_size", 256))
    gamma = float(raw.get("gamma", 0.99))

    # Collect algorithm-specific extras (anything not in the base schema)
    extra: dict[str, Any] = {
        k: v for k, v in raw.items() if k not in _known_base_keys()
    }

    cfg = TrainingConfig(
        algorithm=algorithm,
        schema_version=schema_version,
        runtime=runtime,
        checkpoint=checkpoint,
        logging=log_cfg,
        dataset_path=dataset_path,
        dataset_meta_path=dataset_meta_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        gamma=gamma,
        extra=extra,
    )

    logger.info(
        "Loaded training config: algorithm=%s device=%s epochs=%d batch=%d",
        cfg.algorithm,
        cfg.device,
        cfg.n_epochs,
        cfg.batch_size,
    )

    return cfg


def build_training_config(
    *,
    algorithm: str,
    device: str = "auto",
    dataset_path: str | Path,
    dataset_meta_path: str | Path | None = None,
    n_epochs: int = 100,
    batch_size: int = 256,
    gamma: float = 0.99,
    seed: int = 42,
    checkpoint_dir: str | Path = "checkpoints",
    log_dir: str | Path = "runs",
    experiment_name: str = "mimic_rl",
    extra: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Build a :class:`TrainingConfig` programmatically (for tests / scripts).

    Parameters match the YAML config fields.  All settings have sensible
    defaults so callers only need to provide the mandatory ``dataset_path``.
    """
    torch_device, meta = resolve_device(Backend.from_str(device))

    runtime = RuntimeConfig(
        requested_device=device,
        device=torch_device,
        device_meta=meta,
        seed=seed,
        num_workers=0,
    )
    checkpoint = CheckpointConfig(
        checkpoint_dir=Path(checkpoint_dir),
        save_every_n_epochs=10,
        keep_last_n=3,
    )
    log_cfg = LoggingConfig(
        log_dir=Path(log_dir),
        experiment_name=experiment_name,
        log_every_n_steps=50,
    )

    return TrainingConfig(
        algorithm=algorithm,
        schema_version=CONFIG_SCHEMA_VERSION,
        runtime=runtime,
        checkpoint=checkpoint,
        logging=log_cfg,
        dataset_path=Path(dataset_path),
        dataset_meta_path=Path(dataset_meta_path) if dataset_meta_path else None,
        n_epochs=n_epochs,
        batch_size=batch_size,
        gamma=gamma,
        extra=extra or {},
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RuntimeConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "TrainingConfig",
    "load_training_config",
    "build_training_config",
]
