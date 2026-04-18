"""
Shared training helpers for the MIMIC Sepsis Offline RL stack.

Provides dataset loading, checkpoint management, metric logging, and
config utilities reused by every algorithm trainer (CQL, BCQ, IQL, …).
Algorithm code should never re-implement these helpers independently.

Responsibilities
----------------
- Load replay-buffer Parquet files into in-memory tensor batches.
- Persist and restore model checkpoints with provenance manifests.
- Accumulate and flush scalar training metrics to JSON log files.
- Provide a reproducibility seed-setter that covers Python, NumPy, and
  PyTorch (CPU + CUDA/MPS where supported).

Usage
-----
    from mimic_sepsis_rl.training.common import (
        ReplayDataset,
        CheckpointManager,
        MetricLogger,
        set_global_seed,
    )

Version history
---------------
v1.0.0  2026-03-29  Initial shared training utilities.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

import polars as pl
import torch

from mimic_sepsis_rl.training.config import TrainingConfig

logger = logging.getLogger(__name__)

COMMON_MODULE_VERSION: str = "1.0.0"
LOG_TIMEZONE_NAME: str = "Europe/Istanbul"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Set the global random seed for Python, NumPy, and PyTorch.

    Covers:
    - ``random`` standard library
    - ``numpy`` (if installed)
    - ``torch`` CPU RNG
    - ``torch.cuda`` RNG (all devices, if CUDA is available)
    - ``torch.backends.mps`` does not expose a seed API but the ``torch``
      CPU seed propagates through it on supported versions.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.debug("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Replay dataset
# ---------------------------------------------------------------------------


@dataclass
class TransitionBatch:
    """A mini-batch of MDP transitions ready for gradient computation.

    All tensors share the same device and dtype.

    Attributes
    ----------
    states : torch.Tensor
        Shape ``(B, state_dim)`` – float32 state vectors at step *t*.
    actions : torch.Tensor
        Shape ``(B,)`` – int64 discrete action IDs.
    rewards : torch.Tensor
        Shape ``(B,)`` – float32 scalar rewards.
    next_states : torch.Tensor
        Shape ``(B, state_dim)`` – float32 state vectors at step *t+1*.
    dones : torch.Tensor
        Shape ``(B,)`` – float32 terminal flags (1.0 if done, 0.0 otherwise).
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.states.shape[0]

    @property
    def state_dim(self) -> int:
        return self.states.shape[1]

    def to(self, device: torch.device) -> "TransitionBatch":
        """Move all tensors to *device* in-place and return self."""
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.dones = self.dones.to(device)
        return self


class ReplayDataset:
    """In-memory replay dataset loaded from a Parquet transition file.

    Wraps the flat transition table exported by Phase 6 into shuffled
    mini-batch iterators compatible with all algorithm trainers.

    Parameters
    ----------
    parquet_path:
        Path to a replay-buffer Parquet file produced by
        :func:`mimic_sepsis_rl.datasets.replay_buffer.save_replay_buffer`.
    device:
        Target ``torch.device`` for tensor allocation.
    state_columns:
        Ordered list of state-feature column names.  If ``None``, the
        loader auto-detects columns with the ``s_`` prefix.
    seed:
        Shuffle seed; passed to the sampler on each epoch.
    """

    def __init__(
        self,
        parquet_path: Path,
        *,
        device: torch.device,
        state_columns: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self._path = parquet_path
        self._device = device
        self._seed = seed

        logger.info("Loading replay dataset from %s …", parquet_path)
        df = pl.read_parquet(parquet_path)

        # Auto-detect state columns if not provided
        if state_columns is None:
            state_columns = sorted(
                c for c in df.columns if c.startswith("s_") and not c.startswith("ns_")
            )
        if not state_columns:
            raise ValueError(
                f"No state columns (prefix 's_') found in {parquet_path}. "
                "Pass 'state_columns' explicitly."
            )
        self._state_columns = state_columns
        ns_columns = [f"ns_{c[2:]}" for c in state_columns]

        # Validate required columns
        required = set(state_columns) | set(ns_columns) | {"action", "reward", "done"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Replay Parquet at {parquet_path} is missing columns: {missing}"
            )

        # Extract numpy arrays then convert to tensors once
        states_np = df.select(state_columns).to_numpy()
        next_states_np = df.select(ns_columns).to_numpy()
        actions_np = df["action"].to_numpy()
        rewards_np = df["reward"].to_numpy()
        dones_np = df["done"].to_numpy().astype("float32")

        self._states = torch.tensor(states_np, dtype=torch.float32)
        self._next_states = torch.tensor(next_states_np, dtype=torch.float32)
        self._actions = torch.tensor(actions_np, dtype=torch.int64)
        self._rewards = torch.tensor(rewards_np, dtype=torch.float32)
        self._dones = torch.tensor(dones_np, dtype=torch.float32)

        logger.info(
            "Loaded %d transitions | state_dim=%d | actions=[0,%d)",
            len(df),
            len(state_columns),
            int(self._actions.max().item()) + 1,
        )

    @property
    def n_transitions(self) -> int:
        return self._states.shape[0]

    @property
    def state_dim(self) -> int:
        return self._states.shape[1]

    @property
    def state_columns(self) -> list[str]:
        return list(self._state_columns)

    def __len__(self) -> int:
        return self.n_transitions

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        epoch: int = 0,
    ) -> Iterator[TransitionBatch]:
        """Yield mini-batches over the full dataset.

        Parameters
        ----------
        batch_size:
            Number of transitions per mini-batch.
        shuffle:
            Shuffle indices before batching (recommended for training).
        epoch:
            Current epoch number; combined with the dataset seed to give
            deterministic-but-varied shuffles per epoch.

        Yields
        ------
        TransitionBatch
            Tensors already moved to ``self._device``.
        """
        n = self.n_transitions
        indices = torch.arange(n)

        if shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch)
            indices = indices[torch.randperm(n, generator=g)]

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            batch = TransitionBatch(
                states=self._states[idx].to(self._device),
                actions=self._actions[idx].to(self._device),
                rewards=self._rewards[idx].to(self._device),
                next_states=self._next_states[idx].to(self._device),
                dones=self._dones[idx].to(self._device),
            )
            yield batch

    def sample_batch(self, batch_size: int) -> TransitionBatch:
        """Sample a random mini-batch (with replacement).

        Useful for off-policy algorithms that maintain their own replay
        sampling logic rather than epoch-based iteration.
        """
        idx = torch.randint(0, self.n_transitions, (batch_size,))
        return TransitionBatch(
            states=self._states[idx].to(self._device),
            actions=self._actions[idx].to(self._device),
            rewards=self._rewards[idx].to(self._device),
            next_states=self._next_states[idx].to(self._device),
            dones=self._dones[idx].to(self._device),
        )


def load_replay_dataset(cfg: TrainingConfig) -> ReplayDataset:
    """Load the replay dataset specified in *cfg*.

    Parameters
    ----------
    cfg:
        Resolved :class:`~mimic_sepsis_rl.training.config.TrainingConfig`.

    Raises
    ------
    FileNotFoundError
        If ``cfg.dataset_path`` does not exist.
    """
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.dataset_path}\n"
            "Run Phase 6 (build_transitions) to generate replay buffers."
        )
    return ReplayDataset(
        cfg.dataset_path,
        device=cfg.device,
        seed=cfg.runtime.seed,
    )


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


@dataclass
class CheckpointManifest:
    """Serialisable metadata stored alongside every model checkpoint.

    Attributes
    ----------
    algorithm : str
        Algorithm identifier (e.g. ``"cql"``).
    epoch : int
        Training epoch at which the checkpoint was saved.
    global_step : int
        Total gradient steps completed.
    metrics : dict[str, float]
        Scalar metrics at checkpoint time (e.g. ``{"td_loss": 0.12}``).
    config_dict : dict[str, Any]
        Serialised training config for reproducibility.
    device_meta : dict[str, Any]
        Backend metadata snapshot from :class:`DeviceMetadata`.
    timestamp : float
        Unix timestamp of the checkpoint.
    module_version : str
        ``COMMON_MODULE_VERSION`` sentinel.
    """

    algorithm: str
    epoch: int
    global_step: int
    metrics: dict[str, float]
    config_dict: dict[str, Any]
    device_meta: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    module_version: str = COMMON_MODULE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointManifest":
        return cls(
            algorithm=d["algorithm"],
            epoch=d["epoch"],
            global_step=d["global_step"],
            metrics=d.get("metrics", {}),
            config_dict=d.get("config_dict", {}),
            device_meta=d.get("device_meta", {}),
            timestamp=d.get("timestamp", 0.0),
            module_version=d.get("module_version", COMMON_MODULE_VERSION),
        )


class CheckpointManager:
    """Save and restore model checkpoints with provenance manifests.

    Parameters
    ----------
    checkpoint_dir:
        Root directory where checkpoints are written.
    algorithm:
        Algorithm name, used as a filename prefix.
    keep_last_n:
        Number of most-recent checkpoints to retain (0 = keep all).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        algorithm: str,
        keep_last_n: int = 3,
    ) -> None:
        self._dir = checkpoint_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._algorithm = algorithm
        self._keep_last_n = keep_last_n
        self._saved: list[Path] = []

    def save(
        self,
        model_state_dict: dict[str, Any],
        *,
        epoch: int,
        global_step: int,
        metrics: dict[str, float],
        cfg: TrainingConfig,
        optimizer_state_dict: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a model checkpoint and its manifest.

        Parameters
        ----------
        model_state_dict:
            ``model.state_dict()`` from the PyTorch module.
        epoch:
            Current training epoch.
        global_step:
            Cumulative gradient step count.
        metrics:
            Scalar metrics to embed in the manifest.
        cfg:
            Training config (serialised into the manifest).
        optimizer_state_dict:
            Optional optimizer state for training resumption.

        Returns
        -------
        Path
            Path to the saved ``.pt`` checkpoint file.
        """
        stem = f"{self._algorithm}_epoch{epoch:04d}_step{global_step:07d}"
        ckpt_path = self._dir / f"{stem}.pt"
        manifest_path = self._dir / f"{stem}_manifest.json"

        payload: dict[str, Any] = {
            "model_state_dict": model_state_dict,
            "epoch": epoch,
            "global_step": global_step,
        }
        if optimizer_state_dict is not None:
            payload["optimizer_state_dict"] = optimizer_state_dict

        torch.save(payload, ckpt_path)

        manifest = CheckpointManifest(
            algorithm=self._algorithm,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            config_dict=cfg.to_dict(),
            device_meta=cfg.device_meta.to_dict(),
        )
        manifest_path.write_text(manifest.to_json())

        logger.info("Checkpoint saved: %s", ckpt_path)
        self._saved.append(ckpt_path)
        self._prune()

        return ckpt_path

    def _prune(self) -> None:
        """Remove oldest checkpoints beyond keep_last_n."""
        if self._keep_last_n <= 0:
            return
        while len(self._saved) > self._keep_last_n:
            old = self._saved.pop(0)
            old_manifest = old.with_name(old.stem + "_manifest.json")
            if old.exists():
                old.unlink()
                logger.debug("Pruned checkpoint: %s", old)
            if old_manifest.exists():
                old_manifest.unlink()

    def latest_checkpoint(self) -> Path | None:
        """Return the path to the most-recently saved checkpoint, or None."""
        candidates = sorted(self._dir.glob(f"{self._algorithm}_epoch*.pt"))
        return candidates[-1] if candidates else None

    @staticmethod
    def load(path: Path, *, device: torch.device) -> dict[str, Any]:
        """Load a checkpoint from *path* onto *device*.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint file.
        device:
            Target device for tensor mapping.

        Returns
        -------
        dict
            Raw checkpoint payload containing ``model_state_dict`` and
            optional ``optimizer_state_dict``, ``epoch``, ``global_step``.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=device, weights_only=True)
        logger.info("Checkpoint loaded from %s → device=%s", path, device)
        return payload

    @staticmethod
    def load_manifest(path: Path) -> CheckpointManifest:
        """Load the JSON manifest adjacent to a checkpoint file."""
        stem = path.stem
        manifest_path = path.with_name(f"{stem}_manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
        return CheckpointManifest.from_dict(json.loads(manifest_path.read_text()))


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------


@dataclass
class ScalarMetric:
    """One logged scalar measurement.

    Attributes
    ----------
    step : int
        Global gradient step at which the metric was recorded.
    epoch : int
        Training epoch.
    name : str
        Metric name (e.g. ``"td_loss"``).
    value : float
        Scalar value.
    timestamp : float
        Unix timestamp.
    """

    step: int
    epoch: int
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricLogger:
    """Accumulate scalar metrics and flush them to a JSON-Lines log file.

    Supports both step-level and epoch-level metrics.  The log file is
    flushed after each :meth:`flush` call, making it safe to tail during
    a training run.

    Parameters
    ----------
    log_dir:
        Directory where the ``metrics.jsonl`` file is written.
    experiment_name:
        Used as the log filename prefix.
    log_every_n_steps:
        Emit accumulated step metrics every *N* gradient steps.
    """

    def __init__(
        self,
        log_dir: Path,
        *,
        experiment_name: str = "mimic_rl",
        log_every_n_steps: int = 50,
    ) -> None:
        self._dir = log_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._dir / f"{experiment_name}_metrics.jsonl"
        self._log_every_n = log_every_n_steps
        self._buffer: list[ScalarMetric] = []
        self._step_accum: dict[str, list[float]] = {}

    def log_scalar(
        self,
        name: str,
        value: float,
        *,
        step: int,
        epoch: int,
    ) -> None:
        """Accumulate a scalar metric.

        Flushes to disk automatically when ``step % log_every_n_steps == 0``.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Scalar value (NaN/Inf values are recorded but logged as warnings).
        step:
            Current global gradient step.
        epoch:
            Current training epoch.
        """
        if not math.isfinite(value):
            logger.warning("Non-finite metric '%s'=%s at step %d.", name, value, step)

        self._buffer.append(
            ScalarMetric(step=step, epoch=epoch, name=name, value=value)
        )
        self._step_accum.setdefault(name, []).append(value)

        if step > 0 and step % self._log_every_n == 0:
            self.flush()

    def log_epoch_summary(
        self,
        epoch: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Log a batch of epoch-level summary metrics at once."""
        for name, value in metrics.items():
            self.log_scalar(name, value, step=step, epoch=epoch)
        self.flush()

    def flush(self) -> None:
        """Write buffered metrics to disk and clear the buffer."""
        if not self._buffer:
            return
        with self._log_path.open("a") as fh:
            for m in self._buffer:
                fh.write(json.dumps(m.to_dict()) + "\n")
        logger.debug(
            "Flushed %d metric records to %s.", len(self._buffer), self._log_path
        )
        self._buffer.clear()

    def epoch_mean(self, name: str) -> float | None:
        """Return the mean of accumulated values for *name*, then reset."""
        vals = self._step_accum.pop(name, None)
        if not vals:
            return None
        return sum(vals) / len(vals)

    def reset_accumulators(self) -> None:
        """Clear per-epoch step accumulators (call at the end of each epoch)."""
        self._step_accum.clear()

    @classmethod
    def from_config(cls, cfg: TrainingConfig) -> "MetricLogger":
        """Build a MetricLogger from a :class:`TrainingConfig`."""
        return cls(
            cfg.logging.log_dir,
            experiment_name=cfg.logging.experiment_name,
            log_every_n_steps=cfg.logging.log_every_n_steps,
        )


def _timestamped_now() -> str:
    """Return an ISO 8601 timestamp in the project default timezone."""
    return datetime.now(ZoneInfo(LOG_TIMEZONE_NAME)).isoformat(timespec="seconds")


class EventLogger:
    """Append structured single-line events to a timestamped `.log` file."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Return the backing log file path."""
        return self._log_path

    def log_event(
        self,
        *,
        level: str,
        component: str,
        event: str,
        payload: Mapping[str, Any] | str | None = None,
    ) -> None:
        """Write one append-only event line.

        Parameters
        ----------
        level:
            Human-readable severity level such as ``INFO`` or ``WARNING``.
        component:
            Component name producing the event.
        event:
            Short event identifier.
        payload:
            Optional JSON-serialisable payload or free-form message.
        """
        timestamp = _timestamped_now()
        if payload is None:
            payload_text = "{}"
        elif isinstance(payload, str):
            payload_text = payload.replace("\n", "\\n")
        else:
            payload_text = json.dumps(payload, ensure_ascii=True, sort_keys=True)

        line = f"{timestamp} {level.upper()} {component} {event} {payload_text}\n"
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    @classmethod
    def from_config(
        cls,
        cfg: TrainingConfig,
        *,
        filename: str,
    ) -> "EventLogger":
        """Build an event logger under the experiment-specific artifact folder."""
        artifact_dir = cfg.logging.log_dir / cfg.logging.experiment_name
        return cls(artifact_dir / filename)


# ---------------------------------------------------------------------------
# Training loop utilities
# ---------------------------------------------------------------------------


def build_checkpoint_manager(cfg: TrainingConfig) -> CheckpointManager:
    """Build a :class:`CheckpointManager` from *cfg*."""
    return CheckpointManager(
        cfg.checkpoint.checkpoint_dir,
        algorithm=cfg.algorithm,
        keep_last_n=cfg.checkpoint.keep_last_n,
    )


def should_checkpoint(
    epoch: int, cfg: TrainingConfig, *, is_last: bool = False
) -> bool:
    """Return True if a checkpoint should be saved at *epoch*.

    Parameters
    ----------
    epoch:
        Current (1-based) epoch index.
    cfg:
        Training config.
    is_last:
        Force True for the final epoch regardless of cadence.
    """
    if is_last:
        return True
    every = cfg.checkpoint.save_every_n_epochs
    return every > 0 and epoch % every == 0


def compute_epoch_metrics(
    losses: Sequence[float],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Summarise a list of per-step losses into epoch-level scalars.

    Parameters
    ----------
    losses:
        Per-step scalar loss values recorded during the epoch.
    prefix:
        Optional prefix added to all metric names (e.g. ``"cql_"``).

    Returns
    -------
    dict[str, float]
        Keys: ``{prefix}loss_mean``, ``{prefix}loss_min``, ``{prefix}loss_max``.
    """
    if not losses:
        return {f"{prefix}loss_mean": float("nan")}
    return {
        f"{prefix}loss_mean": sum(losses) / len(losses),
        f"{prefix}loss_min": min(losses),
        f"{prefix}loss_max": max(losses),
    }


__all__ = [
    "COMMON_MODULE_VERSION",
    "LOG_TIMEZONE_NAME",
    "TransitionBatch",
    "ReplayDataset",
    "CheckpointManifest",
    "CheckpointManager",
    "ScalarMetric",
    "MetricLogger",
    "EventLogger",
    "set_global_seed",
    "load_replay_dataset",
    "build_checkpoint_manager",
    "should_checkpoint",
    "compute_epoch_metrics",
]
