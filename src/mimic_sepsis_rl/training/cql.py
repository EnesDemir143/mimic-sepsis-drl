"""
Discrete-action CQL (Conservative Q-Learning) reference trainer.

This module is the Phase 7 reference offline RL implementation.  It trains
a discrete-action CQL policy on the replay-buffer dataset produced by Phase 6
and resolves all device and runtime concerns through the shared abstractions
from :mod:`mimic_sepsis_rl.training.device` and
:mod:`mimic_sepsis_rl.training.common`.

Algorithm summary
-----------------
Discrete CQL adds a conservative regularisation term to the standard DQN
Bellman loss:

    L = L_TD + α · E_s[ logsumexp_a Q(s,a) − Q(s, a_data) ]

The logsumexp term penalises high Q-values for out-of-distribution actions
while the subtracted term rewards the observed (in-dataset) action.  This
keeps the learned policy from exploiting spurious Q-value estimates outside
the data support.

Reference: Kumar et al. (2020) "Conservative Q-Learning for Offline
Reinforcement Learning", NeurIPS 2020.

CLI dry-run
-----------
    python -m mimic_sepsis_rl.training.cql \\
        --config configs/training/cql.yaml \\
        --dry-run

Produces one synthetic mini-batch forward/backward pass to verify the full
training graph without touching real data.

Version history
---------------
v1.0.0  2026-03-29  Initial discrete CQL reference trainer.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimic_sepsis_rl.training.common import (
    CheckpointManager,
    EventLogger,
    MetricLogger,
    ReplayDataset,
    TransitionBatch,
    build_checkpoint_manager,
    compute_epoch_metrics,
    load_replay_dataset,
    set_global_seed,
    should_checkpoint,
)
from mimic_sepsis_rl.training.config import (
    TrainingConfig,
    build_training_config,
    load_training_config,
)
from mimic_sepsis_rl.training.device import resolve_device, validate_mps_ops
from mimic_sepsis_rl.reporting.offline_rl import generate_training_report_artifacts

logger = logging.getLogger(__name__)

CQL_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Hyper-parameter defaults (overridable via config YAML ``extra`` block)
# ---------------------------------------------------------------------------

_DEFAULT_HIDDEN_SIZES: list[int] = [256, 256]
_DEFAULT_CQL_ALPHA: float = 1.0
_DEFAULT_LR: float = 3e-4
_DEFAULT_TARGET_UPDATE_FREQ: int = 10  # epochs between hard target updates
_DEFAULT_POLYAK_TAU: float = 0.005  # for soft target updates (if enabled)
_DEFAULT_GRAD_CLIP: float = 10.0


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Fully-connected Q-network mapping states to per-action Q-values.

    Parameters
    ----------
    state_dim:
        Dimensionality of the input state vector.
    n_actions:
        Number of discrete actions (output dimension).
    hidden_sizes:
        List of hidden layer widths.  Defaults to ``[256, 256]``.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = _DEFAULT_HIDDEN_SIZES

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Parameters
        ----------
        states:
            Float tensor of shape ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, n_actions)`` – Q-value for every action.
        """
        return self.net(states)


# ---------------------------------------------------------------------------
# CQL loss components
# ---------------------------------------------------------------------------


def td_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Compute the standard DQN (Bellman) TD error.

    Parameters
    ----------
    q_values:
        ``(B, n_actions)`` – Q-values from the online network for states s_t.
    actions:
        ``(B,)`` int64 – observed actions a_t.
    rewards:
        ``(B,)`` float32 – immediate rewards r_t.
    next_q_values:
        ``(B, n_actions)`` – Q-values from the target network for s_{t+1}.
    dones:
        ``(B,)`` float32 – terminal flags (1.0 if done).
    gamma:
        Discount factor γ.

    Returns
    -------
    torch.Tensor
        Scalar MSE TD loss.
    """
    # Q(s_t, a_t) – gather observed action
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target: r + γ * max_a Q_target(s_{t+1}, a) (zero-masked at terminal)
    with torch.no_grad():
        next_max_q = next_q_values.max(dim=1).values
        target = rewards + gamma * (1.0 - dones) * next_max_q

    return F.mse_loss(q_taken, target)


def cql_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Compute the CQL conservative regularisation term.

    Penalises high Q-values for all actions (via logsumexp) while
    encouraging high Q-values for the in-dataset observed action.

    Parameters
    ----------
    q_values:
        ``(B, n_actions)`` – Q-values from the online network.
    actions:
        ``(B,)`` int64 – in-dataset observed actions.

    Returns
    -------
    torch.Tensor
        Scalar CQL term (add to TD loss scaled by alpha).
    """
    # logsumexp over all actions (log-sum of exp(Q) is a soft-max)
    logsumexp_q = torch.logsumexp(q_values, dim=1)  # (B,)

    # Q-value of the observed (in-data) action
    q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    return (logsumexp_q - q_data).mean()


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class CQLTrainingResult:
    """Summary of a completed CQL training run.

    Attributes
    ----------
    n_epochs : int
        Number of epochs completed.
    total_steps : int
        Total gradient updates performed.
    final_td_loss : float
        Mean TD loss in the final epoch.
    final_cql_loss : float
        Mean CQL term in the final epoch.
    final_total_loss : float
        Mean combined loss in the final epoch.
    checkpoint_path : Path | None
        Path to the final model checkpoint.
    elapsed_seconds : float
        Wall-clock training time.
    state_dim : int
        State vector dimension.
    n_actions : int
        Number of discrete actions.
    device_backend : str
        Resolved device backend (``"cuda"``, ``"mps"``, or ``"cpu"``).
    """

    n_epochs: int
    total_steps: int
    final_td_loss: float
    final_cql_loss: float
    final_total_loss: float
    checkpoint_path: Path | None
    elapsed_seconds: float
    state_dim: int
    n_actions: int
    device_backend: str
    report_artifacts: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "algorithm": "cql",
            "cql_version": CQL_VERSION,
            "n_epochs": self.n_epochs,
            "total_steps": self.total_steps,
            "final_td_loss": self.final_td_loss,
            "final_cql_loss": self.final_cql_loss,
            "final_total_loss": self.final_total_loss,
            "checkpoint_path": str(self.checkpoint_path)
            if self.checkpoint_path
            else None,
            "elapsed_seconds": self.elapsed_seconds,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "device_backend": self.device_backend,
            "report_artifacts": self.report_artifacts,
        }
        return d


# ---------------------------------------------------------------------------
# Saved policy – inference surface
# ---------------------------------------------------------------------------


@dataclass
class CQLPolicy:
    """Loaded CQL policy ready for held-out inference.

    Attributes
    ----------
    q_network : QNetwork
        Trained Q-network in eval mode.
    device : torch.device
        Device the network lives on.
    state_dim : int
    n_actions : int
    checkpoint_path : Path | None
    """

    q_network: QNetwork
    device: torch.device
    state_dim: int
    n_actions: int
    checkpoint_path: Path | None = None

    def select_action(self, state: list[float] | torch.Tensor) -> int:
        """Select the greedy action for a single state vector.

        Parameters
        ----------
        state:
            1-D state vector (length ``state_dim``).

        Returns
        -------
        int
            Greedy action index (argmax Q).
        """
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = state.to(self.device)

        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        self.q_network.eval()
        with torch.no_grad():
            q = self.q_network(state_t)
        return int(q.argmax(dim=1).item())

    def q_values(self, state: list[float] | torch.Tensor) -> list[float]:
        """Return Q-values for all actions given a single state vector."""
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = state.to(self.device)

        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        self.q_network.eval()
        with torch.no_grad():
            q = self.q_network(state_t)
        return q.squeeze(0).tolist()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CQLTrainer:
    """Discrete-action CQL trainer.

    Wires the Q-network, target network, optimizer, dataset, checkpointer,
    and metric logger into a single training loop.

    Parameters
    ----------
    cfg:
        Resolved :class:`~mimic_sepsis_rl.training.config.TrainingConfig`.
    dataset:
        Pre-loaded :class:`~mimic_sepsis_rl.training.common.ReplayDataset`.
    n_actions:
        Number of discrete actions.  Defaults to 25 (sepsis MDP contract).
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        dataset: ReplayDataset,
        *,
        n_actions: int = 25,
    ) -> None:
        self._cfg = cfg
        self._dataset = dataset
        self._device = cfg.device
        self._n_actions = n_actions

        # Hyper-parameters from config extra block (with defaults)
        extra = cfg.extra
        hidden_sizes: list[int] = extra.get("hidden_sizes", _DEFAULT_HIDDEN_SIZES)
        self._cql_alpha: float = float(extra.get("cql_alpha", _DEFAULT_CQL_ALPHA))
        lr: float = float(extra.get("lr", _DEFAULT_LR))
        self._target_update_freq: int = int(
            extra.get("target_update_freq", _DEFAULT_TARGET_UPDATE_FREQ)
        )
        self._polyak_tau: float = float(extra.get("polyak_tau", _DEFAULT_POLYAK_TAU))
        self._use_soft_update: bool = bool(extra.get("use_soft_update", False))
        self._grad_clip: float = float(extra.get("grad_clip", _DEFAULT_GRAD_CLIP))

        state_dim = dataset.state_dim

        # Networks
        self._q_net = QNetwork(state_dim, n_actions, hidden_sizes).to(self._device)
        self._target_net = copy.deepcopy(self._q_net)
        self._target_net.eval()

        # Optimizer
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=lr)

        # Infra
        self._ckpt_mgr = build_checkpoint_manager(cfg)
        self._metric_logger = MetricLogger.from_config(cfg)
        self._training_event_logger = EventLogger.from_config(
            cfg,
            filename="training.log",
        )
        self._runtime_event_logger = EventLogger.from_config(
            cfg,
            filename="runtime.log",
        )

        self._global_step = 0
        self._start_time = 0.0

        logger.info(
            "CQLTrainer initialised: state_dim=%d n_actions=%d alpha=%.3f "
            "lr=%g device=%s",
            state_dim,
            n_actions,
            self._cql_alpha,
            lr,
            self._device,
        )

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _training_step(self, batch: TransitionBatch) -> dict[str, float]:
        """Compute CQL loss and perform one gradient update.

        Returns
        -------
        dict[str, float]
            ``{"td_loss": …, "cql_loss": …, "total_loss": …}``
        """
        self._q_net.train()

        q_values = self._q_net(batch.states)  # (B, n_actions)
        q_data = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        q_max = q_values.max(dim=1).values
        conservative_gap = (torch.logsumexp(q_values, dim=1) - q_data).mean()
        with torch.no_grad():
            next_q_values = self._target_net(batch.next_states)  # (B, n_actions)

        loss_td = td_loss(
            q_values,
            batch.actions,
            batch.rewards,
            next_q_values,
            batch.dones,
            gamma=self._cfg.gamma,
        )
        loss_cql = cql_loss(q_values, batch.actions)
        total_loss = loss_td + self._cql_alpha * loss_cql

        self._optimizer.zero_grad()
        total_loss.backward()

        if self._grad_clip > 0:
            nn.utils.clip_grad_norm_(self._q_net.parameters(), self._grad_clip)

        self._optimizer.step()
        self._global_step += 1

        return {
            "td_loss": loss_td.item(),
            "cql_loss": loss_cql.item(),
            "total_loss": total_loss.item(),
            "mean_q_dataset": q_data.mean().item(),
            "mean_q_max": q_max.mean().item(),
            "conservative_gap": conservative_gap.item(),
        }

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def _maybe_update_target(self, epoch: int) -> None:
        if self._use_soft_update:
            self._soft_update_target()
        elif epoch % self._target_update_freq == 0:
            self._hard_update_target()

    def _hard_update_target(self) -> None:
        self._target_net.load_state_dict(self._q_net.state_dict())
        logger.debug("Target network hard-updated at step %d.", self._global_step)

    def _soft_update_target(self) -> None:
        tau = self._polyak_tau
        for p_online, p_target in zip(
            self._q_net.parameters(), self._target_net.parameters()
        ):
            p_target.data.mul_(1.0 - tau)
            p_target.data.add_(tau * p_online.data)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self) -> CQLTrainingResult:
        """Run the full CQL training loop.

        Returns
        -------
        CQLTrainingResult
            Summary of the completed run including final losses and
            the path to the last checkpoint.
        """
        cfg = self._cfg
        set_global_seed(cfg.runtime.seed)
        self._start_time = time.time()

        final_td: float = float("nan")
        final_cql: float = float("nan")
        final_total: float = float("nan")
        last_ckpt: Path | None = None

        td_losses: list[float] = []
        cql_losses: list[float] = []
        total_losses: list[float] = []
        epoch_durations: list[float] = []

        self._training_event_logger.log_event(
            level="INFO",
            component="trainer",
            event="run_start",
            payload={
                "algorithm": cfg.algorithm,
                "experiment_name": cfg.logging.experiment_name,
                "seed": cfg.runtime.seed,
                "device_backend": cfg.device_meta.backend,
                "batch_size": cfg.batch_size,
                "gamma": cfg.gamma,
                "n_epochs": cfg.n_epochs,
                "n_actions": self._n_actions,
            },
        )

        logger.info(
            "Starting CQL training: epochs=%d batch_size=%d gamma=%.3f alpha=%.3f",
            cfg.n_epochs,
            cfg.batch_size,
            cfg.gamma,
            self._cql_alpha,
        )

        for epoch in range(1, cfg.n_epochs + 1):
            epoch_started_at = time.time()
            td_losses.clear()
            cql_losses.clear()
            total_losses.clear()

            for batch in self._dataset.iter_batches(
                cfg.batch_size, shuffle=True, epoch=epoch
            ):
                step_metrics = self._training_step(batch)
                td_losses.append(step_metrics["td_loss"])
                cql_losses.append(step_metrics["cql_loss"])
                total_losses.append(step_metrics["total_loss"])

                self._metric_logger.log_scalar(
                    "td_loss",
                    step_metrics["td_loss"],
                    step=self._global_step,
                    epoch=epoch,
                )
                self._metric_logger.log_scalar(
                    "cql_loss",
                    step_metrics["cql_loss"],
                    step=self._global_step,
                    epoch=epoch,
                )
                self._metric_logger.log_scalar(
                    "mean_q_dataset",
                    step_metrics["mean_q_dataset"],
                    step=self._global_step,
                    epoch=epoch,
                )
                self._metric_logger.log_scalar(
                    "mean_q_max",
                    step_metrics["mean_q_max"],
                    step=self._global_step,
                    epoch=epoch,
                )
                self._metric_logger.log_scalar(
                    "conservative_gap",
                    step_metrics["conservative_gap"],
                    step=self._global_step,
                    epoch=epoch,
                )

            # Epoch summary
            epoch_duration = time.time() - epoch_started_at
            epoch_durations.append(epoch_duration)
            epoch_metrics = {
                "td_loss_mean": sum(td_losses) / max(len(td_losses), 1),
                "cql_loss_mean": sum(cql_losses) / max(len(cql_losses), 1),
                "total_loss_mean": sum(total_losses) / max(len(total_losses), 1),
            }
            self._metric_logger.log_epoch_summary(
                epoch, self._global_step, epoch_metrics
            )

            self._maybe_update_target(epoch)

            self._training_event_logger.log_event(
                level="INFO",
                component="trainer",
                event="epoch_end",
                payload={
                    "epoch": epoch,
                    "global_step": self._global_step,
                    "metrics": epoch_metrics,
                },
            )
            self._runtime_event_logger.log_event(
                level="INFO",
                component="runtime",
                event="epoch_runtime",
                payload={
                    "epoch": epoch,
                    "global_step": self._global_step,
                    "epoch_elapsed_seconds": epoch_duration,
                    "device_backend": cfg.device_meta.backend,
                },
            )

            if epoch % 10 == 0 or epoch == cfg.n_epochs:
                logger.info(
                    "Epoch %d/%d | td=%.4f cql=%.4f total=%.4f | steps=%d",
                    epoch,
                    cfg.n_epochs,
                    epoch_metrics["td_loss_mean"],
                    epoch_metrics["cql_loss_mean"],
                    epoch_metrics["total_loss_mean"],
                    self._global_step,
                )

            # Checkpoint
            if should_checkpoint(epoch, cfg, is_last=(epoch == cfg.n_epochs)):
                last_ckpt = self._ckpt_mgr.save(
                    self._q_net.state_dict(),
                    epoch=epoch,
                    global_step=self._global_step,
                    metrics=epoch_metrics,
                    cfg=cfg,
                    optimizer_state_dict=self._optimizer.state_dict(),
                )
                self._training_event_logger.log_event(
                    level="INFO",
                    component="checkpoint",
                    event="saved",
                    payload={
                        "epoch": epoch,
                        "global_step": self._global_step,
                        "checkpoint_path": str(last_ckpt),
                    },
                )

            final_td = epoch_metrics["td_loss_mean"]
            final_cql = epoch_metrics["cql_loss_mean"]
            final_total = epoch_metrics["total_loss_mean"]

        elapsed = time.time() - self._start_time
        logger.info(
            "CQL training complete: %d epochs, %d steps, %.1fs elapsed.",
            cfg.n_epochs,
            self._global_step,
            elapsed,
        )

        report_artifacts: dict[str, Any] | None = None
        try:
            artifacts = generate_training_report_artifacts(
                cfg,
                algorithm=cfg.algorithm,
                state_dim=self._dataset.state_dim,
                n_actions=self._n_actions,
                total_steps=self._global_step,
                elapsed_seconds=elapsed,
                final_metrics={
                    "td_loss_mean": final_td,
                    "cql_loss_mean": final_cql,
                    "total_loss_mean": final_total,
                },
                checkpoint_path=last_ckpt,
                epoch_durations=epoch_durations,
                training_log_path=self._training_event_logger.log_path,
                runtime_log_path=self._runtime_event_logger.log_path,
            )
            report_artifacts = artifacts.to_dict()
        except Exception:
            logger.exception("Failed to generate CQL reporting artifacts.")

        self._training_event_logger.log_event(
            level="INFO",
            component="trainer",
            event="run_complete",
            payload={
                "elapsed_seconds": elapsed,
                "total_steps": self._global_step,
                "checkpoint_path": str(last_ckpt) if last_ckpt else None,
                "report_artifact_dir": report_artifacts["artifact_dir"]
                if report_artifacts
                else None,
            },
        )

        return CQLTrainingResult(
            n_epochs=cfg.n_epochs,
            total_steps=self._global_step,
            final_td_loss=final_td,
            final_cql_loss=final_cql,
            final_total_loss=final_total,
            checkpoint_path=last_ckpt,
            elapsed_seconds=elapsed,
            state_dim=self._dataset.state_dim,
            n_actions=self._n_actions,
            device_backend=cfg.device_meta.backend,
            report_artifacts=report_artifacts,
        )

    def get_policy(self) -> CQLPolicy:
        """Return the current Q-network wrapped in a :class:`CQLPolicy`."""
        return CQLPolicy(
            q_network=copy.deepcopy(self._q_net).eval(),
            device=self._device,
            state_dim=self._dataset.state_dim,
            n_actions=self._n_actions,
            checkpoint_path=self._ckpt_mgr.latest_checkpoint(),
        )


# ---------------------------------------------------------------------------
# Load a saved CQL policy from a checkpoint
# ---------------------------------------------------------------------------


def load_cql_policy(
    checkpoint_path: Path,
    *,
    state_dim: int,
    n_actions: int = 25,
    hidden_sizes: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> CQLPolicy:
    """Reload a saved CQL policy from a checkpoint file.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` checkpoint file saved by :class:`CheckpointManager`.
    state_dim:
        Dimensionality of the state vector (must match training config).
    n_actions:
        Number of discrete actions (default 25).
    hidden_sizes:
        Hidden layer sizes (must match training config).
    device:
        Target device for inference.

    Returns
    -------
    CQLPolicy
        Policy ready for :meth:`CQLPolicy.select_action` calls.
    """
    if isinstance(device, str):
        torch_device, _ = resolve_device(device)
    else:
        torch_device = device

    payload = CheckpointManager.load(checkpoint_path, device=torch_device)

    q_net = QNetwork(state_dim, n_actions, hidden_sizes)
    q_net.load_state_dict(payload["model_state_dict"])
    q_net = q_net.to(torch_device).eval()

    logger.info(
        "CQL policy loaded: state_dim=%d n_actions=%d device=%s",
        state_dim,
        n_actions,
        torch_device,
    )

    return CQLPolicy(
        q_network=q_net,
        device=torch_device,
        state_dim=state_dim,
        n_actions=n_actions,
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------


def _dry_run(cfg: TrainingConfig, n_actions: int = 25) -> None:
    """Execute one synthetic mini-batch forward/backward pass.

    Validates the full computation graph (network → CQL loss → backward)
    without loading real data.  Used for CI and quick environment checks.
    """
    logger.info("=== CQL DRY-RUN ===")
    logger.info(
        "Config: algorithm=%s device=%s epochs=%d batch=%d",
        cfg.algorithm,
        cfg.device,
        cfg.n_epochs,
        cfg.batch_size,
    )

    extra = cfg.extra
    hidden_sizes: list[int] = extra.get("hidden_sizes", _DEFAULT_HIDDEN_SIZES)
    cql_alpha: float = float(extra.get("cql_alpha", _DEFAULT_CQL_ALPHA))
    state_dim: int = int(extra.get("dry_run_state_dim", 33))
    batch_size: int = min(cfg.batch_size, 16)

    # Validate MPS op support early
    if cfg.device.type == "mps":
        issues = validate_mps_ops(cfg.device)
        if issues:
            logger.warning(
                "%d MPS op issue(s) detected. "
                "Set PYTORCH_ENABLE_MPS_FALLBACK=1 if training fails.",
                len(issues),
            )

    q_net = QNetwork(state_dim, n_actions, hidden_sizes).to(cfg.device)
    target_net = copy.deepcopy(q_net).eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

    # Synthetic batch — generate on CPU then move to target device so that
    # a CPU-pinned Generator works correctly on MPS and CUDA alike.
    g = torch.Generator()
    g.manual_seed(42)
    states = torch.randn(batch_size, state_dim, generator=g).to(cfg.device)
    actions = torch.randint(0, n_actions, (batch_size,)).to(cfg.device)
    rewards = torch.randn(batch_size, generator=g).to(cfg.device)
    next_states = torch.randn(batch_size, state_dim, generator=g).to(cfg.device)
    dones = torch.zeros(batch_size, device=cfg.device)
    dones[: batch_size // 4] = 1.0

    batch = TransitionBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    # Forward pass
    q_values = q_net(batch.states)
    with torch.no_grad():
        next_q_values = target_net(batch.next_states)

    loss_td = td_loss(
        q_values,
        batch.actions,
        batch.rewards,
        next_q_values,
        batch.dones,
        gamma=cfg.gamma,
    )
    loss_cql = cql_loss(q_values, batch.actions)
    total_loss = loss_td + cql_alpha * loss_cql

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), _DEFAULT_GRAD_CLIP)
    optimizer.step()

    logger.info(
        "Dry-run forward/backward: td_loss=%.4f cql_loss=%.4f total=%.4f",
        loss_td.item(),
        loss_cql.item(),
        total_loss.item(),
    )

    # Inference check
    policy = CQLPolicy(
        q_network=copy.deepcopy(q_net).eval(),
        device=cfg.device,
        state_dim=state_dim,
        n_actions=n_actions,
    )
    test_state = [0.0] * state_dim
    action = policy.select_action(test_state)
    assert 0 <= action < n_actions, f"select_action returned invalid action: {action}"

    logger.info(
        "Dry-run complete ✅  device=%s backend=%s",
        cfg.device,
        cfg.device_meta.backend,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m mimic_sepsis_rl.training.cql``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.training.cql",
        description="Train a discrete-action CQL policy on the offline sepsis dataset.",
    )
    parser.add_argument(
        "--config",
        default="configs/training/cql.yaml",
        help="Path to the CQL training config YAML (default: configs/training/cql.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute one synthetic mini-batch to verify the training graph, then exit.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override the device in the config (auto, cuda, mps, cpu).",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=25,
        help="Number of discrete actions (default: 25).",
    )
    args = parser.parse_args(argv)

    # Build/load config
    config_path = Path(args.config)
    overrides: dict[str, Any] = {}
    if args.device:
        overrides["runtime"] = {"device": args.device}

    if config_path.exists():
        cfg = load_training_config(
            config_path, overrides=overrides if overrides else None
        )
    else:
        # Minimal in-memory config for dry-run when no file exists yet
        logger.warning(
            "Config file not found at %s. Using defaults for dry-run.", config_path
        )
        cfg = build_training_config(
            algorithm="cql",
            device=args.device or "auto",
            dataset_path=Path("data/replay/replay_train.parquet"),
            n_epochs=100,
            batch_size=256,
            gamma=0.99,
        )

    if args.dry_run:
        _dry_run(cfg, n_actions=args.n_actions)
        sys.exit(0)

    # Full training run
    logger.info("Loading replay dataset from %s …", cfg.dataset_path)
    dataset = load_replay_dataset(cfg)

    trainer = CQLTrainer(cfg, dataset, n_actions=args.n_actions)
    result = trainer.train()

    import json

    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "CQL_VERSION",
    "QNetwork",
    "CQLTrainer",
    "CQLPolicy",
    "CQLTrainingResult",
    "load_cql_policy",
    "td_loss",
    "cql_loss",
]
