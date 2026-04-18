"""
Discrete-action BCQ trainer on the shared offline RL experiment surface.

BCQ (Batch-Constrained Q-learning) constrains value maximization to actions
that remain plausible under a learned behavior model. This implementation
reuses the shared training config, replay dataset, checkpointing, and metric
logging layers introduced for the CQL reference trainer.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimic_sepsis_rl.training.common import (
    EventLogger,
    MetricLogger,
    ReplayDataset,
    TransitionBatch,
    build_checkpoint_manager,
    load_replay_dataset,
    set_global_seed,
    should_checkpoint,
)
from mimic_sepsis_rl.training.config import (
    TrainingConfig,
    build_training_config,
    load_training_config,
)
from mimic_sepsis_rl.training.cql import QNetwork
from mimic_sepsis_rl.training.device import validate_mps_ops
from mimic_sepsis_rl.reporting.offline_rl import generate_training_report_artifacts

logger = logging.getLogger(__name__)

BCQ_VERSION: str = "1.0.0"

_DEFAULT_HIDDEN_SIZES: list[int] = [256, 256]
_DEFAULT_ACTOR_LR: float = 1e-4
_DEFAULT_CRITIC_LR: float = 3e-4
_DEFAULT_POLYAK_TAU: float = 0.005
_DEFAULT_TARGET_UPDATE_FREQ: int = 10
_DEFAULT_IMITATION_THRESHOLD: float = 0.3
_DEFAULT_BEHAVIOR_CLONING_WEIGHT: float = 1.0
_DEFAULT_GRAD_CLIP: float = 10.0


class BehaviorPolicy(nn.Module):
    """Behavior-cloning policy head used to constrain BCQ action selection."""

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
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, n_actions))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def select_bcq_actions(
    q_values: torch.Tensor,
    behavior_logits: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Select BCQ actions using a behavior-support mask."""
    bounded_threshold = min(max(threshold, 0.0), 1.0)
    behavior_probs = F.softmax(behavior_logits, dim=1)
    max_prob = behavior_probs.max(dim=1, keepdim=True).values
    support_mask = behavior_probs >= (bounded_threshold * max_prob)

    masked_q_values = q_values.masked_fill(~support_mask, float("-inf"))
    invalid_rows = ~torch.isfinite(masked_q_values).any(dim=1)
    if invalid_rows.any():
        masked_q_values[invalid_rows] = q_values[invalid_rows]
    return masked_q_values.argmax(dim=1)


@dataclass
class BCQTrainingResult:
    """Summary of a completed BCQ training run."""

    n_epochs: int
    total_steps: int
    final_td_loss: float
    final_imitation_loss: float
    final_total_loss: float
    checkpoint_path: Path | None
    elapsed_seconds: float
    state_dim: int
    n_actions: int
    device_backend: str
    report_artifacts: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "bcq",
            "bcq_version": BCQ_VERSION,
            "n_epochs": self.n_epochs,
            "total_steps": self.total_steps,
            "final_td_loss": self.final_td_loss,
            "final_imitation_loss": self.final_imitation_loss,
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


@dataclass
class BCQPolicy:
    """Loaded BCQ policy surface for greedy discrete-action inference."""

    q_network: QNetwork
    behavior_policy: BehaviorPolicy
    device: torch.device
    state_dim: int
    n_actions: int
    threshold: float
    checkpoint_path: Path | None = None

    def select_action(self, state: list[float] | torch.Tensor) -> int:
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        self.q_network.eval()
        self.behavior_policy.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            behavior_logits = self.behavior_policy(state_tensor)
            actions = select_bcq_actions(
                q_values,
                behavior_logits,
                threshold=self.threshold,
            )
        return int(actions.item())


class BCQTrainer:
    """Discrete-action BCQ trainer using shared replay and artifact contracts."""

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

        extra = cfg.extra
        hidden_sizes: list[int] = extra.get("hidden_sizes", _DEFAULT_HIDDEN_SIZES)
        self._actor_lr = float(extra.get("actor_lr", _DEFAULT_ACTOR_LR))
        self._critic_lr = float(extra.get("critic_lr", _DEFAULT_CRITIC_LR))
        self._polyak_tau = float(extra.get("polyak_tau", _DEFAULT_POLYAK_TAU))
        self._target_update_freq = int(
            extra.get("target_update_freq", _DEFAULT_TARGET_UPDATE_FREQ)
        )
        self._imitation_threshold = float(
            extra.get("imitation_threshold", _DEFAULT_IMITATION_THRESHOLD)
        )
        self._behavior_cloning_weight = float(
            extra.get(
                "behavior_cloning_weight",
                _DEFAULT_BEHAVIOR_CLONING_WEIGHT,
            )
        )
        self._grad_clip = float(extra.get("grad_clip", _DEFAULT_GRAD_CLIP))

        state_dim = dataset.state_dim
        self._q_network = QNetwork(state_dim, n_actions, hidden_sizes).to(self._device)
        self._target_q_network = copy.deepcopy(self._q_network).to(self._device).eval()
        self._behavior_policy = BehaviorPolicy(
            state_dim,
            n_actions,
            hidden_sizes=hidden_sizes,
        ).to(self._device)

        self._critic_optimizer = torch.optim.Adam(
            self._q_network.parameters(),
            lr=self._critic_lr,
        )
        self._actor_optimizer = torch.optim.Adam(
            self._behavior_policy.parameters(),
            lr=self._actor_lr,
        )

        self._checkpoint_manager = build_checkpoint_manager(cfg)
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
            "BCQTrainer initialised: state_dim=%d n_actions=%d threshold=%.3f "
            "critic_lr=%g actor_lr=%g device=%s",
            state_dim,
            n_actions,
            self._imitation_threshold,
            self._critic_lr,
            self._actor_lr,
            self._device,
        )

    def _update_target_network(self) -> None:
        if self._target_update_freq <= 0:
            return
        if self._global_step % self._target_update_freq != 0:
            return

        tau = min(max(self._polyak_tau, 0.0), 1.0)
        for online_param, target_param in zip(
            self._q_network.parameters(),
            self._target_q_network.parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)

    def _training_step(self, batch: TransitionBatch) -> dict[str, float]:
        self._q_network.train()
        self._behavior_policy.train()

        q_values = self._q_network(batch.states)
        chosen_q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self._target_q_network(batch.next_states)
            next_behavior_logits = self._behavior_policy(batch.next_states)
            next_actions = select_bcq_actions(
                next_q_values,
                next_behavior_logits,
                threshold=self._imitation_threshold,
            )
            next_state_values = next_q_values.gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            targets = (
                batch.rewards
                + self._cfg.gamma * (1.0 - batch.dones) * next_state_values
            )

        td_loss = F.mse_loss(chosen_q_values, targets)
        behavior_logits = self._behavior_policy(batch.states)
        behavior_probs = F.softmax(behavior_logits, dim=1)
        behavior_entropy = -(
            behavior_probs * behavior_probs.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        support_mask = behavior_probs >= (
            self._imitation_threshold * behavior_probs.max(dim=1, keepdim=True).values
        )
        imitation_loss = F.cross_entropy(behavior_logits, batch.actions)
        total_loss = td_loss + self._behavior_cloning_weight * imitation_loss

        self._critic_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()
        total_loss.backward()

        if self._grad_clip > 0:
            nn.utils.clip_grad_norm_(self._q_network.parameters(), self._grad_clip)
            nn.utils.clip_grad_norm_(
                self._behavior_policy.parameters(),
                self._grad_clip,
            )

        self._critic_optimizer.step()
        self._actor_optimizer.step()

        self._global_step += 1
        self._update_target_network()

        return {
            "td_loss": td_loss.item(),
            "imitation_loss": imitation_loss.item(),
            "total_loss": total_loss.item(),
            "mean_q_dataset": chosen_q_values.mean().item(),
            "behavior_entropy": behavior_entropy.item(),
            "support_rate": support_mask.float().mean().item(),
        }

    def train(self) -> BCQTrainingResult:
        cfg = self._cfg
        set_global_seed(cfg.runtime.seed)
        self._start_time = time.time()

        final_td_loss = float("nan")
        final_imitation_loss = float("nan")
        final_total_loss = float("nan")
        last_checkpoint: Path | None = None

        td_losses: list[float] = []
        imitation_losses: list[float] = []
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
            "Starting BCQ training: epochs=%d batch_size=%d gamma=%.3f threshold=%.3f",
            cfg.n_epochs,
            cfg.batch_size,
            cfg.gamma,
            self._imitation_threshold,
        )

        for epoch in range(1, cfg.n_epochs + 1):
            epoch_started_at = time.time()
            td_losses.clear()
            imitation_losses.clear()
            total_losses.clear()

            for batch in self._dataset.iter_batches(
                cfg.batch_size,
                shuffle=True,
                epoch=epoch,
            ):
                step_metrics = self._training_step(batch)
                td_losses.append(step_metrics["td_loss"])
                imitation_losses.append(step_metrics["imitation_loss"])
                total_losses.append(step_metrics["total_loss"])

                for name, value in step_metrics.items():
                    self._metric_logger.log_scalar(
                        name,
                        value,
                        step=self._global_step,
                        epoch=epoch,
                    )

            epoch_duration = time.time() - epoch_started_at
            epoch_durations.append(epoch_duration)
            epoch_metrics = {
                "td_loss_mean": sum(td_losses) / max(len(td_losses), 1),
                "imitation_loss_mean": sum(imitation_losses)
                / max(len(imitation_losses), 1),
                "total_loss_mean": sum(total_losses) / max(len(total_losses), 1),
            }
            self._metric_logger.log_epoch_summary(
                epoch,
                self._global_step,
                epoch_metrics,
            )
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
                    "Epoch %d/%d | td=%.4f imitation=%.4f total=%.4f | steps=%d",
                    epoch,
                    cfg.n_epochs,
                    epoch_metrics["td_loss_mean"],
                    epoch_metrics["imitation_loss_mean"],
                    epoch_metrics["total_loss_mean"],
                    self._global_step,
                )

            if should_checkpoint(epoch, cfg, is_last=(epoch == cfg.n_epochs)):
                last_checkpoint = self._checkpoint_manager.save(
                    {
                        "q_network": self._q_network.state_dict(),
                        "target_q_network": self._target_q_network.state_dict(),
                        "behavior_policy": self._behavior_policy.state_dict(),
                    },
                    epoch=epoch,
                    global_step=self._global_step,
                    metrics=epoch_metrics,
                    cfg=cfg,
                    optimizer_state_dict={
                        "critic": self._critic_optimizer.state_dict(),
                        "actor": self._actor_optimizer.state_dict(),
                    },
                )
                self._training_event_logger.log_event(
                    level="INFO",
                    component="checkpoint",
                    event="saved",
                    payload={
                        "epoch": epoch,
                        "global_step": self._global_step,
                        "checkpoint_path": str(last_checkpoint),
                    },
                )

            final_td_loss = epoch_metrics["td_loss_mean"]
            final_imitation_loss = epoch_metrics["imitation_loss_mean"]
            final_total_loss = epoch_metrics["total_loss_mean"]

        elapsed_seconds = time.time() - self._start_time
        logger.info(
            "BCQ training complete: %d epochs, %d steps, %.1fs elapsed.",
            cfg.n_epochs,
            self._global_step,
            elapsed_seconds,
        )

        report_artifacts: dict[str, Any] | None = None
        try:
            artifacts = generate_training_report_artifacts(
                cfg,
                algorithm=cfg.algorithm,
                state_dim=self._dataset.state_dim,
                n_actions=self._n_actions,
                total_steps=self._global_step,
                elapsed_seconds=elapsed_seconds,
                final_metrics={
                    "td_loss_mean": final_td_loss,
                    "imitation_loss_mean": final_imitation_loss,
                    "total_loss_mean": final_total_loss,
                },
                checkpoint_path=last_checkpoint,
                epoch_durations=epoch_durations,
                training_log_path=self._training_event_logger.log_path,
                runtime_log_path=self._runtime_event_logger.log_path,
            )
            report_artifacts = artifacts.to_dict()
        except Exception:
            logger.exception("Failed to generate BCQ reporting artifacts.")

        self._training_event_logger.log_event(
            level="INFO",
            component="trainer",
            event="run_complete",
            payload={
                "elapsed_seconds": elapsed_seconds,
                "total_steps": self._global_step,
                "checkpoint_path": str(last_checkpoint) if last_checkpoint else None,
                "report_artifact_dir": report_artifacts["artifact_dir"]
                if report_artifacts
                else None,
            },
        )

        return BCQTrainingResult(
            n_epochs=cfg.n_epochs,
            total_steps=self._global_step,
            final_td_loss=final_td_loss,
            final_imitation_loss=final_imitation_loss,
            final_total_loss=final_total_loss,
            checkpoint_path=last_checkpoint,
            elapsed_seconds=elapsed_seconds,
            state_dim=self._dataset.state_dim,
            n_actions=self._n_actions,
            device_backend=cfg.device_meta.backend,
            report_artifacts=report_artifacts,
        )

    def get_policy(self) -> BCQPolicy:
        return BCQPolicy(
            q_network=copy.deepcopy(self._q_network).eval(),
            behavior_policy=copy.deepcopy(self._behavior_policy).eval(),
            device=self._device,
            state_dim=self._dataset.state_dim,
            n_actions=self._n_actions,
            threshold=self._imitation_threshold,
            checkpoint_path=self._checkpoint_manager.latest_checkpoint(),
        )


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(
    config_path: Path,
    *,
    device: str | None = None,
) -> TrainingConfig:
    if device is None:
        return load_training_config(config_path)

    import yaml

    with config_path.open("r") as handle:
        raw_payload = yaml.safe_load(handle)
    assert isinstance(raw_payload, dict)
    merged_payload = _deep_merge(raw_payload, {"runtime": {"device": device}})
    return load_training_config(config_path, overrides=merged_payload)


def _dry_run(cfg: TrainingConfig, n_actions: int = 25) -> None:
    logger.info("=== BCQ DRY-RUN ===")
    logger.info(
        "Config: algorithm=%s device=%s epochs=%d batch=%d",
        cfg.algorithm,
        cfg.device,
        cfg.n_epochs,
        cfg.batch_size,
    )

    state_dim = int(cfg.extra.get("dry_run_state_dim", 33))
    hidden_sizes: list[int] = cfg.extra.get("hidden_sizes", _DEFAULT_HIDDEN_SIZES)
    threshold = float(
        cfg.extra.get("imitation_threshold", _DEFAULT_IMITATION_THRESHOLD)
    )
    behavior_cloning_weight = float(
        cfg.extra.get(
            "behavior_cloning_weight",
            _DEFAULT_BEHAVIOR_CLONING_WEIGHT,
        )
    )
    batch_size = min(cfg.batch_size, 16)

    if cfg.device.type == "mps":
        issues = validate_mps_ops(cfg.device)
        if issues:
            logger.warning(
                "%d MPS op issue(s) detected. "
                "Set PYTORCH_ENABLE_MPS_FALLBACK=1 if training fails.",
                len(issues),
            )

    q_network = QNetwork(state_dim, n_actions, hidden_sizes).to(cfg.device)
    target_q_network = copy.deepcopy(q_network).eval()
    behavior_policy = BehaviorPolicy(
        state_dim,
        n_actions,
        hidden_sizes=hidden_sizes,
    ).to(cfg.device)

    critic_optimizer = torch.optim.Adam(q_network.parameters(), lr=_DEFAULT_CRITIC_LR)
    actor_optimizer = torch.optim.Adam(
        behavior_policy.parameters(),
        lr=_DEFAULT_ACTOR_LR,
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    batch = TransitionBatch(
        states=torch.randn(batch_size, state_dim, generator=generator).to(cfg.device),
        actions=torch.randint(0, n_actions, (batch_size,)).to(cfg.device),
        rewards=torch.randn(batch_size, generator=generator).to(cfg.device),
        next_states=torch.randn(batch_size, state_dim, generator=generator).to(
            cfg.device
        ),
        dones=torch.zeros(batch_size, device=cfg.device),
    )
    batch.dones[: batch_size // 4] = 1.0

    q_values = q_network(batch.states)
    chosen_q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_q_network(batch.next_states)
        next_logits = behavior_policy(batch.next_states)
        next_actions = select_bcq_actions(
            next_q_values,
            next_logits,
            threshold=threshold,
        )
        next_state_values = next_q_values.gather(
            1,
            next_actions.unsqueeze(1),
        ).squeeze(1)
        targets = batch.rewards + cfg.gamma * (1.0 - batch.dones) * next_state_values

    td_loss = F.mse_loss(chosen_q_values, targets)
    behavior_logits = behavior_policy(batch.states)
    imitation_loss = F.cross_entropy(behavior_logits, batch.actions)
    total_loss = td_loss + behavior_cloning_weight * imitation_loss

    critic_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), _DEFAULT_GRAD_CLIP)
    nn.utils.clip_grad_norm_(behavior_policy.parameters(), _DEFAULT_GRAD_CLIP)
    critic_optimizer.step()
    actor_optimizer.step()

    policy = BCQPolicy(
        q_network=copy.deepcopy(q_network).eval(),
        behavior_policy=copy.deepcopy(behavior_policy).eval(),
        device=cfg.device,
        state_dim=state_dim,
        n_actions=n_actions,
        threshold=threshold,
    )
    action = policy.select_action([0.0] * state_dim)
    assert 0 <= action < n_actions, f"select_action returned invalid action: {action}"

    logger.info(
        "Dry-run forward/backward: td_loss=%.4f imitation_loss=%.4f total=%.4f",
        td_loss.item(),
        imitation_loss.item(),
        total_loss.item(),
    )
    logger.info(
        "Dry-run complete ✅  device=%s backend=%s",
        cfg.device,
        cfg.device_meta.backend,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.training.bcq",
        description="Train a discrete-action BCQ policy on the offline sepsis dataset.",
    )
    parser.add_argument(
        "--config",
        default="configs/training/bcq.yaml",
        help="Path to the BCQ training config YAML (default: configs/training/bcq.yaml).",
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

    config_path = Path(args.config)
    if config_path.exists():
        cfg = _load_config(config_path, device=args.device)
    else:
        logger.warning(
            "Config file not found at %s. Using defaults for dry-run.",
            config_path,
        )
        cfg = build_training_config(
            algorithm="bcq",
            device=args.device or "auto",
            dataset_path=Path("data/replay/replay_train.parquet"),
            n_epochs=100,
            batch_size=256,
            gamma=0.99,
        )

    if args.dry_run:
        _dry_run(cfg, n_actions=args.n_actions)
        sys.exit(0)

    dataset = load_replay_dataset(cfg)
    trainer = BCQTrainer(cfg, dataset, n_actions=args.n_actions)
    result = trainer.train()
    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "BCQ_VERSION",
    "BehaviorPolicy",
    "BCQPolicy",
    "BCQTrainer",
    "BCQTrainingResult",
    "select_bcq_actions",
]
