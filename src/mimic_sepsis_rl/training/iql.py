"""
Discrete-action IQL trainer on the shared offline RL experiment surface.

IQL (Implicit Q-Learning) separates critic fitting, expectile value
regression, and advantage-weighted behavioral cloning. This implementation
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

logger = logging.getLogger(__name__)

IQL_VERSION: str = "1.0.0"

_DEFAULT_POLICY_HIDDEN_SIZES: list[int] = [256, 256]
_DEFAULT_VALUE_HIDDEN_SIZES: list[int] = [256, 256]
_DEFAULT_CRITIC_HIDDEN_SIZES: list[int] = [256, 256]
_DEFAULT_ACTOR_LR: float = 1e-4
_DEFAULT_CRITIC_LR: float = 3e-4
_DEFAULT_VALUE_LR: float = 3e-4
_DEFAULT_POLYAK_TAU: float = 0.005
_DEFAULT_TARGET_UPDATE_FREQ: int = 10
_DEFAULT_EXPECTILE: float = 0.7
_DEFAULT_TEMPERATURE: float = 3.0
_DEFAULT_MAX_ADV_WEIGHT: float = 100.0
_DEFAULT_GRAD_CLIP: float = 10.0


class PolicyNetwork(nn.Module):
    """Discrete actor that outputs action logits."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = _DEFAULT_POLICY_HIDDEN_SIZES

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


class ValueNetwork(nn.Module):
    """State-value function for discrete IQL."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = _DEFAULT_VALUE_HIDDEN_SIZES

        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def expectile_loss(residual: torch.Tensor, expectile: float) -> torch.Tensor:
    """Compute the asymmetric expectile regression objective."""
    weights = torch.where(
        residual >= 0.0,
        torch.full_like(residual, expectile),
        torch.full_like(residual, 1.0 - expectile),
    )
    return (weights * residual.pow(2)).mean()


@dataclass
class IQLTrainingResult:
    """Summary of a completed IQL training run."""

    n_epochs: int
    total_steps: int
    final_critic_loss: float
    final_value_loss: float
    final_actor_loss: float
    final_total_loss: float
    checkpoint_path: Path | None
    elapsed_seconds: float
    state_dim: int
    n_actions: int
    device_backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "iql",
            "iql_version": IQL_VERSION,
            "n_epochs": self.n_epochs,
            "total_steps": self.total_steps,
            "final_critic_loss": self.final_critic_loss,
            "final_value_loss": self.final_value_loss,
            "final_actor_loss": self.final_actor_loss,
            "final_total_loss": self.final_total_loss,
            "checkpoint_path": str(self.checkpoint_path)
            if self.checkpoint_path
            else None,
            "elapsed_seconds": self.elapsed_seconds,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "device_backend": self.device_backend,
        }


@dataclass
class IQLPolicy:
    """Greedy inference wrapper around the discrete IQL actor."""

    policy_network: PolicyNetwork
    device: torch.device
    state_dim: int
    n_actions: int
    checkpoint_path: Path | None = None

    def select_action(self, state: list[float] | torch.Tensor) -> int:
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        self.policy_network.eval()
        with torch.no_grad():
            logits = self.policy_network(state_tensor)
        return int(logits.argmax(dim=1).item())


class IQLTrainer:
    """Discrete-action IQL trainer using shared replay and artifact contracts."""

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
        policy_hidden_sizes: list[int] = extra.get(
            "policy_hidden_sizes",
            _DEFAULT_POLICY_HIDDEN_SIZES,
        )
        value_hidden_sizes: list[int] = extra.get(
            "value_hidden_sizes",
            _DEFAULT_VALUE_HIDDEN_SIZES,
        )
        critic_hidden_sizes: list[int] = extra.get(
            "critic_hidden_sizes",
            _DEFAULT_CRITIC_HIDDEN_SIZES,
        )
        self._actor_lr = float(extra.get("actor_lr", _DEFAULT_ACTOR_LR))
        self._critic_lr = float(extra.get("critic_lr", _DEFAULT_CRITIC_LR))
        self._value_lr = float(extra.get("value_lr", _DEFAULT_VALUE_LR))
        self._polyak_tau = float(extra.get("polyak_tau", _DEFAULT_POLYAK_TAU))
        self._target_update_freq = int(
            extra.get("target_update_freq", _DEFAULT_TARGET_UPDATE_FREQ)
        )
        self._expectile = float(extra.get("expectile", _DEFAULT_EXPECTILE))
        self._temperature = float(extra.get("temperature", _DEFAULT_TEMPERATURE))
        self._max_adv_weight = float(
            extra.get("max_adv_weight", _DEFAULT_MAX_ADV_WEIGHT)
        )
        self._grad_clip = float(extra.get("grad_clip", _DEFAULT_GRAD_CLIP))

        state_dim = dataset.state_dim
        self._q1 = QNetwork(state_dim, n_actions, critic_hidden_sizes).to(self._device)
        self._q2 = QNetwork(state_dim, n_actions, critic_hidden_sizes).to(self._device)
        self._target_q1 = copy.deepcopy(self._q1).to(self._device).eval()
        self._target_q2 = copy.deepcopy(self._q2).to(self._device).eval()
        self._value_network = ValueNetwork(
            state_dim,
            hidden_sizes=value_hidden_sizes,
        ).to(self._device)
        self._policy_network = PolicyNetwork(
            state_dim,
            n_actions,
            hidden_sizes=policy_hidden_sizes,
        ).to(self._device)

        self._critic_optimizer = torch.optim.Adam(
            list(self._q1.parameters()) + list(self._q2.parameters()),
            lr=self._critic_lr,
        )
        self._value_optimizer = torch.optim.Adam(
            self._value_network.parameters(),
            lr=self._value_lr,
        )
        self._actor_optimizer = torch.optim.Adam(
            self._policy_network.parameters(),
            lr=self._actor_lr,
        )

        self._checkpoint_manager = build_checkpoint_manager(cfg)
        self._metric_logger = MetricLogger.from_config(cfg)

        self._global_step = 0
        self._start_time = 0.0

        logger.info(
            "IQLTrainer initialised: state_dim=%d n_actions=%d expectile=%.2f "
            "temperature=%.2f device=%s",
            state_dim,
            n_actions,
            self._expectile,
            self._temperature,
            self._device,
        )

    def _update_target_networks(self) -> None:
        if self._target_update_freq <= 0:
            return
        if self._global_step % self._target_update_freq != 0:
            return

        tau = min(max(self._polyak_tau, 0.0), 1.0)
        for online_param, target_param in zip(
            self._q1.parameters(),
            self._target_q1.parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)
        for online_param, target_param in zip(
            self._q2.parameters(),
            self._target_q2.parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)

    def _training_step(self, batch: TransitionBatch) -> dict[str, float]:
        self._q1.train()
        self._q2.train()
        self._value_network.train()
        self._policy_network.train()

        with torch.no_grad():
            next_values = self._value_network(batch.next_states).squeeze(1)
            critic_targets = (
                batch.rewards + self._cfg.gamma * (1.0 - batch.dones) * next_values
            )

        q1_values = self._q1(batch.states)
        q2_values = self._q2(batch.states)
        q1_taken = q1_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        q2_taken = q2_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1_taken, critic_targets) + F.mse_loss(
            q2_taken,
            critic_targets,
        )

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        if self._grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self._q1.parameters()) + list(self._q2.parameters()),
                self._grad_clip,
            )
        self._critic_optimizer.step()

        with torch.no_grad():
            target_q1_values = self._target_q1(batch.states)
            target_q2_values = self._target_q2(batch.states)
            target_q1_taken = target_q1_values.gather(
                1,
                batch.actions.unsqueeze(1),
            ).squeeze(1)
            target_q2_taken = target_q2_values.gather(
                1,
                batch.actions.unsqueeze(1),
            ).squeeze(1)
            target_q = torch.minimum(target_q1_taken, target_q2_taken)

        values = self._value_network(batch.states).squeeze(1)
        advantages = target_q - values
        value_loss = expectile_loss(advantages, self._expectile)

        self._value_optimizer.zero_grad()
        value_loss.backward()
        if self._grad_clip > 0:
            nn.utils.clip_grad_norm_(self._value_network.parameters(), self._grad_clip)
        self._value_optimizer.step()

        with torch.no_grad():
            detached_values = self._value_network(batch.states).squeeze(1)
            actor_advantages = target_q - detached_values
            actor_weights = torch.exp(self._temperature * actor_advantages).clamp(
                max=self._max_adv_weight
            )

        logits = self._policy_network(batch.states)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        actor_loss = -(actor_weights * action_log_probs).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        if self._grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                self._grad_clip,
            )
        self._actor_optimizer.step()

        self._global_step += 1
        self._update_target_networks()

        return {
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "total_loss": critic_loss.item()
            + value_loss.item()
            + actor_loss.item(),
        }

    def train(self) -> IQLTrainingResult:
        cfg = self._cfg
        set_global_seed(cfg.runtime.seed)
        self._start_time = time.time()

        final_critic_loss = float("nan")
        final_value_loss = float("nan")
        final_actor_loss = float("nan")
        final_total_loss = float("nan")
        last_checkpoint: Path | None = None

        critic_losses: list[float] = []
        value_losses: list[float] = []
        actor_losses: list[float] = []
        total_losses: list[float] = []

        logger.info(
            "Starting IQL training: epochs=%d batch_size=%d gamma=%.3f expectile=%.2f",
            cfg.n_epochs,
            cfg.batch_size,
            cfg.gamma,
            self._expectile,
        )

        for epoch in range(1, cfg.n_epochs + 1):
            critic_losses.clear()
            value_losses.clear()
            actor_losses.clear()
            total_losses.clear()

            for batch in self._dataset.iter_batches(
                cfg.batch_size,
                shuffle=True,
                epoch=epoch,
            ):
                step_metrics = self._training_step(batch)
                critic_losses.append(step_metrics["critic_loss"])
                value_losses.append(step_metrics["value_loss"])
                actor_losses.append(step_metrics["actor_loss"])
                total_losses.append(step_metrics["total_loss"])

                for name, value in step_metrics.items():
                    self._metric_logger.log_scalar(
                        name,
                        value,
                        step=self._global_step,
                        epoch=epoch,
                    )

            epoch_metrics = {
                "critic_loss_mean": sum(critic_losses) / max(len(critic_losses), 1),
                "value_loss_mean": sum(value_losses) / max(len(value_losses), 1),
                "actor_loss_mean": sum(actor_losses) / max(len(actor_losses), 1),
                "total_loss_mean": sum(total_losses) / max(len(total_losses), 1),
            }
            self._metric_logger.log_epoch_summary(
                epoch,
                self._global_step,
                epoch_metrics,
            )

            if epoch % 10 == 0 or epoch == cfg.n_epochs:
                logger.info(
                    "Epoch %d/%d | critic=%.4f value=%.4f actor=%.4f total=%.4f | "
                    "steps=%d",
                    epoch,
                    cfg.n_epochs,
                    epoch_metrics["critic_loss_mean"],
                    epoch_metrics["value_loss_mean"],
                    epoch_metrics["actor_loss_mean"],
                    epoch_metrics["total_loss_mean"],
                    self._global_step,
                )

            if should_checkpoint(epoch, cfg, is_last=(epoch == cfg.n_epochs)):
                last_checkpoint = self._checkpoint_manager.save(
                    {
                        "q1": self._q1.state_dict(),
                        "q2": self._q2.state_dict(),
                        "target_q1": self._target_q1.state_dict(),
                        "target_q2": self._target_q2.state_dict(),
                        "value_network": self._value_network.state_dict(),
                        "policy_network": self._policy_network.state_dict(),
                    },
                    epoch=epoch,
                    global_step=self._global_step,
                    metrics=epoch_metrics,
                    cfg=cfg,
                    optimizer_state_dict={
                        "critic": self._critic_optimizer.state_dict(),
                        "value": self._value_optimizer.state_dict(),
                        "actor": self._actor_optimizer.state_dict(),
                    },
                )

            final_critic_loss = epoch_metrics["critic_loss_mean"]
            final_value_loss = epoch_metrics["value_loss_mean"]
            final_actor_loss = epoch_metrics["actor_loss_mean"]
            final_total_loss = epoch_metrics["total_loss_mean"]

        elapsed_seconds = time.time() - self._start_time
        logger.info(
            "IQL training complete: %d epochs, %d steps, %.1fs elapsed.",
            cfg.n_epochs,
            self._global_step,
            elapsed_seconds,
        )

        return IQLTrainingResult(
            n_epochs=cfg.n_epochs,
            total_steps=self._global_step,
            final_critic_loss=final_critic_loss,
            final_value_loss=final_value_loss,
            final_actor_loss=final_actor_loss,
            final_total_loss=final_total_loss,
            checkpoint_path=last_checkpoint,
            elapsed_seconds=elapsed_seconds,
            state_dim=self._dataset.state_dim,
            n_actions=self._n_actions,
            device_backend=cfg.device_meta.backend,
        )

    def get_policy(self) -> IQLPolicy:
        return IQLPolicy(
            policy_network=copy.deepcopy(self._policy_network).eval(),
            device=self._device,
            state_dim=self._dataset.state_dim,
            n_actions=self._n_actions,
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
    logger.info("=== IQL DRY-RUN ===")
    logger.info(
        "Config: algorithm=%s device=%s epochs=%d batch=%d",
        cfg.algorithm,
        cfg.device,
        cfg.n_epochs,
        cfg.batch_size,
    )

    state_dim = int(cfg.extra.get("dry_run_state_dim", 33))
    policy_hidden_sizes: list[int] = cfg.extra.get(
        "policy_hidden_sizes",
        _DEFAULT_POLICY_HIDDEN_SIZES,
    )
    value_hidden_sizes: list[int] = cfg.extra.get(
        "value_hidden_sizes",
        _DEFAULT_VALUE_HIDDEN_SIZES,
    )
    critic_hidden_sizes: list[int] = cfg.extra.get(
        "critic_hidden_sizes",
        _DEFAULT_CRITIC_HIDDEN_SIZES,
    )
    expectile = float(cfg.extra.get("expectile", _DEFAULT_EXPECTILE))
    temperature = float(cfg.extra.get("temperature", _DEFAULT_TEMPERATURE))
    batch_size = min(cfg.batch_size, 16)

    if cfg.device.type == "mps":
        issues = validate_mps_ops(cfg.device)
        if issues:
            logger.warning(
                "%d MPS op issue(s) detected. "
                "Set PYTORCH_ENABLE_MPS_FALLBACK=1 if training fails.",
                len(issues),
            )

    q1 = QNetwork(state_dim, n_actions, critic_hidden_sizes).to(cfg.device)
    q2 = QNetwork(state_dim, n_actions, critic_hidden_sizes).to(cfg.device)
    value_network = ValueNetwork(
        state_dim,
        hidden_sizes=value_hidden_sizes,
    ).to(cfg.device)
    policy_network = PolicyNetwork(
        state_dim,
        n_actions,
        hidden_sizes=policy_hidden_sizes,
    ).to(cfg.device)

    critic_optimizer = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=_DEFAULT_CRITIC_LR,
    )
    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=_DEFAULT_VALUE_LR,
    )
    actor_optimizer = torch.optim.Adam(
        policy_network.parameters(),
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

    with torch.no_grad():
        next_values = value_network(batch.next_states).squeeze(1)
        critic_targets = batch.rewards + cfg.gamma * (1.0 - batch.dones) * next_values

    q1_values = q1(batch.states)
    q2_values = q2(batch.states)
    q1_taken = q1_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    q2_taken = q2_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    critic_loss = F.mse_loss(q1_taken, critic_targets) + F.mse_loss(
        q2_taken,
        critic_targets,
    )

    critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(
        list(q1.parameters()) + list(q2.parameters()),
        _DEFAULT_GRAD_CLIP,
    )
    critic_optimizer.step()

    with torch.no_grad():
        target_q = torch.minimum(q1_taken, q2_taken)

    values = value_network(batch.states).squeeze(1)
    value_loss = expectile_loss(target_q - values, expectile)

    value_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(value_network.parameters(), _DEFAULT_GRAD_CLIP)
    value_optimizer.step()

    with torch.no_grad():
        advantages = target_q - value_network(batch.states).squeeze(1)
        weights = torch.exp(temperature * advantages).clamp(max=_DEFAULT_MAX_ADV_WEIGHT)

    logits = policy_network(batch.states)
    log_probs = F.log_softmax(logits, dim=1)
    action_log_probs = log_probs.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    actor_loss = -(weights * action_log_probs).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(policy_network.parameters(), _DEFAULT_GRAD_CLIP)
    actor_optimizer.step()

    policy = IQLPolicy(
        policy_network=copy.deepcopy(policy_network).eval(),
        device=cfg.device,
        state_dim=state_dim,
        n_actions=n_actions,
    )
    action = policy.select_action([0.0] * state_dim)
    assert 0 <= action < n_actions, f"select_action returned invalid action: {action}"

    logger.info(
        "Dry-run forward/backward: critic_loss=%.4f value_loss=%.4f actor_loss=%.4f",
        critic_loss.item(),
        value_loss.item(),
        actor_loss.item(),
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
        prog="python -m mimic_sepsis_rl.training.iql",
        description="Train a discrete-action IQL policy on the offline sepsis dataset.",
    )
    parser.add_argument(
        "--config",
        default="configs/training/iql.yaml",
        help="Path to the IQL training config YAML (default: configs/training/iql.yaml).",
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
            algorithm="iql",
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
    trainer = IQLTrainer(cfg, dataset, n_actions=args.n_actions)
    result = trainer.train()
    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "IQL_VERSION",
    "PolicyNetwork",
    "ValueNetwork",
    "IQLPolicy",
    "IQLTrainer",
    "IQLTrainingResult",
    "expectile_loss",
]
