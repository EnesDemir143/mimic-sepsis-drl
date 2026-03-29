"""
Behavior cloning baseline for the MIMIC Sepsis Offline RL pipeline.

A simple supervised-learning baseline that trains a softmax classifier
to predict the clinician's action from state features. This provides a
meaningful upper bound for imitation: any RL policy that does poorly
on the training distribution should at least match behavior cloning.

The implementation is intentionally lightweight (logistic regression
via gradient descent) to avoid introducing deep learning dependencies
before Phase 7.

Usage (CLI dry run)
-------------------
    python -m mimic_sepsis_rl.baselines.behavior_cloning --dry-run

Version history
---------------
v1.0.0  2026-03-29  Initial behavior cloning baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from mimic_sepsis_rl.datasets.transitions import TransitionRow

logger = logging.getLogger(__name__)

N_ACTIONS: int = 25


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviorCloningResult:
    """Metrics for the behavior cloning baseline.

    Attributes
    ----------
    n_train : int
        Number of training transitions.
    n_eval : int
        Number of evaluation transitions.
    train_accuracy : float
        Top-1 accuracy on training data.
    eval_accuracy : float
        Top-1 accuracy on evaluation data (if separate eval set provided).
    mean_episode_return : float
        Mean return using BC-predicted actions with observed rewards.
    std_episode_return : float
        Std dev of episode returns.
    mortality_rate : float
        Observed mortality rate in the evaluation set.
    n_epochs : int
        Number of training epochs.
    """

    n_train: int
    n_eval: int
    train_accuracy: float
    eval_accuracy: float
    mean_episode_return: float
    std_episode_return: float
    mortality_rate: float
    n_epochs: int

    def to_dict(self) -> dict:
        return {
            "baseline": "behavior_cloning",
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "train_accuracy": self.train_accuracy,
            "eval_accuracy": self.eval_accuracy,
            "mean_episode_return": self.mean_episode_return,
            "std_episode_return": self.std_episode_return,
            "mortality_rate": self.mortality_rate,
            "n_epochs": self.n_epochs,
        }


# ---------------------------------------------------------------------------
# Lightweight softmax classifier
# ---------------------------------------------------------------------------


class SoftmaxClassifier:
    """Minimal multinomial logistic regression for behavior cloning.

    Uses basic gradient descent — no external ML libraries required.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = N_ACTIONS,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr

        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(state_dim)
        self.weights = [
            [rng.gauss(0, scale) for _ in range(state_dim)]
            for _ in range(n_actions)
        ]
        self.biases = [0.0] * n_actions

    def predict_probs(self, state: Sequence[float]) -> list[float]:
        """Compute softmax probabilities over actions."""
        logits = [
            sum(w * s for w, s in zip(self.weights[a], state)) + self.biases[a]
            for a in range(self.n_actions)
        ]
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        return [e / total for e in exp_logits]

    def predict(self, state: Sequence[float]) -> int:
        """Predict the most likely action."""
        probs = self.predict_probs(state)
        return max(range(self.n_actions), key=lambda a: probs[a])

    def train_step(
        self,
        state: Sequence[float],
        action: int,
    ) -> float:
        """One gradient step. Returns cross-entropy loss."""
        probs = self.predict_probs(state)

        # Cross-entropy loss
        loss = -math.log(max(probs[action], 1e-12))

        # Gradient: prob - 1(correct action)
        for a in range(self.n_actions):
            grad = probs[a] - (1.0 if a == action else 0.0)
            for j in range(self.state_dim):
                self.weights[a][j] -= self.lr * grad * state[j]
            self.biases[a] -= self.lr * grad

        return loss

    def accuracy(
        self,
        states: list[tuple[float, ...]],
        actions: list[int],
    ) -> float:
        """Compute top-1 accuracy."""
        if not states:
            return 0.0
        correct = sum(
            1 for s, a in zip(states, actions)
            if self.predict(s) == a
        )
        return correct / len(states)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train_behavior_cloning(
    train_transitions: Sequence[TransitionRow],
    eval_transitions: Sequence[TransitionRow] | None = None,
    *,
    n_epochs: int = 10,
    lr: float = 0.01,
    seed: int = 42,
) -> tuple[BehaviorCloningResult, SoftmaxClassifier]:
    """Train and evaluate a behavior cloning baseline.

    Parameters
    ----------
    train_transitions:
        Training-split transitions.
    eval_transitions:
        Optional evaluation-split transitions. If None, evaluation
        metrics are computed on train data.
    n_epochs:
        Number of training epochs.
    lr:
        Learning rate for gradient descent.
    seed:
        Random seed for weight init and data shuffling.

    Returns
    -------
    (result, model)
    """
    if not train_transitions:
        return BehaviorCloningResult(
            n_train=0,
            n_eval=0,
            train_accuracy=0.0,
            eval_accuracy=0.0,
            mean_episode_return=0.0,
            std_episode_return=0.0,
            mortality_rate=0.0,
            n_epochs=n_epochs,
        ), SoftmaxClassifier(state_dim=1)

    state_dim = len(train_transitions[0].state)
    model = SoftmaxClassifier(
        state_dim=state_dim,
        n_actions=N_ACTIONS,
        lr=lr,
        seed=seed,
    )

    # Prepare data
    train_states = [t.state for t in train_transitions]
    train_actions = [t.action for t in train_transitions]

    # Training
    rng = random.Random(seed)
    indices = list(range(len(train_transitions)))

    for epoch in range(n_epochs):
        rng.shuffle(indices)
        epoch_loss = 0.0
        for idx in indices:
            loss = model.train_step(train_states[idx], train_actions[idx])
            epoch_loss += loss
        avg_loss = epoch_loss / len(indices)
        if (epoch + 1) % max(1, n_epochs // 5) == 0 or epoch == 0:
            acc = model.accuracy(train_states, train_actions)
            logger.info(
                "  epoch %d/%d: loss=%.4f, train_acc=%.4f",
                epoch + 1,
                n_epochs,
                avg_loss,
                acc,
            )

    # Evaluate
    train_acc = model.accuracy(train_states, train_actions)

    eval_trans = eval_transitions if eval_transitions is not None else train_transitions
    eval_states = [t.state for t in eval_trans]
    eval_actions = [t.action for t in eval_trans]
    eval_acc = model.accuracy(eval_states, eval_actions)

    # Episode returns (using observed rewards, not counterfactual)
    episode_rewards: dict[int, float] = {}
    terminal_rewards: dict[int, float] = {}
    for t in eval_trans:
        episode_rewards[t.stay_id] = episode_rewards.get(t.stay_id, 0.0) + t.reward
        if t.done:
            terminal_rewards[t.stay_id] = t.reward

    returns = list(episode_rewards.values())
    n_ep = len(returns)
    mean_return = sum(returns) / n_ep if n_ep else 0.0
    var_return = sum((r - mean_return) ** 2 for r in returns) / n_ep if n_ep else 0.0
    std_return = math.sqrt(var_return)

    died = sum(1 for tr in terminal_rewards.values() if tr < 0)
    mortality_rate = died / n_ep if n_ep > 0 else 0.0

    result = BehaviorCloningResult(
        n_train=len(train_transitions),
        n_eval=len(eval_trans),
        train_accuracy=train_acc,
        eval_accuracy=eval_acc,
        mean_episode_return=mean_return,
        std_episode_return=std_return,
        mortality_rate=mortality_rate,
        n_epochs=n_epochs,
    )

    logger.info(
        "BC baseline: train_acc=%.4f, eval_acc=%.4f, mean_return=%.4f",
        result.train_accuracy,
        result.eval_accuracy,
        result.mean_episode_return,
    )

    return result, model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.baselines.behavior_cloning",
        description="Validate behavior cloning baseline with a dry run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic data to verify the pipeline works.",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of training epochs. Default: 10.",
    )
    return p


def _dry_run(n_epochs: int) -> None:
    """Smoke test with synthetic transitions."""
    from mimic_sepsis_rl.datasets.transitions import TransitionRow

    logger.info("Running behavior cloning dry-run...")

    rng = random.Random(42)
    state_dim = 8
    n_episodes = 10
    steps_per_ep = 6

    transitions: list[TransitionRow] = []
    for ep in range(n_episodes):
        mortality = rng.choice([0, 1])
        for step in range(steps_per_ep):
            state = tuple(rng.gauss(0, 1) for _ in range(state_dim))
            is_last = step == steps_per_ep - 1
            reward = (15.0 if mortality == 0 else -15.0) if is_last else rng.uniform(-0.5, 0.5)
            transitions.append(
                TransitionRow(
                    stay_id=ep * 100,
                    step_index=step,
                    state=state,
                    action=rng.randint(0, 24),
                    reward=reward,
                    next_state=state,  # simplified
                    done=is_last,
                )
            )

    result, model = train_behavior_cloning(
        transitions,
        n_epochs=n_epochs,
        seed=42,
    )

    logger.info("Result: %s", result.to_dict())
    logger.info("✅ Behavior cloning dry-run PASSED.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        _dry_run(args.n_epochs)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
