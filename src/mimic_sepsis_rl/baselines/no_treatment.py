"""
No-treatment baseline.

A static policy that always selects action 0 (no vasopressor, no IV fluid).
This is the lower-bound comparator: any learned policy that performs worse
than doing nothing is clearly pathological.

The evaluator replaces each observed action with action 0, recomputes
episode returns using only the terminal reward component (since
intermediate shaping is action-independent in the current contract),
and reports aggregate metrics.

Usage
-----
    from mimic_sepsis_rl.baselines.no_treatment import evaluate_no_treatment_baseline

Version history
---------------
v1.0.0  2026-03-29  Initial no-treatment baseline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from mimic_sepsis_rl.datasets.transitions import TransitionRow

logger = logging.getLogger(__name__)

NO_TREATMENT_ACTION: int = 0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoTreatmentBaselineResult:
    """Aggregate metrics for the no-treatment baseline.

    Attributes
    ----------
    n_episodes : int
        Number of episodes evaluated.
    n_transitions : int
        Total transitions.
    mean_episode_return : float
        Mean undiscounted return per episode.
    std_episode_return : float
        Std dev of undiscounted return across episodes.
    mean_reward : float
        Mean per-step reward.
    policy_action : int
        The constant action selected (always 0).
    mortality_rate : float
        Fraction of episodes ending with negative terminal reward.
    """

    n_episodes: int
    n_transitions: int
    mean_episode_return: float
    std_episode_return: float
    mean_reward: float
    policy_action: int
    mortality_rate: float

    def to_dict(self) -> dict:
        return {
            "baseline": "no_treatment",
            "n_episodes": self.n_episodes,
            "n_transitions": self.n_transitions,
            "mean_episode_return": self.mean_episode_return,
            "std_episode_return": self.std_episode_return,
            "mean_reward": self.mean_reward,
            "policy_action": self.policy_action,
            "mortality_rate": self.mortality_rate,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def evaluate_no_treatment_baseline(
    transitions: Sequence[TransitionRow],
) -> NoTreatmentBaselineResult:
    """Evaluate the no-treatment baseline.

    Since the terminal reward depends on outcome (not action taken) and
    intermediate shaping is already computed, the no-treatment baseline
    uses the same rewards as observed. In a real counterfactual setting
    we cannot know the true outcome under no treatment — this baseline
    reports the observed rewards as a *lower bound* reference.

    Parameters
    ----------
    transitions:
        Flat transition list from the shared transition contract.

    Returns
    -------
    NoTreatmentBaselineResult
    """
    if not transitions:
        return NoTreatmentBaselineResult(
            n_episodes=0,
            n_transitions=0,
            mean_episode_return=0.0,
            std_episode_return=0.0,
            mean_reward=0.0,
            policy_action=NO_TREATMENT_ACTION,
            mortality_rate=0.0,
        )

    # Group rewards by episode (using observed rewards as-is)
    episode_rewards: dict[int, float] = {}
    terminal_rewards: dict[int, float] = {}

    for t in transitions:
        episode_rewards[t.stay_id] = episode_rewards.get(t.stay_id, 0.0) + t.reward
        if t.done:
            terminal_rewards[t.stay_id] = t.reward

    returns = list(episode_rewards.values())
    n_episodes = len(returns)
    mean_return = sum(returns) / n_episodes
    var_return = sum((r - mean_return) ** 2 for r in returns) / n_episodes
    std_return = math.sqrt(var_return)

    all_rewards = [t.reward for t in transitions]
    mean_reward = sum(all_rewards) / len(all_rewards)

    died = sum(1 for tr in terminal_rewards.values() if tr < 0)
    mortality_rate = died / n_episodes if n_episodes > 0 else 0.0

    result = NoTreatmentBaselineResult(
        n_episodes=n_episodes,
        n_transitions=len(transitions),
        mean_episode_return=mean_return,
        std_episode_return=std_return,
        mean_reward=mean_reward,
        policy_action=NO_TREATMENT_ACTION,
        mortality_rate=mortality_rate,
    )

    logger.info(
        "No-treatment baseline: %d episodes, mean_return=%.4f, mortality=%.2f%%",
        result.n_episodes,
        result.mean_episode_return,
        result.mortality_rate * 100,
    )

    return result
