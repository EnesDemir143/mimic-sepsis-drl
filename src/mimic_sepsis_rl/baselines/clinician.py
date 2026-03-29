"""
Clinician-action replay baseline.

This baseline simply replays the observed clinician actions from the
transition dataset. It is the direct comparator: any offline RL policy
that cannot match the clinician's observed reward distribution brings
no actionable improvement.

The clinician baseline does **not** learn any policy — it reports
aggregate metrics over the actions that were actually taken in the data.

Usage
-----
    from mimic_sepsis_rl.baselines.clinician import evaluate_clinician_baseline

Version history
---------------
v1.0.0  2026-03-29  Initial clinician baseline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from mimic_sepsis_rl.datasets.transitions import TransitionRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClinicianBaselineResult:
    """Aggregate metrics for the clinician-action replay baseline.

    Attributes
    ----------
    n_episodes : int
        Number of episodes evaluated.
    n_transitions : int
        Total transitions in the dataset.
    mean_episode_return : float
        Mean undiscounted return per episode.
    std_episode_return : float
        Std dev of undiscounted return across episodes.
    mean_reward : float
        Mean per-step reward across all transitions.
    action_distribution : dict[int, int]
        Frequency count per action ID.
    mortality_rate : float
        Fraction of episodes ending with terminal reward < 0.
    """

    n_episodes: int
    n_transitions: int
    mean_episode_return: float
    std_episode_return: float
    mean_reward: float
    action_distribution: dict[int, int]
    mortality_rate: float

    def to_dict(self) -> dict:
        return {
            "baseline": "clinician",
            "n_episodes": self.n_episodes,
            "n_transitions": self.n_transitions,
            "mean_episode_return": self.mean_episode_return,
            "std_episode_return": self.std_episode_return,
            "mean_reward": self.mean_reward,
            "action_distribution": self.action_distribution,
            "mortality_rate": self.mortality_rate,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def evaluate_clinician_baseline(
    transitions: Sequence[TransitionRow],
) -> ClinicianBaselineResult:
    """Evaluate the clinician baseline on a set of transitions.

    This computes summary statistics over the *observed* clinician actions
    without rewriting any action decisions.

    Parameters
    ----------
    transitions:
        Flat transition list from the shared transition contract.

    Returns
    -------
    ClinicianBaselineResult
    """
    if not transitions:
        return ClinicianBaselineResult(
            n_episodes=0,
            n_transitions=0,
            mean_episode_return=0.0,
            std_episode_return=0.0,
            mean_reward=0.0,
            action_distribution={},
            mortality_rate=0.0,
        )

    # Group by episode
    episode_rewards: dict[int, float] = {}
    terminal_rewards: dict[int, float] = {}
    action_counts: dict[int, int] = {}

    for t in transitions:
        episode_rewards[t.stay_id] = episode_rewards.get(t.stay_id, 0.0) + t.reward
        action_counts[t.action] = action_counts.get(t.action, 0) + 1
        if t.done:
            terminal_rewards[t.stay_id] = t.reward

    returns = list(episode_rewards.values())
    n_episodes = len(returns)
    mean_return = sum(returns) / n_episodes
    var_return = sum((r - mean_return) ** 2 for r in returns) / n_episodes
    std_return = math.sqrt(var_return)

    all_rewards = [t.reward for t in transitions]
    mean_reward = sum(all_rewards) / len(all_rewards)

    # Mortality: fraction of episodes with negative terminal reward
    died = sum(1 for tr in terminal_rewards.values() if tr < 0)
    mortality_rate = died / n_episodes if n_episodes > 0 else 0.0

    result = ClinicianBaselineResult(
        n_episodes=n_episodes,
        n_transitions=len(transitions),
        mean_episode_return=mean_return,
        std_episode_return=std_return,
        mean_reward=mean_reward,
        action_distribution=dict(sorted(action_counts.items())),
        mortality_rate=mortality_rate,
    )

    logger.info(
        "Clinician baseline: %d episodes, mean_return=%.4f, mortality=%.2f%%",
        result.n_episodes,
        result.mean_episode_return,
        result.mortality_rate * 100,
    )

    return result
