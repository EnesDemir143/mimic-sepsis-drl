"""
Typed reward models for the MIMIC Sepsis Offline RL pipeline.

Frozen data classes that define the reward contract:

- ``RewardConfig``   – configurable shaping parameters and reward version
- ``StepReward``     – computed reward for a single transition
- ``RewardSummary``  – diagnostic statistics for an episode or split

Version history
---------------
v1.0.0  2026-03-29  Initial reward contract types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

REWARD_SPEC_VERSION: Final[str] = "1.0.0"


# ---------------------------------------------------------------------------
# Reward variant
# ---------------------------------------------------------------------------


class RewardVariant(str, Enum):
    """Available reward formulations for ablation comparison."""

    SPARSE = "sparse"
    """Terminal 90-day mortality only (+15 survived / −15 died)."""

    SOFA_SHAPED = "sofa_shaped"
    """Sparse + conservative SOFA-delta shaping at intermediate steps."""

    FULL_SHAPED = "full_shaped"
    """Sparse + SOFA-delta + lactate clearance + MAP maintenance."""


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Configurable parameters for the reward function.

    Attributes
    ----------
    variant : RewardVariant
        Which reward formulation to apply.
    terminal_reward_survived : float
        Reward for surviving 90 days post-onset.
    terminal_reward_died : float
        Reward for 90-day mortality.
    sofa_delta_weight : float
        Shaping coefficient for SOFA delta (negative delta = improvement).
    lactate_clearance_weight : float
        Shaping coefficient for lactate reduction.
    map_stability_weight : float
        Shaping coefficient for MAP ≥ 65 mmHg maintenance.
    map_threshold : float
        MAP threshold below which the penalty is applied (mmHg).
    version : str
        Reward specification version for audit trail.
    """

    variant: RewardVariant = RewardVariant.SOFA_SHAPED
    terminal_reward_survived: float = 15.0
    terminal_reward_died: float = -15.0
    sofa_delta_weight: float = -0.025
    lactate_clearance_weight: float = 0.0
    map_stability_weight: float = 0.0
    map_threshold: float = 65.0
    version: str = REWARD_SPEC_VERSION


# ---------------------------------------------------------------------------
# Per-step reward output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepReward:
    """Computed reward for a single MDP transition.

    Attributes
    ----------
    stay_id : int
        ICU stay this transition belongs to.
    step_index : int
        0-based step index within the episode.
    terminal : float
        Terminal component (non-zero only at the last step).
    sofa_shaping : float
        SOFA-delta shaping component.
    lactate_shaping : float
        Lactate-clearance shaping component.
    map_shaping : float
        MAP-stability shaping component.
    total : float
        Sum of all reward components.
    is_terminal : bool
        Whether this is the last step of the episode.
    """

    stay_id: int
    step_index: int
    terminal: float
    sofa_shaping: float
    lactate_shaping: float
    map_shaping: float
    total: float
    is_terminal: bool


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardSummary:
    """Aggregate diagnostic statistics for a set of computed rewards.

    Attributes
    ----------
    variant : RewardVariant
        Reward formulation used.
    n_episodes : int
        Number of episodes.
    n_transitions : int
        Total number of transitions.
    n_survived : int
        Terminal episodes with survival outcome.
    n_died : int
        Terminal episodes with mortality outcome.
    mean_total : float
        Mean total reward across all transitions.
    std_total : float
        Std dev of total reward.
    mean_sofa_shaping : float
        Mean SOFA shaping component.
    mean_terminal : float
        Mean terminal reward.
    """

    variant: RewardVariant
    n_episodes: int
    n_transitions: int
    n_survived: int
    n_died: int
    mean_total: float
    std_total: float
    mean_sofa_shaping: float
    mean_terminal: float
