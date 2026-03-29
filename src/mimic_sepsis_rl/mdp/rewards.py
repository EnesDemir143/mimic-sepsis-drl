"""
Reward computation for the MIMIC Sepsis Offline RL pipeline.

Implements terminal 90-day mortality rewards and configurable intermediate
shaping terms.  Three reward variants are supported for ablation:

1. **Sparse** — terminal reward only (+15 survived / −15 died).
2. **SOFA-shaped** — sparse + conservative SOFA-delta shaping.
3. **Full-shaped** — SOFA-delta + lactate clearance + MAP stability.

The reward contract is deterministic: the same episode data and
``RewardConfig`` always produce the same reward sequence.

Usage (CLI validation)
----------------------
    python -m mimic_sepsis_rl.mdp.rewards --dry-run

Version history
---------------
v1.0.0  2026-03-29  Initial reward contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Final

import polars as pl

from mimic_sepsis_rl.mdp.reward_models import (
    REWARD_SPEC_VERSION,
    RewardConfig,
    RewardSummary,
    RewardVariant,
    StepReward,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_REWARD_CONFIG: Final[RewardConfig] = RewardConfig()


# ---------------------------------------------------------------------------
# Core reward computation
# ---------------------------------------------------------------------------


def compute_step_reward(
    *,
    is_terminal: bool,
    survived_90d: bool | None,
    sofa_current: float | None,
    sofa_previous: float | None,
    lactate_current: float | None,
    lactate_previous: float | None,
    map_current: float | None,
    config: RewardConfig,
) -> StepReward:
    """Compute reward for a single transition.

    Parameters
    ----------
    is_terminal:
        Whether this is the last step of the episode.
    survived_90d:
        Whether the patient survived 90 days. Required when
        ``is_terminal`` is True.
    sofa_current:
        SOFA score at the current step.
    sofa_previous:
        SOFA score at the previous step.
    lactate_current:
        Serum lactate at the current step (mmol/L).
    lactate_previous:
        Serum lactate at the previous step (mmol/L).
    map_current:
        Mean arterial pressure at the current step (mmHg).
    config:
        Reward configuration with shaping weights.

    Returns
    -------
    StepReward
        Struct with individual components and total reward.
    """
    # Terminal component
    terminal = 0.0
    if is_terminal and survived_90d is not None:
        terminal = (
            config.terminal_reward_survived
            if survived_90d
            else config.terminal_reward_died
        )

    # SOFA-delta shaping (negative delta = improvement → positive reward)
    sofa_shaping = 0.0
    if config.variant in (RewardVariant.SOFA_SHAPED, RewardVariant.FULL_SHAPED):
        if sofa_current is not None and sofa_previous is not None:
            sofa_delta = sofa_current - sofa_previous
            sofa_shaping = config.sofa_delta_weight * sofa_delta

    # Lactate clearance shaping
    lactate_shaping = 0.0
    if config.variant == RewardVariant.FULL_SHAPED:
        if (
            lactate_current is not None
            and lactate_previous is not None
            and lactate_previous > 0
        ):
            clearance = (lactate_previous - lactate_current) / lactate_previous
            lactate_shaping = config.lactate_clearance_weight * clearance

    # MAP stability shaping
    map_shaping = 0.0
    if config.variant == RewardVariant.FULL_SHAPED:
        if map_current is not None:
            if map_current < config.map_threshold:
                deficit = config.map_threshold - map_current
                map_shaping = -config.map_stability_weight * deficit

    total = terminal + sofa_shaping + lactate_shaping + map_shaping

    return StepReward(
        stay_id=0,  # Placeholder — set by caller
        step_index=0,  # Placeholder — set by caller
        terminal=terminal,
        sofa_shaping=sofa_shaping,
        lactate_shaping=lactate_shaping,
        map_shaping=map_shaping,
        total=total,
        is_terminal=is_terminal,
    )


# ---------------------------------------------------------------------------
# Episode-level computation
# ---------------------------------------------------------------------------


def compute_episode_rewards(
    episode_df: pl.DataFrame,
    *,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
    stay_id_col: str = "stay_id",
    step_col: str = "step_index",
    sofa_col: str = "sofa_score",
    lactate_col: str = "lactate",
    map_col: str = "map",
    mortality_col: str = "mortality_90d",
) -> list[StepReward]:
    """Compute rewards for all transitions in a single episode.

    Parameters
    ----------
    episode_df:
        DataFrame sorted by step_index for one stay_id. Must contain
        columns named by the *_col parameters. ``mortality_90d`` should
        be 1 (died) or 0 (survived) and be constant across the episode.
    config:
        Reward configuration.

    Returns
    -------
    list[StepReward]
        One reward per step, in step_index order.
    """
    if episode_df.is_empty():
        return []

    sorted_ep = episode_df.sort(step_col)
    n_steps = sorted_ep.height
    stay_id = int(sorted_ep.get_column(stay_id_col)[0])

    # Determine mortality outcome
    mortality_val = sorted_ep.get_column(mortality_col)[0]
    survived_90d = bool(int(mortality_val) == 0) if mortality_val is not None else None

    rewards: list[StepReward] = []

    for i in range(n_steps):
        row = sorted_ep.row(i, named=True)
        is_terminal = i == n_steps - 1

        sofa_current = _safe_float(row.get(sofa_col))
        sofa_previous = (
            _safe_float(sorted_ep.row(i - 1, named=True).get(sofa_col))
            if i > 0
            else None
        )

        lactate_current = _safe_float(row.get(lactate_col))
        lactate_previous = (
            _safe_float(sorted_ep.row(i - 1, named=True).get(lactate_col))
            if i > 0
            else None
        )

        map_current = _safe_float(row.get(map_col))

        step_reward = compute_step_reward(
            is_terminal=is_terminal,
            survived_90d=survived_90d if is_terminal else None,
            sofa_current=sofa_current,
            sofa_previous=sofa_previous,
            lactate_current=lactate_current,
            lactate_previous=lactate_previous,
            map_current=map_current,
            config=config,
        )

        # Patch step metadata
        rewards.append(
            StepReward(
                stay_id=stay_id,
                step_index=int(row[step_col]),
                terminal=step_reward.terminal,
                sofa_shaping=step_reward.sofa_shaping,
                lactate_shaping=step_reward.lactate_shaping,
                map_shaping=step_reward.map_shaping,
                total=step_reward.total,
                is_terminal=is_terminal,
            )
        )

    return rewards


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------


def compute_rewards_batch(
    state_df: pl.DataFrame,
    *,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
    stay_id_col: str = "stay_id",
    **kwargs: str,
) -> list[StepReward]:
    """Compute rewards for all episodes in a dataset.

    Parameters
    ----------
    state_df:
        State table with all episodes. Must contain ``stay_id``,
        ``step_index``, ``sofa_score``, ``lactate``, ``map``,
        ``mortality_90d``.
    config:
        Reward configuration.

    Returns
    -------
    list[StepReward]
        All rewards across all episodes, ordered by stay_id then step_index.
    """
    all_rewards: list[StepReward] = []

    stay_ids = state_df.get_column(stay_id_col).unique().sort().to_list()
    for sid in stay_ids:
        episode = state_df.filter(pl.col(stay_id_col) == sid)
        ep_rewards = compute_episode_rewards(
            episode, config=config, stay_id_col=stay_id_col, **kwargs
        )
        all_rewards.extend(ep_rewards)

    return all_rewards


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------


def reward_summary(
    rewards: list[StepReward],
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> RewardSummary:
    """Compute aggregate statistics from computed rewards.

    Parameters
    ----------
    rewards:
        List of StepReward from :func:`compute_rewards_batch`.
    config:
        The reward config used.

    Returns
    -------
    RewardSummary
    """
    if not rewards:
        return RewardSummary(
            variant=config.variant,
            n_episodes=0,
            n_transitions=0,
            n_survived=0,
            n_died=0,
            mean_total=0.0,
            std_total=0.0,
            mean_sofa_shaping=0.0,
            mean_terminal=0.0,
        )

    totals = [r.total for r in rewards]
    sofa_components = [r.sofa_shaping for r in rewards]
    terminal_components = [r.terminal for r in rewards]

    n = len(totals)
    mean_total = sum(totals) / n
    var_total = sum((t - mean_total) ** 2 for t in totals) / n
    std_total = math.sqrt(var_total)

    terminal_rewards = [r for r in rewards if r.is_terminal]
    n_survived = sum(1 for r in terminal_rewards if r.terminal > 0)
    n_died = sum(1 for r in terminal_rewards if r.terminal < 0)

    episode_ids = {r.stay_id for r in rewards}

    return RewardSummary(
        variant=config.variant,
        n_episodes=len(episode_ids),
        n_transitions=n,
        n_survived=n_survived,
        n_died=n_died,
        mean_total=mean_total,
        std_total=std_total,
        mean_sofa_shaping=sum(sofa_components) / n if n else 0.0,
        mean_terminal=sum(terminal_components) / n if n else 0.0,
    )


# ---------------------------------------------------------------------------
# Reward table export
# ---------------------------------------------------------------------------


def rewards_to_dataframe(rewards: list[StepReward]) -> pl.DataFrame:
    """Convert reward list to a Polars DataFrame."""
    if not rewards:
        return pl.DataFrame(
            schema={
                "stay_id": pl.Int64,
                "step_index": pl.Int32,
                "terminal": pl.Float64,
                "sofa_shaping": pl.Float64,
                "lactate_shaping": pl.Float64,
                "map_shaping": pl.Float64,
                "total": pl.Float64,
                "is_terminal": pl.Boolean,
            }
        )

    return pl.DataFrame(
        {
            "stay_id": [r.stay_id for r in rewards],
            "step_index": [r.step_index for r in rewards],
            "terminal": [r.terminal for r in rewards],
            "sofa_shaping": [r.sofa_shaping for r in rewards],
            "lactate_shaping": [r.lactate_shaping for r in rewards],
            "map_shaping": [r.map_shaping for r in rewards],
            "total": [r.total for r in rewards],
            "is_terminal": [r.is_terminal for r in rewards],
        }
    )


# ---------------------------------------------------------------------------
# Reward config persistence
# ---------------------------------------------------------------------------


def save_reward_config(config: RewardConfig, output_path: Path) -> None:
    """Save reward config as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": config.version,
        "variant": config.variant.value,
        "terminal_reward_survived": config.terminal_reward_survived,
        "terminal_reward_died": config.terminal_reward_died,
        "sofa_delta_weight": config.sofa_delta_weight,
        "lactate_clearance_weight": config.lactate_clearance_weight,
        "map_stability_weight": config.map_stability_weight,
        "map_threshold": config.map_threshold,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved reward config to %s", output_path)


def load_reward_config(path: Path) -> RewardConfig:
    """Load reward config from JSON."""
    payload: dict[str, Any] = json.loads(path.read_text())
    return RewardConfig(
        variant=RewardVariant(payload["variant"]),
        terminal_reward_survived=float(payload["terminal_reward_survived"]),
        terminal_reward_died=float(payload["terminal_reward_died"]),
        sofa_delta_weight=float(payload["sofa_delta_weight"]),
        lactate_clearance_weight=float(payload.get("lactate_clearance_weight", 0.0)),
        map_stability_weight=float(payload.get("map_stability_weight", 0.0)),
        map_threshold=float(payload.get("map_threshold", 65.0)),
        version=str(payload.get("version", REWARD_SPEC_VERSION)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Coerce to float, returning None for null/NaN."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.mdp.rewards",
        description="Validate reward computation with a dry run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic data to verify the pipeline works.",
    )
    p.add_argument(
        "--variant",
        choices=["sparse", "sofa_shaped", "full_shaped"],
        default="sofa_shaped",
        help="Reward variant to test. Default: sofa_shaped.",
    )
    return p


def _dry_run(variant: str) -> None:
    """Smoke test with synthetic episode data."""
    logger.info("Running reward computation dry-run (variant=%s)...", variant)

    config = RewardConfig(variant=RewardVariant(variant))

    # Synthetic episode: 5 steps, patient dies
    episode = pl.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1],
            "step_index": [0, 1, 2, 3, 4],
            "sofa_score": [4.0, 6.0, 8.0, 7.0, 5.0],
            "lactate": [2.0, 3.5, 4.0, 3.0, 2.5],
            "map": [75.0, 60.0, 55.0, 68.0, 72.0],
            "mortality_90d": [1, 1, 1, 1, 1],
        }
    )

    rewards = compute_episode_rewards(episode, config=config)
    logger.info("Computed %d step rewards:", len(rewards))
    for r in rewards:
        logger.info(
            "  step=%d  terminal=%.2f  sofa=%.4f  lactate=%.4f  map=%.4f → total=%.4f%s",
            r.step_index,
            r.terminal,
            r.sofa_shaping,
            r.lactate_shaping,
            r.map_shaping,
            r.total,
            "  [TERMINAL]" if r.is_terminal else "",
        )

    summary = reward_summary(rewards, config)
    logger.info(
        "Summary: episodes=%d  transitions=%d  survived=%d  died=%d  "
        "mean_total=%.4f  std=%.4f",
        summary.n_episodes,
        summary.n_transitions,
        summary.n_survived,
        summary.n_died,
        summary.mean_total,
        summary.std_total,
    )

    logger.info("✅ Reward computation dry-run PASSED.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        _dry_run(args.variant)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
