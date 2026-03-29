"""
Regression tests for the reward computation pipeline.

Covers:
- Terminal reward behaviour (survived vs died)
- SOFA-delta shaping mechanics
- Lactate clearance shaping
- MAP stability shaping
- Sparse / SOFA-shaped / full-shaped variant switching
- Edge cases (missing values, single step, empty episodes)
- RewardConfig serialisation round-trip
- RewardSummary correctness
- Determinism (same input → same output)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import polars as pl
import pytest

from mimic_sepsis_rl.mdp.reward_models import (
    REWARD_SPEC_VERSION,
    RewardConfig,
    RewardSummary,
    RewardVariant,
    StepReward,
)
from mimic_sepsis_rl.mdp.rewards import (
    compute_step_reward,
    compute_episode_rewards,
    compute_rewards_batch,
    reward_summary,
    rewards_to_dataframe,
    save_reward_config,
    load_reward_config,
)


# ===================================================================
# Terminal Reward Tests
# ===================================================================


class TestTerminalReward:
    """Verify terminal 90-day mortality reward logic."""

    def test_survived_gets_positive(self) -> None:
        """Surviving patient gets +15 terminal reward."""
        config = RewardConfig(variant=RewardVariant.SPARSE)
        r = compute_step_reward(
            is_terminal=True,
            survived_90d=True,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        assert r.terminal == pytest.approx(15.0)
        assert r.total == pytest.approx(15.0)

    def test_died_gets_negative(self) -> None:
        """Deceased patient gets −15 terminal reward."""
        config = RewardConfig(variant=RewardVariant.SPARSE)
        r = compute_step_reward(
            is_terminal=True,
            survived_90d=False,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        assert r.terminal == pytest.approx(-15.0)
        assert r.total == pytest.approx(-15.0)

    def test_non_terminal_zero(self) -> None:
        """Non-terminal steps have zero terminal reward."""
        config = RewardConfig(variant=RewardVariant.SPARSE)
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=True,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        assert r.terminal == pytest.approx(0.0)
        assert r.total == pytest.approx(0.0)

    def test_custom_terminal_values(self) -> None:
        """Custom terminal rewards are respected."""
        config = RewardConfig(
            variant=RewardVariant.SPARSE,
            terminal_reward_survived=100.0,
            terminal_reward_died=-50.0,
        )
        r_survived = compute_step_reward(
            is_terminal=True,
            survived_90d=True,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        assert r_survived.terminal == pytest.approx(100.0)

        r_died = compute_step_reward(
            is_terminal=True,
            survived_90d=False,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        assert r_died.terminal == pytest.approx(-50.0)


# ===================================================================
# SOFA-Delta Shaping Tests
# ===================================================================


class TestSofaShaping:
    """Verify SOFA-delta intermediate shaping."""

    def test_sofa_improvement_positive_reward(self) -> None:
        """SOFA decrease (improvement) gives positive shaping."""
        config = RewardConfig(
            variant=RewardVariant.SOFA_SHAPED,
            sofa_delta_weight=-0.025,
        )
        # SOFA dropped from 8 to 6 → delta = -2
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=6.0,
            sofa_previous=8.0,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        # sofa_shaping = -0.025 * (6 - 8) = -0.025 * -2 = +0.05
        assert r.sofa_shaping == pytest.approx(0.05)

    def test_sofa_worsening_negative_reward(self) -> None:
        """SOFA increase (worsening) gives negative shaping."""
        config = RewardConfig(
            variant=RewardVariant.SOFA_SHAPED,
            sofa_delta_weight=-0.025,
        )
        # SOFA rose from 4 to 8 → delta = +4
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=8.0,
            sofa_previous=4.0,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        # sofa_shaping = -0.025 * 4 = -0.10
        assert r.sofa_shaping == pytest.approx(-0.10)

    def test_no_sofa_shaping_in_sparse(self) -> None:
        """Sparse variant ignores SOFA shaping."""
        config = RewardConfig(variant=RewardVariant.SPARSE)
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=10.0,
            sofa_previous=5.0,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        assert r.sofa_shaping == pytest.approx(0.0)

    def test_missing_sofa_zero_shaping(self) -> None:
        """Missing SOFA values produce zero shaping."""
        config = RewardConfig(variant=RewardVariant.SOFA_SHAPED)
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=None,
            sofa_previous=5.0,
            lactate_current=None,
            lactate_previous=None,
            map_current=None,
            config=config,
        )
        assert r.sofa_shaping == pytest.approx(0.0)


# ===================================================================
# Lactate & MAP Shaping Tests
# ===================================================================


class TestFullShaping:
    """Verify lactate clearance and MAP stability shaping."""

    def test_lactate_clearance_positive(self) -> None:
        """Lactate decrease gives positive shaping in full_shaped."""
        config = RewardConfig(
            variant=RewardVariant.FULL_SHAPED,
            lactate_clearance_weight=0.5,
        )
        # Lactate: 4.0 → 2.0 = 50% clearance
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=2.0,
            lactate_previous=4.0,
            map_current=None,
            config=config,
        )
        # clearance = (4-2)/4 = 0.5, shaping = 0.5 * 0.5 = 0.25
        assert r.lactate_shaping == pytest.approx(0.25)

    def test_map_below_threshold_penalised(self) -> None:
        """MAP below 65 mmHg incurs a penalty in full_shaped."""
        config = RewardConfig(
            variant=RewardVariant.FULL_SHAPED,
            map_stability_weight=0.1,
            map_threshold=65.0,
        )
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=None,
            lactate_previous=None,
            map_current=55.0,
            config=config,
        )
        # deficit = 65 - 55 = 10, shaping = -0.1 * 10 = -1.0
        assert r.map_shaping == pytest.approx(-1.0)

    def test_map_above_threshold_no_penalty(self) -> None:
        """MAP above threshold produces zero MAP shaping."""
        config = RewardConfig(
            variant=RewardVariant.FULL_SHAPED,
            map_stability_weight=0.1,
        )
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=None,
            lactate_previous=None,
            map_current=80.0,
            config=config,
        )
        assert r.map_shaping == pytest.approx(0.0)

    def test_no_lactate_map_in_sofa_shaped(self) -> None:
        """SOFA-shaped variant ignores lactate and MAP shaping."""
        config = RewardConfig(
            variant=RewardVariant.SOFA_SHAPED,
            lactate_clearance_weight=1.0,
            map_stability_weight=1.0,
        )
        r = compute_step_reward(
            is_terminal=False,
            survived_90d=None,
            sofa_current=None,
            sofa_previous=None,
            lactate_current=2.0,
            lactate_previous=4.0,
            map_current=50.0,
            config=config,
        )
        assert r.lactate_shaping == pytest.approx(0.0)
        assert r.map_shaping == pytest.approx(0.0)


# ===================================================================
# Episode-Level Tests
# ===================================================================


class TestEpisodeRewards:
    """Test episode-level reward computation."""

    def _make_episode(self, mortality: int, n_steps: int = 5) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "stay_id": [1] * n_steps,
                "step_index": list(range(n_steps)),
                "sofa_score": [4.0 + i for i in range(n_steps)],
                "lactate": [2.0 + 0.5 * i for i in range(n_steps)],
                "map": [75.0 - 3.0 * i for i in range(n_steps)],
                "mortality_90d": [mortality] * n_steps,
            }
        )

    def test_correct_number_of_rewards(self) -> None:
        """One reward per step."""
        ep = self._make_episode(mortality=0, n_steps=5)
        rewards = compute_episode_rewards(ep)
        assert len(rewards) == 5

    def test_terminal_on_last_step_only(self) -> None:
        """Only the last step has is_terminal=True."""
        ep = self._make_episode(mortality=0, n_steps=5)
        rewards = compute_episode_rewards(ep)
        assert rewards[-1].is_terminal is True
        assert all(r.is_terminal is False for r in rewards[:-1])

    def test_survived_episode_terminal(self) -> None:
        """Survived episode gets positive terminal on last step."""
        ep = self._make_episode(mortality=0)
        rewards = compute_episode_rewards(ep)
        assert rewards[-1].terminal == pytest.approx(15.0)

    def test_died_episode_terminal(self) -> None:
        """Died episode gets negative terminal on last step."""
        ep = self._make_episode(mortality=1)
        rewards = compute_episode_rewards(ep)
        assert rewards[-1].terminal == pytest.approx(-15.0)

    def test_first_step_no_sofa_shaping(self) -> None:
        """First step has no previous SOFA → zero shaping."""
        ep = self._make_episode(mortality=0)
        config = RewardConfig(variant=RewardVariant.SOFA_SHAPED)
        rewards = compute_episode_rewards(ep, config=config)
        assert rewards[0].sofa_shaping == pytest.approx(0.0)

    def test_empty_episode_returns_empty(self) -> None:
        """Empty DataFrame returns empty list."""
        empty = pl.DataFrame(
            schema={
                "stay_id": pl.Int64,
                "step_index": pl.Int32,
                "sofa_score": pl.Float64,
                "lactate": pl.Float64,
                "map": pl.Float64,
                "mortality_90d": pl.Int64,
            }
        )
        rewards = compute_episode_rewards(empty)
        assert rewards == []


# ===================================================================
# Batch & Summary Tests
# ===================================================================


class TestBatchAndSummary:
    """Test multi-episode computation and diagnostics."""

    def _make_batch(self) -> pl.DataFrame:
        # Episode 1: survived, 3 steps
        ep1 = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "step_index": [0, 1, 2],
                "sofa_score": [4.0, 3.0, 2.0],
                "lactate": [2.0, 1.5, 1.0],
                "map": [80.0, 75.0, 78.0],
                "mortality_90d": [0, 0, 0],
            }
        )
        # Episode 2: died, 2 steps
        ep2 = pl.DataFrame(
            {
                "stay_id": [2, 2],
                "step_index": [0, 1],
                "sofa_score": [6.0, 10.0],
                "lactate": [3.0, 5.0],
                "map": [60.0, 50.0],
                "mortality_90d": [1, 1],
            }
        )
        return pl.concat([ep1, ep2])

    def test_batch_all_episodes_covered(self) -> None:
        """All episodes produce rewards."""
        batch = self._make_batch()
        rewards = compute_rewards_batch(batch)
        assert len(rewards) == 5  # 3 + 2

    def test_summary_counts(self) -> None:
        """Summary has correct episode and survival counts."""
        batch = self._make_batch()
        rewards = compute_rewards_batch(batch)
        summary = reward_summary(rewards)
        assert summary.n_episodes == 2
        assert summary.n_transitions == 5
        assert summary.n_survived == 1
        assert summary.n_died == 1

    def test_empty_summary(self) -> None:
        """Empty rewards produce zero-filled summary."""
        summary = reward_summary([])
        assert summary.n_episodes == 0
        assert summary.n_transitions == 0

    def test_rewards_to_dataframe(self) -> None:
        """Reward list converts to valid DataFrame."""
        batch = self._make_batch()
        rewards = compute_rewards_batch(batch)
        df = rewards_to_dataframe(rewards)
        assert df.height == 5
        assert "total" in df.columns
        assert "is_terminal" in df.columns


# ===================================================================
# Config Serialisation Tests
# ===================================================================


class TestConfigSerialisation:
    """Verify RewardConfig JSON round-trip."""

    def test_save_load_roundtrip(self) -> None:
        """Config survives JSON round-trip."""
        config = RewardConfig(
            variant=RewardVariant.FULL_SHAPED,
            terminal_reward_survived=20.0,
            terminal_reward_died=-20.0,
            sofa_delta_weight=-0.05,
            lactate_clearance_weight=0.3,
            map_stability_weight=0.15,
            map_threshold=60.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reward_config.json"
            save_reward_config(config, path)
            loaded = load_reward_config(path)

        assert loaded.variant == config.variant
        assert loaded.terminal_reward_survived == pytest.approx(config.terminal_reward_survived)
        assert loaded.terminal_reward_died == pytest.approx(config.terminal_reward_died)
        assert loaded.sofa_delta_weight == pytest.approx(config.sofa_delta_weight)
        assert loaded.lactate_clearance_weight == pytest.approx(config.lactate_clearance_weight)
        assert loaded.map_stability_weight == pytest.approx(config.map_stability_weight)

    def test_frozen_config(self) -> None:
        """RewardConfig is immutable."""
        config = RewardConfig()
        with pytest.raises(AttributeError):
            config.variant = RewardVariant.SPARSE  # type: ignore[misc]


# ===================================================================
# Determinism Tests
# ===================================================================


class TestDeterminism:
    """Ensure deterministic reward output."""

    def test_identical_inputs_identical_rewards(self) -> None:
        """Same episode data and config → same rewards."""
        ep = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "step_index": [0, 1, 2],
                "sofa_score": [4.0, 6.0, 3.0],
                "lactate": [2.0, 3.0, 1.5],
                "map": [70.0, 55.0, 75.0],
                "mortality_90d": [0, 0, 0],
            }
        )
        config = RewardConfig(variant=RewardVariant.SOFA_SHAPED)

        r1 = compute_episode_rewards(ep, config=config)
        r2 = compute_episode_rewards(ep, config=config)

        for a, b in zip(r1, r2):
            assert a.total == pytest.approx(b.total)
            assert a.sofa_shaping == pytest.approx(b.sofa_shaping)
            assert a.terminal == pytest.approx(b.terminal)
