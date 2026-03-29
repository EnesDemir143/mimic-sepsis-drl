"""
Regression tests for the transition dataset and replay buffer pipeline.

Covers:
- Single-episode transition shape and done handling
- Multi-episode batch transitions
- State vector extraction (null / NaN → 0.0)
- Episode boundary integrity (done only at last step)
- Absorbing next_state at terminal step
- Action range constraints
- DataFrame export round-trip
- Metadata construction and serialisation
- Replay buffer grouping and validation
- Empty input edge cases
- Step alignment (transitions align with step_index)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import polars as pl
import pytest

from mimic_sepsis_rl.datasets.transitions import (
    TRANSITION_SPEC_VERSION,
    TransitionRow,
    TransitionDatasetMeta,
    build_episode_transitions,
    build_transitions,
    build_dataset_meta,
    transitions_to_dataframe,
    save_transitions,
    load_transition_meta,
)
from mimic_sepsis_rl.datasets.replay_buffer import (
    REPLAY_BUFFER_VERSION,
    EpisodeBuffer,
    ReplayBuffer,
    ReplayBufferValidationError,
    build_replay_buffer,
    validate_replay_buffer,
    save_replay_buffer,
    load_replay_buffer_meta,
)


# ===================================================================
# Helpers
# ===================================================================


FEATURE_COLS = ["feat_a", "feat_b", "feat_c"]


def _make_episode(
    stay_id: int = 100,
    n_steps: int = 4,
    mortality: int = 0,
) -> pl.DataFrame:
    """Create a synthetic single-episode DataFrame."""
    return pl.DataFrame(
        {
            "stay_id": [stay_id] * n_steps,
            "step_index": list(range(n_steps)),
            "feat_a": [1.0 + i for i in range(n_steps)],
            "feat_b": [2.0 - 0.5 * i for i in range(n_steps)],
            "feat_c": [0.5] * n_steps,
            "action_id": [i % 25 for i in range(n_steps)],
            "reward_total": [0.0] * (n_steps - 1)
            + [15.0 if mortality == 0 else -15.0],
            "mortality_90d": [mortality] * n_steps,
        }
    )


def _make_batch(
    n_episodes: int = 3,
    steps_per_episode: int = 4,
) -> pl.DataFrame:
    """Create a multi-episode batch DataFrame."""
    frames = []
    for i in range(n_episodes):
        ep = _make_episode(
            stay_id=(i + 1) * 100,
            n_steps=steps_per_episode,
            mortality=i % 2,
        )
        frames.append(ep)
    return pl.concat(frames)


# ===================================================================
# Episode Transition Tests
# ===================================================================


class TestEpisodeTransitions:
    """Verify single-episode transition generation."""

    def test_correct_count(self) -> None:
        """One transition per step."""
        ep = _make_episode(n_steps=5)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        assert len(trans) == 5

    def test_done_only_at_last_step(self) -> None:
        """Only the final transition has done=True."""
        ep = _make_episode(n_steps=4)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        assert trans[-1].done is True
        assert all(t.done is False for t in trans[:-1])

    def test_absorbing_next_state(self) -> None:
        """Terminal next_state mirrors current state."""
        ep = _make_episode(n_steps=3)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        assert trans[-1].next_state == trans[-1].state

    def test_next_state_links_to_next_step(self) -> None:
        """Non-terminal next_state equals the next step's state."""
        ep = _make_episode(n_steps=4)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        for i in range(len(trans) - 1):
            assert trans[i].next_state == trans[i + 1].state

    def test_state_dimension(self) -> None:
        """State vectors have the correct dimension."""
        ep = _make_episode(n_steps=3)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        for t in trans:
            assert len(t.state) == len(FEATURE_COLS)
            assert len(t.next_state) == len(FEATURE_COLS)

    def test_stay_id_preserved(self) -> None:
        """Each transition carries the correct stay_id."""
        ep = _make_episode(stay_id=999, n_steps=3)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        assert all(t.stay_id == 999 for t in trans)

    def test_step_index_alignment(self) -> None:
        """Transition step_index matches input step_index."""
        ep = _make_episode(n_steps=5)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        for i, t in enumerate(trans):
            assert t.step_index == i

    def test_empty_episode(self) -> None:
        """Empty DataFrame returns empty list."""
        empty = pl.DataFrame(
            schema={
                "stay_id": pl.Int64,
                "step_index": pl.Int32,
                "feat_a": pl.Float64,
                "feat_b": pl.Float64,
                "feat_c": pl.Float64,
                "action_id": pl.Int32,
                "reward_total": pl.Float64,
            }
        )
        trans = build_episode_transitions(empty, feature_columns=FEATURE_COLS)
        assert trans == []


# ===================================================================
# State Vector Extraction Tests
# ===================================================================


class TestStateExtraction:
    """Verify null/NaN handling in state vectors."""

    def test_null_feature_becomes_zero(self) -> None:
        """Null values in feature columns become 0.0 in state vector."""
        ep = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "step_index": [0, 1],
                "feat_a": [1.0, None],
                "feat_b": [None, 2.0],
                "feat_c": [3.0, 3.0],
                "action_id": [0, 1],
                "reward_total": [0.0, 15.0],
            }
        )
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        # Step 0: feat_b is None → 0.0
        assert trans[0].state == (1.0, 0.0, 3.0)
        # Step 1: feat_a is None → 0.0
        assert trans[1].state == (0.0, 2.0, 3.0)

    def test_nan_feature_becomes_zero(self) -> None:
        """NaN values become 0.0 in state vector."""
        ep = pl.DataFrame(
            {
                "stay_id": [1],
                "step_index": [0],
                "feat_a": [float("nan")],
                "feat_b": [2.0],
                "feat_c": [3.0],
                "action_id": [5],
                "reward_total": [15.0],
            }
        )
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        assert trans[0].state[0] == 0.0


# ===================================================================
# Batch Transition Tests
# ===================================================================


class TestBatchTransitions:
    """Verify multi-episode batch transitions."""

    def test_total_count(self) -> None:
        """Total transitions equals sum of episode steps."""
        batch = _make_batch(n_episodes=3, steps_per_episode=4)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        assert len(trans) == 12  # 3 × 4

    def test_done_count_matches_episodes(self) -> None:
        """Number of done=True transitions equals number of episodes."""
        batch = _make_batch(n_episodes=5, steps_per_episode=3)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        done_count = sum(1 for t in trans if t.done)
        assert done_count == 5

    def test_ordered_by_stay_id(self) -> None:
        """Transitions are ordered by stay_id."""
        batch = _make_batch(n_episodes=3)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        stay_ids = [t.stay_id for t in trans]
        assert stay_ids == sorted(stay_ids)


# ===================================================================
# DataFrame Export Tests
# ===================================================================


class TestDataFrameExport:
    """Verify transition-to-DataFrame export."""

    def test_correct_columns(self) -> None:
        """Output has expected column structure."""
        ep = _make_episode(n_steps=3)
        trans = build_episode_transitions(ep, feature_columns=FEATURE_COLS)
        df = transitions_to_dataframe(trans, FEATURE_COLS)
        assert "stay_id" in df.columns
        assert "action" in df.columns
        assert "reward" in df.columns
        assert "done" in df.columns
        for col in FEATURE_COLS:
            assert f"s_{col}" in df.columns
            assert f"ns_{col}" in df.columns

    def test_row_count(self) -> None:
        """Output rows match transitions count."""
        batch = _make_batch(n_episodes=2, steps_per_episode=5)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        df = transitions_to_dataframe(trans, FEATURE_COLS)
        assert df.height == 10

    def test_empty_transitions(self) -> None:
        """Empty transitions produce empty DataFrame with correct schema."""
        df = transitions_to_dataframe([], FEATURE_COLS)
        assert df.height == 0
        assert "stay_id" in df.columns
        for col in FEATURE_COLS:
            assert f"s_{col}" in df.columns


# ===================================================================
# Metadata Tests
# ===================================================================


class TestMetadata:
    """Verify transition metadata construction."""

    def test_meta_counts(self) -> None:
        """Metadata reflects correct episode/transition counts."""
        batch = _make_batch(n_episodes=3, steps_per_episode=4)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        meta = build_dataset_meta(
            trans,
            feature_columns=FEATURE_COLS,
            split_label="train",
            manifest_seed=42,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
        )
        assert meta.n_episodes == 3
        assert meta.n_transitions == 12
        assert meta.state_dim == 3
        assert meta.n_actions == 25

    def test_meta_serialisation_roundtrip(self) -> None:
        """Metadata survives dict round-trip."""
        meta = TransitionDatasetMeta(
            spec_version="1.0.0",
            n_episodes=5,
            n_transitions=25,
            state_dim=8,
            n_actions=25,
            split_label="test",
            manifest_seed=99,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
            feature_columns=("a", "b", "c"),
        )
        loaded = TransitionDatasetMeta.from_dict(meta.to_dict())
        assert loaded == meta

    def test_meta_json_persistence(self) -> None:
        """Metadata saves and loads from JSON file."""
        batch = _make_batch(n_episodes=2, steps_per_episode=3)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)

        with tempfile.TemporaryDirectory() as tmpdir:
            _, meta_path = save_transitions(
                trans,
                FEATURE_COLS,
                Path(tmpdir),
                split_label="train",
                manifest_seed=42,
                action_spec_version="1.0.0",
                reward_spec_version="1.0.0",
            )
            loaded = load_transition_meta(meta_path)

        assert loaded.n_episodes == 2
        assert loaded.n_transitions == 6


# ===================================================================
# Replay Buffer Tests
# ===================================================================


class TestReplayBuffer:
    """Verify episode-aware replay buffer."""

    def test_episode_grouping(self) -> None:
        """Buffer groups transitions by stay_id."""
        batch = _make_batch(n_episodes=3, steps_per_episode=4)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        buffer = build_replay_buffer(
            trans,
            feature_columns=FEATURE_COLS,
            split_label="train",
            manifest_seed=42,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
        )
        assert buffer.n_episodes == 3
        assert buffer.n_transitions == 12

    def test_flat_transitions_preserves_order(self) -> None:
        """flat_transitions() returns all transitions in episode order."""
        batch = _make_batch(n_episodes=2, steps_per_episode=3)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        buffer = build_replay_buffer(
            trans,
            feature_columns=FEATURE_COLS,
            split_label="train",
            manifest_seed=42,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
        )
        flat = buffer.flat_transitions()
        assert len(flat) == 6

    def test_validation_passes(self) -> None:
        """Valid buffer passes validation without error."""
        batch = _make_batch(n_episodes=2, steps_per_episode=4)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        buffer = build_replay_buffer(
            trans,
            feature_columns=FEATURE_COLS,
            split_label="train",
            manifest_seed=42,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
        )
        validate_replay_buffer(
            buffer,
            expected_state_dim=len(FEATURE_COLS),
            expected_n_actions=25,
        )

    def test_validation_bad_action_range(self) -> None:
        """Validation fails on out-of-range actions."""
        bad_transition = TransitionRow(
            stay_id=1,
            step_index=0,
            state=(1.0, 2.0, 3.0),
            action=30,
            reward=0.0,
            next_state=(1.0, 2.0, 3.0),
            done=True,
        )
        buffer = ReplayBuffer(
            episodes=(
                EpisodeBuffer(
                    stay_id=1,
                    transitions=(bad_transition,),
                    n_steps=1,
                ),
            ),
            meta=TransitionDatasetMeta(
                spec_version="1.0.0",
                n_episodes=1,
                n_transitions=1,
                state_dim=3,
                n_actions=25,
                split_label="train",
                manifest_seed=42,
                action_spec_version="1.0.0",
                reward_spec_version="1.0.0",
                feature_columns=("a", "b", "c"),
            ),
        )
        with pytest.raises(ReplayBufferValidationError, match="out of range"):
            validate_replay_buffer(buffer, expected_n_actions=25)

    def test_validation_bad_done_flag(self) -> None:
        """Validation fails when last transition is not done."""
        t = TransitionRow(
            stay_id=1,
            step_index=0,
            state=(1.0, 2.0, 3.0),
            action=0,
            reward=0.0,
            next_state=(1.0, 2.0, 3.0),
            done=False,
        )
        buffer = ReplayBuffer(
            episodes=(
                EpisodeBuffer(
                    stay_id=1,
                    transitions=(t,),
                    n_steps=1,
                ),
            ),
            meta=TransitionDatasetMeta(
                spec_version="1.0.0",
                n_episodes=1,
                n_transitions=1,
                state_dim=3,
                n_actions=25,
                split_label="train",
                manifest_seed=42,
                action_spec_version="1.0.0",
                reward_spec_version="1.0.0",
                feature_columns=("a", "b", "c"),
            ),
        )
        with pytest.raises(ReplayBufferValidationError, match="not marked done"):
            validate_replay_buffer(buffer)


# ===================================================================
# Persistence Tests
# ===================================================================


class TestPersistence:
    """Verify save/load round-trips."""

    def test_save_transitions(self) -> None:
        """Transitions save as Parquet + JSON meta."""
        batch = _make_batch(n_episodes=2, steps_per_episode=3)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path, meta_path = save_transitions(
                trans,
                FEATURE_COLS,
                Path(tmpdir),
                split_label="train",
                manifest_seed=42,
                action_spec_version="1.0.0",
                reward_spec_version="1.0.0",
            )
            assert pq_path.exists()
            assert meta_path.exists()

            loaded_df = pl.read_parquet(pq_path)
            assert loaded_df.height == 6

    def test_save_replay_buffer(self) -> None:
        """Replay buffer saves as Parquet + JSON meta."""
        batch = _make_batch(n_episodes=2, steps_per_episode=3)
        trans = build_transitions(batch, feature_columns=FEATURE_COLS)
        buffer = build_replay_buffer(
            trans,
            feature_columns=FEATURE_COLS,
            split_label="test",
            manifest_seed=99,
            action_spec_version="1.0.0",
            reward_spec_version="1.0.0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path, meta_path = save_replay_buffer(buffer, Path(tmpdir))
            assert pq_path.exists()
            assert meta_path.exists()

            meta = load_replay_buffer_meta(meta_path)
            assert meta["n_episode_buffers"] == 2
            assert meta["replay_buffer_version"] == REPLAY_BUFFER_VERSION


# ===================================================================
# Determinism Tests
# ===================================================================


class TestDeterminism:
    """Ensure deterministic output."""

    def test_same_input_same_output(self) -> None:
        """Identical inputs produce identical transitions."""
        batch = _make_batch(n_episodes=2, steps_per_episode=3)
        t1 = build_transitions(batch, feature_columns=FEATURE_COLS)
        t2 = build_transitions(batch, feature_columns=FEATURE_COLS)

        for a, b in zip(t1, t2):
            assert a.state == b.state
            assert a.next_state == b.next_state
            assert a.action == b.action
            assert a.reward == pytest.approx(b.reward)
            assert a.done == b.done
