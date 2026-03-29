"""
Episode-aware replay buffer serialisation for offline RL trainers.

This module wraps :mod:`mimic_sepsis_rl.datasets.transitions` into
structures that preserve episode boundaries, making them suitable
for trajectory-aware offline RL algorithms.

Key capabilities:

- Group transitions by episode into ``EpisodeBuffer`` instances.
- Serialise / deserialise the full replay buffer as a portable directory.
- Validate buffer integrity (episode boundaries, state dimensions, action
  range) before training begins.

Usage (CLI validation)
----------------------
    python -m mimic_sepsis_rl.cli.build_transitions --dry-run

Version history
---------------
v1.0.0  2026-03-29  Initial replay buffer contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import polars as pl

from mimic_sepsis_rl.datasets.transitions import (
    TransitionDatasetMeta,
    TransitionRow,
    build_dataset_meta,
    transitions_to_dataframe,
    load_transition_meta,
)

logger = logging.getLogger(__name__)

REPLAY_BUFFER_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Episode buffer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeBuffer:
    """All transitions for a single ICU episode.

    Attributes
    ----------
    stay_id : int
        ICU stay identifier.
    transitions : tuple[TransitionRow, ...]
        Ordered transitions within the episode (by step_index).
    n_steps : int
        Number of transitions.
    """

    stay_id: int
    transitions: tuple[TransitionRow, ...]
    n_steps: int


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayBuffer:
    """Episode-aware replay buffer wrapping the shared transition contract.

    Attributes
    ----------
    episodes : tuple[EpisodeBuffer, ...]
        Per-episode transition groups, sorted by stay_id.
    meta : TransitionDatasetMeta
        Provenance metadata for the underlying dataset.
    """

    episodes: tuple[EpisodeBuffer, ...]
    meta: TransitionDatasetMeta

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def n_transitions(self) -> int:
        return sum(ep.n_steps for ep in self.episodes)

    def flat_transitions(self) -> list[TransitionRow]:
        """Return all transitions in a flat list, episode order preserved."""
        result: list[TransitionRow] = []
        for ep in self.episodes:
            result.extend(ep.transitions)
        return result


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_replay_buffer(
    transitions: list[TransitionRow],
    *,
    feature_columns: Sequence[str],
    split_label: str,
    manifest_seed: int,
    action_spec_version: str,
    reward_spec_version: str,
    n_actions: int = 25,
) -> ReplayBuffer:
    """Group transitions into an episode-aware replay buffer.

    Parameters
    ----------
    transitions:
        Flat list of transitions (all episodes combined).
    feature_columns:
        Ordered feature column names (for metadata).

    Returns
    -------
    ReplayBuffer
    """
    # Group by stay_id
    grouped: dict[int, list[TransitionRow]] = {}
    for t in transitions:
        grouped.setdefault(t.stay_id, []).append(t)

    episodes: list[EpisodeBuffer] = []
    for stay_id in sorted(grouped):
        ep_transitions = sorted(grouped[stay_id], key=lambda t: t.step_index)
        episodes.append(
            EpisodeBuffer(
                stay_id=stay_id,
                transitions=tuple(ep_transitions),
                n_steps=len(ep_transitions),
            )
        )

    meta = build_dataset_meta(
        transitions,
        feature_columns=feature_columns,
        split_label=split_label,
        manifest_seed=manifest_seed,
        action_spec_version=action_spec_version,
        reward_spec_version=reward_spec_version,
        n_actions=n_actions,
    )

    return ReplayBuffer(episodes=tuple(episodes), meta=meta)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ReplayBufferValidationError(Exception):
    """Raised when buffer integrity checks fail."""


def validate_replay_buffer(
    buffer: ReplayBuffer,
    *,
    expected_state_dim: int | None = None,
    expected_n_actions: int = 25,
) -> None:
    """Validate replay buffer structural integrity.

    Raises
    ------
    ReplayBufferValidationError
        If any structural check fails.
    """
    errors: list[str] = []

    # Episode boundary check: last transition in each episode must be done
    for ep in buffer.episodes:
        if not ep.transitions:
            errors.append(f"Episode {ep.stay_id} has zero transitions.")
            continue
        if not ep.transitions[-1].done:
            errors.append(
                f"Episode {ep.stay_id}: last transition is not marked done."
            )
        for t in ep.transitions[:-1]:
            if t.done:
                errors.append(
                    f"Episode {ep.stay_id}, step {t.step_index}: "
                    "non-terminal transition marked as done."
                )

    # State dimension check
    if expected_state_dim is not None:
        for ep in buffer.episodes:
            for t in ep.transitions:
                if len(t.state) != expected_state_dim:
                    errors.append(
                        f"Episode {ep.stay_id}, step {t.step_index}: "
                        f"state dim {len(t.state)} != expected {expected_state_dim}."
                    )
                    break
                if len(t.next_state) != expected_state_dim:
                    errors.append(
                        f"Episode {ep.stay_id}, step {t.step_index}: "
                        f"next_state dim {len(t.next_state)} != expected {expected_state_dim}."
                    )
                    break

    # Action range check
    for ep in buffer.episodes:
        for t in ep.transitions:
            if not (0 <= t.action < expected_n_actions):
                errors.append(
                    f"Episode {ep.stay_id}, step {t.step_index}: "
                    f"action {t.action} out of range [0, {expected_n_actions})."
                )

    if errors:
        msg = "Replay buffer validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ReplayBufferValidationError(msg)

    logger.info(
        "✅ Replay buffer validation passed: %d episodes, %d transitions.",
        buffer.n_episodes,
        buffer.n_transitions,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_replay_buffer(
    buffer: ReplayBuffer,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save replay buffer to disk.

    Outputs
    -------
    (parquet_path, meta_path)
        Parquet file with flat transitions and JSON with metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    split = buffer.meta.split_label

    all_transitions = buffer.flat_transitions()
    df = transitions_to_dataframe(all_transitions, buffer.meta.feature_columns)
    parquet_path = output_dir / f"replay_{split}.parquet"
    df.write_parquet(parquet_path)

    meta_path = output_dir / f"replay_{split}_meta.json"
    meta_dict = buffer.meta.to_dict()
    meta_dict["replay_buffer_version"] = REPLAY_BUFFER_VERSION
    meta_dict["n_episode_buffers"] = buffer.n_episodes
    meta_path.write_text(json.dumps(meta_dict, indent=2))

    logger.info(
        "Saved replay buffer (%d episodes, %d transitions) to %s",
        buffer.n_episodes,
        buffer.n_transitions,
        output_dir,
    )
    return parquet_path, meta_path


def load_replay_buffer_meta(path: Path) -> dict[str, Any]:
    """Load replay buffer metadata from JSON."""
    return json.loads(path.read_text())


__all__ = [
    "REPLAY_BUFFER_VERSION",
    "EpisodeBuffer",
    "ReplayBuffer",
    "ReplayBufferValidationError",
    "build_replay_buffer",
    "validate_replay_buffer",
    "save_replay_buffer",
    "load_replay_buffer_meta",
]
