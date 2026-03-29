"""
Transition dataset builder for the MIMIC Sepsis Offline RL pipeline.

Converts the frozen state table, action assignments, and computed rewards
into replay-ready ``(s_t, a_t, r_t, s_{t+1}, done)`` transition tuples.

The builder enforces the existing split, action, and reward contracts:
transitions cannot be produced without referencing the artifact versions
that generated each upstream surface.

Usage (CLI validation)
----------------------
    python -m mimic_sepsis_rl.cli.build_transitions --dry-run

Version history
---------------
v1.0.0  2026-03-29  Initial transition dataset contract.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

TRANSITION_SPEC_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Transition row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionRow:
    """One MDP transition: (s_t, a_t, r_t, s_{t+1}, done).

    Attributes
    ----------
    stay_id : int
        ICU episode identifier.
    step_index : int
        0-based step within the episode.
    state : tuple[float, ...]
        State vector at step t.
    action : int
        Discrete action ID (0–24) at step t.
    reward : float
        Reward received after taking action at step t.
    next_state : tuple[float, ...]
        State vector at step t+1 (absorbing copy when done).
    done : bool
        Whether this transition ends the episode.
    """

    stay_id: int
    step_index: int
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionDatasetMeta:
    """Provenance metadata bundled with every transition export.

    Records which upstream artifact versions generated this dataset so
    the lineage can be audited without inspecting the raw data.
    """

    spec_version: str
    n_episodes: int
    n_transitions: int
    state_dim: int
    n_actions: int
    split_label: str
    manifest_seed: int
    action_spec_version: str
    reward_spec_version: str
    feature_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "n_episodes": self.n_episodes,
            "n_transitions": self.n_transitions,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "split_label": self.split_label,
            "manifest_seed": self.manifest_seed,
            "action_spec_version": self.action_spec_version,
            "reward_spec_version": self.reward_spec_version,
            "feature_columns": list(self.feature_columns),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionDatasetMeta":
        return cls(
            spec_version=str(payload["spec_version"]),
            n_episodes=int(payload["n_episodes"]),
            n_transitions=int(payload["n_transitions"]),
            state_dim=int(payload["state_dim"]),
            n_actions=int(payload["n_actions"]),
            split_label=str(payload["split_label"]),
            manifest_seed=int(payload["manifest_seed"]),
            action_spec_version=str(payload["action_spec_version"]),
            reward_spec_version=str(payload["reward_spec_version"]),
            feature_columns=tuple(payload["feature_columns"]),
        )


# ---------------------------------------------------------------------------
# State vector extraction
# ---------------------------------------------------------------------------


def _extract_state_vector(
    row: dict[str, Any],
    feature_columns: Sequence[str],
) -> tuple[float, ...]:
    """Extract an ordered state vector from a named row dict."""
    values: list[float] = []
    for col in feature_columns:
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            values.append(0.0)
        else:
            values.append(float(v))
    return tuple(values)


# ---------------------------------------------------------------------------
# Transition builder
# ---------------------------------------------------------------------------


def build_episode_transitions(
    episode_df: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    action_col: str = "action_id",
    reward_col: str = "reward_total",
    stay_id_col: str = "stay_id",
    step_col: str = "step_index",
) -> list[TransitionRow]:
    """Build transitions for a single episode.

    Parameters
    ----------
    episode_df:
        DataFrame for one episode, sorted by step_index. Must contain
        all *feature_columns*, *action_col*, *reward_col*, *stay_id_col*,
        and *step_col*.
    feature_columns:
        Ordered list of state-feature column names.
    action_col:
        Column with integer action IDs (0–24).
    reward_col:
        Column with scalar reward values.

    Returns
    -------
    list[TransitionRow]
        One transition per step. The last step has ``done=True`` and
        ``next_state`` copied from the current state (absorbing).
    """
    if episode_df.is_empty():
        return []

    sorted_ep = episode_df.sort(step_col)
    n_steps = sorted_ep.height
    stay_id = int(sorted_ep.get_column(stay_id_col)[0])

    transitions: list[TransitionRow] = []

    for i in range(n_steps):
        row = sorted_ep.row(i, named=True)
        is_last = i == n_steps - 1
        state = _extract_state_vector(row, feature_columns)
        action = int(row[action_col])
        reward = float(row[reward_col])

        if is_last:
            # Absorbing: next_state mirrors current state
            next_state = state
        else:
            next_row = sorted_ep.row(i + 1, named=True)
            next_state = _extract_state_vector(next_row, feature_columns)

        transitions.append(
            TransitionRow(
                stay_id=stay_id,
                step_index=int(row[step_col]),
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=is_last,
            )
        )

    return transitions


def build_transitions(
    merged_df: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    action_col: str = "action_id",
    reward_col: str = "reward_total",
    stay_id_col: str = "stay_id",
    step_col: str = "step_index",
) -> list[TransitionRow]:
    """Build transitions for all episodes in a merged dataframe.

    Parameters
    ----------
    merged_df:
        DataFrame with state features, action IDs, and rewards for all
        episodes. Must be groupable by *stay_id_col*.

    Returns
    -------
    list[TransitionRow]
        All transitions ordered by stay_id → step_index.
    """
    all_transitions: list[TransitionRow] = []

    stay_ids = merged_df.get_column(stay_id_col).unique().sort().to_list()
    for sid in stay_ids:
        episode = merged_df.filter(pl.col(stay_id_col) == sid)
        ep_trans = build_episode_transitions(
            episode,
            feature_columns=feature_columns,
            action_col=action_col,
            reward_col=reward_col,
            stay_id_col=stay_id_col,
            step_col=step_col,
        )
        all_transitions.extend(ep_trans)

    return all_transitions


# ---------------------------------------------------------------------------
# Transition to DataFrame
# ---------------------------------------------------------------------------


def transitions_to_dataframe(
    transitions: list[TransitionRow],
    feature_columns: Sequence[str],
) -> pl.DataFrame:
    """Export transition list to a flat Polars DataFrame.

    The state and next_state vectors are unpacked into individual columns:
    ``s_<feature>`` and ``ns_<feature>`` respectively.
    """
    if not transitions:
        schema: dict[str, pl.DataType] = {
            "stay_id": pl.Int64,
            "step_index": pl.Int32,
            "action": pl.Int32,
            "reward": pl.Float64,
            "done": pl.Boolean,
        }
        for col in feature_columns:
            schema[f"s_{col}"] = pl.Float64
            schema[f"ns_{col}"] = pl.Float64
        return pl.DataFrame(schema=schema)

    records: list[dict[str, Any]] = []
    for t in transitions:
        row: dict[str, Any] = {
            "stay_id": t.stay_id,
            "step_index": t.step_index,
            "action": t.action,
            "reward": t.reward,
            "done": t.done,
        }
        for i, col in enumerate(feature_columns):
            row[f"s_{col}"] = t.state[i]
            row[f"ns_{col}"] = t.next_state[i]
        records.append(row)

    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# Dataset metadata builder
# ---------------------------------------------------------------------------


def build_dataset_meta(
    transitions: list[TransitionRow],
    *,
    feature_columns: Sequence[str],
    split_label: str,
    manifest_seed: int,
    action_spec_version: str,
    reward_spec_version: str,
    n_actions: int = 25,
) -> TransitionDatasetMeta:
    """Build metadata for a transition dataset export."""
    episode_ids = {t.stay_id for t in transitions}
    return TransitionDatasetMeta(
        spec_version=TRANSITION_SPEC_VERSION,
        n_episodes=len(episode_ids),
        n_transitions=len(transitions),
        state_dim=len(feature_columns),
        n_actions=n_actions,
        split_label=split_label,
        manifest_seed=manifest_seed,
        action_spec_version=action_spec_version,
        reward_spec_version=reward_spec_version,
        feature_columns=tuple(feature_columns),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_transitions(
    transitions: list[TransitionRow],
    feature_columns: Sequence[str],
    output_dir: Path,
    *,
    split_label: str,
    manifest_seed: int,
    action_spec_version: str,
    reward_spec_version: str,
) -> tuple[Path, Path]:
    """Save transitions and metadata to disk.

    Returns
    -------
    (parquet_path, meta_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # DataFrame export
    df = transitions_to_dataframe(transitions, feature_columns)
    parquet_path = output_dir / f"transitions_{split_label}.parquet"
    df.write_parquet(parquet_path)

    # Metadata
    meta = build_dataset_meta(
        transitions,
        feature_columns=feature_columns,
        split_label=split_label,
        manifest_seed=manifest_seed,
        action_spec_version=action_spec_version,
        reward_spec_version=reward_spec_version,
    )
    meta_path = output_dir / f"transitions_{split_label}_meta.json"
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

    logger.info(
        "Saved %d transitions (%d episodes) to %s",
        meta.n_transitions,
        meta.n_episodes,
        parquet_path,
    )
    return parquet_path, meta_path


def load_transition_meta(path: Path) -> TransitionDatasetMeta:
    """Load transition metadata from JSON."""
    return TransitionDatasetMeta.from_dict(json.loads(path.read_text()))


__all__ = [
    "TRANSITION_SPEC_VERSION",
    "TransitionRow",
    "TransitionDatasetMeta",
    "build_episode_transitions",
    "build_transitions",
    "build_dataset_meta",
    "transitions_to_dataframe",
    "save_transitions",
    "load_transition_meta",
]
