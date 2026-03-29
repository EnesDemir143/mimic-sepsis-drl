"""
CLI entrypoint for building transition datasets and replay buffers.

This CLI consumes the frozen state, action, and reward surfaces to produce
replay-ready transition artifacts.

Usage
-----
    python -m mimic_sepsis_rl.cli.build_transitions --dry-run
    python -m mimic_sepsis_rl.cli.build_transitions --help

Version history
---------------
v1.0.0  2026-03-29  Initial transition CLI.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.cli.build_transitions",
        description="Build transition datasets and replay buffers.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic data to verify the pipeline works.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/transitions",
        help="Output directory for transition artifacts. Default: outputs/transitions.",
    )
    return p


def _make_synthetic_data(
    n_episodes: int = 5,
    steps_per_episode: int = 6,
    n_features: int = 8,
) -> tuple[pl.DataFrame, list[str]]:
    """Generate synthetic merged data for dry-run validation."""
    random.seed(42)
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    records: list[dict] = []
    for ep in range(1, n_episodes + 1):
        mortality = random.choice([0, 1])
        for step in range(steps_per_episode):
            row = {
                "stay_id": ep * 100,
                "step_index": step,
                "action_id": random.randint(0, 24),
                "reward_total": random.uniform(-1.0, 1.0),
                "mortality_90d": mortality,
            }
            for feat in feature_cols:
                row[feat] = random.gauss(0, 1)
            records.append(row)

    # Set terminal reward for last step
    for ep in range(1, n_episodes + 1):
        idx = ep * steps_per_episode - 1
        mortality = records[idx]["mortality_90d"]
        records[idx]["reward_total"] = 15.0 if mortality == 0 else -15.0

    return pl.DataFrame(records), feature_cols


def _dry_run() -> None:
    """Smoke test with synthetic data."""
    from mimic_sepsis_rl.datasets.transitions import (
        TRANSITION_SPEC_VERSION,
        build_transitions,
        build_dataset_meta,
        transitions_to_dataframe,
    )
    from mimic_sepsis_rl.datasets.replay_buffer import (
        build_replay_buffer,
        validate_replay_buffer,
    )

    logger.info("Running transition builder dry-run...")

    merged_df, feature_cols = _make_synthetic_data()
    logger.info(
        "Synthetic data: %d rows, %d episodes, %d features",
        merged_df.height,
        merged_df.get_column("stay_id").n_unique(),
        len(feature_cols),
    )

    # Build transitions
    transitions = build_transitions(
        merged_df,
        feature_columns=feature_cols,
    )
    logger.info("Built %d transitions.", len(transitions))

    # Build metadata
    meta = build_dataset_meta(
        transitions,
        feature_columns=feature_cols,
        split_label="train",
        manifest_seed=42,
        action_spec_version="1.0.0",
        reward_spec_version="1.0.0",
    )
    logger.info(
        "Metadata: %d episodes, %d transitions, state_dim=%d",
        meta.n_episodes,
        meta.n_transitions,
        meta.state_dim,
    )

    # Verify transitions to DataFrame
    df = transitions_to_dataframe(transitions, feature_cols)
    logger.info("Flat DataFrame: %d rows, %d columns", df.height, df.width)

    # Verify episode boundaries
    done_count = sum(1 for t in transitions if t.done)
    logger.info("Done transitions: %d (should equal episode count)", done_count)
    assert done_count == meta.n_episodes, "Done count mismatch!"

    # Build replay buffer
    buffer = build_replay_buffer(
        transitions,
        feature_columns=feature_cols,
        split_label="train",
        manifest_seed=42,
        action_spec_version="1.0.0",
        reward_spec_version="1.0.0",
    )
    logger.info(
        "Replay buffer: %d episodes, %d transitions",
        buffer.n_episodes,
        buffer.n_transitions,
    )

    # Validate
    validate_replay_buffer(
        buffer,
        expected_state_dim=len(feature_cols),
        expected_n_actions=25,
    )

    logger.info("✅ Transition builder dry-run PASSED.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        _dry_run()
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
