"""
Train-only action binning and 25-action mapping for vasopressors × IV fluids.

This module learns action bin edges from the *training split only* and
maps each (vasopressor_dose, fluid_volume) pair to one of 25 discrete
actions using a 5×5 grid:

    action_id = vaso_bin * 5 + fluid_bin

Bin layout
----------
- **Bin 0**: zero-dose (no treatment administered)
- **Bins 1–4**: non-zero-dose quartile thresholds learned from train data

This ensures that the zero-dose case is handled explicitly, not merged
into the lowest dose quartile.

The frozen thresholds are serialisable to JSON via ``ActionBinArtifacts``
so downstream phases (baselines, RL training, evaluation) reuse the
exact same action boundaries without refitting.

Usage (CLI validation)
----------------------
    python -m mimic_sepsis_rl.mdp.actions.bins --dry-run

Version history
---------------
v1.0.0  2026-03-29  Initial 5×5 action binning contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

ACTION_SPEC_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS: Final[int] = 5          # 0 = zero-dose, 1–4 = quartiles
N_ACTIONS: Final[int] = 25     # 5 × 5
N_NONZERO_BINS: Final[int] = 4  # quartiles for non-zero doses

# Quartile cut-points (relative fractions of non-zero distribution)
QUARTILE_CUTS: Final[tuple[float, ...]] = (0.25, 0.50, 0.75)


# ---------------------------------------------------------------------------
# Frozen artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionBinArtifacts:
    """Serialisable artifact bundle for the trained 5×5 action encoder.

    Attributes
    ----------
    spec_version : str
        Version of the binning schema.
    manifest_seed : int
        Split manifest seed used during fitting.
    vaso_edges : tuple[float, ...]
        Three internal bin edges for non-zero vasopressor doses (Q25, Q50, Q75).
    fluid_edges : tuple[float, ...]
        Three internal bin edges for non-zero fluid volumes (Q25, Q50, Q75).
    n_train_vaso_nonzero : int
        Number of non-zero vasopressor observations used for fitting.
    n_train_fluid_nonzero : int
        Number of non-zero fluid observations used for fitting.
    """

    spec_version: str
    manifest_seed: int
    vaso_edges: tuple[float, ...]
    fluid_edges: tuple[float, ...]
    n_train_vaso_nonzero: int
    n_train_fluid_nonzero: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable mapping."""
        return {
            "spec_version": self.spec_version,
            "manifest_seed": self.manifest_seed,
            "vaso_edges": list(self.vaso_edges),
            "fluid_edges": list(self.fluid_edges),
            "n_train_vaso_nonzero": self.n_train_vaso_nonzero,
            "n_train_fluid_nonzero": self.n_train_fluid_nonzero,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionBinArtifacts":
        """Reconstruct from serialised data."""
        return cls(
            spec_version=str(payload["spec_version"]),
            manifest_seed=int(payload["manifest_seed"]),
            vaso_edges=tuple(float(e) for e in payload["vaso_edges"]),
            fluid_edges=tuple(float(e) for e in payload["fluid_edges"]),
            n_train_vaso_nonzero=int(payload["n_train_vaso_nonzero"]),
            n_train_fluid_nonzero=int(payload["n_train_fluid_nonzero"]),
        )


# ---------------------------------------------------------------------------
# Bin learning
# ---------------------------------------------------------------------------


def _learn_edges(
    values: pl.Series,
    label: str,
) -> tuple[float, ...]:
    """Learn quartile edges from non-zero values.

    Returns
    -------
    tuple of three floats
        (Q25, Q50, Q75) thresholds for non-zero values.

    Raises
    ------
    ValueError
        If there are fewer than 4 non-zero observations (one per quartile).
    """
    nonzero = values.filter(values > 0.0)
    if nonzero.len() < N_NONZERO_BINS:
        raise ValueError(
            f"Cannot learn {label} edges: need at least {N_NONZERO_BINS} "
            f"non-zero observations, got {nonzero.len()}."
        )

    edges: list[float] = []
    for q in QUARTILE_CUTS:
        edge = nonzero.quantile(q)
        if edge is None:
            raise ValueError(f"Quantile {q} returned None for {label}.")
        edges.append(float(edge))

    # Ensure strict monotonicity (deduplicate ties)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    return tuple(edges)


def _assign_bin(value: float, edges: tuple[float, ...]) -> int:
    """Assign a single value to a bin index (0–4).

    - 0 if value ≤ 0 (zero dose)
    - 1 if 0 < value ≤ Q25
    - 2 if Q25 < value ≤ Q50
    - 3 if Q50 < value ≤ Q75
    - 4 if value > Q75
    """
    if value <= 0.0 or math.isnan(value):
        return 0
    for i, edge in enumerate(edges):
        if value <= edge:
            return i + 1
    return N_NONZERO_BINS  # bin 4 (highest)


# ---------------------------------------------------------------------------
# Action Binner
# ---------------------------------------------------------------------------


class ActionBinner:
    """Learn and apply the 5×5 discrete action map.

    Call :meth:`fit` on training-split treatment levels, then :meth:`transform`
    on any split to produce integer action IDs.

    The fitted state is captured in :attr:`artifacts` for serialisation.
    """

    def __init__(self) -> None:
        self._artifacts: ActionBinArtifacts | None = None

    @property
    def artifacts(self) -> ActionBinArtifacts:
        """Access fitted bin artifacts (raises if not fitted)."""
        if self._artifacts is None:
            raise RuntimeError("ActionBinner has not been fitted yet.")
        return self._artifacts

    @property
    def is_fitted(self) -> bool:
        """Whether the binner has been fitted."""
        return self._artifacts is not None

    def fit(
        self,
        train_df: pl.DataFrame,
        *,
        manifest_seed: int,
        vaso_col: str = "vaso_dose_4h",
        fluid_col: str = "fluid_volume_4h",
    ) -> "ActionBinner":
        """Fit bin edges on training split treatment columns.

        Parameters
        ----------
        train_df:
            DataFrame containing treatment levels for training-split steps.
        manifest_seed:
            Seed of the split manifest that produced ``train_df``.
        vaso_col:
            Column with standardised vasopressor dose.
        fluid_col:
            Column with IV fluid volume.

        Returns
        -------
        self
        """
        vaso_series = train_df.get_column(vaso_col).cast(pl.Float64)
        fluid_series = train_df.get_column(fluid_col).cast(pl.Float64)

        vaso_edges = _learn_edges(vaso_series, "vasopressor")
        fluid_edges = _learn_edges(fluid_series, "IV_fluid")

        self._artifacts = ActionBinArtifacts(
            spec_version=ACTION_SPEC_VERSION,
            manifest_seed=manifest_seed,
            vaso_edges=vaso_edges,
            fluid_edges=fluid_edges,
            n_train_vaso_nonzero=int(
                vaso_series.filter(vaso_series > 0.0).len()
            ),
            n_train_fluid_nonzero=int(
                fluid_series.filter(fluid_series > 0.0).len()
            ),
        )

        logger.info(
            "Fitted action bins: vaso_edges=%s, fluid_edges=%s",
            self._artifacts.vaso_edges,
            self._artifacts.fluid_edges,
        )
        return self

    def load(self, artifacts: ActionBinArtifacts) -> "ActionBinner":
        """Load previously fitted bin artifacts."""
        self._artifacts = artifacts
        return self

    def transform(
        self,
        df: pl.DataFrame,
        *,
        vaso_col: str = "vaso_dose_4h",
        fluid_col: str = "fluid_volume_4h",
    ) -> pl.DataFrame:
        """Map treatment columns to discrete action IDs.

        Parameters
        ----------
        df:
            DataFrame with *vaso_col* and *fluid_col*.

        Returns
        -------
        pl.DataFrame
            Original DataFrame with added columns:
            ``vaso_bin`` (0–4), ``fluid_bin`` (0–4), ``action_id`` (0–24).
        """
        arts = self.artifacts  # raises if not fitted

        vaso_vals = df.get_column(vaso_col).to_list()
        fluid_vals = df.get_column(fluid_col).to_list()

        vaso_bins = [_assign_bin(float(v) if v is not None else 0.0, arts.vaso_edges) for v in vaso_vals]
        fluid_bins = [_assign_bin(float(v) if v is not None else 0.0, arts.fluid_edges) for v in fluid_vals]
        action_ids = [vb * N_BINS + fb for vb, fb in zip(vaso_bins, fluid_bins)]

        return df.with_columns(
            pl.Series("vaso_bin", vaso_bins, dtype=pl.Int32),
            pl.Series("fluid_bin", fluid_bins, dtype=pl.Int32),
            pl.Series("action_id", action_ids, dtype=pl.Int32),
        )

    def decode_action(self, action_id: int) -> tuple[int, int]:
        """Decode a single action ID into (vaso_bin, fluid_bin)."""
        if not 0 <= action_id < N_ACTIONS:
            raise ValueError(
                f"action_id must be in [0, {N_ACTIONS}), got {action_id}."
            )
        return divmod(action_id, N_BINS)

    def action_label(self, action_id: int) -> str:
        """Human-readable label for an action ID."""
        vb, fb = self.decode_action(action_id)
        vaso_labels = ["no_vaso", "vaso_Q1", "vaso_Q2", "vaso_Q3", "vaso_Q4"]
        fluid_labels = ["no_fluid", "fluid_Q1", "fluid_Q2", "fluid_Q3", "fluid_Q4"]
        return f"{vaso_labels[vb]}×{fluid_labels[fb]}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_action_bin_artifacts(
    artifacts: ActionBinArtifacts,
    output_path: Path,
) -> None:
    """Persist action bin artifacts as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifacts.to_dict(), indent=2))
    logger.info("Saved action bin artifacts to %s", output_path)


def load_action_bin_artifacts(path: Path) -> ActionBinArtifacts:
    """Load previously persisted action bin artifacts."""
    return ActionBinArtifacts.from_dict(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.mdp.actions.bins",
        description="Validate action binning logic with a dry run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic data to verify the pipeline works.",
    )
    return p


def _dry_run() -> None:
    """Smoke test with synthetic treatment data."""
    import random

    logger.info("Running action binning dry-run...")
    random.seed(42)

    # Simulate train data with some zero-dose and non-zero doses
    n = 200
    vaso_vals = [0.0] * 60 + [random.uniform(0.01, 1.5) for _ in range(140)]
    fluid_vals = [0.0] * 40 + [random.uniform(10.0, 2000.0) for _ in range(160)]
    random.shuffle(vaso_vals)
    random.shuffle(fluid_vals)

    train_df = pl.DataFrame(
        {
            "vaso_dose_4h": vaso_vals[:n],
            "fluid_volume_4h": fluid_vals[:n],
        }
    )

    binner = ActionBinner()
    binner.fit(train_df, manifest_seed=42)

    arts = binner.artifacts
    logger.info("Vaso edges (Q25, Q50, Q75): %s", arts.vaso_edges)
    logger.info("Fluid edges (Q25, Q50, Q75): %s", arts.fluid_edges)
    logger.info(
        "Non-zero train counts: vaso=%d, fluid=%d",
        arts.n_train_vaso_nonzero,
        arts.n_train_fluid_nonzero,
    )

    result = binner.transform(train_df)
    action_dist = result.get_column("action_id").value_counts().sort("action_id")
    logger.info("Action distribution:\n%s", action_dist)

    # Verify decode round-trip
    for aid in range(N_ACTIONS):
        vb, fb = binner.decode_action(aid)
        assert vb * 5 + fb == aid, f"Decode failed for action {aid}"
        label = binner.action_label(aid)
        logger.info("  action %2d → %s", aid, label)

    logger.info("✅ Action binning dry-run PASSED.")


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
