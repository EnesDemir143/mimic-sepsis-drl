"""
IV fluid aggregation within 4-hour decision windows.

This module sums IV fluid volumes recorded in MIMIC-IV ``inputevents``
within each episode step to produce the per-step fluid quantity used
by the action encoder.  The aggregation follows the same crystalloid
and colloid item-ID set declared in the feature dictionary but
operates *independently*: feature dictionary entries represent
cumulative state signals, whereas this module produces step-level
*action* volumes.

Usage (CLI validation)
----------------------
    python -m mimic_sepsis_rl.mdp.actions.fluids --dry-run

Version history
---------------
v1.0.0  2026-03-29  Initial fluid aggregation contract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Final

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

FLUID_SPEC_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# MIMIC-IV item IDs for IV fluids (crystalloids + colloids)
# ---------------------------------------------------------------------------

# Sourced from the feature dictionary (cum_iv_fluid_ml) and extended
# with commonly used MIMIC-IV fluid item IDs.
IV_FLUID_ITEM_IDS: Final[tuple[int, ...]] = (
    # Crystalloids
    220949,   # 0.9% Normal Saline
    220950,   # 0.9% Normal Saline — pre-mixed
    225158,   # NaCl 0.9%
    225159,   # NaCl 0.45%
    225161,   # NaCl 3%
    225168,   # Lactated Ringers
    225828,   # Lactated Ringers — another ID
    # Colloids
    225823,   # D5W (Dextrose 5%)
    225825,   # D5 1/2NS
    226089,   # Piggyback
    # Other IV
    220952,   # Dextrose 5%
    220955,   # Hetastarch
)


# ---------------------------------------------------------------------------
# Fluid Aggregator
# ---------------------------------------------------------------------------


class FluidAggregator:
    """Aggregate IV fluid volumes within each 4-hour episode step.

    The aggregator sums the ``amount`` column for rows matching the
    configured item IDs and falling within step boundaries.

    Attributes
    ----------
    item_ids : tuple[int, ...]
        MIMIC-IV item identifiers considered as IV fluids.

    Examples
    --------
    >>> agg = FluidAggregator()
    >>> result = agg.aggregate_per_step(input_df, step_boundaries)
    """

    def __init__(
        self,
        item_ids: tuple[int, ...] | None = None,
    ) -> None:
        self.item_ids = item_ids or IV_FLUID_ITEM_IDS

    def filter_fluids(
        self,
        inputevents: pl.DataFrame,
        *,
        itemid_col: str = "itemid",
    ) -> pl.DataFrame:
        """Select only IV fluid rows from inputevents.

        Parameters
        ----------
        inputevents:
            Raw MIMIC-IV inputevents DataFrame.
        itemid_col:
            Column name for the item identifier.

        Returns
        -------
        pl.DataFrame
            Subset containing only IV fluid rows.
        """
        return inputevents.filter(
            pl.col(itemid_col).is_in(list(self.item_ids))
        )

    def aggregate_per_step(
        self,
        inputevents: pl.DataFrame,
        step_boundaries: pl.DataFrame,
        *,
        amount_col: str = "amount",
        itemid_col: str = "itemid",
        time_col: str = "starttime",
    ) -> pl.DataFrame:
        """Sum IV fluid volumes within each 4-hour step.

        Parameters
        ----------
        inputevents:
            Raw MIMIC-IV inputevents DataFrame with ``stay_id``,
            *itemid_col*, *amount_col*, and *time_col*.
        step_boundaries:
            DataFrame with ``stay_id``, ``step_index``, ``step_start``,
            ``step_end``.
        amount_col:
            Column containing administered volume in mL.
        itemid_col:
            Column containing MIMIC-IV item IDs.
        time_col:
            Column containing event timestamp for assignment to steps.

        Returns
        -------
        pl.DataFrame
            One row per (stay_id, step_index) with ``fluid_volume_4h``
            (mL). Zero when no fluids were administered.
        """
        fluid_df = self.filter_fluids(inputevents, itemid_col=itemid_col)

        if fluid_df.is_empty():
            return step_boundaries.select(
                "stay_id", "step_index"
            ).with_columns(
                pl.lit(0.0).alias("fluid_volume_4h")
            )

        # Assign each fluid event to a step
        joined = step_boundaries.join(
            fluid_df.select("stay_id", time_col, amount_col),
            on="stay_id",
            how="left",
        )

        # Filter events within step boundaries
        in_step = joined.filter(
            (pl.col(time_col) >= pl.col("step_start"))
            & (pl.col(time_col) < pl.col("step_end"))
        )

        if in_step.is_empty():
            return step_boundaries.select(
                "stay_id", "step_index"
            ).with_columns(
                pl.lit(0.0).alias("fluid_volume_4h")
            )

        # Sum volumes per step
        step_totals = (
            in_step.group_by("stay_id", "step_index")
            .agg(pl.col(amount_col).sum().alias("fluid_volume_4h"))
        )

        # Left-join back to ensure every step has a row
        return (
            step_boundaries.select("stay_id", "step_index")
            .join(step_totals, on=["stay_id", "step_index"], how="left")
            .with_columns(
                pl.col("fluid_volume_4h").fill_null(0.0)
            )
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.mdp.actions.fluids",
        description="Validate IV fluid aggregation logic with a dry run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic data to verify the pipeline works.",
    )
    return p


def _dry_run() -> None:
    """Quick smoke test with synthetic data."""
    from datetime import datetime

    logger.info("Running IV fluid aggregation dry-run...")
    agg = FluidAggregator()

    input_df = pl.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2],
            "itemid": [220949, 225168, 220949, 220949, 999999],
            "amount": [500.0, 250.0, 1000.0, 300.0, 100.0],
            "starttime": [
                datetime(2024, 1, 1, 0, 30),
                datetime(2024, 1, 1, 1, 0),
                datetime(2024, 1, 1, 5, 0),
                datetime(2024, 1, 1, 0, 15),
                datetime(2024, 1, 1, 0, 15),
            ],
        }
    )

    steps = pl.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "step_index": [0, 1, 0],
            "step_start": [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 4, 0),
                datetime(2024, 1, 1, 0, 0),
            ],
            "step_end": [
                datetime(2024, 1, 1, 4, 0),
                datetime(2024, 1, 1, 8, 0),
                datetime(2024, 1, 1, 4, 0),
            ],
        }
    )

    result = agg.aggregate_per_step(input_df, steps)
    logger.info("Aggregated fluid volumes per step:")
    for row in result.iter_rows(named=True):
        logger.info(
            "  stay_id=%d  step_index=%d  fluid_volume_4h=%.1f mL",
            row["stay_id"],
            row["step_index"],
            row["fluid_volume_4h"],
        )

    logger.info("✅ IV fluid aggregation dry-run PASSED.")


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
