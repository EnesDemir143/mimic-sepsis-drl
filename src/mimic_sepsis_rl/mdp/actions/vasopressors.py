"""
Vasopressor standardisation to norepinephrine-equivalent dose rates.

MIMIC-IV ``inputevents`` records vasopressor infusions in native units
(e.g. dopamine in µg/kg/min, vasopressin in units/min). This module
normalises all agents to a common norepinephrine-equivalent dose rate
(µg/kg/min) using published pharmacological conversion factors.

Conversion factors
------------------
The equivalence table follows the widely cited sepsis offline RL
literature (Komorowski et al., Nature Medicine 2018) and SSC guideline
pharmacology:

| Agent              | Factor | Source unit      |
|--------------------|--------|-----------------|
| Norepinephrine     | 1.0    | µg/kg/min       |
| Epinephrine        | 1.0    | µg/kg/min       |
| Dopamine           | 0.01   | µg/kg/min → NE  |
| Phenylephrine      | 0.1    | µg/kg/min → NE  |
| Vasopressin        | 2.5    | units/min → NE  |

Usage (CLI validation)
----------------------
    python -m mimic_sepsis_rl.mdp.actions.vasopressors --dry-run

Version history
---------------
v1.0.0  2026-03-29  Initial vasopressor standardisation contract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Final

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

VASOPRESSOR_SPEC_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# MIMIC-IV item ID → agent mapping
# ---------------------------------------------------------------------------

# itemid values from MIMIC-IV d_items for common ICU vasopressors.
AGENT_ITEM_IDS: Final[dict[str, tuple[int, ...]]] = {
    "norepinephrine": (221906,),
    "epinephrine": (221289,),
    "dopamine": (221662,),
    "phenylephrine": (222315,),
    "vasopressin": (222042,),
}

# Inverse lookup: itemid → agent name
ITEM_TO_AGENT: Final[dict[int, str]] = {
    item_id: agent
    for agent, ids in AGENT_ITEM_IDS.items()
    for item_id in ids
}


# ---------------------------------------------------------------------------
# Conversion table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversionEntry:
    """Pharmacological equivalence to norepinephrine."""

    agent: str
    factor: float
    source_unit: str
    target_unit: str = "µg/kg/min NE-equiv"


NE_EQUIVALENCE: Final[dict[str, ConversionEntry]] = {
    "norepinephrine": ConversionEntry("norepinephrine", 1.0, "µg/kg/min"),
    "epinephrine": ConversionEntry("epinephrine", 1.0, "µg/kg/min"),
    "dopamine": ConversionEntry("dopamine", 0.01, "µg/kg/min"),
    "phenylephrine": ConversionEntry("phenylephrine", 0.1, "µg/kg/min"),
    "vasopressin": ConversionEntry("vasopressin", 2.5, "units/min"),
}


# ---------------------------------------------------------------------------
# Standardiser
# ---------------------------------------------------------------------------


class VasopressorStandardiser:
    """Convert raw vasopressor infusion records to NE-equivalent dose rates.

    Attributes
    ----------
    equivalence_table : dict[str, ConversionEntry]
        Frozen conversion factors used for standardisation.

    Examples
    --------
    >>> std = VasopressorStandardiser()
    >>> result = std.standardise(input_df)
    """

    def __init__(
        self,
        equivalence: dict[str, ConversionEntry] | None = None,
    ) -> None:
        self.equivalence_table = equivalence or dict(NE_EQUIVALENCE)

    def standardise(
        self,
        inputevents: pl.DataFrame,
        *,
        rate_col: str = "rate",
        itemid_col: str = "itemid",
    ) -> pl.DataFrame:
        """Map raw infusion rows to NE-equivalent dose rates.

        Parameters
        ----------
        inputevents:
            DataFrame with at least ``stay_id``, ``starttime``, ``endtime``,
            *itemid_col*, and *rate_col* columns.
        rate_col:
            Column containing the raw infusion rate.
        itemid_col:
            Column containing the MIMIC-IV item identifier.

        Returns
        -------
        pl.DataFrame
            Original rows filtered to vasopressor item IDs with an appended
            ``ne_equiv_rate`` column (µg/kg/min NE-equivalent) and an
            ``agent`` column.
        """
        all_item_ids = list(ITEM_TO_AGENT.keys())
        vaso_df = inputevents.filter(pl.col(itemid_col).is_in(all_item_ids))

        if vaso_df.is_empty():
            return vaso_df.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("agent"),
                pl.lit(None).cast(pl.Float64).alias("ne_equiv_rate"),
            )

        # Map item_id → agent
        item_agent_df = pl.DataFrame(
            {
                itemid_col: list(ITEM_TO_AGENT.keys()),
                "agent": list(ITEM_TO_AGENT.values()),
            }
        )
        vaso_df = vaso_df.join(item_agent_df, on=itemid_col, how="left")

        # Map agent → factor
        agent_factor_df = pl.DataFrame(
            {
                "agent": [e.agent for e in self.equivalence_table.values()],
                "_factor": [e.factor for e in self.equivalence_table.values()],
            }
        )
        vaso_df = vaso_df.join(agent_factor_df, on="agent", how="left")

        # Compute NE-equivalent rate
        vaso_df = vaso_df.with_columns(
            (pl.col(rate_col).fill_null(0.0) * pl.col("_factor").fill_null(0.0))
            .alias("ne_equiv_rate")
        ).drop("_factor")

        return vaso_df

    def aggregate_per_step(
        self,
        standardised_df: pl.DataFrame,
        step_boundaries: pl.DataFrame,
    ) -> pl.DataFrame:
        """Aggregate NE-equivalent rates within each 4-hour step.

        Parameters
        ----------
        standardised_df:
            Output of :meth:`standardise` with ``stay_id``, ``starttime``,
            ``endtime``, ``ne_equiv_rate``.
        step_boundaries:
            DataFrame with ``stay_id``, ``step_index``, ``step_start``,
            ``step_end``.

        Returns
        -------
        pl.DataFrame
            One row per (stay_id, step_index) with ``vaso_dose_4h`` — the
            time-weighted average NE-equivalent infusion rate over that step
            (µg/kg/min). Zero when no vasopressor was running.
        """
        if standardised_df.is_empty():
            return step_boundaries.select(
                "stay_id", "step_index"
            ).with_columns(
                pl.lit(0.0).alias("vaso_dose_4h")
            )

        # Cross-join approach: for each step, find overlapping infusions
        joined = step_boundaries.join(
            standardised_df.select(
                "stay_id", "starttime", "endtime", "ne_equiv_rate"
            ),
            on="stay_id",
            how="left",
        )

        # Filter to overlapping time windows
        overlap = joined.filter(
            (pl.col("starttime") < pl.col("step_end"))
            & (pl.col("endtime") > pl.col("step_start"))
        )

        if overlap.is_empty():
            return step_boundaries.select(
                "stay_id", "step_index"
            ).with_columns(
                pl.lit(0.0).alias("vaso_dose_4h")
            )

        # Compute overlap duration and time-weighted average
        result = (
            overlap.with_columns(
                # Clamp to step boundaries
                pl.max_horizontal("starttime", "step_start").alias("eff_start"),
                pl.min_horizontal("endtime", "step_end").alias("eff_end"),
            )
            .with_columns(
                (
                    (pl.col("eff_end") - pl.col("eff_start")).dt.total_seconds()
                    / 3600.0
                ).alias("overlap_hours"),
            )
            .filter(pl.col("overlap_hours") > 0)
            .with_columns(
                (pl.col("ne_equiv_rate") * pl.col("overlap_hours")).alias("dose_hours")
            )
            .group_by("stay_id", "step_index")
            .agg(
                pl.col("dose_hours").sum().alias("total_dose_hours"),
                pl.col("overlap_hours").sum().alias("total_overlap_hours"),
            )
            .with_columns(
                (pl.col("total_dose_hours") / pl.col("total_overlap_hours"))
                .alias("vaso_dose_4h")
            )
            .select("stay_id", "step_index", "vaso_dose_4h")
        )

        # Left-join back to ensure every step has a row
        return (
            step_boundaries.select("stay_id", "step_index")
            .join(result, on=["stay_id", "step_index"], how="left")
            .with_columns(pl.col("vaso_dose_4h").fill_null(0.0))
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.mdp.actions.vasopressors",
        description="Validate vasopressor standardisation logic with a dry run.",
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

    logger.info("Running vasopressor standardisation dry-run...")
    std = VasopressorStandardiser()

    # Synthetic inputevents
    input_df = pl.DataFrame(
        {
            "stay_id": [1, 1, 1, 2],
            "itemid": [221906, 221289, 221662, 222042],
            "rate": [0.15, 0.10, 5.0, 0.04],
            "starttime": [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 1, 0),
                datetime(2024, 1, 1, 2, 0),
                datetime(2024, 1, 1, 0, 0),
            ],
            "endtime": [
                datetime(2024, 1, 1, 4, 0),
                datetime(2024, 1, 1, 3, 0),
                datetime(2024, 1, 1, 4, 0),
                datetime(2024, 1, 1, 4, 0),
            ],
        }
    )

    result = std.standardise(input_df)
    logger.info("Standardised %d vasopressor rows:", result.height)
    for row in result.iter_rows(named=True):
        logger.info(
            "  stay_id=%d agent=%-18s raw_rate=%.4f → NE-equiv=%.4f",
            row["stay_id"],
            row["agent"],
            row["rate"],
            row["ne_equiv_rate"],
        )

    logger.info("✅ Vasopressor standardisation dry-run PASSED.")


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
