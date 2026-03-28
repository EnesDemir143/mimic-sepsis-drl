"""
Feature extraction scaffolding for Phase 4 state representation.

This module provides the protocol, base class, and concrete extractor
stubs that translate raw MIMIC-IV measurements into per-step feature
values driven entirely by the ``FeatureSpec`` contract in ``dictionary.py``.

No extraction logic depends on hard-coded clinical constants — every
decision about source tables, item IDs, aggregation rules, valid ranges,
and imputation strategies is read from the ``FeatureSpec`` objects at
runtime.

Architecture
------------
FeatureExtractor (Protocol)
    Structural interface — any callable object that accepts a window
    DataFrame and a FeatureSpec and returns a scalar ``float | None``.

BaseWindowExtractor
    Abstract class that wires ``FeatureSpec`` reading, range validation,
    and missing-value handling around a single ``_extract_raw`` hook.
    Concrete extractors only override ``_extract_raw``.

ChartEventsExtractor
    Extracts values from a pre-filtered chartevents window.

LabEventsExtractor
    Extracts values from a pre-filtered labevents window.

InputEventsExtractor
    Computes cumulative or per-window totals from inputevents.

OutputEventsExtractor
    Sums output quantities (urine) from outputevents.

DerivedExtractor
    Computes features derived from already-extracted base features.

StateVectorBuilder
    Orchestrates all extractors for every episode step and returns a
    Polars DataFrame with one row per (stay_id, step_index) and one
    column per feature.

Usage
-----
    from mimic_sepsis_rl.mdp.features.extractors import StateVectorBuilder
    from mimic_sepsis_rl.mdp.features.dictionary import load_feature_registry

    registry = load_feature_registry(config)
    builder = StateVectorBuilder(registry)
    state_df = builder.build(step_windows)

Version history
---------------
v1.0.0  2026-04-01  Initial extractor scaffolding for Phase 4.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import polars as pl

from mimic_sepsis_rl.mdp.features.dictionary import (
    AggregationRule,
    FeatureFamily,
    FeatureSpec,
    MissingStrategy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extractor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureExtractor(Protocol):
    """Structural interface for all feature extractors.

    An extractor accepts a window-level measurement DataFrame
    (already filtered to a single 4-hour step) and the ``FeatureSpec``
    that defines what to extract, then returns a raw scalar or ``None``
    when no measurement is available in the window.

    The caller (``BaseWindowExtractor``) is responsible for applying
    range validation, aggregation, and imputation around this value.
    """

    def extract(
        self,
        window_df: pl.DataFrame,
        spec: FeatureSpec,
        context: "ExtractionContext",
        impute: bool = True,
    ) -> float | None:
        """Return the aggregated value for ``spec`` from ``window_df``.

        Parameters
        ----------
        window_df:
            Measurements restricted to a single (stay_id, step) window.
            Must contain at least ``itemid`` and ``valuenum`` columns.
        spec:
            Feature contract describing source, item IDs, aggregation, etc.
        context:
            Episode-level context (prior values for forward-fill, weight, etc.)
        impute:
            When True, apply the feature's missing-data policy after raw
            extraction. When False, return only directly observed values.

        Returns
        -------
        float | None
            Aggregated scalar, or ``None`` if no valid measurement exists.
        """
        ...


# ---------------------------------------------------------------------------
# Extraction context
# ---------------------------------------------------------------------------


@dataclass
class ExtractionContext:
    """Mutable per-step context threaded through the extraction pipeline.

    Attributes
    ----------
    stay_id : int
        Current ICU stay being processed.
    step_index : int
        Current 4-hour step index within the episode (0-based).
    prior_values : dict[str, float]
        Most recent successfully extracted value per feature_id from all
        earlier steps in this episode.  Used for forward-fill imputation.
    episode_start_time : Any
        Absolute datetime of the first step in the episode.  Used by
        cumulative extractors to correctly bound their aggregation window.
    weight_kg : float | None
        Patient weight in kg; required for dose normalisation.
    train_medians : dict[str, float]
        Feature-level medians fit on the training split only.  Populated
        before inference; absent during unit tests (treated as 0.0).
    extracted_this_step : dict[str, float | None]
        Collects extracted values for the *current* step so that derived
        features can reference base features already computed this step.
    """

    stay_id: int
    step_index: int
    prior_values: dict[str, float] = field(default_factory=dict)
    episode_start_time: Any = None
    weight_kg: float | None = None
    train_medians: dict[str, float] = field(default_factory=dict)
    extracted_this_step: dict[str, float | None] = field(default_factory=dict)

    def get_prior(self, feature_id: str) -> float | None:
        """Return the most recent value for ``feature_id``, or ``None``."""
        return self.prior_values.get(feature_id)

    def record(self, feature_id: str, value: float | None) -> None:
        """Record an extracted value so that later features can reference it."""
        self.extracted_this_step[feature_id] = value
        if value is not None:
            self.prior_values[feature_id] = value

    def get_current(self, feature_id: str) -> float | None:
        """Return the value extracted for ``feature_id`` earlier this step."""
        return self.extracted_this_step.get(feature_id)


# ---------------------------------------------------------------------------
# Base extractor
# ---------------------------------------------------------------------------


class BaseWindowExtractor(ABC):
    """Abstract base that wires the full extraction pipeline around a hook.

    Subclasses implement ``_extract_raw`` to return a scalar from raw
    measurements; this class handles:

    1. Filtering ``window_df`` to the item IDs declared in ``spec``.
    2. Dropping rows outside ``[spec.valid_low, spec.valid_high]``.
    3. Calling the subclass ``_extract_raw`` hook.
    4. Applying forward-fill, median, or normal-value imputation when the
       hook returns ``None``.
    5. Clipping the final value to ``[spec.clip_low, spec.clip_high]``.
    """

    def extract(
        self,
        window_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
        impute: bool = True,
    ) -> float | None:
        """Full extraction pipeline for one feature in one step window."""
        # Step 1 — filter to item IDs (skip for derived / demographics)
        if spec.item_ids and not window_df.is_empty() and "itemid" in window_df.columns:
            item_df = window_df.filter(pl.col("itemid").is_in(list(spec.item_ids)))
        elif spec.item_ids and (
            window_df.is_empty() or "itemid" not in window_df.columns
        ):
            item_df = pl.DataFrame()
        else:
            item_df = window_df

        # Step 2 — range validation (drop physiologically implausible values)
        if "valuenum" in item_df.columns:
            item_df = self._apply_range_filter(item_df, spec)

        # Step 3 — raw extraction hook
        raw = self._extract_raw(item_df, spec, context)

        # Step 4 — imputation when raw is None
        if raw is None and impute:
            raw = self._impute(spec, context)

        # Step 5 — clip
        if raw is not None:
            raw = self._clip(raw, spec)

        context.record(spec.feature_id, raw)
        return raw

    # ------------------------------------------------------------------
    # Abstract hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_raw(
        self,
        item_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
    ) -> float | None:
        """Extract and aggregate raw measurements.

        Parameters
        ----------
        item_df:
            Measurement rows already filtered to ``spec.item_ids`` and
            valid range.  May be empty.
        spec:
            Full feature contract.
        context:
            Episode-level context.

        Returns
        -------
        float | None
            Aggregated value, or ``None`` if ``item_df`` is empty.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_range_filter(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
        """Drop rows where valuenum is outside [valid_low, valid_high]."""
        mask = pl.lit(True)
        if spec.valid_low is not None:
            mask = mask & (pl.col("valuenum") >= spec.valid_low)
        if spec.valid_high is not None:
            mask = mask & (pl.col("valuenum") <= spec.valid_high)
        return df.filter(mask)

    @staticmethod
    def _aggregate(series: pl.Series, rule: AggregationRule) -> float | None:
        """Apply ``rule`` to a Polars Series and return a Python float."""
        if series.is_empty():
            return None
        valid = series.drop_nulls()
        if valid.is_empty():
            return None
        if rule == AggregationRule.LAST:
            return float(valid[-1])
        if rule == AggregationRule.MEAN:
            return float(valid.mean())  # type: ignore[arg-type]
        if rule == AggregationRule.MAX:
            return float(valid.max())  # type: ignore[arg-type]
        if rule == AggregationRule.MIN:
            return float(valid.min())  # type: ignore[arg-type]
        if rule in (AggregationRule.SUM, AggregationRule.CUMULATIVE):
            return float(valid.sum())
        raise ValueError(f"Unknown aggregation rule: {rule}")

    def _impute(self, spec: FeatureSpec, context: ExtractionContext) -> float | None:
        """Return an imputed value for a missing measurement."""
        strategy = spec.missing_strategy

        if strategy == MissingStrategy.FORWARD_FILL:
            prior = context.get_prior(spec.feature_id)
            if prior is not None:
                return prior
            # Fall through to median/normal if no prior exists
            fallback_median = context.train_medians.get(spec.feature_id)
            if fallback_median is not None:
                return fallback_median
            return spec.normal_value

        if strategy == MissingStrategy.MEDIAN_TRAIN:
            median = context.train_medians.get(spec.feature_id)
            if median is not None:
                return median
            return spec.normal_value

        if strategy == MissingStrategy.ZERO:
            return 0.0

        if strategy == MissingStrategy.NORMAL_VALUE:
            return spec.normal_value

        logger.warning(
            "Unhandled missing_strategy '%s' for feature '%s'; returning None.",
            strategy,
            spec.feature_id,
        )
        return None

    @staticmethod
    def _clip(value: float, spec: FeatureSpec) -> float:
        """Apply hard clip to ``[clip_low, clip_high]``."""
        if spec.clip_low is not None:
            value = max(value, spec.clip_low)
        if spec.clip_high is not None:
            value = min(value, spec.clip_high)
        return value


# ---------------------------------------------------------------------------
# Concrete extractors
# ---------------------------------------------------------------------------


class ChartEventsExtractor(BaseWindowExtractor):
    """Extract vitals and monitor data from a chartevents window.

    Expected ``window_df`` columns: ``itemid``, ``valuenum``, ``charttime``.
    Rows must already be restricted to the current step's time boundaries.
    """

    def _extract_raw(
        self,
        item_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
    ) -> float | None:
        if item_df.is_empty() or "valuenum" not in item_df.columns:
            return None

        # Sort by charttime so LAST aggregation returns most-recent value
        if "charttime" in item_df.columns:
            item_df = item_df.sort("charttime")

        return self._aggregate(item_df["valuenum"], spec.aggregation)


class LabEventsExtractor(BaseWindowExtractor):
    """Extract laboratory values from a labevents window.

    Expected ``window_df`` columns: ``itemid``, ``valuenum``, ``charttime``.
    """

    def _extract_raw(
        self,
        item_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
    ) -> float | None:
        if item_df.is_empty() or "valuenum" not in item_df.columns:
            return None

        if "charttime" in item_df.columns:
            item_df = item_df.sort("charttime")

        return self._aggregate(item_df["valuenum"], spec.aggregation)


class InputEventsExtractor(BaseWindowExtractor):
    """Extract fluid and vasopressor data from inputevents.

    Handles both ``SUM`` (per-window totals) and ``CUMULATIVE`` (episode
    totals up to window end) aggregation rules.

    Expected ``window_df`` columns: ``itemid``, ``amount``, ``starttime``,
    ``endtime``.  For cumulative features the caller must pass all events
    from episode start, not just the current window.
    """

    def _extract_raw(
        self,
        item_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
    ) -> float | None:
        if item_df.is_empty():
            return None

        amount_col = "amount" if "amount" in item_df.columns else "valuenum"
        if amount_col not in item_df.columns:
            return None

        valid = item_df[amount_col].drop_nulls()
        if valid.is_empty():
            return None

        return float(valid.sum())


class OutputEventsExtractor(BaseWindowExtractor):
    """Extract urine output and other output quantities from outputevents.

    Expected ``window_df`` columns: ``itemid``, ``value``, ``charttime``.
    """

    def _extract_raw(
        self,
        item_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
    ) -> float | None:
        if item_df.is_empty():
            return None

        value_col = "value" if "value" in item_df.columns else "valuenum"
        if value_col not in item_df.columns:
            return None

        valid = item_df[value_col].drop_nulls()
        if valid.is_empty():
            return None

        return float(valid.sum())


class DerivedExtractor(BaseWindowExtractor):
    """Compute features derived from already-extracted base features.

    Derived features (``FeatureFamily.DERIVED``) reference values
    extracted earlier in the same step via ``context.get_current()``.
    No ``window_df`` rows are consumed.

    Supported derived feature IDs
    ------------------------------
    pf_ratio
        ``pao2 / fio2_vent``  (returns None when either parent is missing)
    shock_index
        ``heart_rate / sbp``  (returns None when sbp ≤ 0)
    hours_since_onset
        Read directly from ``EpisodeStep.hours_relative_to_onset``; the
        builder injects this via ``context.extracted_this_step`` before
        calling this extractor.
    """

    _DERIVATION_MAP: dict[str, tuple[str, ...]] = {
        "pf_ratio": ("pao2", "fio2_vent"),
        "shock_index": ("heart_rate", "sbp"),
        "hours_since_onset": (),
    }

    def _extract_raw(
        self,
        item_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
    ) -> float | None:
        fid = spec.feature_id

        if fid == "pf_ratio":
            pao2 = context.get_current("pao2")
            fio2 = context.get_current("fio2_vent")
            if pao2 is None or fio2 is None or fio2 <= 0.0:
                return None
            return pao2 / fio2

        if fid == "shock_index":
            hr = context.get_current("heart_rate")
            sbp = context.get_current("sbp")
            if hr is None or sbp is None or sbp <= 0.0:
                return None
            return hr / sbp

        if fid == "hours_since_onset":
            # Injected by StateVectorBuilder before extraction
            return context.get_current("hours_since_onset")

        logger.warning("DerivedExtractor: unknown derived feature_id '%s'.", fid)
        return None


class DemographicsExtractor(BaseWindowExtractor):
    """Inject static per-episode demographic values.

    Demographics do not vary within an episode.  The builder injects
    ``age_years`` and ``weight_kg`` into the context before extraction
    so this extractor simply reads from ``context``.
    """

    def _extract_raw(
        self,
        item_df: pl.DataFrame,
        spec: FeatureSpec,
        context: ExtractionContext,
    ) -> float | None:
        if spec.feature_id == "weight_kg":
            return context.weight_kg
        if spec.feature_id == "age_years":
            return context.get_current("age_years")
        return None


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------

#: Maps source_table values to the extractor class responsible for that table.
_EXTRACTOR_REGISTRY: dict[str, type[BaseWindowExtractor]] = {
    "chartevents": ChartEventsExtractor,
    "labevents": LabEventsExtractor,
    "inputevents": InputEventsExtractor,
    "outputevents": OutputEventsExtractor,
    "derived": DerivedExtractor,
    "patients": DemographicsExtractor,
}


def get_extractor_for_spec(spec: FeatureSpec) -> BaseWindowExtractor:
    """Return the appropriate extractor instance for ``spec.source_table``."""
    extractor_cls = _EXTRACTOR_REGISTRY.get(spec.source_table)
    if extractor_cls is None:
        raise ValueError(
            f"No extractor registered for source_table='{spec.source_table}' "
            f"(feature_id='{spec.feature_id}')."
        )
    return extractor_cls()


# ---------------------------------------------------------------------------
# Step window container
# ---------------------------------------------------------------------------


@dataclass
class StepWindowData:
    """All raw measurement DataFrames for a single (stay_id, step_index).

    Attributes
    ----------
    stay_id : int
        ICU stay identifier.
    step_index : int
        0-based step index within the episode.
    hours_relative_to_onset : float
        Hours offset from sepsis onset for this step start.
    chartevents : pl.DataFrame
        Chartevents rows in the 4-hour window.
    labevents : pl.DataFrame
        Labevents rows in the 4-hour window (plus any carry-over for
        cumulative features — handled by caller).
    inputevents : pl.DataFrame
        Inputevents rows from episode start to window end (for cumulative
        aggregation).
    outputevents : pl.DataFrame
        Outputevents rows in the 4-hour window.
    age_years : float | None
        Patient age at ICU admission (static per episode).
    weight_kg : float | None
        Admission body weight (static per episode).
    subject_id : int | None
        Optional patient identifier for manifest-aware downstream builders.
    """

    stay_id: int
    step_index: int
    hours_relative_to_onset: float
    chartevents: pl.DataFrame
    labevents: pl.DataFrame
    inputevents: pl.DataFrame
    outputevents: pl.DataFrame
    age_years: float | None = None
    weight_kg: float | None = None
    subject_id: int | None = None

    def get_df_for_table(self, table: str) -> pl.DataFrame:
        """Return the appropriate DataFrame for ``table``."""
        mapping = {
            "chartevents": self.chartevents,
            "labevents": self.labevents,
            "inputevents": self.inputevents,
            "outputevents": self.outputevents,
        }
        return mapping.get(table, pl.DataFrame())


# ---------------------------------------------------------------------------
# State vector builder
# ---------------------------------------------------------------------------


class StateVectorBuilder:
    """Orchestrate all feature extractors to produce episode state vectors.

    The builder iterates over episode steps in chronological order.
    For each step it:

    1. Sets up an ``ExtractionContext`` with the current prior_values
       (forward-fill state), weight, medians, and temporal metadata.
    2. Injects ``hours_since_onset`` and demographic values.
    3. Calls the appropriate extractor for each ``FeatureSpec``.
    4. Emits a row ``{stay_id, step_index, feature_id: value, ...}``.
    5. Optionally emits ``{feature_id}_missing`` binary columns for
       specs with ``include_missingness_flag=True``.

    Parameters
    ----------
    registry : dict[str, FeatureSpec]
        Feature contract loaded via ``load_feature_registry``.
    train_medians : dict[str, float]
        Median values fit on the training split only.  Pass ``{}`` during
        unit tests.
    emit_missingness_flags : bool
        When True, emit ``{feature_id}_missing`` columns alongside imputed
        values.  Defaults to True.
    """

    def __init__(
        self,
        registry: dict[str, FeatureSpec],
        train_medians: dict[str, float] | None = None,
        emit_missingness_flags: bool = True,
    ) -> None:
        self._registry = registry
        self._train_medians = train_medians or {}
        self._emit_flags = emit_missingness_flags
        self._extractors: dict[str, BaseWindowExtractor] = {
            fid: get_extractor_for_spec(spec) for fid, spec in registry.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        step_windows: list[StepWindowData],
    ) -> pl.DataFrame:
        """Build a state-vector DataFrame from a list of step windows.

        Parameters
        ----------
        step_windows:
            One ``StepWindowData`` per (stay_id, step_index), sorted by
            (stay_id, step_index) ascending.

        Returns
        -------
        pl.DataFrame
            One row per step.  Columns: ``stay_id``, ``step_index``,
            then one column per feature (and optional ``_missing`` flags).
        """
        if not step_windows:
            return self._empty_schema()

        rows: list[dict[str, Any]] = []

        # Group by stay_id to thread prior_values through each episode
        grouped: dict[int, list[StepWindowData]] = {}
        for sw in step_windows:
            grouped.setdefault(sw.stay_id, []).append(sw)

        for stay_id, windows in grouped.items():
            windows_sorted = sorted(windows, key=lambda w: w.step_index)
            prior_values: dict[str, float] = {}

            for sw in windows_sorted:
                row = self._process_step(sw, prior_values)
                rows.append(row)

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_step(
        self,
        sw: StepWindowData,
        prior_values: dict[str, float],
    ) -> dict[str, Any]:
        """Extract all features for one step window."""
        context = ExtractionContext(
            stay_id=sw.stay_id,
            step_index=sw.step_index,
            prior_values=prior_values,
            weight_kg=sw.weight_kg,
            train_medians=self._train_medians,
        )

        # Inject derived temporal feature before processing
        context.extracted_this_step["hours_since_onset"] = float(
            sw.hours_relative_to_onset
        )
        if sw.age_years is not None:
            context.extracted_this_step["age_years"] = sw.age_years

        row: dict[str, Any] = {
            "stay_id": sw.stay_id,
            "step_index": sw.step_index,
        }

        for fid, spec in self._registry.items():
            window_df = sw.get_df_for_table(spec.source_table)
            extractor = self._extractors[fid]

            before_extract = set(context.prior_values.keys())
            value = extractor.extract(window_df, spec, context)

            row[fid] = value

            if self._emit_flags and spec.include_missingness_flag:
                # True (1) when the value was imputed (no raw measurement)
                was_imputed = fid not in context.prior_values or (
                    fid in before_extract and context.prior_values.get(fid) == value
                )
                # Simpler heuristic: flag is set when window_df had no matching rows
                raw_present = self._had_raw_measurement(window_df, spec)
                row[f"{fid}_missing"] = int(not raw_present)

        # Carry forward updated priors to next step (mutate in place)
        prior_values.update(context.prior_values)

        return row

    @staticmethod
    def _had_raw_measurement(window_df: pl.DataFrame, spec: FeatureSpec) -> bool:
        """Return True if ``window_df`` contained at least one valid row."""
        if window_df.is_empty() or not spec.item_ids:
            return False
        if "itemid" not in window_df.columns:
            return False
        matching = window_df.filter(pl.col("itemid").is_in(list(spec.item_ids)))
        if matching.is_empty():
            return False
        value_col = "valuenum" if "valuenum" in matching.columns else "value"
        if value_col not in matching.columns:
            return False
        return matching[value_col].drop_nulls().len() > 0

    def _empty_schema(self) -> pl.DataFrame:
        """Return an empty DataFrame with the correct schema."""
        schema: dict[str, type] = {
            "stay_id": int,
            "step_index": int,
        }
        for fid, spec in self._registry.items():
            schema[fid] = float
            if self._emit_flags and spec.include_missingness_flag:
                schema[f"{fid}_missing"] = int
        return pl.DataFrame(schema={k: pl.Float64 for k in schema})


# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    "FeatureExtractor",
    "ExtractionContext",
    "BaseWindowExtractor",
    "ChartEventsExtractor",
    "LabEventsExtractor",
    "InputEventsExtractor",
    "OutputEventsExtractor",
    "DerivedExtractor",
    "DemographicsExtractor",
    "StepWindowData",
    "StateVectorBuilder",
    "get_extractor_for_spec",
]
