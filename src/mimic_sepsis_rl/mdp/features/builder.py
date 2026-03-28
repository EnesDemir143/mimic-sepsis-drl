"""
Deterministic state-table construction for Phase 4.

This module lifts the low-level extractor scaffolding into a split-aware
builder that materialises one row per episode step. It keeps the project's
missing-data contract explicit:

1. Use observed measurements within the current step when available.
2. Forward-fill from earlier steps in the same episode when configured.
3. Fall back to train-only medians when no prior value exists.
4. Emit optional missingness flags based on whether a raw measurement was
   present in the current step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import polars as pl

from mimic_sepsis_rl.data.split_models import SplitLabel, SplitManifest
from mimic_sepsis_rl.mdp.features.dictionary import FeatureSpec
from mimic_sepsis_rl.mdp.features.extractors import (
    BaseWindowExtractor,
    ExtractionContext,
    StepWindowData,
    get_extractor_for_spec,
)


@dataclass(frozen=True)
class _ResolvedStepWindow:
    """Step window paired with the subject and split it belongs to."""

    window: StepWindowData
    subject_id: int | None
    split_label: SplitLabel | None


class StateTableBuilder:
    """Build a deterministic step-level state table from episode windows."""

    def __init__(
        self,
        registry: Mapping[str, FeatureSpec],
        train_medians: Mapping[str, float] | None = None,
        emit_missingness_flags: bool = True,
        impute_missing: bool = True,
    ) -> None:
        self._registry = dict(registry)
        self._train_medians = {key: float(value) for key, value in (train_medians or {}).items()}
        self._emit_flags = emit_missingness_flags
        self._impute_missing = impute_missing
        self._extractors: dict[str, BaseWindowExtractor] = {
            feature_id: get_extractor_for_spec(spec)
            for feature_id, spec in self._registry.items()
        }

    def build(
        self,
        step_windows: list[StepWindowData],
        split_manifest: SplitManifest | None = None,
        stay_to_subject_id: Mapping[int, int] | None = None,
    ) -> pl.DataFrame:
        """Build one state row per step window."""
        resolved_steps = self._resolve_step_windows(
            step_windows=step_windows,
            split_manifest=split_manifest,
            stay_to_subject_id=stay_to_subject_id,
        )
        if not resolved_steps:
            return self._empty_schema(
                include_subject_id=split_manifest is not None or stay_to_subject_id is not None,
                include_split=split_manifest is not None,
            )

        grouped: dict[int, list[_ResolvedStepWindow]] = {}
        for resolved in resolved_steps:
            grouped.setdefault(resolved.window.stay_id, []).append(resolved)

        rows: list[dict[str, Any]] = []
        for stay_id in sorted(grouped):
            stay_windows = sorted(
                grouped[stay_id],
                key=lambda item: (item.window.step_index, item.window.hours_relative_to_onset),
            )
            self._validate_episode_windows(stay_id, stay_windows)
            prior_values: dict[str, float] = {}
            for resolved in stay_windows:
                rows.append(self._process_step(resolved, prior_values))

        sort_columns = ["stay_id", "step_index"]
        if "subject_id" in rows[0]:
            sort_columns = ["subject_id", *sort_columns]
        return pl.DataFrame(rows).sort(sort_columns)

    def _resolve_step_windows(
        self,
        step_windows: list[StepWindowData],
        split_manifest: SplitManifest | None,
        stay_to_subject_id: Mapping[int, int] | None,
    ) -> list[_ResolvedStepWindow]:
        resolved: list[_ResolvedStepWindow] = []
        for window in step_windows:
            subject_id = window.subject_id
            if subject_id is None and stay_to_subject_id is not None:
                subject_id = stay_to_subject_id.get(window.stay_id)

            split_label: SplitLabel | None = None
            if split_manifest is not None:
                if subject_id is None:
                    raise ValueError(
                        "split_manifest was provided but no subject_id could be "
                        f"resolved for stay_id={window.stay_id}."
                    )
                split_label = split_manifest.split_for(subject_id)
                if split_label is None:
                    raise ValueError(
                        f"subject_id={subject_id} for stay_id={window.stay_id} "
                        "is not present in the provided split manifest."
                    )

            resolved.append(
                _ResolvedStepWindow(
                    window=window,
                    subject_id=subject_id,
                    split_label=split_label,
                )
            )
        return resolved

    @staticmethod
    def _validate_episode_windows(
        stay_id: int,
        stay_windows: list[_ResolvedStepWindow],
    ) -> None:
        seen_step_indices: set[int] = set()
        subject_ids = {
            resolved.subject_id
            for resolved in stay_windows
            if resolved.subject_id is not None
        }
        if len(subject_ids) > 1:
            raise ValueError(
                f"stay_id={stay_id} resolved to multiple subject_ids: {sorted(subject_ids)}."
            )

        for resolved in stay_windows:
            step_index = resolved.window.step_index
            if step_index in seen_step_indices:
                raise ValueError(
                    f"Duplicate step_index={step_index} encountered for stay_id={stay_id}."
                )
            seen_step_indices.add(step_index)

    def _process_step(
        self,
        resolved: _ResolvedStepWindow,
        prior_values: dict[str, float],
    ) -> dict[str, Any]:
        window = resolved.window
        context = ExtractionContext(
            stay_id=window.stay_id,
            step_index=window.step_index,
            prior_values=prior_values,
            weight_kg=window.weight_kg,
            train_medians=dict(self._train_medians),
        )
        context.extracted_this_step["hours_since_onset"] = float(
            window.hours_relative_to_onset
        )
        if window.age_years is not None:
            context.extracted_this_step["age_years"] = float(window.age_years)

        row: dict[str, Any] = {
            "stay_id": window.stay_id,
            "step_index": window.step_index,
        }
        if resolved.subject_id is not None:
            row["subject_id"] = resolved.subject_id
        if resolved.split_label is not None:
            row["split"] = resolved.split_label.value

        for feature_id, spec in self._registry.items():
            source_df = window.get_df_for_table(spec.source_table)
            raw_present = self._had_raw_measurement(source_df, spec)
            extractor = self._extractors[feature_id]
            value = extractor.extract(
                source_df,
                spec,
                context,
                impute=self._impute_missing,
            )
            row[feature_id] = value
            if self._emit_flags and spec.include_missingness_flag:
                row[f"{feature_id}_missing"] = int(not raw_present)

        prior_values.update(context.prior_values)
        return row

    @staticmethod
    def _had_raw_measurement(window_df: pl.DataFrame, spec: FeatureSpec) -> bool:
        """Return True when the current step contains a usable raw value."""
        if window_df.is_empty() or not spec.item_ids:
            return False
        if "itemid" not in window_df.columns:
            return False

        matching = window_df.filter(pl.col("itemid").is_in(list(spec.item_ids)))
        if matching.is_empty():
            return False

        value_column = None
        for candidate in ("valuenum", "value", "amount"):
            if candidate in matching.columns:
                value_column = candidate
                break
        if value_column is None:
            return False

        filtered = matching.get_column(value_column).drop_nulls()
        if filtered.is_empty():
            return False

        if spec.valid_low is not None:
            filtered = filtered.filter(filtered >= spec.valid_low)
        if spec.valid_high is not None:
            filtered = filtered.filter(filtered <= spec.valid_high)
        return not filtered.is_empty()

    def _empty_schema(
        self,
        include_subject_id: bool,
        include_split: bool,
    ) -> pl.DataFrame:
        schema: dict[str, pl.DataType] = {
            "stay_id": pl.Int64,
            "step_index": pl.Int64,
        }
        if include_subject_id:
            schema["subject_id"] = pl.Int64
        if include_split:
            schema["split"] = pl.Utf8
        for feature_id, spec in self._registry.items():
            schema[feature_id] = pl.Float64
            if self._emit_flags and spec.include_missingness_flag:
                schema[f"{feature_id}_missing"] = pl.Int64
        return pl.DataFrame(schema=schema)


def build_state_table(
    step_windows: list[StepWindowData],
    registry: Mapping[str, FeatureSpec],
    split_manifest: SplitManifest | None = None,
    stay_to_subject_id: Mapping[int, int] | None = None,
    train_medians: Mapping[str, float] | None = None,
    emit_missingness_flags: bool = True,
    impute_missing: bool = True,
) -> pl.DataFrame:
    """Convenience wrapper around :class:`StateTableBuilder`."""
    builder = StateTableBuilder(
        registry=registry,
        train_medians=train_medians,
        emit_missingness_flags=emit_missingness_flags,
        impute_missing=impute_missing,
    )
    return builder.build(
        step_windows=step_windows,
        split_manifest=split_manifest,
        stay_to_subject_id=stay_to_subject_id,
    )


__all__ = ["StateTableBuilder", "build_state_table"]
