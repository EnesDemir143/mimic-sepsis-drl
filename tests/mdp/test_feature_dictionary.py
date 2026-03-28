"""
Tests for the Phase 4 feature dictionary contract and extraction wiring.

Covers:
- FEATURE_REGISTRY completeness and uniqueness
- Every FeatureSpec has required metadata (display_name, unit, description)
- Valid range constraints are consistent (low < high)
- Clip range constraints are consistent (low < high)
- normal_value is set when missing_strategy == NORMAL_VALUE
- Aggregation rules are valid enum values
- Missing strategies are valid enum values
- FEATURE_SPEC_VERSION is non-empty
- load_feature_registry filtering (include / exclude / flag override)
- validate_registry passes on the full default registry
- registry_summary reports correct totals
- BaseWindowExtractor range filter drops out-of-bounds rows
- BaseWindowExtractor aggregation rules (last, mean, max, min, sum)
- BaseWindowExtractor imputation strategies
  (forward_fill, median_train, zero, normal_value)
- BaseWindowExtractor clip applied after aggregation
- ChartEventsExtractor wired to correct source_table
- LabEventsExtractor wired to correct source_table
- InputEventsExtractor sums amount column
- OutputEventsExtractor sums value column
- DerivedExtractor computes pf_ratio, shock_index, hours_since_onset
- DemographicsExtractor reads from context
- get_extractor_for_spec dispatches correctly
- get_extractor_for_spec raises on unknown table
- ExtractionContext forward-fill state management
- StateVectorBuilder produces correct column set
- StateVectorBuilder threads prior_values across steps
- StateVectorBuilder emits missingness flags when configured
- StateVectorBuilder returns empty DataFrame on empty input
- FeatureSpec is immutable (frozen dataclass)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from mimic_sepsis_rl.mdp.features.dictionary import (
    FEATURE_REGISTRY,
    FEATURE_SPEC_VERSION,
    AggregationRule,
    FeatureFamily,
    FeatureSpec,
    MissingStrategy,
    load_feature_registry,
    registry_summary,
    validate_registry,
)
from mimic_sepsis_rl.mdp.features.extractors import (
    BaseWindowExtractor,
    ChartEventsExtractor,
    DemographicsExtractor,
    DerivedExtractor,
    ExtractionContext,
    InputEventsExtractor,
    LabEventsExtractor,
    OutputEventsExtractor,
    StateVectorBuilder,
    StepWindowData,
    get_extractor_for_spec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    feature_id: str = "test_feat",
    display_name: str = "Test Feature",
    family: FeatureFamily = FeatureFamily.VITALS,
    source_table: str = "chartevents",
    item_ids: tuple[int, ...] = (99999,),
    unit: str = "unit",
    aggregation: AggregationRule = AggregationRule.LAST,
    missing_strategy: MissingStrategy = MissingStrategy.FORWARD_FILL,
    valid_low: float | None = 0.0,
    valid_high: float | None = 500.0,
    clip_low: float | None = 0.0,
    clip_high: float | None = 400.0,
    normal_value: float | None = 100.0,
    include_missingness_flag: bool = False,
    description: str = "A test feature.",
) -> FeatureSpec:
    return FeatureSpec(
        feature_id=feature_id,
        display_name=display_name,
        family=family,
        source_table=source_table,
        item_ids=item_ids,
        unit=unit,
        aggregation=aggregation,
        missing_strategy=missing_strategy,
        valid_low=valid_low,
        valid_high=valid_high,
        clip_low=clip_low,
        clip_high=clip_high,
        normal_value=normal_value,
        include_missingness_flag=include_missingness_flag,
        description=description,
    )


def _make_context(
    stay_id: int = 1,
    step_index: int = 0,
    prior_values: dict[str, float] | None = None,
    train_medians: dict[str, float] | None = None,
    weight_kg: float | None = 70.0,
) -> ExtractionContext:
    return ExtractionContext(
        stay_id=stay_id,
        step_index=step_index,
        prior_values=prior_values or {},
        train_medians=train_medians or {},
        weight_kg=weight_kg,
    )


def _make_chart_df(item_ids: list[int], values: list[float]) -> pl.DataFrame:
    """Tiny chartevents-like DataFrame for unit tests."""
    base = datetime(2150, 1, 1, 12, 0, 0)
    return pl.DataFrame(
        {
            "itemid": item_ids,
            "valuenum": values,
            "charttime": [
                base + timedelta(minutes=i * 10) for i in range(len(item_ids))
            ],
        }
    )


def _empty_step_window(stay_id: int = 1, step_index: int = 0) -> StepWindowData:
    return StepWindowData(
        stay_id=stay_id,
        step_index=step_index,
        hours_relative_to_onset=float(step_index * 4 - 24),
        chartevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
        labevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
        inputevents=pl.DataFrame({"itemid": [], "amount": []}),
        outputevents=pl.DataFrame({"itemid": [], "value": []}),
        age_years=65.0,
        weight_kg=70.0,
    )


# ---------------------------------------------------------------------------
# 1. Registry metadata completeness
# ---------------------------------------------------------------------------


class TestRegistryMetadata:
    """Every entry in FEATURE_REGISTRY must meet basic metadata requirements."""

    def test_registry_is_non_empty(self):
        assert len(FEATURE_REGISTRY) > 0

    def test_all_feature_ids_are_unique(self):
        ids = list(FEATURE_REGISTRY.keys())
        assert len(ids) == len(set(ids)), "Duplicate feature_id found in registry"

    def test_feature_id_key_matches_spec_attribute(self):
        for fid, spec in FEATURE_REGISTRY.items():
            assert fid == spec.feature_id, (
                f"Key '{fid}' does not match spec.feature_id '{spec.feature_id}'"
            )

    def test_all_specs_have_non_empty_display_name(self):
        for fid, spec in FEATURE_REGISTRY.items():
            assert spec.display_name, f"{fid}: display_name is empty"

    def test_all_specs_have_non_empty_unit(self):
        for fid, spec in FEATURE_REGISTRY.items():
            assert spec.unit, f"{fid}: unit is empty"

    def test_all_specs_have_non_empty_description(self):
        for fid, spec in FEATURE_REGISTRY.items():
            assert spec.description, f"{fid}: description is empty"

    def test_all_feature_ids_are_snake_case(self):
        for fid in FEATURE_REGISTRY:
            assert fid == fid.lower(), f"feature_id '{fid}' is not lower-case"
            assert " " not in fid, f"feature_id '{fid}' contains a space"

    def test_all_aggregation_rules_are_valid_enum(self):
        valid = set(AggregationRule)
        for fid, spec in FEATURE_REGISTRY.items():
            assert spec.aggregation in valid, (
                f"{fid}: aggregation '{spec.aggregation}' is not a valid AggregationRule"
            )

    def test_all_missing_strategies_are_valid_enum(self):
        valid = set(MissingStrategy)
        for fid, spec in FEATURE_REGISTRY.items():
            assert spec.missing_strategy in valid, (
                f"{fid}: missing_strategy '{spec.missing_strategy}' is not valid"
            )

    def test_all_families_are_valid_enum(self):
        valid = set(FeatureFamily)
        for fid, spec in FEATURE_REGISTRY.items():
            assert spec.family in valid, (
                f"{fid}: family '{spec.family}' is not a valid FeatureFamily"
            )

    def test_all_source_tables_are_known(self):
        known = {
            "chartevents",
            "labevents",
            "inputevents",
            "outputevents",
            "patients",
            "derived",
        }
        for fid, spec in FEATURE_REGISTRY.items():
            assert spec.source_table in known, (
                f"{fid}: unknown source_table '{spec.source_table}'"
            )


# ---------------------------------------------------------------------------
# 2. Range constraint consistency
# ---------------------------------------------------------------------------


class TestRangeConstraints:
    """valid_low < valid_high and clip_low < clip_high for all specs."""

    def test_valid_range_low_less_than_high(self):
        for fid, spec in FEATURE_REGISTRY.items():
            if spec.valid_low is not None and spec.valid_high is not None:
                assert spec.valid_low < spec.valid_high, (
                    f"{fid}: valid_low ({spec.valid_low}) >= valid_high ({spec.valid_high})"
                )

    def test_clip_range_low_less_than_high(self):
        for fid, spec in FEATURE_REGISTRY.items():
            if spec.clip_low is not None and spec.clip_high is not None:
                assert spec.clip_low < spec.clip_high, (
                    f"{fid}: clip_low ({spec.clip_low}) >= clip_high ({spec.clip_high})"
                )

    def test_normal_value_set_when_required(self):
        """Features with NORMAL_VALUE strategy must declare a normal_value."""
        for fid, spec in FEATURE_REGISTRY.items():
            if spec.missing_strategy == MissingStrategy.NORMAL_VALUE:
                assert spec.normal_value is not None, (
                    f"{fid}: missing_strategy=NORMAL_VALUE but normal_value is None"
                )


# ---------------------------------------------------------------------------
# 3. Known features present
# ---------------------------------------------------------------------------


class TestKnownFeaturesPresent:
    """Spot-check that critical clinical features are in the registry."""

    @pytest.mark.parametrize(
        "fid",
        [
            "heart_rate",
            "map",
            "sbp",
            "resp_rate",
            "temperature",
            "spo2",
            "gcs_total",
            "lactate",
            "arterial_ph",
            "bicarbonate",
            "pao2",
            "creatinine",
            "bilirubin_total",
            "sodium",
            "potassium",
            "glucose",
            "wbc",
            "platelets",
            "haemoglobin",
            "cum_iv_fluid_ml",
            "cum_vasopressor_dose_nor_equiv",
            "urine_output_4h",
            "age_years",
            "weight_kg",
            "pf_ratio",
            "shock_index",
            "hours_since_onset",
        ],
    )
    def test_feature_present(self, fid: str):
        assert fid in FEATURE_REGISTRY, f"Expected feature '{fid}' not in registry"


# ---------------------------------------------------------------------------
# 4. Specific spec attributes for high-priority features
# ---------------------------------------------------------------------------


class TestCriticalFeatureAttributes:
    def test_lactate_uses_max_aggregation(self):
        assert FEATURE_REGISTRY["lactate"].aggregation == AggregationRule.MAX

    def test_spo2_uses_min_aggregation(self):
        assert FEATURE_REGISTRY["spo2"].aggregation == AggregationRule.MIN

    def test_cum_iv_fluid_uses_cumulative_aggregation(self):
        assert (
            FEATURE_REGISTRY["cum_iv_fluid_ml"].aggregation
            == AggregationRule.CUMULATIVE
        )

    def test_urine_output_uses_sum_aggregation(self):
        assert FEATURE_REGISTRY["urine_output_4h"].aggregation == AggregationRule.SUM

    def test_cum_iv_fluid_missing_strategy_is_zero(self):
        assert (
            FEATURE_REGISTRY["cum_iv_fluid_ml"].missing_strategy == MissingStrategy.ZERO
        )

    def test_vasopressor_missing_strategy_is_zero(self):
        assert (
            FEATURE_REGISTRY["cum_vasopressor_dose_nor_equiv"].missing_strategy
            == MissingStrategy.ZERO
        )

    def test_map_has_missingness_flag(self):
        assert FEATURE_REGISTRY["map"].include_missingness_flag is True

    def test_heart_rate_no_missingness_flag(self):
        assert FEATURE_REGISTRY["heart_rate"].include_missingness_flag is False

    def test_pf_ratio_family_is_derived(self):
        assert FEATURE_REGISTRY["pf_ratio"].family == FeatureFamily.DERIVED

    def test_age_years_family_is_demographics(self):
        assert FEATURE_REGISTRY["age_years"].family == FeatureFamily.DEMOGRAPHICS

    def test_derived_features_have_empty_item_ids(self):
        derived = ["pf_ratio", "shock_index", "hours_since_onset"]
        for fid in derived:
            spec = FEATURE_REGISTRY[fid]
            assert spec.item_ids == (), (
                f"Derived feature '{fid}' should have empty item_ids, got {spec.item_ids}"
            )

    def test_hours_since_onset_valid_range(self):
        spec = FEATURE_REGISTRY["hours_since_onset"]
        assert spec.valid_low == -24.0
        assert spec.valid_high == 48.0


# ---------------------------------------------------------------------------
# 5. FEATURE_SPEC_VERSION
# ---------------------------------------------------------------------------


class TestSpecVersion:
    def test_version_is_non_empty_string(self):
        assert FEATURE_SPEC_VERSION
        assert isinstance(FEATURE_SPEC_VERSION, str)

    def test_version_follows_semver_pattern(self):
        parts = FEATURE_SPEC_VERSION.split(".")
        assert len(parts) == 3, "Expected MAJOR.MINOR.PATCH format"
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' is not an integer"


# ---------------------------------------------------------------------------
# 6. validate_registry
# ---------------------------------------------------------------------------


class TestValidateRegistry:
    def test_default_registry_passes_validation(self):
        passed, errors = validate_registry()
        assert passed, f"Registry validation failed with errors: {errors}"

    def test_empty_registry_passes(self):
        passed, errors = validate_registry({})
        assert passed

    def test_spec_with_inverted_valid_range_fails(self):
        bad = _make_spec(valid_low=100.0, valid_high=10.0)
        passed, errors = validate_registry({bad.feature_id: bad})
        assert not passed
        assert any("valid_low" in e for e in errors)

    def test_spec_with_inverted_clip_range_fails(self):
        bad = _make_spec(clip_low=200.0, clip_high=50.0)
        passed, errors = validate_registry({bad.feature_id: bad})
        assert not passed
        assert any("clip_low" in e for e in errors)

    def test_spec_with_normal_value_strategy_but_no_value_fails(self):
        bad = _make_spec(
            missing_strategy=MissingStrategy.NORMAL_VALUE,
            normal_value=None,
        )
        passed, errors = validate_registry({bad.feature_id: bad})
        assert not passed
        assert any("normal_value" in e for e in errors)

    def test_spec_with_empty_description_fails(self):
        bad = _make_spec(description="")
        passed, errors = validate_registry({bad.feature_id: bad})
        assert not passed


# ---------------------------------------------------------------------------
# 7. load_feature_registry
# ---------------------------------------------------------------------------


class TestLoadFeatureRegistry:
    def test_no_config_returns_full_registry(self):
        reg = load_feature_registry()
        assert reg == FEATURE_REGISTRY

    def test_include_features_filters_correctly(self):
        reg = load_feature_registry({"include_features": ["heart_rate", "map"]})
        assert set(reg.keys()) == {"heart_rate", "map"}

    def test_include_features_unknown_id_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            load_feature_registry({"include_features": ["nonexistent_feature"]})

    def test_exclude_features_removes_entries(self):
        reg = load_feature_registry({"exclude_features": ["albumin", "base_excess"]})
        assert "albumin" not in reg
        assert "base_excess" not in reg
        assert "heart_rate" in reg

    def test_exclude_unknown_id_does_not_raise(self):
        # Should log a warning but not raise
        reg = load_feature_registry({"exclude_features": ["does_not_exist"]})
        assert len(reg) == len(FEATURE_REGISTRY)

    def test_missingness_flags_default_true_overrides_all(self):
        reg = load_feature_registry({"missingness_flags_default": True})
        for spec in reg.values():
            assert spec.include_missingness_flag is True

    def test_missingness_flags_default_false_overrides_all(self):
        reg = load_feature_registry({"missingness_flags_default": False})
        for spec in reg.values():
            assert spec.include_missingness_flag is False

    def test_missingness_flags_default_null_respects_per_feature(self):
        reg = load_feature_registry({"missingness_flags_default": None})
        # map should have flag=True, heart_rate should have flag=False
        assert reg["map"].include_missingness_flag is True
        assert reg["heart_rate"].include_missingness_flag is False

    def test_include_and_exclude_combined(self):
        reg = load_feature_registry(
            {
                "include_features": ["heart_rate", "map", "lactate"],
                "exclude_features": ["lactate"],
            }
        )
        assert "lactate" not in reg
        assert "heart_rate" in reg
        assert "map" in reg


# ---------------------------------------------------------------------------
# 8. registry_summary
# ---------------------------------------------------------------------------


class TestRegistrySummary:
    def test_total_features_matches_registry(self):
        summary = registry_summary()
        assert summary["total_features"] == len(FEATURE_REGISTRY)

    def test_spec_version_in_summary(self):
        summary = registry_summary()
        assert summary["spec_version"] == FEATURE_SPEC_VERSION

    def test_families_cover_all_entries(self):
        summary = registry_summary()
        total_in_families = sum(summary["families"].values())
        assert total_in_families == len(FEATURE_REGISTRY)

    def test_source_tables_cover_all_entries(self):
        summary = registry_summary()
        total_in_tables = sum(summary["source_tables"].values())
        assert total_in_tables == len(FEATURE_REGISTRY)

    def test_missingness_flag_features_are_subset(self):
        summary = registry_summary()
        flag_ids = set(summary["missingness_flag_features"])
        all_ids = set(FEATURE_REGISTRY.keys())
        assert flag_ids <= all_ids

    def test_missingness_flag_count_matches_list(self):
        summary = registry_summary()
        assert summary["total_missingness_flag_features"] == len(
            summary["missingness_flag_features"]
        )


# ---------------------------------------------------------------------------
# 9. FeatureSpec immutability
# ---------------------------------------------------------------------------


class TestFeatureSpecImmutability:
    def test_frozen_dataclass_rejects_attribute_mutation(self):
        spec = _make_spec()
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            spec.feature_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 10. ExtractionContext
# ---------------------------------------------------------------------------


class TestExtractionContext:
    def test_get_prior_returns_none_for_unseen_feature(self):
        ctx = _make_context()
        assert ctx.get_prior("heart_rate") is None

    def test_record_updates_prior_values(self):
        ctx = _make_context()
        ctx.record("heart_rate", 80.0)
        assert ctx.get_prior("heart_rate") == pytest.approx(80.0)

    def test_record_none_does_not_update_prior_values(self):
        ctx = _make_context()
        ctx.record("heart_rate", None)
        assert ctx.get_prior("heart_rate") is None

    def test_get_current_returns_value_set_this_step(self):
        ctx = _make_context()
        ctx.extracted_this_step["pao2"] = 90.0
        assert ctx.get_current("pao2") == pytest.approx(90.0)

    def test_get_current_returns_none_for_unseen(self):
        ctx = _make_context()
        assert ctx.get_current("nonexistent") is None


# ---------------------------------------------------------------------------
# 11. BaseWindowExtractor — range filter
# ---------------------------------------------------------------------------


class TestBaseWindowExtractorRangeFilter:
    class _Impl(BaseWindowExtractor):
        """Minimal concrete subclass to test base class logic."""

        def _extract_raw(self, item_df, spec, context):
            if item_df.is_empty():
                return None
            return float(item_df["valuenum"].drop_nulls()[-1])

    def test_values_below_valid_low_are_dropped(self):
        spec = _make_spec(
            valid_low=0.0, valid_high=300.0, clip_low=0.0, clip_high=300.0
        )
        df = _make_chart_df([99999, 99999], [-5.0, 100.0])
        extractor = self._Impl()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(100.0)

    def test_values_above_valid_high_are_dropped(self):
        spec = _make_spec(
            valid_low=0.0, valid_high=300.0, clip_low=0.0, clip_high=300.0
        )
        df = _make_chart_df([99999, 99999], [80.0, 9999.0])
        extractor = self._Impl()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(80.0)

    def test_all_values_out_of_range_triggers_imputation(self):
        spec = _make_spec(
            valid_low=0.0,
            valid_high=300.0,
            clip_low=0.0,
            clip_high=300.0,
            missing_strategy=MissingStrategy.NORMAL_VALUE,
            normal_value=75.0,
        )
        df = _make_chart_df([99999], [9999.0])
        extractor = self._Impl()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# 12. BaseWindowExtractor — aggregation rules
# ---------------------------------------------------------------------------


class TestBaseWindowExtractorAggregation:
    class _DirectImpl(BaseWindowExtractor):
        """Uses _aggregate directly on filtered series."""

        def _extract_raw(self, item_df, spec, context):
            if item_df.is_empty():
                return None
            if "valuenum" not in item_df.columns:
                return None
            return self._aggregate(item_df["valuenum"].drop_nulls(), spec.aggregation)

    @pytest.mark.parametrize(
        "rule, values, expected",
        [
            (AggregationRule.LAST, [10.0, 20.0, 30.0], 30.0),
            (AggregationRule.MEAN, [10.0, 20.0, 30.0], 20.0),
            (AggregationRule.MAX, [10.0, 20.0, 30.0], 30.0),
            (AggregationRule.MIN, [10.0, 20.0, 30.0], 10.0),
            (AggregationRule.SUM, [10.0, 20.0, 30.0], 60.0),
            (AggregationRule.CUMULATIVE, [10.0, 20.0, 30.0], 60.0),
        ],
    )
    def test_aggregation_rule(self, rule, values, expected):
        spec = _make_spec(
            aggregation=rule,
            valid_low=None,
            valid_high=None,
            clip_low=None,
            clip_high=None,
        )
        df = _make_chart_df([99999] * len(values), values)
        extractor = self._DirectImpl()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(expected)

    def test_empty_series_returns_none_from_aggregate(self):
        impl = TestBaseWindowExtractorAggregation._DirectImpl()
        series = pl.Series("valuenum", [], dtype=pl.Float64)
        result = impl._aggregate(series, AggregationRule.LAST)
        assert result is None


# ---------------------------------------------------------------------------
# 13. BaseWindowExtractor — imputation strategies
# ---------------------------------------------------------------------------


class TestBaseWindowExtractorImputation:
    class _AlwaysMissImpl(BaseWindowExtractor):
        """Simulates a missing value by always returning None from _extract_raw."""

        def _extract_raw(self, item_df, spec, context):
            return None

    def test_forward_fill_uses_prior_value(self):
        spec = _make_spec(
            missing_strategy=MissingStrategy.FORWARD_FILL,
            clip_low=None,
            clip_high=None,
        )
        ctx = _make_context(prior_values={"test_feat": 88.0})
        extractor = self._AlwaysMissImpl()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(88.0)

    def test_forward_fill_falls_back_to_median_when_no_prior(self):
        spec = _make_spec(
            missing_strategy=MissingStrategy.FORWARD_FILL,
            clip_low=None,
            clip_high=None,
            normal_value=None,
        )
        ctx = _make_context(train_medians={"test_feat": 55.0})
        extractor = self._AlwaysMissImpl()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(55.0)

    def test_forward_fill_falls_back_to_normal_value_last(self):
        spec = _make_spec(
            missing_strategy=MissingStrategy.FORWARD_FILL,
            clip_low=None,
            clip_high=None,
            normal_value=42.0,
        )
        ctx = _make_context()  # no prior, no median
        extractor = self._AlwaysMissImpl()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(42.0)

    def test_median_train_uses_train_median(self):
        spec = _make_spec(
            missing_strategy=MissingStrategy.MEDIAN_TRAIN,
            clip_low=None,
            clip_high=None,
        )
        ctx = _make_context(train_medians={"test_feat": 33.5})
        extractor = self._AlwaysMissImpl()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(33.5)

    def test_zero_imputation(self):
        spec = _make_spec(
            missing_strategy=MissingStrategy.ZERO,
            clip_low=None,
            clip_high=None,
        )
        ctx = _make_context()
        extractor = self._AlwaysMissImpl()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(0.0)

    def test_normal_value_imputation(self):
        spec = _make_spec(
            missing_strategy=MissingStrategy.NORMAL_VALUE,
            normal_value=37.0,
            clip_low=None,
            clip_high=None,
        )
        ctx = _make_context()
        extractor = self._AlwaysMissImpl()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(37.0)


# ---------------------------------------------------------------------------
# 14. BaseWindowExtractor — clipping
# ---------------------------------------------------------------------------


class TestBaseWindowExtractorClipping:
    class _FixedImpl(BaseWindowExtractor):
        """Returns a fixed raw value for testing clip logic."""

        def __init__(self, raw_value: float):
            self._raw = raw_value

        def _extract_raw(self, item_df, spec, context):
            return self._raw

    def test_value_clipped_to_clip_high(self):
        spec = _make_spec(
            valid_low=None,
            valid_high=None,
            clip_low=0.0,
            clip_high=100.0,
        )
        extractor = self._FixedImpl(raw_value=500.0)
        ctx = _make_context()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(100.0)

    def test_value_clipped_to_clip_low(self):
        spec = _make_spec(
            valid_low=None,
            valid_high=None,
            clip_low=10.0,
            clip_high=200.0,
        )
        extractor = self._FixedImpl(raw_value=-5.0)
        ctx = _make_context()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(10.0)

    def test_value_within_clip_range_unchanged(self):
        spec = _make_spec(
            valid_low=None,
            valid_high=None,
            clip_low=0.0,
            clip_high=100.0,
        )
        extractor = self._FixedImpl(raw_value=50.0)
        ctx = _make_context()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 15. ChartEventsExtractor
# ---------------------------------------------------------------------------


class TestChartEventsExtractor:
    def test_extracts_last_value_sorted_by_charttime(self):
        spec = FEATURE_REGISTRY["heart_rate"]
        df = _make_chart_df([220045, 220045, 220045], [60.0, 80.0, 100.0])
        extractor = ChartEventsExtractor()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(100.0)

    def test_returns_imputed_when_no_matching_items(self):
        spec = FEATURE_REGISTRY["heart_rate"]
        df = _make_chart_df([99999], [80.0])  # wrong item ID
        extractor = ChartEventsExtractor()
        ctx = _make_context(prior_values={"heart_rate": 72.0})
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(72.0)  # forward-fill from prior

    def test_source_table_is_chartevents(self):
        assert FEATURE_REGISTRY["heart_rate"].source_table == "chartevents"
        assert FEATURE_REGISTRY["temperature"].source_table == "chartevents"
        assert FEATURE_REGISTRY["gcs_total"].source_table == "chartevents"


# ---------------------------------------------------------------------------
# 16. LabEventsExtractor
# ---------------------------------------------------------------------------


class TestLabEventsExtractor:
    def test_extracts_max_lactate(self):
        spec = FEATURE_REGISTRY["lactate"]
        df = _make_chart_df([50813, 50813, 50813], [1.5, 4.2, 2.0])
        extractor = LabEventsExtractor()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(4.2)

    def test_source_table_is_labevents(self):
        for fid in ["lactate", "creatinine", "bilirubin_total", "wbc", "platelets"]:
            assert FEATURE_REGISTRY[fid].source_table == "labevents"


# ---------------------------------------------------------------------------
# 17. InputEventsExtractor
# ---------------------------------------------------------------------------


class TestInputEventsExtractor:
    def test_sums_amount_column(self):
        spec = FEATURE_REGISTRY["cum_iv_fluid_ml"]
        df = pl.DataFrame(
            {
                "itemid": [225158, 225158, 220949],
                "amount": [500.0, 250.0, 1000.0],
            }
        )
        extractor = InputEventsExtractor()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(1750.0)

    def test_zero_imputation_on_empty_input(self):
        spec = FEATURE_REGISTRY["cum_iv_fluid_ml"]
        df = pl.DataFrame({"itemid": [], "amount": []})
        extractor = InputEventsExtractor()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(0.0)  # zero missing strategy


# ---------------------------------------------------------------------------
# 18. OutputEventsExtractor
# ---------------------------------------------------------------------------


class TestOutputEventsExtractor:
    def test_sums_value_column(self):
        spec = FEATURE_REGISTRY["urine_output_4h"]
        df = pl.DataFrame(
            {
                "itemid": [226559, 226559],
                "value": [150.0, 200.0],
            }
        )
        extractor = OutputEventsExtractor()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(350.0)

    def test_zero_imputation_on_empty(self):
        spec = FEATURE_REGISTRY["urine_output_4h"]
        df = pl.DataFrame({"itemid": [], "value": []})
        extractor = OutputEventsExtractor()
        ctx = _make_context()
        result = extractor.extract(df, spec, ctx)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 19. DerivedExtractor
# ---------------------------------------------------------------------------


class TestDerivedExtractor:
    def test_pf_ratio_computed_correctly(self):
        spec = FEATURE_REGISTRY["pf_ratio"]
        ctx = _make_context()
        ctx.extracted_this_step["pao2"] = 150.0
        ctx.extracted_this_step["fio2_vent"] = 0.5
        extractor = DerivedExtractor()
        result = extractor.extract(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(300.0)

    def test_pf_ratio_returns_none_when_fio2_missing(self):
        spec = FEATURE_REGISTRY["pf_ratio"]
        ctx = _make_context()
        ctx.extracted_this_step["pao2"] = 150.0
        # fio2_vent not set
        extractor = DerivedExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result is None

    def test_pf_ratio_returns_none_when_pao2_missing(self):
        spec = FEATURE_REGISTRY["pf_ratio"]
        ctx = _make_context()
        ctx.extracted_this_step["fio2_vent"] = 0.4
        # pao2 not set
        extractor = DerivedExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result is None

    def test_pf_ratio_returns_none_when_fio2_zero(self):
        spec = FEATURE_REGISTRY["pf_ratio"]
        ctx = _make_context()
        ctx.extracted_this_step["pao2"] = 100.0
        ctx.extracted_this_step["fio2_vent"] = 0.0
        extractor = DerivedExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result is None

    def test_shock_index_computed_correctly(self):
        spec = FEATURE_REGISTRY["shock_index"]
        ctx = _make_context()
        ctx.extracted_this_step["heart_rate"] = 120.0
        ctx.extracted_this_step["sbp"] = 80.0
        extractor = DerivedExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(1.5)

    def test_shock_index_returns_none_when_sbp_zero(self):
        spec = FEATURE_REGISTRY["shock_index"]
        ctx = _make_context()
        ctx.extracted_this_step["heart_rate"] = 100.0
        ctx.extracted_this_step["sbp"] = 0.0
        extractor = DerivedExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result is None

    def test_hours_since_onset_reads_from_context(self):
        spec = FEATURE_REGISTRY["hours_since_onset"]
        ctx = _make_context()
        ctx.extracted_this_step["hours_since_onset"] = -8.0
        extractor = DerivedExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(-8.0)


# ---------------------------------------------------------------------------
# 20. DemographicsExtractor
# ---------------------------------------------------------------------------


class TestDemographicsExtractor:
    def test_weight_kg_read_from_context(self):
        spec = FEATURE_REGISTRY["weight_kg"]
        ctx = _make_context(weight_kg=85.0)
        extractor = DemographicsExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(85.0)

    def test_age_years_read_from_context(self):
        spec = FEATURE_REGISTRY["age_years"]
        ctx = _make_context()
        ctx.extracted_this_step["age_years"] = 68.0
        extractor = DemographicsExtractor()
        result = extractor._extract_raw(pl.DataFrame(), spec, ctx)
        assert result == pytest.approx(68.0)


# ---------------------------------------------------------------------------
# 21. get_extractor_for_spec dispatch
# ---------------------------------------------------------------------------


class TestGetExtractorForSpec:
    @pytest.mark.parametrize(
        "table,expected_type",
        [
            ("chartevents", ChartEventsExtractor),
            ("labevents", LabEventsExtractor),
            ("inputevents", InputEventsExtractor),
            ("outputevents", OutputEventsExtractor),
            ("derived", DerivedExtractor),
            ("patients", DemographicsExtractor),
        ],
    )
    def test_dispatch_returns_correct_type(self, table, expected_type):
        spec = _make_spec(source_table=table)
        extractor = get_extractor_for_spec(spec)
        assert isinstance(extractor, expected_type)

    def test_unknown_table_raises_value_error(self):
        spec = _make_spec(source_table="unknown_table")
        with pytest.raises(ValueError, match="unknown_table"):
            get_extractor_for_spec(spec)


# ---------------------------------------------------------------------------
# 22. StateVectorBuilder
# ---------------------------------------------------------------------------


class TestStateVectorBuilder:
    def _minimal_registry(self) -> dict[str, FeatureSpec]:
        """Tiny registry with one chartevents feature and one derived feature."""
        return {
            "heart_rate": FEATURE_REGISTRY["heart_rate"],
            "hours_since_onset": FEATURE_REGISTRY["hours_since_onset"],
        }

    def test_empty_input_returns_empty_dataframe(self):
        builder = StateVectorBuilder(self._minimal_registry())
        result = builder.build([])
        assert isinstance(result, pl.DataFrame)

    def test_single_step_produces_one_row(self):
        builder = StateVectorBuilder(
            self._minimal_registry(), emit_missingness_flags=False
        )
        sw = _empty_step_window(stay_id=1, step_index=0)
        result = builder.build([sw])
        assert result.height == 1

    def test_output_has_stay_id_and_step_index_columns(self):
        builder = StateVectorBuilder(
            self._minimal_registry(), emit_missingness_flags=False
        )
        sw = _empty_step_window()
        result = builder.build([sw])
        assert "stay_id" in result.columns
        assert "step_index" in result.columns

    def test_output_has_feature_columns(self):
        registry = self._minimal_registry()
        builder = StateVectorBuilder(registry, emit_missingness_flags=False)
        sw = _empty_step_window()
        result = builder.build([sw])
        for fid in registry:
            assert fid in result.columns, f"Column '{fid}' missing from output"

    def test_hours_since_onset_is_injected_correctly(self):
        registry = {"hours_since_onset": FEATURE_REGISTRY["hours_since_onset"]}
        builder = StateVectorBuilder(registry, emit_missingness_flags=False)
        sw = _empty_step_window(step_index=3)  # step 3 → offset = 3*4-24 = -12h
        result = builder.build([sw])
        assert result["hours_since_onset"][0] == pytest.approx(-12.0)

    def test_forward_fill_threads_across_steps(self):
        """A value extracted in step 0 should forward-fill into step 1."""
        spec = FEATURE_REGISTRY["heart_rate"]
        registry = {"heart_rate": spec}
        builder = StateVectorBuilder(registry, emit_missingness_flags=False)

        # Step 0: heart rate is 90
        step0 = StepWindowData(
            stay_id=1,
            step_index=0,
            hours_relative_to_onset=-24.0,
            chartevents=_make_chart_df([220045], [90.0]),
            labevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
            inputevents=pl.DataFrame({"itemid": [], "amount": []}),
            outputevents=pl.DataFrame({"itemid": [], "value": []}),
        )
        # Step 1: no heart rate measurement → should forward-fill 90
        step1 = StepWindowData(
            stay_id=1,
            step_index=1,
            hours_relative_to_onset=-20.0,
            chartevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
            labevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
            inputevents=pl.DataFrame({"itemid": [], "amount": []}),
            outputevents=pl.DataFrame({"itemid": [], "value": []}),
        )

        result = builder.build([step0, step1])
        assert result.height == 2
        row0_hr = result.filter(pl.col("step_index") == 0)["heart_rate"][0]
        row1_hr = result.filter(pl.col("step_index") == 1)["heart_rate"][0]
        assert row0_hr == pytest.approx(90.0)
        assert row1_hr == pytest.approx(90.0)

    def test_missingness_flags_emitted_for_flagged_features(self):
        registry = {"map": FEATURE_REGISTRY["map"]}
        builder = StateVectorBuilder(registry, emit_missingness_flags=True)
        sw = _empty_step_window()
        result = builder.build([sw])
        assert "map_missing" in result.columns

    def test_no_missingness_flags_when_disabled(self):
        registry = {"map": FEATURE_REGISTRY["map"]}
        builder = StateVectorBuilder(registry, emit_missingness_flags=False)
        sw = _empty_step_window()
        result = builder.build([sw])
        assert "map_missing" not in result.columns

    def test_multiple_episodes_do_not_share_prior_values(self):
        """Prior values from stay 1 must not bleed into stay 2."""
        registry = {"heart_rate": FEATURE_REGISTRY["heart_rate"]}
        builder = StateVectorBuilder(registry, emit_missingness_flags=False)

        step_stay1 = StepWindowData(
            stay_id=1,
            step_index=0,
            hours_relative_to_onset=-24.0,
            chartevents=_make_chart_df([220045], [120.0]),
            labevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
            inputevents=pl.DataFrame({"itemid": [], "amount": []}),
            outputevents=pl.DataFrame({"itemid": [], "value": []}),
        )
        step_stay2 = StepWindowData(
            stay_id=2,
            step_index=0,
            hours_relative_to_onset=-24.0,
            chartevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
            labevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
            inputevents=pl.DataFrame({"itemid": [], "amount": []}),
            outputevents=pl.DataFrame({"itemid": [], "value": []}),
        )

        result = builder.build([step_stay1, step_stay2])
        hr_stay2 = result.filter(pl.col("stay_id") == 2)["heart_rate"][0]
        hr_stay1 = result.filter(pl.col("stay_id") == 1)["heart_rate"][0]

        assert hr_stay1 == pytest.approx(120.0)
        # Stay 2 has no measurement and no prior → must use normal_value (75)
        assert hr_stay2 == pytest.approx(75.0)

    def test_correct_number_of_rows_for_multi_step_episode(self):
        registry = {"heart_rate": FEATURE_REGISTRY["heart_rate"]}
        builder = StateVectorBuilder(registry, emit_missingness_flags=False)
        windows = [_empty_step_window(stay_id=42, step_index=i) for i in range(6)]
        result = builder.build(windows)
        assert result.height == 6

    def test_stay_id_and_step_index_preserved(self):
        registry = {"heart_rate": FEATURE_REGISTRY["heart_rate"]}
        builder = StateVectorBuilder(registry, emit_missingness_flags=False)
        windows = [_empty_step_window(stay_id=99, step_index=i) for i in range(3)]
        result = builder.build(windows)
        assert list(result["stay_id"]) == [99, 99, 99]
        assert list(result["step_index"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 23. StepWindowData — table dispatch
# ---------------------------------------------------------------------------


class TestStepWindowData:
    def test_get_df_for_table_returns_correct_df(self):
        sw = _empty_step_window()
        chart = sw.get_df_for_table("chartevents")
        lab = sw.get_df_for_table("labevents")
        assert isinstance(chart, pl.DataFrame)
        assert isinstance(lab, pl.DataFrame)

    def test_get_df_for_unknown_table_returns_empty(self):
        sw = _empty_step_window()
        result = sw.get_df_for_table("unknown_table")
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()
