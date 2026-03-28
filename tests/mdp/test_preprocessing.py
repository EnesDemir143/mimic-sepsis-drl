from __future__ import annotations

import math

import polars as pl
import pytest

from mimic_sepsis_rl.data.split_models import SplitManifest
from mimic_sepsis_rl.mdp.features.dictionary import (
    AggregationRule,
    FeatureFamily,
    FeatureSpec,
    MissingStrategy,
)
from mimic_sepsis_rl.mdp.preprocessing import (
    fit_preprocessing_artifacts,
    fit_train_feature_medians,
    load_preprocessing_artifacts,
    save_preprocessing_artifacts,
    transform_state_table,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        spec_version="1.0.0",
        seed=42,
        source_episode_set="tests",
        train_ids=frozenset({1}),
        validation_ids=frozenset({2}),
        test_ids=frozenset({3}),
    )


def _registry() -> dict[str, FeatureSpec]:
    return {
        "heart_rate": FeatureSpec(
            feature_id="heart_rate",
            display_name="Heart Rate",
            family=FeatureFamily.VITALS,
            source_table="chartevents",
            item_ids=(220045,),
            unit="bpm",
            aggregation=AggregationRule.LAST,
            missing_strategy=MissingStrategy.FORWARD_FILL,
            valid_low=20.0,
            valid_high=250.0,
            clip_low=0.0,
            clip_high=100.0,
            normal_value=75.0,
            include_missingness_flag=False,
            description="Test feature.",
        )
    }


def test_fit_train_feature_medians_ignores_validation_and_test_rows():
    state_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 3],
            "stay_id": [11, 11, 11, 22, 33],
            "step_index": [0, 1, 2, 0, 0],
            "heart_rate": [10.0, 20.0, 30.0, 999.0, 888.0],
        }
    )

    medians = fit_train_feature_medians(state_df, _registry(), _manifest())
    assert medians["heart_rate"] == pytest.approx(20.0)


def test_fit_preprocessing_artifacts_uses_train_only_statistics():
    state_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2],
            "stay_id": [11, 11, 11, 22],
            "step_index": [0, 1, 2, 0],
            "heart_rate": [10.0, 20.0, 500.0, 1000.0],
        }
    )

    artifacts = fit_preprocessing_artifacts(
        state_df=state_df,
        registry=_registry(),
        split_manifest=_manifest(),
        train_medians={"heart_rate": 20.0},
    )

    transform = artifacts.transforms["heart_rate"]
    expected_values = [10.0, 20.0, 100.0]
    expected_mean = sum(expected_values) / len(expected_values)
    expected_scale = math.sqrt(
        sum((value - expected_mean) ** 2 for value in expected_values)
        / len(expected_values)
    )

    assert transform.mean == pytest.approx(expected_mean)
    assert transform.scale == pytest.approx(expected_scale)
    assert transform.clip_high == pytest.approx(100.0)


def test_transform_state_table_round_trips_through_serialization(tmp_path):
    state_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "stay_id": [11, 11, 22],
            "step_index": [0, 1, 0],
            "heart_rate": [10.0, 30.0, 90.0],
        }
    )
    artifacts = fit_preprocessing_artifacts(
        state_df=state_df,
        registry=_registry(),
        split_manifest=_manifest(),
        train_medians={"heart_rate": 20.0},
    )
    output_path = tmp_path / "preprocessing.json"

    save_preprocessing_artifacts(artifacts, output_path)
    loaded = load_preprocessing_artifacts(output_path)

    first = transform_state_table(state_df, loaded)
    second = transform_state_table(state_df, loaded)

    assert first.equals(second)


def test_transform_state_table_fills_nulls_and_preserves_missing_flags():
    state_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "stay_id": [11, 11, 22],
            "step_index": [0, 1, 0],
            "heart_rate": [10.0, 30.0, None],
            "heart_rate_missing": [0, 0, 1],
        }
    )
    artifacts = fit_preprocessing_artifacts(
        state_df=state_df,
        registry=_registry(),
        split_manifest=_manifest(),
        train_medians={"heart_rate": 20.0},
    )

    transformed = transform_state_table(state_df, artifacts)

    assert transformed["heart_rate"][2] == pytest.approx(0.0)
    assert list(transformed["heart_rate_missing"]) == [0, 0, 1]


def test_zero_variance_train_feature_uses_unit_scale():
    state_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "stay_id": [11, 11, 22],
            "step_index": [0, 1, 0],
            "heart_rate": [5.0, 5.0, 100.0],
        }
    )

    artifacts = fit_preprocessing_artifacts(
        state_df=state_df,
        registry=_registry(),
        split_manifest=_manifest(),
        train_medians={"heart_rate": 5.0},
    )

    assert artifacts.transforms["heart_rate"].scale == pytest.approx(1.0)
