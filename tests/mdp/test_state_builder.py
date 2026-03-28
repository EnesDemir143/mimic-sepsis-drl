from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from mimic_sepsis_rl.data.split_models import SplitManifest
from mimic_sepsis_rl.mdp.features.builder import StateTableBuilder
from mimic_sepsis_rl.mdp.features.dictionary import FEATURE_REGISTRY
from mimic_sepsis_rl.mdp.features.extractors import StepWindowData


def _manifest() -> SplitManifest:
    return SplitManifest(
        spec_version="1.0.0",
        seed=42,
        source_episode_set="tests",
        train_ids=frozenset({11}),
        validation_ids=frozenset({22}),
        test_ids=frozenset({33}),
    )


def _chart_df(item_ids: list[int], values: list[float | None]) -> pl.DataFrame:
    base = datetime(2150, 1, 1, 12, 0, 0)
    return pl.DataFrame(
        {
            "itemid": item_ids,
            "valuenum": values,
            "charttime": [base + timedelta(minutes=index * 5) for index in range(len(values))],
        }
    )


def _empty_chart_df() -> pl.DataFrame:
    return pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []})


def _step_window(
    *,
    stay_id: int,
    subject_id: int,
    step_index: int,
    hours_relative_to_onset: float,
    chartevents: pl.DataFrame | None = None,
) -> StepWindowData:
    if chartevents is None:
        chartevents = _empty_chart_df()
    return StepWindowData(
        stay_id=stay_id,
        step_index=step_index,
        hours_relative_to_onset=hours_relative_to_onset,
        chartevents=chartevents,
        labevents=pl.DataFrame({"itemid": [], "valuenum": [], "charttime": []}),
        inputevents=pl.DataFrame({"itemid": [], "amount": []}),
        outputevents=pl.DataFrame({"itemid": [], "value": []}),
        age_years=64.0,
        weight_kg=78.0,
        subject_id=subject_id,
    )


def test_builder_is_deterministic_and_manifest_aware():
    registry = {
        "heart_rate": FEATURE_REGISTRY["heart_rate"],
        "hours_since_onset": FEATURE_REGISTRY["hours_since_onset"],
    }
    builder = StateTableBuilder(
        registry=registry,
        train_medians={"heart_rate": 55.0, "hours_since_onset": 0.0},
        emit_missingness_flags=False,
    )
    manifest = _manifest()
    step_zero = _step_window(
        stay_id=101,
        subject_id=11,
        step_index=0,
        hours_relative_to_onset=-24.0,
        chartevents=_chart_df([220045], [92.0]),
    )
    step_one = _step_window(
        stay_id=101,
        subject_id=11,
        step_index=1,
        hours_relative_to_onset=-20.0,
    )

    first = builder.build([step_one, step_zero], split_manifest=manifest)
    second = builder.build([step_zero, step_one], split_manifest=manifest)

    assert first.equals(second)
    assert list(first["stay_id"]) == [101, 101]
    assert list(first["step_index"]) == [0, 1]
    assert list(first["subject_id"]) == [11, 11]
    assert list(first["split"]) == ["train", "train"]


def test_builder_uses_forward_fill_before_train_median():
    builder = StateTableBuilder(
        registry={"heart_rate": FEATURE_REGISTRY["heart_rate"]},
        train_medians={"heart_rate": 55.0},
        emit_missingness_flags=False,
    )

    observed = _step_window(
        stay_id=201,
        subject_id=11,
        step_index=0,
        hours_relative_to_onset=-24.0,
        chartevents=_chart_df([220045], [90.0]),
    )
    missing = _step_window(
        stay_id=201,
        subject_id=11,
        step_index=1,
        hours_relative_to_onset=-20.0,
    )

    result = builder.build([observed, missing], split_manifest=_manifest())
    assert list(result["heart_rate"]) == pytest.approx([90.0, 90.0])


def test_builder_uses_train_median_at_episode_boundary():
    builder = StateTableBuilder(
        registry={"heart_rate": FEATURE_REGISTRY["heart_rate"]},
        train_medians={"heart_rate": 55.0},
        emit_missingness_flags=False,
    )

    first = _step_window(
        stay_id=301,
        subject_id=11,
        step_index=0,
        hours_relative_to_onset=-24.0,
    )
    second = _step_window(
        stay_id=301,
        subject_id=11,
        step_index=1,
        hours_relative_to_onset=-20.0,
    )

    result = builder.build([first, second], split_manifest=_manifest())
    assert list(result["heart_rate"]) == pytest.approx([55.0, 55.0])


def test_builder_missingness_flags_track_invalid_and_missing_windows():
    builder = StateTableBuilder(
        registry={"map": FEATURE_REGISTRY["map"]},
        train_medians={"map": 70.0},
        emit_missingness_flags=True,
    )

    observed = _step_window(
        stay_id=401,
        subject_id=11,
        step_index=0,
        hours_relative_to_onset=-24.0,
        chartevents=_chart_df([220052], [68.0]),
    )
    out_of_range = _step_window(
        stay_id=401,
        subject_id=11,
        step_index=1,
        hours_relative_to_onset=-20.0,
        chartevents=_chart_df([220052], [9999.0]),
    )
    missing = _step_window(
        stay_id=401,
        subject_id=11,
        step_index=2,
        hours_relative_to_onset=-16.0,
    )

    result = builder.build([observed, out_of_range, missing], split_manifest=_manifest())
    assert list(result["map"]) == pytest.approx([68.0, 68.0, 68.0])
    assert list(result["map_missing"]) == [0, 1, 1]


def test_builder_reads_weight_from_static_context():
    builder = StateTableBuilder(
        registry={"weight_kg": FEATURE_REGISTRY["weight_kg"]},
        train_medians={"weight_kg": 70.0},
        emit_missingness_flags=False,
    )
    step = _step_window(
        stay_id=451,
        subject_id=11,
        step_index=0,
        hours_relative_to_onset=-24.0,
    )

    result = builder.build([step], split_manifest=_manifest())
    assert result["weight_kg"][0] == pytest.approx(78.0)


def test_builder_rejects_subjects_missing_from_manifest():
    builder = StateTableBuilder(
        registry={"heart_rate": FEATURE_REGISTRY["heart_rate"]},
        train_medians={"heart_rate": 55.0},
        emit_missingness_flags=False,
    )
    rogue_step = _step_window(
        stay_id=501,
        subject_id=999,
        step_index=0,
        hours_relative_to_onset=-24.0,
    )

    with pytest.raises(ValueError, match="not present"):
        builder.build([rogue_step], split_manifest=_manifest())
