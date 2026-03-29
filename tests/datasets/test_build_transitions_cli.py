from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import polars as pl
import yaml

from mimic_sepsis_rl.cli.build_transitions import main as build_transitions_main


def _write_gzip_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_split_manifests(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "split": ["train", "train"],
            "episode_keys": [[101], [202]],
            "n_episodes": [1, 1],
        }
    ).write_parquet(base_dir / "train_manifest.parquet")
    pl.DataFrame(
        {
            "subject_id": [3],
            "split": ["validation"],
            "episode_keys": [[303]],
            "n_episodes": [1],
        }
    ).write_parquet(base_dir / "validation_manifest.parquet")
    pl.DataFrame(
        {
            "subject_id": [4],
            "split": ["test"],
            "episode_keys": [[404]],
            "n_episodes": [1],
        }
    ).write_parquet(base_dir / "test_manifest.parquet")
    (base_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "spec_version": "1.0.0",
                "seed": 42,
                "source_episode_set": "data/processed/episodes/episodes.parquet",
                "has_leakage": False,
                "n_total_patients": 4,
                "splits": {
                    "train": {"n_patients": 2, "n_episodes": 2},
                    "validation": {"n_patients": 1, "n_episodes": 1},
                    "test": {"n_patients": 1, "n_episodes": 1},
                },
            },
            indent=2,
        )
    )


def _prepare_synthetic_workspace(root: Path) -> None:
    processed_dir = root / "data" / "processed"
    raw_root = root / "data" / "raw" / "physionet.org" / "files" / "mimiciv" / "3.1"

    episode_rows = []
    step_rows = []
    cohort_rows = []
    admissions_rows = []
    chartevents_rows = []
    labevents_rows = []
    inputevents_rows = []
    outputevents_rows: list[dict[str, object]] = []

    base_date = pl.datetime(2024, 1, 1, 0, 0, 0)
    stays = [
        (101, 1, 1001, 60, None),
        (202, 2, 1002, 65, None),
        (303, 3, 1003, 70, "2024-01-20 04:00:00"),
        (404, 4, 1004, 75, None),
    ]
    step_offsets = [
        ("2024-01-01 00:00:00", "2024-01-01 04:00:00", -4),
        ("2024-01-01 04:00:00", "2024-01-01 08:00:00", 0),
        ("2024-01-01 08:00:00", "2024-01-01 12:00:00", 4),
    ]

    fluid_amounts = {
        101: [100.0, 200.0, 300.0],
        202: [400.0, 500.0, 600.0],
        303: [150.0, 250.0, 350.0],
        404: [180.0, 280.0, 380.0],
    }
    vaso_rates = {
        101: [0.05, 0.10, 0.15],
        202: [0.20, 0.25, 0.30],
        303: [0.08, 0.12, 0.18],
        404: [0.09, 0.14, 0.19],
    }

    for index, (stay_id, subject_id, hadm_id, age, deathtime) in enumerate(stays):
        day = index + 1
        onset = f"2024-01-{day:02d} 04:00:00"
        window_start = f"2024-01-{day:02d} 00:00:00"
        window_end = f"2024-01-{day:02d} 12:00:00"
        episode_rows.append(
            {
                "stay_id": stay_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "onset_time": onset,
                "window_start": window_start,
                "window_end": window_end,
                "actual_end": window_end,
                "n_steps": 3,
                "is_truncated": False,
                "truncation_reason": None,
                "truncation_step": None,
            }
        )
        cohort_rows.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "anchor_age": age,
            }
        )
        admissions_rows.append({"hadm_id": hadm_id, "deathtime": deathtime})

        for step_index, (start_str, end_str, hours_rel) in enumerate(step_offsets):
            start = start_str.replace("2024-01-01", f"2024-01-{day:02d}")
            end = end_str.replace("2024-01-01", f"2024-01-{day:02d}")
            step_rows.append(
                {
                    "stay_id": stay_id,
                    "step_index": step_index,
                    "step_start": start,
                    "step_end": end,
                    "hours_relative_to_onset": hours_rel,
                    "is_pre_onset": hours_rel < 0,
                }
            )
            event_time = start.replace(":00:00", ":30:00")
            chartevents_rows.extend(
                [
                    {
                        "stay_id": stay_id,
                        "charttime": event_time,
                        "itemid": 220052,
                        "value": 75 - step_index * 5 - index,
                        "valuenum": 75 - step_index * 5 - index,
                    },
                    {
                        "stay_id": stay_id,
                        "charttime": event_time,
                        "itemid": 220277,
                        "value": 97 - step_index,
                        "valuenum": 97 - step_index,
                    },
                    {
                        "stay_id": stay_id,
                        "charttime": event_time,
                        "itemid": 223901,
                        "value": 15 - (step_index == 1),
                        "valuenum": 15 - (step_index == 1),
                    },
                ]
            )
            labevents_rows.append(
                {
                    "hadm_id": hadm_id,
                    "charttime": event_time,
                    "itemid": 50813,
                    "value": 2.0 + step_index + index * 0.1,
                    "valuenum": 2.0 + step_index + index * 0.1,
                }
            )
            inputevents_rows.extend(
                [
                    {
                        "stay_id": stay_id,
                        "starttime": event_time,
                        "endtime": end,
                        "itemid": 220949,
                        "amount": fluid_amounts[stay_id][step_index],
                        "rate": 0.0,
                        "patientweight": 70.0 + index,
                    },
                    {
                        "stay_id": stay_id,
                        "starttime": event_time,
                        "endtime": end,
                        "itemid": 221906,
                        "amount": 1.0,
                        "rate": vaso_rates[stay_id][step_index],
                        "patientweight": 70.0 + index,
                    },
                ]
            )

    episodes_df = pl.DataFrame(episode_rows).with_columns(
        pl.col("onset_time").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        pl.col("window_start").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        pl.col("window_end").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        pl.col("actual_end").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    )
    steps_df = pl.DataFrame(step_rows).with_columns(
        pl.col("step_start").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        pl.col("step_end").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    )
    cohort_df = pl.DataFrame(cohort_rows)

    (processed_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (processed_dir / "cohort").mkdir(parents=True, exist_ok=True)
    episodes_df.write_parquet(processed_dir / "episodes" / "episodes.parquet")
    steps_df.write_parquet(processed_dir / "episodes" / "episode_steps.parquet")
    cohort_df.write_parquet(processed_dir / "cohort" / "cohort.parquet")

    _write_split_manifests(root / "data" / "splits")

    feature_config = {
        "spec_version": "1.0.0",
        "include_features": [
            "map",
            "spo2",
            "gcs_total",
            "lactate",
            "age_years",
            "weight_kg",
            "hours_since_onset",
        ],
        "exclude_features": [],
        "missingness_flags_default": False,
        "imputation": {
            "train_medians_path": "data/processed/features/train_medians.json",
        },
        "output": {
            "state_table_dir": "data/processed/features/state_vectors",
        },
    }
    (root / "configs" / "features").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "features" / "test.yaml").write_text(yaml.safe_dump(feature_config))

    _write_gzip_csv(
        raw_root / "icu" / "chartevents.csv.gz",
        chartevents_rows,
        ["stay_id", "charttime", "itemid", "value", "valuenum"],
    )
    _write_gzip_csv(
        raw_root / "hosp" / "labevents.csv.gz",
        labevents_rows,
        ["hadm_id", "charttime", "itemid", "value", "valuenum"],
    )
    _write_gzip_csv(
        raw_root / "icu" / "inputevents.csv.gz",
        inputevents_rows,
        ["stay_id", "starttime", "endtime", "itemid", "amount", "rate", "patientweight"],
    )
    _write_gzip_csv(
        raw_root / "icu" / "outputevents.csv.gz",
        outputevents_rows,
        ["stay_id", "charttime", "itemid", "value"],
    )
    _write_gzip_csv(
        raw_root / "hosp" / "admissions.csv.gz",
        admissions_rows,
        ["hadm_id", "deathtime"],
    )


def test_live_build_transitions_exports_replay_files(tmp_path, monkeypatch) -> None:
    _prepare_synthetic_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    exit_code = build_transitions_main(
        [
            "--raw-root",
            "data/raw/physionet.org/files/mimiciv/3.1",
            "--cohort-path",
            "data/processed/cohort/cohort.parquet",
            "--episodes-path",
            "data/processed/episodes/episodes.parquet",
            "--steps-path",
            "data/processed/episodes/episode_steps.parquet",
            "--split-manifest-dir",
            "data/splits",
            "--features-config",
            "configs/features/test.yaml",
            "--output-dir",
            "data/replay",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "data" / "replay" / "replay_train.parquet").exists()
    assert (tmp_path / "data" / "replay" / "replay_train_meta.json").exists()
    assert (tmp_path / "data" / "replay" / "replay_validation.parquet").exists()
    assert (tmp_path / "data" / "replay" / "replay_test.parquet").exists()
    assert (tmp_path / "data" / "processed" / "actions" / "action_bins.json").exists()
    assert (tmp_path / "data" / "processed" / "rewards" / "reward_config.json").exists()

    meta = json.loads((tmp_path / "data" / "replay" / "replay_train_meta.json").read_text())
    assert meta["split_label"] == "train"
    assert meta["n_actions"] == 25
    assert meta["n_episodes"] == 2
    assert meta["state_dim"] > 0
