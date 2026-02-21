"""
MIMIC-IV Sepsis DRL — Polars Lazy Preprocessing Pipeline (Memory-Optimized)
===========================================================================
Ham chartevents + labevents verilerini:
  1. itemid bazlı filtrele
  2. icustays ile birleştir (stay_id ata)
  3. Saatlik bloklara yuvarla  (hourly binning)
  4. Tek-geçişli group_by + conditional aggregation (pivot-free, join-free)
  5. Vitals + Labs tek join ile birleştir
  6. Forward-fill (stay_id bazında)
  7. Parquet'e yaz (streaming sink)

Önceki versiyon her feature için ayrı LazyFrame oluşturup N-way full join
yapıyordu → bellek patlamasına neden oluyordu (~60GB).
Bu versiyon tek group_by ile tüm feature'ları çıkarır ve sink_parquet
ile streaming yazarak belleği minimize eder.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.preprocess.config import (
    ALL_LABS_IDS,
    ALL_VITALS_IDS,
    CHARTEVENTS_CSV,
    ICUSTAYS_CSV,
    LABEVENTS_CSV,
    LABS,
    OUT_DIR,
    OUT_PARQUET,
    VITALS,
)


# ─── Helpers ──────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Basit zaman damgalı loglama."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _build_agg_exprs(feature_map: dict[str, list[int]]) -> list[pl.Expr]:
    """
    Feature map'ten tek-geçişli conditional aggregation ifadeleri oluşturur.

    Her feature için:
      when(itemid ∈ ids).then(valuenum).mean()  →  feature_name

    Bu sayede N ayrı frame + N-1 full join yerine tek group_by yeterli olur.
    """
    exprs = []
    for name, ids in feature_map.items():
        expr = (
            pl.when(pl.col("itemid").is_in(ids))
            .then(pl.col("valuenum"))
            .otherwise(None)
            .mean()
            .alias(name)
        )
        exprs.append(expr)
    return exprs


# ─── ICU Stays ────────────────────────────────────────────────────

def load_icustays() -> pl.LazyFrame:
    """icustays tablosunu lazy scan et — stay_id, hadm_id, subject_id, intime."""
    _log("icustays.csv.gz okunuyor...")
    return (
        pl.scan_csv(
            ICUSTAYS_CSV,
            dtypes={"stay_id": pl.Int64, "hadm_id": pl.Int64, "subject_id": pl.Int64},
        )
        .with_columns(pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"))
        .select("stay_id", "hadm_id", "subject_id", "intime")
    )


# ─── Vitals (chartevents) — tek geçiş ───────────────────────────

def build_vitals_hourly(icustays_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    chartevents.csv.gz'den vital sign feature'larını saatlik bloklara böl.

    Tek group_by + conditional aggregation ile tüm feature'ları çıkarır.
    N-way full join YAPILMAZ → bellek dostu.
    """
    _log("chartevents.csv.gz okunuyor (vitals)...")

    chart_lf = (
        pl.scan_csv(
            CHARTEVENTS_CSV,
            dtypes={
                "stay_id": pl.Int64,
                "itemid": pl.Int64,
                "valuenum": pl.Float64,
            },
        )
        .filter(pl.col("itemid").is_in(ALL_VITALS_IDS))
        .filter(pl.col("valuenum").is_not_null())
        .with_columns(
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        )
        .with_columns(
            pl.col("charttime").dt.truncate("1h").alias("hour_bin"),
        )
        .select("stay_id", "hour_bin", "itemid", "valuenum")
    )

    _log("Vital feature'lar oluşturuluyor (tek geçiş)...")
    agg_exprs = _build_agg_exprs(VITALS)

    return (
        chart_lf
        .group_by("stay_id", "hour_bin")
        .agg(agg_exprs)
    )


# ─── Labs (labevents) — tek geçiş ───────────────────────────────

def build_labs_hourly(icustays_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    labevents.csv.gz'den lab feature'larını saatlik bloklara böl.

    labevents'te stay_id yok → hadm_id üzerinden icustays ile join yapılır.
    Sonra tek group_by + conditional aggregation ile feature'lar çıkarılır.
    """
    _log("labevents.csv.gz okunuyor (labs)...")

    lab_lf = (
        pl.scan_csv(
            LABEVENTS_CSV,
            dtypes={
                "hadm_id": pl.Int64,
                "itemid": pl.Int64,
                "valuenum": pl.Float64,
            },
        )
        .filter(pl.col("itemid").is_in(ALL_LABS_IDS))
        .filter(pl.col("valuenum").is_not_null())
        .filter(pl.col("hadm_id").is_not_null())
        .with_columns(
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        )
        .with_columns(
            pl.col("charttime").dt.truncate("1h").alias("hour_bin"),
        )
    )

    # labevents → icustays join (hadm_id üzerinden stay_id al)
    icu_map = icustays_lf.select("stay_id", "hadm_id", "intime")
    lab_lf = (
        lab_lf
        .join(icu_map, on="hadm_id", how="inner")
        .filter(pl.col("charttime") >= pl.col("intime"))
        .select("stay_id", "hour_bin", "itemid", "valuenum")
    )

    _log("Lab feature'lar oluşturuluyor (tek geçiş)...")
    agg_exprs = _build_agg_exprs(LABS)

    return (
        lab_lf
        .group_by("stay_id", "hour_bin")
        .agg(agg_exprs)
    )


# ─── Forward-Fill & Sink ─────────────────────────────────────────

def merge_and_forward_fill(
    vitals_lf: pl.LazyFrame,
    labs_lf: pl.LazyFrame,
    out_path: Path = OUT_PARQUET,
) -> None:
    """
    Vitals + Labs tablosunu birleştir, sırala, forward-fill uygula, parquet'e yaz.
    Sadece tek bir full join yapılır (vitals ↔ labs).
    """
    _log("Vitals + Labs birleştiriliyor...")

    feature_cols = list(VITALS.keys()) + list(LABS.keys())

    combined = (
        vitals_lf
        .join(labs_lf, on=["stay_id", "hour_bin"], how="full", coalesce=True)
        .sort("stay_id", "hour_bin")
    )

    # Forward-fill: her stay_id grubunda, her feature sütununda
    _log("Forward-fill uygulanıyor...")
    combined = combined.with_columns(
        [
            pl.col(c).forward_fill().over("stay_id").alias(c)
            for c in feature_cols
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Streaming sink ile parquet'e yaz — collect YAPILMAZ, bellek dostu
    _log(f"Parquet'e yazılıyor (streaming) → {out_path}")
    try:
        combined.sink_parquet(out_path)
        _log("✅ Pipeline tamamlandı (streaming sink)!")
    except Exception:
        # sink_parquet bazı join/sort kombinasyonlarında desteklenmeyebilir
        # Bu durumda low_memory collect ile fallback yap
        _log("⚠️  sink_parquet desteklenmiyor, low_memory collect ile yazılıyor...")
        df = combined.collect(streaming=True)
        df.write_parquet(out_path)
        _log(f"✅ Pipeline tamamlandı!  Satır: {df.shape[0]:,}  |  Sütun: {df.shape[1]}")


# ─── Pipeline Orchestrator ───────────────────────────────────────

def run_pipeline() -> None:
    """Ana pipeline'ı çalıştır."""
    _log("=" * 60)
    _log("  MIMIC-IV Sepsis DRL — Faz 1 Preprocessing")
    _log("=" * 60)

    steps = tqdm(
        ["icustays", "vitals", "labs", "merge & forward-fill"],
        desc="🚀 Pipeline",
        unit="step",
    )

    steps.set_postfix(stage="icustays")
    icustays_lf = load_icustays()
    steps.update(1)

    steps.set_postfix(stage="vitals")
    vitals_lf = build_vitals_hourly(icustays_lf)
    steps.update(1)

    steps.set_postfix(stage="labs")
    labs_lf = build_labs_hourly(icustays_lf)
    steps.update(1)

    steps.set_postfix(stage="merge & write")
    merge_and_forward_fill(vitals_lf, labs_lf)
    steps.update(1)
    steps.close()
