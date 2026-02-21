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
    ADMISSIONS_CSV,
    ALL_FLUID_IDS,
    ALL_GCS_IDS,
    ALL_LABS_IDS,
    ALL_URINE_IDS,
    ALL_VASO_IDS,
    ALL_VITALS_IDS,
    CHARTEVENTS_CSV,
    CRYSTALLOIDS,
    GCS,
    ICUSTAYS_CSV,
    INPUTEVENTS_CSV,
    LABEVENTS_CSV,
    LABS,
    OUT_DIR,
    OUT_PARQUET,
    OUTPUTEVENTS_CSV,
    PATIENTS_CSV,
    URINE_OUTPUT,
    VASOPRESSORS,
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


# ─── Urine Output (outputevents) ──────────────────────────────────
def build_urine_output_hourly(icustays_lf: pl.LazyFrame) -> pl.LazyFrame:
    """outputevents.csv.gz'den saatlik idrar çıkışını hesapla (ml)."""
    _log("outputevents.csv.gz okunuyor (urine)...")
    
    return (
        pl.scan_csv(
            OUTPUTEVENTS_CSV,
            dtypes={"stay_id": pl.Int64, "itemid": pl.Int64, "value": pl.Float64},
        )
        .filter(pl.col("itemid").is_in(ALL_URINE_IDS))
        .filter(pl.col("value").is_not_null())
        .with_columns(
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        )
        .with_columns(
            pl.col("charttime").dt.truncate("1h").alias("hour_bin"),
        )
        .group_by("stay_id", "hour_bin")
        .agg(pl.col("value").sum().alias("urine_output"))  # Saatlik toplam ml
    )


# ─── Vasopressors & Fluids (inputevents) ──────────────────────────
def build_inputs_hourly(icustays_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    inputevents.csv.gz'den:
    1. Vasopressor dozları (mcg/kg/min toplamı)
    2. IV Sıvı miktarları (ml toplamı)
    
    Not: Vasopressor'ler farklı ünitelerde olabilir, şimdilik amountu alıyoruz.
    RL'de aksiyon olarak 'ne kadar verildiğini' bilmek yeterli, ünite normalize sonrası yapılır.
    """
    _log("inputevents.csv.gz okunuyor (vasopressors & fluids)...")
    
    inputs = (
        pl.scan_csv(
            INPUTEVENTS_CSV,
            dtypes={
                "stay_id": pl.Int64, 
                "itemid": pl.Int64, 
                "amount": pl.Float64,
                "rate": pl.Float64,
                "starttime": pl.Utf8,
                "endtime": pl.Utf8,
            },
        )
        .filter(pl.col("itemid").is_in(ALL_VASO_IDS + ALL_FLUID_IDS))
        .filter(pl.col("amount").is_not_null())
        .with_columns(
            pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        )
        # Her kaydın başlangıç saatini al (4 saatlik pencere için basitleştirme)
        .with_columns(
            pl.col("starttime").dt.truncate("1h").alias("hour_bin"),
        )
    )
    
    # Vasopressor'leri ayrı, sıvıları ayrı topla
    vaso_exprs = []
    for name, ids in VASOPRESSORS.items():
        expr = (
            pl.when(pl.col("itemid").is_in(ids))
            .then(pl.col("amount"))  # veya rate, hangisi daha uygunsa
            .otherwise(0)
            .sum()
            .alias(f"{name}_dose")
        )
        vaso_exprs.append(expr)
    
    # Sıvılar için toplam
    fluid_expr = (
        pl.when(pl.col("itemid").is_in(ALL_FLUID_IDS))
        .then(pl.col("amount"))
        .otherwise(0)
        .sum()
        .alias("crystalloid_ml")
    )
    
    return (
        inputs
        .group_by("stay_id", "hour_bin")
        .agg(vaso_exprs + [fluid_expr])
    )


# ─── Demographics (patients + admissions) ─────────────────────────
def build_demographics(icustays_lf: pl.LazyFrame) -> pl.LazyFrame:
    """patients + admissions'tan demografik verileri çıkar."""
    _log("Demographics (patients + admissions) okunuyor...")
    
    patients = (
        pl.scan_csv(PATIENTS_CSV, dtypes={"subject_id": pl.Int64, "anchor_age": pl.Int64})
        .select("subject_id", "gender", "anchor_age")
    )
    
    admissions = (
        pl.scan_csv(ADMISSIONS_CSV, dtypes={"hadm_id": pl.Int64, "subject_id": pl.Int64})
        .select("hadm_id", "subject_id", "admission_type", "insurance")
    )
    
    # ICU stays ile birleştirerek stay_id bazında demografik bilgi oluştur
    return (
        icustays_lf
        .join(admissions, on="hadm_id", how="left")
        .join(patients, on="subject_id", how="left")
        .select(
            "stay_id",
            "gender",
            pl.col("anchor_age").alias("age"),  # MIMIC-IV'te anchor_age kullanılır
            "admission_type",
        )
    )


# ─── GCS (Glasgow Coma Scale) ─────────────────────────────────────
def build_gcs_hourly(icustays_lf: pl.LazyFrame) -> pl.LazyFrame:
    """chartevents'ten GCS skorlarını al (eye + motor + verbal)."""
    _log("GCS verileri oluşturuluyor...")
    
    gcs_lf = (
        pl.scan_csv(
            CHARTEVENTS_CSV,
            dtypes={"stay_id": pl.Int64, "itemid": pl.Int64, "valuenum": pl.Float64},
        )
        .filter(pl.col("itemid").is_in(ALL_GCS_IDS))
        .filter(pl.col("valuenum").is_not_null())
        .with_columns(
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        )
        .with_columns(
            pl.col("charttime").dt.truncate("1h").alias("hour_bin"),
        )
    )
    
    # Her bileşen için ortalama al (saatlik)
    agg_exprs = []
    for name, ids in GCS.items():
        expr = (
            pl.when(pl.col("itemid").is_in(ids))
            .then(pl.col("valuenum"))
            .otherwise(None)
            .mean()
            .alias(name)
        )
        agg_exprs.append(expr)
    
    return (
        gcs_lf
        .group_by("stay_id", "hour_bin")
        .agg(agg_exprs)
        .with_columns(
            # Toplam GCS skoru (eğer eksik varsa null kalır)
            pl.sum_horizontal([pl.col("gcs_eye"), pl.col("gcs_motor"), pl.col("gcs_verbal")])
            .alias("gcs_total")
        )
    )


# ─── Forward-Fill & Sink ─────────────────────────────────────────

def merge_and_forward_fill_enhanced(
    vitals_lf: pl.LazyFrame,
    labs_lf: pl.LazyFrame,
    urine_lf: pl.LazyFrame,
    inputs_lf: pl.LazyFrame,
    gcs_lf: pl.LazyFrame,
    demo_lf: pl.LazyFrame,
    out_path: Path = OUT_PARQUET,
) -> None:
    """
    Tüm tabloları birleştir:
    1. Vitals + Labs + Urine + Inputs + GCS → hourly join
    2. Demographics → stay_id bazında left join (zaman bağımsız)
    3. Forward fill
    4. Parquet'e yaz
    """
    _log("Tüm veriler birleştiriliyor (vitals + labs + urine + inputs + gcs)...")
    
    # Zaman bazlı tabloları birleştir
    time_based = (
        vitals_lf
        .join(labs_lf, on=["stay_id", "hour_bin"], how="full", coalesce=True)
        .join(urine_lf, on=["stay_id", "hour_bin"], how="full", coalesce=True)
        .join(inputs_lf, on=["stay_id", "hour_bin"], how="full", coalesce=True)
        .join(gcs_lf, on=["stay_id", "hour_bin"], how="full", coalesce=True)
    )
    
    # Demografik bilgileri ekle (zaman bağımsız, stay_id bazında)
    combined = time_based.join(demo_lf, on="stay_id", how="left")
    
    # Tüm feature kolonlarını belirle (stay_id, hour_bin hariç)
    feature_cols = [
        *VITALS.keys(), *LABS.keys(), 
        *URINE_OUTPUT.keys(),
        *[f"{k}_dose" for k in VASOPRESSORS.keys()], "crystalloid_ml",
        "gcs_eye", "gcs_motor", "gcs_verbal", "gcs_total",
        "age", "gender", "admission_type"
    ]
    
    # Sırala ve forward-fill uygula
    _log("Forward-fill uygulanıyor (stay_id bazında)...")
    combined = combined.sort("stay_id", "hour_bin")
    
    # Forward fill: stay_id içindeki eksik değerleri son bilinen değerle doldur
    fill_exprs = []
    for col in feature_cols:
        if col in ["gender", "admission_type"]:  # Kategorik kolonlar için farklı işlem
            fill_exprs.append(pl.col(col).forward_fill().over("stay_id").alias(col))
        else:  # Numerik kolonlar
            fill_exprs.append(pl.col(col).forward_fill().over("stay_id").alias(col))
    
    combined = combined.with_columns(fill_exprs)
    
    # Output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Parquet'e yazılıyor → {out_path}")
    
    try:
        combined.sink_parquet(out_path)
        _log("✅ Pipeline tamamlandı!")
    except Exception as e:
        _log(f"⚠️ sink_parquet başarısız, collect ile yazılıyor... {e}")
        df = combined.collect(streaming=True)
        df.write_parquet(out_path)
        _log(f"✅ Tamamlandı! Satır: {df.shape[0]:,}")


# ─── Pipeline Orchestrator ───────────────────────────────────────

def run_pipeline() -> None:
    """Ana pipeline'ı çalıştır (genişletilmiş versiyon)."""
    _log("=" * 60)
    _log("  MIMIC-IV Sepsis DRL — Full Preprocessing Pipeline")
    _log("=" * 60)

    # 1. Temel tablolar
    icustays_lf = load_icustays()
    demo_lf = build_demographics(icustays_lf)
    
    # 2. Zaman serisi verileri (hepsi lazy, paralel tanımlanabilir)
    vitals_lf = build_vitals_hourly(icustays_lf)
    labs_lf = build_labs_hourly(icustays_lf)
    urine_lf = build_urine_output_hourly(icustays_lf)
    inputs_lf = build_inputs_hourly(icustays_lf)
    gcs_lf = build_gcs_hourly(icustays_lf)
    
    _log("Adım 3: Birleştir ve kaydet")
    merge_and_forward_fill_enhanced(
        vitals_lf, labs_lf, urine_lf, inputs_lf, gcs_lf, demo_lf
    )

if __name__ == "__main__":
    run_pipeline()
