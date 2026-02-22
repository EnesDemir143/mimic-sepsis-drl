"""
MIMIC-IV Sepsis DRL — 48-Feature State Vector Builder
======================================================
Mevcut ``mimic_hourly_binned.parquet`` ve raw MIMIC-IV dosyalarından
48 klinik özelliği çıkararak ``data/processed/state.parquet`` üretir.

Kullanım
--------
    uv run python -m src.preprocess
"""
from __future__ import annotations

import polars as pl
from tqdm import tqdm

from src.preprocess.config import (
    ADMISSIONS_PATH,
    CHARTEVENTS_PATH,
    DIAGNOSES_ICD_PATH,
    ELIXHAUSER_ICD9,
    ELIXHAUSER_ICD10,
    EXTRA_LAB_ITEMS,
    HOURLY_BINNED_PATH,
    ICUSTAYS_PATH,
    LABEVENTS_PATH,
    META_COLUMNS,
    OMR_PATH,
    STATE_FEATURES,
    STATE_PARQUET_PATH,
    VASOPRESSOR_CONVERSION,
    WEIGHT_CHART_ITEMIDS,
)


# ═════════════════════════════════════════════
# 1. Mevcut Hourly-Binned Parquet Yükleme
# ═════════════════════════════════════════════
def load_hourly_binned() -> pl.DataFrame:
    """``mimic_hourly_binned.parquet`` dosyasını yükler."""
    print("📂 Hourly-binned parquet yükleniyor …")
    df = pl.read_parquet(HOURLY_BINNED_PATH)
    print(f"   ✅ {df.shape[0]:,} satır, {df.shape[1]} sütun yüklendi.")
    return df


# ═════════════════════════════════════════════
# 2. Eksik Lab Parametrelerini Çekme
# ═════════════════════════════════════════════
def extract_extra_labs(stay_ids: set[int], icustay_map: pl.DataFrame) -> pl.DataFrame:
    """
    ``labevents.csv.gz`` dosyasından 10 eksik lab parametresini çeker,
    saatlik bin'lere ortalamasını alır ve (stay_id, hour_bin, feature)
    formatında döndürür.

    Parameters
    ----------
    stay_ids : set[int]
        İlgilenilen stay_id kümesi.
    icustay_map : pl.DataFrame
        ``stay_id → (subject_id, hadm_id, intime, outtime)`` eşlemesi.
    """
    print("🧪 Eksik lab parametreleri labevents'ten çekiliyor …")

    # Hedef item ID'lerini düzle
    all_item_ids: list[int] = []
    item_to_name: dict[int, str] = {}
    for name, ids in EXTRA_LAB_ITEMS.items():
        for iid in ids:
            all_item_ids.append(iid)
            item_to_name[iid] = name

    # labevents lazy scan — sadece gerekli sütunlar
    labs_raw = (
        pl.scan_csv(
            LABEVENTS_PATH,
            schema_overrides={
                "hadm_id": pl.Utf8,      # null olabiliyor
                "valuenum": pl.Float64,
                "itemid": pl.Int64,
                "subject_id": pl.Int64,
            },
            infer_schema_length=10000,
        )
        .select(["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
        .filter(pl.col("itemid").is_in(all_item_ids))
        .filter(pl.col("valuenum").is_not_null())
        .with_columns(
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        )
        .collect()
    )
    print(f"   📊 {labs_raw.shape[0]:,} lab ölçümü çekildi.")

    # ICU stay sürelerine göre filtrele
    labs_with_stay = (
        labs_raw
        .join(icustay_map, on=["subject_id", "hadm_id"], how="inner")
        .filter(
            (pl.col("charttime") >= pl.col("intime"))
            & (pl.col("charttime") <= pl.col("outtime"))
        )
    )

    # Feature ismi ekle
    labs_with_stay = labs_with_stay.with_columns(
        pl.col("itemid").replace_strict(item_to_name, default=None).alias("feature_name")
    )

    # Saatlik bin'e yuvarlama
    labs_with_stay = labs_with_stay.with_columns(
        pl.col("charttime").dt.truncate("1h").alias("hour_bin")
    )

    # Saatlik ortalama → pivot
    labs_hourly = (
        labs_with_stay
        .group_by(["stay_id", "hour_bin", "feature_name"])
        .agg(pl.col("valuenum").mean().alias("value"))
        .pivot(on="feature_name", index=["stay_id", "hour_bin"], values="value")
    )

    print(f"   ✅ {labs_hourly.shape[0]:,} satır lab verisi pivot edildi.")
    return labs_hourly


# ═════════════════════════════════════════════
# 3. Ağırlık Çekme
# ═════════════════════════════════════════════
def extract_weight(icustay_map: pl.DataFrame) -> pl.DataFrame:
    """
    Hasta başına ağırlık (kg) çeker.
    Önce chartevents'ten Admission Weight → yoksa OMR'den Weight (Lbs) → kg.
    Sonuç: ``stay_id → weight_kg`` (tek değer / hasta).
    """
    print("⚖️  Ağırlık bilgisi çekiliyor …")

    # 1) chartevents'ten
    weight_chart = (
        pl.scan_csv(
            CHARTEVENTS_PATH,
            infer_schema_length=10000,
            schema_overrides={"valuenum": pl.Float64, "itemid": pl.Int64,
                              "stay_id": pl.Int64},
        )
        .select(["stay_id", "itemid", "valuenum"])
        .filter(
            pl.col("itemid").is_in(WEIGHT_CHART_ITEMIDS)
            & pl.col("valuenum").is_not_null()
            & (pl.col("valuenum") > 10)     # saçma değerleri ele
            & (pl.col("valuenum") < 300)
        )
        .group_by("stay_id")
        .agg(pl.col("valuenum").first().alias("weight_kg"))
        .collect()
    )
    print(f"   📊 chartevents'ten {weight_chart.shape[0]:,} hasta ağırlığı bulundu.")

    # 2) OMR'den eksik kalan hastalar için
    found_stays = set(weight_chart["stay_id"].to_list())
    missing_subject_ids = (
        icustay_map
        .filter(~pl.col("stay_id").is_in(found_stays))
        .select("subject_id")
        .unique()
    )

    if missing_subject_ids.shape[0] > 0:
        omr = pl.read_csv(OMR_PATH)
        omr_weight = (
            omr
            .filter(pl.col("result_name").str.contains("Weight"))
            .filter(pl.col("result_name").str.contains("Lbs"))
            .with_columns(
                pl.col("result_value")
                .cast(pl.Float64, strict=False)
                .alias("weight_lbs")
            )
            .filter(pl.col("weight_lbs").is_not_null())
            .group_by("subject_id")
            .agg((pl.col("weight_lbs").first() * 0.453592).alias("weight_kg"))
        )

        # subject_id → stay_id mapping
        omr_with_stay = (
            icustay_map
            .filter(~pl.col("stay_id").is_in(found_stays))
            .select(["stay_id", "subject_id"])
            .join(omr_weight, on="subject_id", how="inner")
            .select(["stay_id", "weight_kg"])
        )

        weight_chart = pl.concat([weight_chart, omr_with_stay])
        print(f"   📊 OMR'den {omr_with_stay.shape[0]:,} ek ağırlık eklendi.")

    print(f"   ✅ Toplam {weight_chart.shape[0]:,} hasta için ağırlık mevcut.")
    return weight_chart


# ═════════════════════════════════════════════
# 4. Elixhauser Komorbidite Skoru
# ═════════════════════════════════════════════
def compute_elixhauser(icustay_map: pl.DataFrame) -> pl.DataFrame:
    """
    ``diagnoses_icd.csv.gz``'den ICD-9/10 kodlarını okuyarak
    hasta başına Elixhauser komorbidite sayısı hesaplar.
    Sonuç: ``stay_id → elixhauser_score``.
    """
    print("🏥 Elixhauser komorbidite skoru hesaplanıyor …")

    diag = pl.read_csv(
        DIAGNOSES_ICD_PATH,
        schema_overrides={"icd_code": pl.Utf8, "icd_version": pl.Int64},
    )
    diag = diag.with_columns(pl.col("icd_code").cast(pl.Utf8).str.strip_chars())

    # hadm_id → stay_id mapping
    stay_hadm = icustay_map.select(["stay_id", "hadm_id"]).unique()
    diag_with_stay = diag.join(stay_hadm, on="hadm_id", how="inner")

    # Her ICD kodu için Elixhauser kategorilerini eşle
    def _match_categories(row_icd_code: str, row_icd_version: int) -> set[str]:
        """Tek bir ICD kodunun hangi Elixhauser kategorilerine düştüğünü döndürür."""
        mapping = ELIXHAUSER_ICD9 if row_icd_version == 9 else ELIXHAUSER_ICD10
        matched: set[str] = set()
        code_str = str(row_icd_code).strip()
        for cat, prefixes in mapping.items():
            for prefix in prefixes:
                if code_str.startswith(prefix):
                    matched.add(cat)
                    break
        return matched

    # Polars UDF yerine Python-level hesaplama (diagnoses genelde küçük)
    records: dict[int, set[str]] = {}
    for row in diag_with_stay.iter_rows(named=True):
        sid = row["stay_id"]
        cats = _match_categories(row["icd_code"], row["icd_version"])
        if sid not in records:
            records[sid] = set()
        records[sid].update(cats)

    elix_data = [
        {"stay_id": sid, "elixhauser_score": len(cats)}
        for sid, cats in records.items()
    ]

    if not elix_data:
        # Hiç eşleşme yoksa boş DF
        return pl.DataFrame({"stay_id": pl.Series([], dtype=pl.Int64),
                             "elixhauser_score": pl.Series([], dtype=pl.Int32)})

    result = pl.DataFrame(elix_data).with_columns(
        pl.col("elixhauser_score").cast(pl.Int32)
    )
    print(f"   ✅ {result.shape[0]:,} hasta için Elixhauser skoru hesaplandı.")
    print(f"   📊 Skor dağılımı: min={result['elixhauser_score'].min()}, "
          f"max={result['elixhauser_score'].max()}, "
          f"mean={result['elixhauser_score'].mean():.1f}")
    return result


# ═════════════════════════════════════════════
# 5. ICU Readmission Flag
# ═════════════════════════════════════════════
def compute_icu_readmission(icustay_map: pl.DataFrame) -> pl.DataFrame:
    """
    Aynı ``hadm_id`` altında birden fazla ICU stay varsa
    ilki hariç diğerleri readmission=1 olarak işaretlenir.
    Sonuç: ``stay_id → icu_readmission`` (0/1).
    """
    print("🔄 ICU readmission flag hesaplanıyor …")

    ordered = icustay_map.sort(["hadm_id", "intime"])

    readmit = (
        ordered
        .with_columns(
            pl.col("stay_id")
            .cum_count()
            .over("hadm_id")
            .alias("_seq")
        )
        .with_columns(
            pl.when(pl.col("_seq") > 1)
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("icu_readmission")
        )
        .select(["stay_id", "icu_readmission"])
    )

    n_readmit = readmit.filter(pl.col("icu_readmission") == 1).shape[0]
    print(f"   ✅ {n_readmit:,} / {readmit.shape[0]:,} stay readmission olarak işaretlendi.")
    return readmit


# ═════════════════════════════════════════════
# 6. Türetilen Klinik Özellikler
# ═════════════════════════════════════════════
def compute_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Mevcut sütunlardan türetilen özellikleri hesaplar:
    - Total Vasopressor Equivalent
    - SOFA Score (6 organ)
    - SIRS Score
    - Shock Index
    - PaO2/FiO2 Ratio
    - Mechanical Ventilation flag
    - Cumulative Fluid Balance
    - HCO3 (= bicarbonate alias)
    """
    print("🔬 Türetilen klinik özellikler hesaplanıyor …")

    # ── Total Vasopressor Equivalent ──
    vaso_cols = list(VASOPRESSOR_CONVERSION.keys())
    df = df.with_columns(
        [pl.col(c).fill_null(0.0) for c in vaso_cols if c in df.columns]
    )
    vaso_expr = [
        pl.col(col) * rate
        for col, rate in VASOPRESSOR_CONVERSION.items()
        if col in df.columns
    ]
    df = df.with_columns(
        pl.sum_horizontal(vaso_expr).alias("total_vaso_equiv")
    )

    # ── SOFA Score ──
    df = _compute_sofa(df)

    # ── SIRS Score ──
    df = _compute_sirs(df)

    # ── Shock Index ──
    df = df.with_columns(
        pl.when(pl.col("sbp") > 0)
        .then(pl.col("heart_rate") / pl.col("sbp"))
        .otherwise(None)
        .alias("shock_index")
    )

    # ── PaO2/FiO2 Oranı ──
    df = df.with_columns(
        pl.when(
            pl.col("fio2").is_not_null() & (pl.col("fio2") > 0)
            & pl.col("pao2").is_not_null()
        )
        .then(
            pl.col("pao2") / pl.when(pl.col("fio2") > 1)
            .then(pl.col("fio2") / 100.0)
            .otherwise(pl.col("fio2"))
        )
        .otherwise(None)
        .alias("pf_ratio")
    )

    # ── Mechanical Ventilation ──
    df = df.with_columns(
        pl.when(pl.col("fio2").is_not_null() & (pl.col("fio2") > 21))
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("mechanical_ventilation")
    )

    # ── Cumulative Fluid Balance ──
    df = df.with_columns(
        pl.col("crystalloid_ml").fill_null(0.0).alias("_fluid_in"),
        pl.col("urine_output").fill_null(0.0).alias("_fluid_out"),
    )
    df = df.sort(["stay_id", "hour_bin"])
    df = df.with_columns(
        (pl.col("_fluid_in") - pl.col("_fluid_out"))
        .cum_sum()
        .over("stay_id")
        .alias("cumulative_fluid_balance")
    )
    df = df.drop(["_fluid_in", "_fluid_out"])

    # ── HCO3 (bicarbonate alias) ──
    df = df.with_columns(
        pl.col("bicarbonate").alias("hco3")
    )

    print("   ✅ Tüm türetilen özellikler hesaplandı.")
    return df


def _compute_sofa(df: pl.DataFrame) -> pl.DataFrame:
    """SOFA skoru: 6 organ sistemi (0–24)."""

    # Önceden fio2_ratio hesapla (% ise /100)
    fio2_ratio = (
        pl.when(pl.col("fio2") > 1)
        .then(pl.col("fio2") / 100.0)
        .otherwise(pl.col("fio2"))
    )
    pf = pl.col("pao2") / fio2_ratio

    # Mekanik ventilasyon tahmini
    is_mv = pl.col("fio2").is_not_null() & (pl.col("fio2") > 21)

    # 1. Respiratory
    sofa_resp = (
        pl.when(pf.is_null() | pl.col("pao2").is_null() | pl.col("fio2").is_null())
        .then(0)
        .when((pf <= 100) & is_mv).then(4)
        .when((pf <= 200) & is_mv).then(3)
        .when(pf <= 200).then(2)
        .when(pf <= 300).then(1)
        .when(pf <= 400).then(1)
        .otherwise(0)
    )

    # 2. Cardiovascular (MAP + vasopressor)
    sofa_cardio = (
        pl.when(pl.col("total_vaso_equiv") > 0.5).then(4)
        .when(pl.col("total_vaso_equiv") > 0.1).then(3)
        .when(pl.col("total_vaso_equiv") > 0).then(2)
        .when(pl.col("mbp").is_not_null() & (pl.col("mbp") < 70)).then(1)
        .otherwise(0)
    )

    # 3. Renal (Creatinine)
    sofa_renal = (
        pl.when(pl.col("creatinine").is_null()).then(0)
        .when(pl.col("creatinine") >= 5.0).then(4)
        .when(pl.col("creatinine") >= 3.5).then(3)
        .when(pl.col("creatinine") >= 2.0).then(2)
        .when(pl.col("creatinine") >= 1.2).then(1)
        .otherwise(0)
    )

    # 4. Neurological (GCS)
    sofa_neuro = (
        pl.when(pl.col("gcs_total").is_null()).then(0)
        .when(pl.col("gcs_total") < 6).then(4)
        .when(pl.col("gcs_total") <= 9).then(3)
        .when(pl.col("gcs_total") <= 12).then(2)
        .when(pl.col("gcs_total") <= 14).then(1)
        .otherwise(0)
    )

    # 5. Coagulation (Platelets)
    sofa_coag = (
        pl.when(pl.col("platelet").is_null()).then(0)
        .when(pl.col("platelet") <= 20).then(4)
        .when(pl.col("platelet") <= 50).then(3)
        .when(pl.col("platelet") <= 100).then(2)
        .when(pl.col("platelet") <= 150).then(1)
        .otherwise(0)
    )

    # 6. Liver (Bilirubin)
    sofa_liver = (
        pl.when(pl.col("bilirubin_total").is_null()).then(0)
        .when(pl.col("bilirubin_total") >= 12.0).then(4)
        .when(pl.col("bilirubin_total") >= 6.0).then(3)
        .when(pl.col("bilirubin_total") >= 2.0).then(2)
        .when(pl.col("bilirubin_total") >= 1.2).then(1)
        .otherwise(0)
    )

    df = df.with_columns(
        (sofa_resp + sofa_cardio + sofa_renal + sofa_neuro + sofa_coag + sofa_liver)
        .cast(pl.Int32)
        .alias("sofa_score")
    )
    return df


def _compute_sirs(df: pl.DataFrame) -> pl.DataFrame:
    """SIRS skoru: 4 kriter (0–4)."""

    sirs_temp = (
        pl.when(pl.col("temp_c").is_null()).then(0)
        .when((pl.col("temp_c") > 38.0) | (pl.col("temp_c") < 36.0)).then(1)
        .otherwise(0)
    )

    sirs_hr = (
        pl.when(pl.col("heart_rate").is_null()).then(0)
        .when(pl.col("heart_rate") > 90).then(1)
        .otherwise(0)
    )

    sirs_rr = (
        pl.when(pl.col("resp_rate").is_null()).then(0)
        .when(pl.col("resp_rate") > 20).then(1)
        .otherwise(0)
    )

    sirs_wbc = (
        pl.when(pl.col("wbc").is_null()).then(0)
        .when((pl.col("wbc") > 12.0) | (pl.col("wbc") < 4.0)).then(1)
        .otherwise(0)
    )

    df = df.with_columns(
        (sirs_temp + sirs_hr + sirs_rr + sirs_wbc)
        .cast(pl.Int32)
        .alias("sirs_score")
    )
    return df


# ═════════════════════════════════════════════
# 7. Imputation (LOCF + Median)
# ═════════════════════════════════════════════
def apply_imputation(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """
    1. LOCF (Last Observation Carried Forward) — ``stay_id`` bazında forward-fill.
    2. Kalan null'lar → kolon medyanı ile doldurma.
    """
    print("🩹 Imputation uygulanıyor (LOCF + median) …")

    numeric_features = [
        c for c in feature_cols
        if c in df.columns and df[c].dtype in (pl.Float64, pl.Float32, pl.Int32, pl.Int64)
    ]

    # LOCF — per stay_id
    df = df.sort(["stay_id", "hour_bin"])
    df = df.with_columns(
        [pl.col(c).forward_fill().over("stay_id").alias(c) for c in numeric_features]
    )

    # Median fill — kalan null'lar
    medians = {c: df[c].median() for c in numeric_features}
    df = df.with_columns(
        [pl.col(c).fill_null(medians[c] if medians[c] is not None else 0)
         for c in numeric_features]
    )

    null_counts = {c: df[c].null_count() for c in numeric_features}
    still_null = {k: v for k, v in null_counts.items() if v > 0}
    if still_null:
        print(f"   ⚠️  Hâlâ null olan sütunlar: {still_null}")
    else:
        print("   ✅ Tüm null'lar dolduruldu.")

    return df


# ═════════════════════════════════════════════
# 8. ICU Stay Mapping Yükleme
# ═════════════════════════════════════════════
def load_icustay_map() -> pl.DataFrame:
    """``icustays.csv.gz`` dosyasını yükler ve gerekli sütunları döndürür."""
    print("🏥 ICU stay mapping yükleniyor …")
    icu = pl.read_csv(ICUSTAYS_PATH)
    icu = icu.with_columns(
        pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    )
    print(f"   ✅ {icu.shape[0]:,} ICU stay yüklendi.")
    return icu


# ═════════════════════════════════════════════
# 9. Ana Orchestrator
# ═════════════════════════════════════════════
def run() -> None:
    """48-feature state vektörü pipeline'ını çalıştırır."""
    print("=" * 60)
    print("  MIMIC-IV Sepsis DRL — 48-Feature State Builder")
    print("=" * 60)

    # 1) Mevcut veriyi yükle
    df = load_hourly_binned()
    stay_ids = set(df["stay_id"].unique().to_list())

    # 2) ICU stay mapping
    icustay_map = load_icustay_map()
    # Sadece bizim stay_id'lerimizi tut
    icustay_map = icustay_map.filter(pl.col("stay_id").is_in(stay_ids))

    # 3) Eksik lab parametreleri
    extra_labs = extract_extra_labs(stay_ids, icustay_map)
    df = df.join(extra_labs, on=["stay_id", "hour_bin"], how="left")
    print(f"   📊 Lab merge sonrası: {df.shape[1]} sütun")

    # 4) Ağırlık
    weight_df = extract_weight(icustay_map)
    df = df.join(weight_df, on="stay_id", how="left")

    # 5) Elixhauser skoru
    elix_df = compute_elixhauser(icustay_map)
    df = df.join(elix_df, on="stay_id", how="left")
    df = df.with_columns(pl.col("elixhauser_score").fill_null(0))

    # 6) ICU readmission
    readmit_df = compute_icu_readmission(icustay_map)
    df = df.join(readmit_df, on="stay_id", how="left")
    df = df.with_columns(pl.col("icu_readmission").fill_null(0))

    # 7) Gender encoding: M=0, F=1
    df = df.with_columns(
        pl.when(pl.col("gender") == "F")
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("gender")
    )

    # 8) Türetilen özellikler
    df = compute_derived_features(df)

    # 9) Imputation
    df = apply_imputation(df, STATE_FEATURES)

    # 10) Final sütun seçimi
    available = set(df.columns)
    missing_cols = [c for c in STATE_FEATURES if c not in available]
    if missing_cols:
        print(f"   ⚠️  Eksik sütunlar (0 ile doldurulacak): {missing_cols}")
        for c in missing_cols:
            df = df.with_columns(pl.lit(0.0).alias(c))

    final_cols = META_COLUMNS + STATE_FEATURES
    df = df.select(final_cols)

    # 11) Kaydet
    print(f"\n💾 state.parquet kaydediliyor → {STATE_PARQUET_PATH}")
    df.write_parquet(STATE_PARQUET_PATH)

    # 12) Özet
    print("\n" + "=" * 60)
    print("  ✅ Pipeline tamamlandı!")
    print(f"  📊 Shape: {df.shape[0]:,} satır × {df.shape[1]} sütun")
    print(f"  📂 Çıktı: {STATE_PARQUET_PATH}")
    print("=" * 60)

    # Null özet
    null_summary = df.null_count()
    print("\n📋 Null sayıları:")
    for col in STATE_FEATURES:
        nc = null_summary[col][0]
        if nc > 0:
            pct = nc / df.shape[0] * 100
            print(f"   {col}: {nc:,} ({pct:.1f}%)")

    print("\n🎯 İlk 3 satır:")
    print(df.head(3))
