# 02 — Feature Engineering (MIMIC-IV Sepsis DRL)

> **Input:** `mimic_hourly_binned.parquet`  
> **Output:** `mimic_hourly_binned_feature_engineered.parquet`  

2024-2025 literatürüne uygun ~20-25 feature'lık standart state vektörü oluşturma:

| # | Adım | Yeni Feature(lar) |
|---|------|--------------------|
| 1 | Veri yükleme & kontrol | — |
| 2 | Norepinefrin eşdeğeri | `total_vaso_equiv` |
| 3 | Sıvı dengesi (4h) | `fluid_balance_4h` |
| 4 | SOFA skoru (6 organ) | `sofa_score` |
| 5 | Mek. ventilasyon & Şok indeksi | `mechanical_ventilation`, `shock_index` |
| 6 | Lag features | `prev_fluid_dose`, `prev_vaso_dose` |
| 7 | Final state vector & kayıt | ~20-25 feature → parquet |


```python
import polars as pl
from pathlib import Path

# ─── Paths ─────────────────────────────────────────
PROJECT_ROOT = Path.cwd().parent  # notebooks/ → proje kökü
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
INPUT_PATH   = DATA_DIR / "mimic_hourly_binned.parquet"
OUTPUT_PATH  = DATA_DIR / "mimic_hourly_binned_feature_engineered.parquet"

print(f"Input  : {INPUT_PATH}")
print(f"Output : {OUTPUT_PATH}")
print(f"Dosya mevcut: {INPUT_PATH.exists()}")
```

    Input  : /Users/enesdemir/Documents/mimic-sepsis-drl/data/processed/mimic_hourly_binned.parquet
    Output : /Users/enesdemir/Documents/mimic-sepsis-drl/data/processed/mimic_hourly_binned_feature_engineered.parquet
    Dosya mevcut: True


## 1. Veri Yükleme & Şema Kontrolü


```python
df = pl.read_parquet(INPUT_PATH)

print(f"Shape: {df.shape}")
print(f"Sütunlar ({len(df.columns)}):")
for col in df.columns:
    null_pct = df[col].null_count() / len(df) * 100
    print(f"  {col:30s}  dtype={str(df[col].dtype):12s}  null={null_pct:.1f}%")
```

    Shape: (8808129, 43)
    Sütunlar (43):
      stay_id                         dtype=Int64         null=0.0%
      hour_bin                        dtype=Datetime(time_unit='us', time_zone=None)  null=0.0%
      heart_rate                      dtype=Float64       null=0.4%
      sbp                             dtype=Float64       null=0.8%
      dbp                             dtype=Float64       null=0.8%
      mbp                             dtype=Float64       null=0.8%
      resp_rate                       dtype=Float64       null=0.5%
      spo2                            dtype=Float64       null=0.5%
      temp_c                          dtype=Float64       null=83.4%
      fio2                            dtype=Float64       null=32.4%
      lactate                         dtype=Float64       null=28.0%
      creatinine                      dtype=Float64       null=5.5%
      bilirubin_total                 dtype=Float64       null=37.3%
      platelet                        dtype=Float64       null=5.8%
      wbc                             dtype=Float64       null=5.8%
      bun                             dtype=Float64       null=5.5%
      glucose                         dtype=Float64       null=6.8%
      sodium                          dtype=Float64       null=6.1%
      potassium                       dtype=Float64       null=5.8%
      hemoglobin                      dtype=Float64       null=5.8%
      hematocrit                      dtype=Float64       null=5.6%
      bicarbonate                     dtype=Float64       null=5.5%
      chloride                        dtype=Float64       null=5.4%
      anion_gap                       dtype=Float64       null=5.7%
      inr                             dtype=Float64       null=12.0%
      pao2                            dtype=Float64       null=25.9%
      paco2                           dtype=Float64       null=25.9%
      ph                              dtype=Float64       null=24.6%
      urine_output                    dtype=Float64       null=5.8%
      norepinephrine_dose             dtype=Float64       null=18.3%
      epinephrine_dose                dtype=Float64       null=18.3%
      phenylephrine_dose              dtype=Float64       null=18.3%
      vasopressin_dose                dtype=Float64       null=18.3%
      dopamine_dose                   dtype=Float64       null=18.3%
      dobutamine_dose                 dtype=Float64       null=18.3%
      crystalloid_ml                  dtype=Float64       null=18.3%
      gcs_eye                         dtype=Float64       null=1.2%
      gcs_motor                       dtype=Float64       null=1.3%
      gcs_verbal                      dtype=Float64       null=1.2%
      gcs_total                       dtype=Float64       null=1.2%
      gender                          dtype=String        null=0.0%
      age                             dtype=Int64         null=0.0%
      admission_type                  dtype=String        null=0.0%



```python
df.head(5)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 43)</small><table border="1" class="dataframe"><thead><tr><th>stay_id</th><th>hour_bin</th><th>heart_rate</th><th>sbp</th><th>dbp</th><th>mbp</th><th>resp_rate</th><th>spo2</th><th>temp_c</th><th>fio2</th><th>lactate</th><th>creatinine</th><th>bilirubin_total</th><th>platelet</th><th>wbc</th><th>bun</th><th>glucose</th><th>sodium</th><th>potassium</th><th>hemoglobin</th><th>hematocrit</th><th>bicarbonate</th><th>chloride</th><th>anion_gap</th><th>inr</th><th>pao2</th><th>paco2</th><th>ph</th><th>urine_output</th><th>norepinephrine_dose</th><th>epinephrine_dose</th><th>phenylephrine_dose</th><th>vasopressin_dose</th><th>dopamine_dose</th><th>dobutamine_dose</th><th>crystalloid_ml</th><th>gcs_eye</th><th>gcs_motor</th><th>gcs_verbal</th><th>gcs_total</th><th>gender</th><th>age</th><th>admission_type</th></tr><tr><td>i64</td><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>str</td><td>i64</td><td>str</td></tr></thead><tbody><tr><td>30000153</td><td>2174-09-29 12:00:00</td><td>100.0</td><td>136.0</td><td>74.0</td><td>89.0</td><td>18.0</td><td>100.0</td><td>null</td><td>75.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>35.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>280.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 13:00:00</td><td>104.0</td><td>132.0</td><td>74.5</td><td>84.0</td><td>16.0</td><td>100.0</td><td>null</td><td>75.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>35.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>221.0</td><td>45.0</td><td>7.3</td><td>280.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 14:00:00</td><td>83.0</td><td>131.0</td><td>61.0</td><td>80.0</td><td>16.0</td><td>100.0</td><td>null</td><td>75.0</td><td>2.1</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>35.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>263.0</td><td>45.0</td><td>7.3</td><td>45.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 15:00:00</td><td>92.0</td><td>123.0</td><td>65.0</td><td>84.0</td><td>14.0</td><td>100.0</td><td>null</td><td>50.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>31.7</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>263.0</td><td>45.0</td><td>7.3</td><td>50.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 16:00:00</td><td>83.0</td><td>109.0</td><td>55.0</td><td>71.0</td><td>16.0</td><td>100.0</td><td>null</td><td>50.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>31.7</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>215.0</td><td>42.0</td><td>7.31</td><td>50.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>941.299999</td><td>4.0</td><td>6.0</td><td>1.0</td><td>11.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr></tbody></table></div>



## 2. Norepinefrin Eşdeğeri (Vazopressor Standardizasyonu)

Farklı vazopressorleri tek skalaya indirgeme (2024-2025 standardı):

| İlaç | Dönüşüm Oranı |
|------|---------------|
| Norepinefrin | ×1.0 |
| Epinefrin | ×0.1 |
| Phenylefrin | ×0.1 |
| Vasopressin | ×0.4 |
| Dopamin | ×0.01 |
| Dobutamin | ×0.0 (inotrop, ayrı tutulur) |


```python
# ─── Norepinefrin eşdeğeri dönüşüm oranları ───────
VASO_CONVERSION = {
    "norepinephrine_dose": 1.0,
    "epinephrine_dose":    0.1,
    "phenylephrine_dose":  0.1,
    "vasopressin_dose":    0.4,
    "dopamine_dose":       0.01,
    "dobutamine_dose":     0.0,   # İnotrop etki — vazopressor değil
}

# Mevcut vazo kolonlarını kontrol et
available_vaso_cols = [c for c in VASO_CONVERSION if c in df.columns]
print(f"Mevcut vazopressor kolonları: {available_vaso_cols}")

# Her bir ilacın eşdeğerini hesapla
equiv_exprs = []
for col, ratio in VASO_CONVERSION.items():
    if col in df.columns:
        equiv_exprs.append(
            (pl.col(col).fill_null(0) * ratio).alias(col.replace("_dose", "_equiv"))
        )

df = df.with_columns(equiv_exprs)

# Toplam vazopressor eşdeğeri
equiv_cols = [col.replace("_dose", "_equiv") for col in available_vaso_cols]
df = df.with_columns(
    pl.sum_horizontal([pl.col(c) for c in equiv_cols]).alias("total_vaso_equiv")
)

print(f"\ntotal_vaso_equiv istatistikleri:")
df.select("total_vaso_equiv").describe()
```

    Mevcut vazopressor kolonları: ['norepinephrine_dose', 'epinephrine_dose', 'phenylephrine_dose', 'vasopressin_dose', 'dopamine_dose', 'dobutamine_dose']
    
    total_vaso_equiv istatistikleri:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>total_vaso_equiv</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>8.808129e6</td></tr><tr><td>&quot;null_count&quot;</td><td>0.0</td></tr><tr><td>&quot;mean&quot;</td><td>0.277242</td></tr><tr><td>&quot;std&quot;</td><td>1.735346</td></tr><tr><td>&quot;min&quot;</td><td>0.0</td></tr><tr><td>&quot;25%&quot;</td><td>0.0</td></tr><tr><td>&quot;50%&quot;</td><td>0.0</td></tr><tr><td>&quot;75%&quot;</td><td>0.0</td></tr><tr><td>&quot;max&quot;</td><td>1100.354014</td></tr></tbody></table></div>



## 3. Sıvı Dengesi (Net Fluid Balance — 4h)

```
fluid_balance_4h = crystalloid_ml − urine_output
```

- **Pozitif:** Ödem riski  
- **Negatif:** Hipovolemi


```python
df = df.with_columns(
    (
        pl.col("crystalloid_ml").fill_null(0) 
        - pl.col("urine_output").fill_null(0)
    ).alias("fluid_balance_4h")
)

print("fluid_balance_4h istatistikleri:")
df.select("fluid_balance_4h").describe()
```

    fluid_balance_4h istatistikleri:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>fluid_balance_4h</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>8.808129e6</td></tr><tr><td>&quot;null_count&quot;</td><td>0.0</td></tr><tr><td>&quot;mean&quot;</td><td>31.625266</td></tr><tr><td>&quot;std&quot;</td><td>1466.576125</td></tr><tr><td>&quot;min&quot;</td><td>-876537.000001</td></tr><tr><td>&quot;25%&quot;</td><td>-145.931016</td></tr><tr><td>&quot;50%&quot;</td><td>-16.889828</td></tr><tr><td>&quot;75%&quot;</td><td>114.250006</td></tr><tr><td>&quot;max&quot;</td><td>1.0003e6</td></tr></tbody></table></div>



## 4. SOFA Skoru (Sequential Organ Failure Assessment)

Vincent et al. (1996) — 6 organ, her biri 0-4 puan, toplam **0-24**.

| Organ | Metrik | Puan Aralığı |
|-------|--------|--------------|
| Solunum | PaO2/FiO2 | 0-4 |
| Kardiyovasküler | MAP + Vazo dozu | 0-4 |
| Böbrek | Kreatinin ∨ İdrar | 0-4 |
| Nörolojik | GCS | 0-4 |
| Koagülasyon | Trombosit | 0-4 |
| Karaciğer | Bilirubin | 0-4 |


```python
# ═══════════════════════════════════════════════════
# 4A. SOFA — Solunum (PaO2/FiO2)
# ═══════════════════════════════════════════════════

# FiO2: chartevents'te % olarak (21-100), orana çevir
df = df.with_columns(
    pl.when(pl.col("fio2") > 1.0)
    .then(pl.col("fio2") / 100.0)   # 21% → 0.21
    .otherwise(pl.col("fio2"))        # Zaten oran ise olduğu gibi
    .alias("fio2_ratio")
)

# PF oranı
df = df.with_columns(
    (pl.col("pao2") / pl.col("fio2_ratio")).alias("pf_ratio")
)

# Mekanik ventilasyon flag (SOFA resp 3-4 için gerekli)
df = df.with_columns(
    pl.when(
        (pl.col("fio2").is_not_null() & (pl.col("fio2") > 21))
    )
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .alias("mechanical_ventilation")
)

# SOFA Respiratory skoru
df = df.with_columns(
    pl.when(pl.col("pf_ratio").is_null())
    .then(pl.lit(None).cast(pl.Int32))
    .when(pl.col("pf_ratio") > 400)
    .then(pl.lit(0))
    .when(pl.col("pf_ratio") > 300)
    .then(pl.lit(1))
    .when(pl.col("pf_ratio") > 200)
    .then(pl.lit(2))
    .when((pl.col("pf_ratio") > 100) & (pl.col("mechanical_ventilation") == 1))
    .then(pl.lit(3))
    .when((pl.col("pf_ratio") <= 100) & (pl.col("mechanical_ventilation") == 1))
    .then(pl.lit(4))
    .otherwise(pl.lit(2))  # PF<=200 ama ventilasyon yoksa max 2
    .alias("sofa_resp")
)

print("SOFA Respiratory dağılımı:")
df.group_by("sofa_resp").len().sort("sofa_resp")
```

    SOFA Respiratory dağılımı:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (6, 2)</small><table border="1" class="dataframe"><thead><tr><th>sofa_resp</th><th>len</th></tr><tr><td>i32</td><td>u32</td></tr></thead><tbody><tr><td>null</td><td>3338801</td></tr><tr><td>0</td><td>683094</td></tr><tr><td>1</td><td>842241</td></tr><tr><td>2</td><td>1496423</td></tr><tr><td>3</td><td>1714091</td></tr><tr><td>4</td><td>733479</td></tr></tbody></table></div>




```python
# ═══════════════════════════════════════════════════
# 4B. SOFA — Kardiyovasküler
#     MAP + vazopressor dozu (norepinefrin eşdeğeri)
# ═══════════════════════════════════════════════════

df = df.with_columns(
    pl.when(pl.col("mbp").is_null() & pl.col("total_vaso_equiv").is_null())
    .then(pl.lit(None).cast(pl.Int32))
    .when((pl.col("mbp").fill_null(70) >= 70) & (pl.col("total_vaso_equiv").fill_null(0) == 0))
    .then(pl.lit(0))
    .when(pl.col("mbp").fill_null(70) < 70)
    .then(pl.lit(1))
    .when(pl.col("total_vaso_equiv") <= 0.1)
    .then(pl.lit(2))
    .when(pl.col("total_vaso_equiv") <= 0.5)
    .then(pl.lit(3))
    .otherwise(pl.lit(4))
    .alias("sofa_cardio")
)

print("SOFA Cardiovascular dağılımı:")
df.group_by("sofa_cardio").len().sort("sofa_cardio")
```

    SOFA Cardiovascular dağılımı:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 2)</small><table border="1" class="dataframe"><thead><tr><th>sofa_cardio</th><th>len</th></tr><tr><td>i32</td><td>u32</td></tr></thead><tbody><tr><td>0</td><td>5795830</td></tr><tr><td>1</td><td>2309005</td></tr><tr><td>2</td><td>94269</td></tr><tr><td>3</td><td>239549</td></tr><tr><td>4</td><td>369476</td></tr></tbody></table></div>




```python
# ═══════════════════════════════════════════════════
# 4C. SOFA — Böbrek (Kreatinin)
#     İdrar 24h kriteri burada uygulanmaz (saatlik veri),
#     yalnızca kreatinin kullanılır.
# ═══════════════════════════════════════════════════

df = df.with_columns(
    pl.when(pl.col("creatinine").is_null())
    .then(pl.lit(None).cast(pl.Int32))
    .when(pl.col("creatinine") < 1.2)
    .then(pl.lit(0))
    .when(pl.col("creatinine") < 2.0)
    .then(pl.lit(1))
    .when(pl.col("creatinine") < 3.5)
    .then(pl.lit(2))
    .when(pl.col("creatinine") < 5.0)
    .then(pl.lit(3))
    .otherwise(pl.lit(4))
    .alias("sofa_renal")
)

print("SOFA Renal dağılımı:")
df.group_by("sofa_renal").len().sort("sofa_renal")
```

    SOFA Renal dağılımı:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (6, 2)</small><table border="1" class="dataframe"><thead><tr><th>sofa_renal</th><th>len</th></tr><tr><td>i32</td><td>u32</td></tr></thead><tbody><tr><td>null</td><td>480550</td></tr><tr><td>0</td><td>4975961</td></tr><tr><td>1</td><td>1686184</td></tr><tr><td>2</td><td>979055</td></tr><tr><td>3</td><td>378714</td></tr><tr><td>4</td><td>307665</td></tr></tbody></table></div>




```python
# ═══════════════════════════════════════════════════
# 4D. SOFA — Nörolojik (GCS)
#     Sedasyonlu hasta: forward-fill zaten pipeline'da uygulandı
# ═══════════════════════════════════════════════════

df = df.with_columns(
    pl.when(pl.col("gcs_total").is_null())
    .then(pl.lit(None).cast(pl.Int32))
    .when(pl.col("gcs_total") >= 15)
    .then(pl.lit(0))
    .when(pl.col("gcs_total") >= 13)
    .then(pl.lit(1))
    .when(pl.col("gcs_total") >= 10)
    .then(pl.lit(2))
    .when(pl.col("gcs_total") >= 6)
    .then(pl.lit(3))
    .otherwise(pl.lit(4))
    .alias("sofa_neuro")
)

print("SOFA Neurological dağılımı:")
df.group_by("sofa_neuro").len().sort("sofa_neuro")
```

    SOFA Neurological dağılımı:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (6, 2)</small><table border="1" class="dataframe"><thead><tr><th>sofa_neuro</th><th>len</th></tr><tr><td>i32</td><td>u32</td></tr></thead><tbody><tr><td>null</td><td>105917</td></tr><tr><td>0</td><td>3826558</td></tr><tr><td>1</td><td>1304030</td></tr><tr><td>2</td><td>1538324</td></tr><tr><td>3</td><td>1223074</td></tr><tr><td>4</td><td>810226</td></tr></tbody></table></div>




```python
# ═══════════════════════════════════════════════════
# 4E. SOFA — Koagülasyon (Trombosit)
# ═══════════════════════════════════════════════════

df = df.with_columns(
    pl.when(pl.col("platelet").is_null())
    .then(pl.lit(None).cast(pl.Int32))
    .when(pl.col("platelet") > 150)
    .then(pl.lit(0))
    .when(pl.col("platelet") > 100)
    .then(pl.lit(1))
    .when(pl.col("platelet") > 50)
    .then(pl.lit(2))
    .when(pl.col("platelet") > 20)
    .then(pl.lit(3))
    .otherwise(pl.lit(4))
    .alias("sofa_coag")
)

print("SOFA Coagulation dağılımı:")
df.group_by("sofa_coag").len().sort("sofa_coag")
```

    SOFA Coagulation dağılımı:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (6, 2)</small><table border="1" class="dataframe"><thead><tr><th>sofa_coag</th><th>len</th></tr><tr><td>i32</td><td>u32</td></tr></thead><tbody><tr><td>null</td><td>509026</td></tr><tr><td>0</td><td>5604087</td></tr><tr><td>1</td><td>1422701</td></tr><tr><td>2</td><td>882591</td></tr><tr><td>3</td><td>321949</td></tr><tr><td>4</td><td>67775</td></tr></tbody></table></div>




```python
# ═══════════════════════════════════════════════════
# 4F. SOFA — Karaciğer (Bilirubin)
# ═══════════════════════════════════════════════════

df = df.with_columns(
    pl.when(pl.col("bilirubin_total").is_null())
    .then(pl.lit(None).cast(pl.Int32))
    .when(pl.col("bilirubin_total") < 1.2)
    .then(pl.lit(0))
    .when(pl.col("bilirubin_total") < 2.0)
    .then(pl.lit(1))
    .when(pl.col("bilirubin_total") < 6.0)
    .then(pl.lit(2))
    .when(pl.col("bilirubin_total") < 12.0)
    .then(pl.lit(3))
    .otherwise(pl.lit(4))
    .alias("sofa_liver")
)

print("SOFA Liver dağılımı:")
df.group_by("sofa_liver").len().sort("sofa_liver")
```

    SOFA Liver dağılımı:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (6, 2)</small><table border="1" class="dataframe"><thead><tr><th>sofa_liver</th><th>len</th></tr><tr><td>i32</td><td>u32</td></tr></thead><tbody><tr><td>null</td><td>3284098</td></tr><tr><td>0</td><td>3892956</td></tr><tr><td>1</td><td>585822</td></tr><tr><td>2</td><td>592125</td></tr><tr><td>3</td><td>210916</td></tr><tr><td>4</td><td>242212</td></tr></tbody></table></div>




```python
# ═══════════════════════════════════════════════════
# 4G. Toplam SOFA Skoru (0-24)
# ═══════════════════════════════════════════════════

sofa_components = [
    "sofa_resp", "sofa_cardio", "sofa_renal",
    "sofa_neuro", "sofa_coag", "sofa_liver"
]

df = df.with_columns(
    pl.sum_horizontal([pl.col(c).fill_null(0) for c in sofa_components])
    .alias("sofa_score")
)

print("SOFA Score istatistikleri:")
df.select("sofa_score").describe()
```

    SOFA Score istatistikleri:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>sofa_score</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>8.808129e6</td></tr><tr><td>&quot;null_count&quot;</td><td>0.0</td></tr><tr><td>&quot;mean&quot;</td><td>4.734691</td></tr><tr><td>&quot;std&quot;</td><td>3.566224</td></tr><tr><td>&quot;min&quot;</td><td>0.0</td></tr><tr><td>&quot;25%&quot;</td><td>2.0</td></tr><tr><td>&quot;50%&quot;</td><td>4.0</td></tr><tr><td>&quot;75%&quot;</td><td>7.0</td></tr><tr><td>&quot;max&quot;</td><td>23.0</td></tr></tbody></table></div>




```python
# Hızlı doğrulama
assert df["sofa_score"].min() >= 0, "SOFA min < 0!"
assert df["sofa_score"].max() <= 24, "SOFA max > 24!"
print(f"✅ SOFA skoru aralığı: [{df['sofa_score'].min()}, {df['sofa_score'].max()}]")
```

    ✅ SOFA skoru aralığı: [0, 23]


## 5. Mekanik Ventilasyon & Şok İndeksi

- **Mekanik Ventilasyon:** Zaten yukarıda (SOFA Resp) hesaplandı — `FiO2 > 21%` ise 1  
- **Şok İndeksi:** `HR / SBP` — Normal 0.5-0.7, Yüksek >1.0 (şok belirtisi)


```python
# ─── Şok İndeksi ──────────────────────────────────
df = df.with_columns(
    (pl.col("heart_rate") / pl.col("sbp")).alias("shock_index")
)

print("Shock Index istatistikleri:")
df.select("shock_index").describe()
```

    Shock Index istatistikleri:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>shock_index</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>8.739273e6</td></tr><tr><td>&quot;null_count&quot;</td><td>68856.0</td></tr><tr><td>&quot;mean&quot;</td><td>NaN</td></tr><tr><td>&quot;std&quot;</td><td>NaN</td></tr><tr><td>&quot;min&quot;</td><td>-3094.807692</td></tr><tr><td>&quot;25%&quot;</td><td>0.589147</td></tr><tr><td>&quot;50%&quot;</td><td>0.712121</td></tr><tr><td>&quot;75%&quot;</td><td>0.86</td></tr><tr><td>&quot;max&quot;</td><td>inf</td></tr></tbody></table></div>



## 6. Lag Features (Önceki Timestep Dozları)

Agent "şimdi ne yapmalıyım?" derken "az önce ne yaptım?" bilmeli.  
İlaç kümülasyonu nedeniyle zorunlu.

```
prev_fluid_dose(t) = crystalloid_ml(t-1)
prev_vaso_dose(t)  = total_vaso_equiv(t-1)
```

İlk timestep → `null` kalır (sonradan impute edilecek).


```python
# Sıralama garanti
df = df.sort("stay_id", "hour_bin")

# Lag features (stay_id içinde shift)
df = df.with_columns([
    pl.col("crystalloid_ml").shift(1).over("stay_id").alias("prev_fluid_dose"),
    pl.col("total_vaso_equiv").shift(1).over("stay_id").alias("prev_vaso_dose"),
])

print("Lag features (ilk 10 satır, tek stay_id):")
sample_stay = df["stay_id"].drop_nulls()[0]
df.filter(pl.col("stay_id") == sample_stay).select(
    "stay_id", "hour_bin", "crystalloid_ml", "prev_fluid_dose",
    "total_vaso_equiv", "prev_vaso_dose"
).head(10)
```

    Lag features (ilk 10 satır, tek stay_id):





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 6)</small><table border="1" class="dataframe"><thead><tr><th>stay_id</th><th>hour_bin</th><th>crystalloid_ml</th><th>prev_fluid_dose</th><th>total_vaso_equiv</th><th>prev_vaso_dose</th></tr><tr><td>i64</td><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>30000153</td><td>2174-09-29 12:00:00</td><td>30.0</td><td>null</td><td>0.0</td><td>null</td></tr><tr><td>30000153</td><td>2174-09-29 13:00:00</td><td>30.0</td><td>30.0</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 14:00:00</td><td>30.0</td><td>30.0</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 15:00:00</td><td>30.0</td><td>30.0</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 16:00:00</td><td>941.299999</td><td>30.0</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 17:00:00</td><td>941.299999</td><td>941.299999</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 18:00:00</td><td>941.299999</td><td>941.299999</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 19:00:00</td><td>941.299999</td><td>941.299999</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 20:00:00</td><td>941.299999</td><td>941.299999</td><td>0.0</td><td>0.0</td></tr><tr><td>30000153</td><td>2174-09-29 21:00:00</td><td>199.999995</td><td>941.299999</td><td>0.0</td><td>0.0</td></tr></tbody></table></div>



## 7. Final State Vector & Parquet Kayıt

2024-2025 MIMIC-IV Sepsis DRL standardı — ~20-25 feature:

| Kategori | Feature'lar |
|----------|-------------|
| **Lag** | `prev_fluid_dose`, `prev_vaso_dose` |
| **Vitals** | `heart_rate`, `sbp`, `dbp`, `mbp`, `resp_rate`, `spo2`, `temp_c` |
| **Labs** | `lactate`, `creatinine`, `platelet`, `bun`, `wbc`, `bilirubin_total` |
| **Organ** | `sofa_score`, `gcs_total`, `urine_output` |
| **Hemodinamik** | `shock_index`, `mechanical_ventilation` |
| **Sıvı** | `fluid_balance_4h` |
| **Demografi** | `age`, `gender` |


```python
# ─── State vector tanımı ───────────────────────────
STATE_FEATURES = [
    # Lag
    "prev_fluid_dose", "prev_vaso_dose",
    # Vitals
    "heart_rate", "sbp", "dbp", "mbp", "resp_rate", "spo2", "temp_c",
    # Labs
    "lactate", "creatinine", "platelet", "bun", "wbc", "bilirubin_total",
    # Organ function
    "sofa_score", "gcs_total", "urine_output",
    # Hemodynamic indices
    "shock_index", "mechanical_ventilation",
    # Fluid
    "fluid_balance_4h",
    # Demographics
    "age", "gender",
]

# Meta sütunlar (ID + zaman)
META_COLS = ["stay_id", "hour_bin"]

# Mevcut olan feature'ları filtrele
available_features = [f for f in STATE_FEATURES if f in df.columns]
missing_features   = [f for f in STATE_FEATURES if f not in df.columns]

print(f"State vector boyutu: {len(available_features)} feature")
if missing_features:
    print(f"⚠️  Eksik feature'lar: {missing_features}")
else:
    print("✅ Tüm feature'lar mevcut!")
```

    State vector boyutu: 23 feature
    ✅ Tüm feature'lar mevcut!



```python
# ─── Gender encode (M=0, F=1) ─────────────────────
if "gender" in df.columns:
    df = df.with_columns(
        pl.when(pl.col("gender") == "M")
        .then(pl.lit(0))
        .when(pl.col("gender") == "F")
        .then(pl.lit(1))
        .otherwise(pl.lit(None))
        .cast(pl.Int32)
        .alias("gender")
    )

print("Gender dağılımı:")
df.group_by("gender").len().sort("gender")
```

    Gender dağılımı:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (2, 2)</small><table border="1" class="dataframe"><thead><tr><th>gender</th><th>len</th></tr><tr><td>i32</td><td>u32</td></tr></thead><tbody><tr><td>0</td><td>5074695</td></tr><tr><td>1</td><td>3733434</td></tr></tbody></table></div>




```python
# ─── Final DataFrame oluştur ──────────────────────
df_final = df.select(META_COLS + available_features)

print(f"Final shape: {df_final.shape}")
print(f"Sütunlar ({len(df_final.columns)}): {df_final.columns}")
print("\nÖzet istatistikler:")
df_final.describe()
```

    Final shape: (8808129, 25)
    Sütunlar (25): ['stay_id', 'hour_bin', 'prev_fluid_dose', 'prev_vaso_dose', 'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2', 'temp_c', 'lactate', 'creatinine', 'platelet', 'bun', 'wbc', 'bilirubin_total', 'sofa_score', 'gcs_total', 'urine_output', 'shock_index', 'mechanical_ventilation', 'fluid_balance_4h', 'age', 'gender']
    
    Özet istatistikler:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 26)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>stay_id</th><th>hour_bin</th><th>prev_fluid_dose</th><th>prev_vaso_dose</th><th>heart_rate</th><th>sbp</th><th>dbp</th><th>mbp</th><th>resp_rate</th><th>spo2</th><th>temp_c</th><th>lactate</th><th>creatinine</th><th>platelet</th><th>bun</th><th>wbc</th><th>bilirubin_total</th><th>sofa_score</th><th>gcs_total</th><th>urine_output</th><th>shock_index</th><th>mechanical_ventilation</th><th>fluid_balance_4h</th><th>age</th><th>gender</th></tr><tr><td>str</td><td>f64</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>8.808129e6</td><td>&quot;8808129&quot;</td><td>7.117971e6</td><td>8.713671e6</td><td>8.770031e6</td><td>8.740315e6</td><td>8.740024e6</td><td>8.741935e6</td><td>8.760778e6</td><td>8.764642e6</td><td>1.465057e6</td><td>6.340328e6</td><td>8.327579e6</td><td>8.299103e6</td><td>8.326449e6</td><td>8.295935e6</td><td>5.524031e6</td><td>8.808129e6</td><td>8.702212e6</td><td>8.293822e6</td><td>8.739273e6</td><td>8.808129e6</td><td>8.808129e6</td><td>8.808129e6</td><td>8.808129e6</td></tr><tr><td>&quot;null_count&quot;</td><td>0.0</td><td>&quot;0&quot;</td><td>1.690158e6</td><td>94458.0</td><td>38098.0</td><td>67814.0</td><td>68105.0</td><td>66194.0</td><td>47351.0</td><td>43487.0</td><td>7.343072e6</td><td>2.467801e6</td><td>480550.0</td><td>509026.0</td><td>481680.0</td><td>512194.0</td><td>3.284098e6</td><td>0.0</td><td>105917.0</td><td>514307.0</td><td>68856.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;mean&quot;</td><td>3.4974e7</td><td>&quot;2153-10-15 03:38:30.535507&quot;</td><td>251.733846</td><td>0.27857</td><td>87.819938</td><td>120.478139</td><td>65.184255</td><td>84.512153</td><td>21.169934</td><td>137.767256</td><td>38.354066</td><td>3.263103</td><td>1.481856</td><td>219.682421</td><td>30.943375</td><td>12.125175</td><td>2.209595</td><td>4.734691</td><td>11.923467</td><td>184.733394</td><td>NaN</td><td>0.663667</td><td>31.625266</td><td>62.641503</td><td>0.423862</td></tr><tr><td>&quot;std&quot;</td><td>2.8843e6</td><td>null</td><td>1580.41945</td><td>1.732863</td><td>3797.403494</td><td>491.228532</td><td>259.152687</td><td>4828.722159</td><td>2407.519375</td><td>19406.703678</td><td>9.811226</td><td>1433.422544</td><td>1.46838</td><td>132.416995</td><td>25.204517</td><td>8.458586</td><td>5.210624</td><td>3.566224</td><td>3.831584</td><td>364.241888</td><td>NaN</td><td>0.472454</td><td>1466.576125</td><td>16.111648</td><td>0.494169</td></tr><tr><td>&quot;min&quot;</td><td>3.0000153e7</td><td>&quot;2110-01-11 10:00:00&quot;</td><td>0.0</td><td>0.0</td><td>-241395.0</td><td>-94.0</td><td>-40.0</td><td>-9806.0</td><td>0.0</td><td>-951234.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>5.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1.0</td><td>-3765.0</td><td>-3094.807692</td><td>0.0</td><td>-876537.000001</td><td>18.0</td><td>0.0</td></tr><tr><td>&quot;25%&quot;</td><td>3.2477246e7</td><td>&quot;2133-12-07 02:00:00&quot;</td><td>47.900002</td><td>0.0</td><td>73.0</td><td>104.0</td><td>53.0</td><td>69.0</td><td>16.0</td><td>95.0</td><td>36.6</td><td>1.0</td><td>0.7</td><td>131.0</td><td>14.0</td><td>7.7</td><td>0.4</td><td>2.0</td><td>10.0</td><td>50.0</td><td>0.589147</td><td>0.0</td><td>-145.931016</td><td>53.0</td><td>0.0</td></tr><tr><td>&quot;50%&quot;</td><td>3.4965363e7</td><td>&quot;2153-08-17 00:00:00&quot;</td><td>100.0</td><td>0.0</td><td>85.0</td><td>118.0</td><td>62.0</td><td>78.0</td><td>19.25</td><td>97.0</td><td>37.1</td><td>1.4</td><td>1.0</td><td>196.0</td><td>23.0</td><td>10.6</td><td>0.6</td><td>4.0</td><td>14.0</td><td>120.0</td><td>0.712121</td><td>1.0</td><td>-16.889828</td><td>64.0</td><td>0.0</td></tr><tr><td>&quot;75%&quot;</td><td>3.7460082e7</td><td>&quot;2173-11-27 22:00:00&quot;</td><td>295.000009</td><td>0.0</td><td>98.0</td><td>134.0</td><td>73.0</td><td>89.0</td><td>24.0</td><td>99.0</td><td>37.6</td><td>1.9</td><td>1.7</td><td>280.0</td><td>39.0</td><td>14.5</td><td>1.4</td><td>7.0</td><td>15.0</td><td>250.0</td><td>0.86</td><td>1.0</td><td>114.250006</td><td>75.0</td><td>1.0</td></tr><tr><td>&quot;max&quot;</td><td>3.9999858e7</td><td>&quot;2214-08-11 05:00:00&quot;</td><td>1.0004e6</td><td>1100.354014</td><td>1e7</td><td>1.00311e6</td><td>114109.0</td><td>8.99909e6</td><td>7.0004e6</td><td>9.9e6</td><td>987.4</td><td>1.276103e6</td><td>80.0</td><td>2385.0</td><td>305.0</td><td>572.5</td><td>87.2</td><td>23.0</td><td>15.0</td><td>876587.0</td><td>inf</td><td>1.0</td><td>1.0003e6</td><td>91.0</td><td>1.0</td></tr></tbody></table></div>




```python
# ─── Null yüzdeleri ────────────────────────────────
print("Null yüzdeleri (%):\n")
for col in available_features:
    null_pct = df_final[col].null_count() / len(df_final) * 100
    bar = "█" * int(null_pct // 2)
    print(f"  {col:30s} {null_pct:6.1f}%  {bar}")
```

    Null yüzdeleri (%):
    
      prev_fluid_dose                  19.2%  █████████
      prev_vaso_dose                    1.1%  
      heart_rate                        0.4%  
      sbp                               0.8%  
      dbp                               0.8%  
      mbp                               0.8%  
      resp_rate                         0.5%  
      spo2                              0.5%  
      temp_c                           83.4%  █████████████████████████████████████████
      lactate                          28.0%  ██████████████
      creatinine                        5.5%  ██
      platelet                          5.8%  ██
      bun                               5.5%  ██
      wbc                               5.8%  ██
      bilirubin_total                  37.3%  ██████████████████
      sofa_score                        0.0%  
      gcs_total                         1.2%  
      urine_output                      5.8%  ██
      shock_index                       0.8%  
      mechanical_ventilation            0.0%  
      fluid_balance_4h                  0.0%  
      age                               0.0%  
      gender                            0.0%  



```python
# ─── Parquet'e yaz ─────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_final.write_parquet(OUTPUT_PATH)

# Doğrulama
file_size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
print(f"\n✅ Kaydedildi: {OUTPUT_PATH}")
print(f"   Boyut: {file_size_mb:.1f} MB")
print(f"   Satır: {df_final.shape[0]:,}")
print(f"   Sütun: {df_final.shape[1]}")
```

    
    ✅ Kaydedildi: /Users/enesdemir/Documents/mimic-sepsis-drl/data/processed/mimic_hourly_binned_feature_engineered.parquet
       Boyut: 176.1 MB
       Satır: 8,808,129
       Sütun: 25



```python
# ─── Okuma doğrulaması ─────────────────────────────
df_check = pl.read_parquet(OUTPUT_PATH)
assert df_check.shape == df_final.shape, "Shape mismatch!"
assert df_check.columns == df_final.columns, "Column mismatch!"
print(f"✅ Okuma doğrulaması başarılı: {df_check.shape}")
df_check.head(5)
```

    ✅ Okuma doğrulaması başarılı: (8808129, 25)





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 25)</small><table border="1" class="dataframe"><thead><tr><th>stay_id</th><th>hour_bin</th><th>prev_fluid_dose</th><th>prev_vaso_dose</th><th>heart_rate</th><th>sbp</th><th>dbp</th><th>mbp</th><th>resp_rate</th><th>spo2</th><th>temp_c</th><th>lactate</th><th>creatinine</th><th>platelet</th><th>bun</th><th>wbc</th><th>bilirubin_total</th><th>sofa_score</th><th>gcs_total</th><th>urine_output</th><th>shock_index</th><th>mechanical_ventilation</th><th>fluid_balance_4h</th><th>age</th><th>gender</th></tr><tr><td>i64</td><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i32</td><td>f64</td><td>f64</td><td>f64</td><td>i32</td><td>f64</td><td>i64</td><td>i32</td></tr></thead><tbody><tr><td>30000153</td><td>2174-09-29 12:00:00</td><td>null</td><td>null</td><td>100.0</td><td>136.0</td><td>74.0</td><td>89.0</td><td>18.0</td><td>100.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>3</td><td>9.0</td><td>280.0</td><td>0.735294</td><td>1</td><td>-250.0</td><td>61</td><td>0</td></tr><tr><td>30000153</td><td>2174-09-29 13:00:00</td><td>30.0</td><td>0.0</td><td>104.0</td><td>132.0</td><td>74.5</td><td>84.0</td><td>16.0</td><td>100.0</td><td>null</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>5</td><td>9.0</td><td>280.0</td><td>0.787879</td><td>1</td><td>-250.0</td><td>61</td><td>0</td></tr><tr><td>30000153</td><td>2174-09-29 14:00:00</td><td>30.0</td><td>0.0</td><td>83.0</td><td>131.0</td><td>61.0</td><td>80.0</td><td>16.0</td><td>100.0</td><td>null</td><td>2.1</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>4</td><td>9.0</td><td>45.0</td><td>0.633588</td><td>1</td><td>-15.0</td><td>61</td><td>0</td></tr><tr><td>30000153</td><td>2174-09-29 15:00:00</td><td>30.0</td><td>0.0</td><td>92.0</td><td>123.0</td><td>65.0</td><td>84.0</td><td>14.0</td><td>100.0</td><td>null</td><td>2.1</td><td>0.9</td><td>173.0</td><td>22.0</td><td>17.0</td><td>null</td><td>3</td><td>9.0</td><td>50.0</td><td>0.747967</td><td>1</td><td>-20.0</td><td>61</td><td>0</td></tr><tr><td>30000153</td><td>2174-09-29 16:00:00</td><td>30.0</td><td>0.0</td><td>83.0</td><td>109.0</td><td>55.0</td><td>71.0</td><td>16.0</td><td>100.0</td><td>null</td><td>2.1</td><td>0.9</td><td>173.0</td><td>22.0</td><td>17.0</td><td>null</td><td>2</td><td>11.0</td><td>50.0</td><td>0.761468</td><td>1</td><td>891.299999</td><td>61</td><td>0</td></tr></tbody></table></div>


