# 01 — Data Exploration
> MIMIC-IV Sepsis DRL — Hourly-Binned Parquet Verisi Keşfi

Bu notebook, `data/processed/mimic_hourly_binned.parquet` dosyasını **Polars + PyArrow** ile açıp temel kontrolleri yapar:
1. Dosya doğru açılıyor mu?
2. Kaç satır × kaç sütun?
3. Sütun isimleri ve veri tipleri (schema)
4. Null / missing oranları
5. Her sütundaki unique değer sayısı
6. Temel istatistikler (describe)
7. `stay_id` başına satır sayısı dağılımı


```python
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path

PARQUET_PATH = Path("..") / "data" / "processed" / "mimic_hourly_binned.parquet"
print(f"Dosya mevcut mu? → {PARQUET_PATH.exists()}")
print(f"Dosya boyutu  → {PARQUET_PATH.stat().st_size / 1e6:.1f} MB")
```

    Dosya mevcut mu? → True
    Dosya boyutu  → 182.1 MB


## 1 · PyArrow ile Schema (Hızlı Metadata Kontrolü)
Dosyayı belleğe yüklemeden sadece metadata'yı okuyalım.


```python
pf = pq.ParquetFile(PARQUET_PATH)

print(f"Satır sayısı   : {pf.metadata.num_rows:,}")
print(f"Row-group sayısı: {pf.metadata.num_row_groups}")
print(f"Sütun sayısı   : {pf.metadata.num_columns}")
print()
print("=== PyArrow Schema ===")
print(pf.schema_arrow)
```

    Satır sayısı   : 8,808,129
    Row-group sayısı: 72
    Sütun sayısı   : 43
    
    === PyArrow Schema ===
    stay_id: int64
    hour_bin: timestamp[us]
    heart_rate: double
    sbp: double
    dbp: double
    mbp: double
    resp_rate: double
    spo2: double
    temp_c: double
    fio2: double
    lactate: double
    creatinine: double
    bilirubin_total: double
    platelet: double
    wbc: double
    bun: double
    glucose: double
    sodium: double
    potassium: double
    hemoglobin: double
    hematocrit: double
    bicarbonate: double
    chloride: double
    anion_gap: double
    inr: double
    pao2: double
    paco2: double
    ph: double
    urine_output: double
    norepinephrine_dose: double
    epinephrine_dose: double
    phenylephrine_dose: double
    vasopressin_dose: double
    dopamine_dose: double
    dobutamine_dose: double
    crystalloid_ml: double
    gcs_eye: double
    gcs_motor: double
    gcs_verbal: double
    gcs_total: double
    gender: large_string
    age: int64
    admission_type: large_string


## 2 · Polars ile Yükleme & İlk Bakış


```python
df = pl.read_parquet(PARQUET_PATH)
print(f"Shape: {df.shape}  →  {df.shape[0]:,} satır × {df.shape[1]} sütun")
df.head(10)
```

    Shape: (8808129, 43)  →  8,808,129 satır × 43 sütun





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 43)</small><table border="1" class="dataframe"><thead><tr><th>stay_id</th><th>hour_bin</th><th>heart_rate</th><th>sbp</th><th>dbp</th><th>mbp</th><th>resp_rate</th><th>spo2</th><th>temp_c</th><th>fio2</th><th>lactate</th><th>creatinine</th><th>bilirubin_total</th><th>platelet</th><th>wbc</th><th>bun</th><th>glucose</th><th>sodium</th><th>potassium</th><th>hemoglobin</th><th>hematocrit</th><th>bicarbonate</th><th>chloride</th><th>anion_gap</th><th>inr</th><th>pao2</th><th>paco2</th><th>ph</th><th>urine_output</th><th>norepinephrine_dose</th><th>epinephrine_dose</th><th>phenylephrine_dose</th><th>vasopressin_dose</th><th>dopamine_dose</th><th>dobutamine_dose</th><th>crystalloid_ml</th><th>gcs_eye</th><th>gcs_motor</th><th>gcs_verbal</th><th>gcs_total</th><th>gender</th><th>age</th><th>admission_type</th></tr><tr><td>i64</td><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>str</td><td>i64</td><td>str</td></tr></thead><tbody><tr><td>30000153</td><td>2174-09-29 12:00:00</td><td>100.0</td><td>136.0</td><td>74.0</td><td>89.0</td><td>18.0</td><td>100.0</td><td>null</td><td>75.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>35.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>280.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 13:00:00</td><td>104.0</td><td>132.0</td><td>74.5</td><td>84.0</td><td>16.0</td><td>100.0</td><td>null</td><td>75.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>35.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>221.0</td><td>45.0</td><td>7.3</td><td>280.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 14:00:00</td><td>83.0</td><td>131.0</td><td>61.0</td><td>80.0</td><td>16.0</td><td>100.0</td><td>null</td><td>75.0</td><td>2.1</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>35.0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>263.0</td><td>45.0</td><td>7.3</td><td>45.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 15:00:00</td><td>92.0</td><td>123.0</td><td>65.0</td><td>84.0</td><td>14.0</td><td>100.0</td><td>null</td><td>50.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>31.7</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>263.0</td><td>45.0</td><td>7.3</td><td>50.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 16:00:00</td><td>83.0</td><td>109.0</td><td>55.0</td><td>71.0</td><td>16.0</td><td>100.0</td><td>null</td><td>50.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>31.7</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>215.0</td><td>42.0</td><td>7.31</td><td>50.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>941.299999</td><td>4.0</td><td>6.0</td><td>1.0</td><td>11.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 17:00:00</td><td>103.0</td><td>111.0</td><td>56.0</td><td>71.0</td><td>20.0</td><td>100.0</td><td>null</td><td>50.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>31.7</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>215.0</td><td>42.0</td><td>7.31</td><td>45.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>941.299999</td><td>4.0</td><td>6.0</td><td>1.0</td><td>11.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 18:00:00</td><td>111.0</td><td>133.0</td><td>63.0</td><td>83.0</td><td>19.0</td><td>99.0</td><td>null</td><td>50.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>31.7</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>215.0</td><td>42.0</td><td>7.31</td><td>70.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>941.299999</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 19:00:00</td><td>123.0</td><td>155.0</td><td>68.0</td><td>91.0</td><td>21.0</td><td>96.0</td><td>null</td><td>50.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>32.1</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>215.0</td><td>42.0</td><td>7.31</td><td>70.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>941.299999</td><td>3.0</td><td>5.0</td><td>1.0</td><td>9.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 20:00:00</td><td>128.0</td><td>122.0</td><td>67.0</td><td>83.0</td><td>21.0</td><td>98.0</td><td>null</td><td>40.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>32.1</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>215.0</td><td>42.0</td><td>7.31</td><td>70.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>941.299999</td><td>3.0</td><td>6.0</td><td>3.0</td><td>12.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>30000153</td><td>2174-09-29 21:00:00</td><td>123.0</td><td>136.0</td><td>67.0</td><td>87.0</td><td>22.0</td><td>96.0</td><td>null</td><td>40.0</td><td>2.1</td><td>0.9</td><td>null</td><td>173.0</td><td>17.0</td><td>22.0</td><td>192.0</td><td>142.0</td><td>4.4</td><td>10.8</td><td>32.1</td><td>19.0</td><td>115.0</td><td>12.0</td><td>1.1</td><td>215.0</td><td>42.0</td><td>7.31</td><td>80.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>199.999995</td><td>3.0</td><td>6.0</td><td>3.0</td><td>12.0</td><td>&quot;M&quot;</td><td>61</td><td>&quot;EW EMER.&quot;</td></tr></tbody></table></div>



## 3 · Sütun İsimleri ve Veri Tipleri


```python
schema_df = pl.DataFrame({
    "column": df.columns,
    "dtype": [str(dt) for dt in df.dtypes],
})
schema_df
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (43, 2)</small><table border="1" class="dataframe"><thead><tr><th>column</th><th>dtype</th></tr><tr><td>str</td><td>str</td></tr></thead><tbody><tr><td>&quot;stay_id&quot;</td><td>&quot;Int64&quot;</td></tr><tr><td>&quot;hour_bin&quot;</td><td>&quot;Datetime(time_unit=&#x27;us&#x27;, time_…</td></tr><tr><td>&quot;heart_rate&quot;</td><td>&quot;Float64&quot;</td></tr><tr><td>&quot;sbp&quot;</td><td>&quot;Float64&quot;</td></tr><tr><td>&quot;dbp&quot;</td><td>&quot;Float64&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td></tr><tr><td>&quot;gcs_verbal&quot;</td><td>&quot;Float64&quot;</td></tr><tr><td>&quot;gcs_total&quot;</td><td>&quot;Float64&quot;</td></tr><tr><td>&quot;gender&quot;</td><td>&quot;String&quot;</td></tr><tr><td>&quot;age&quot;</td><td>&quot;Int64&quot;</td></tr><tr><td>&quot;admission_type&quot;</td><td>&quot;String&quot;</td></tr></tbody></table></div>



## 4 · Null / Missing Oranları
Her sütundaki null sayısı ve yüzdesi.


```python
null_counts = df.null_count()
total_rows = df.shape[0]

null_df = pl.DataFrame({
    "column": df.columns,
    "null_count": [null_counts[col][0] for col in df.columns],
    "null_pct": [round(null_counts[col][0] / total_rows * 100, 2) for col in df.columns],
}).sort("null_pct", descending=True)

print(f"Toplam satır: {total_rows:,}")
null_df
```

    Toplam satır: 8,808,129





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (43, 3)</small><table border="1" class="dataframe"><thead><tr><th>column</th><th>null_count</th><th>null_pct</th></tr><tr><td>str</td><td>i64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;temp_c&quot;</td><td>7343072</td><td>83.37</td></tr><tr><td>&quot;bilirubin_total&quot;</td><td>3284098</td><td>37.28</td></tr><tr><td>&quot;fio2&quot;</td><td>2850062</td><td>32.36</td></tr><tr><td>&quot;lactate&quot;</td><td>2467801</td><td>28.02</td></tr><tr><td>&quot;paco2&quot;</td><td>2281818</td><td>25.91</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>&quot;stay_id&quot;</td><td>0</td><td>0.0</td></tr><tr><td>&quot;hour_bin&quot;</td><td>0</td><td>0.0</td></tr><tr><td>&quot;gender&quot;</td><td>0</td><td>0.0</td></tr><tr><td>&quot;age&quot;</td><td>0</td><td>0.0</td></tr><tr><td>&quot;admission_type&quot;</td><td>0</td><td>0.0</td></tr></tbody></table></div>



## 5 · Unique Değer Sayıları
Her sütundaki unique (benzersiz) eleman sayısı.


```python
unique_df = pl.DataFrame({
    "column": df.columns,
    "n_unique": [df[col].n_unique() for col in df.columns],
}).sort("n_unique", descending=True)

unique_df
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (43, 2)</small><table border="1" class="dataframe"><thead><tr><th>column</th><th>n_unique</th></tr><tr><td>str</td><td>i64</td></tr></thead><tbody><tr><td>&quot;hour_bin&quot;</td><td>846870</td></tr><tr><td>&quot;crystalloid_ml&quot;</td><td>739641</td></tr><tr><td>&quot;norepinephrine_dose&quot;</td><td>320794</td></tr><tr><td>&quot;phenylephrine_dose&quot;</td><td>141675</td></tr><tr><td>&quot;stay_id&quot;</td><td>94458</td></tr><tr><td>&hellip;</td><td>&hellip;</td></tr><tr><td>&quot;gcs_motor&quot;</td><td>28</td></tr><tr><td>&quot;gcs_verbal&quot;</td><td>27</td></tr><tr><td>&quot;gcs_eye&quot;</td><td>22</td></tr><tr><td>&quot;admission_type&quot;</td><td>9</td></tr><tr><td>&quot;gender&quot;</td><td>2</td></tr></tbody></table></div>



## 6 · Temel İstatistikler (describe)


```python
df.describe()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 44)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>stay_id</th><th>hour_bin</th><th>heart_rate</th><th>sbp</th><th>dbp</th><th>mbp</th><th>resp_rate</th><th>spo2</th><th>temp_c</th><th>fio2</th><th>lactate</th><th>creatinine</th><th>bilirubin_total</th><th>platelet</th><th>wbc</th><th>bun</th><th>glucose</th><th>sodium</th><th>potassium</th><th>hemoglobin</th><th>hematocrit</th><th>bicarbonate</th><th>chloride</th><th>anion_gap</th><th>inr</th><th>pao2</th><th>paco2</th><th>ph</th><th>urine_output</th><th>norepinephrine_dose</th><th>epinephrine_dose</th><th>phenylephrine_dose</th><th>vasopressin_dose</th><th>dopamine_dose</th><th>dobutamine_dose</th><th>crystalloid_ml</th><th>gcs_eye</th><th>gcs_motor</th><th>gcs_verbal</th><th>gcs_total</th><th>gender</th><th>age</th><th>admission_type</th></tr><tr><td>str</td><td>f64</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>str</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>8.808129e6</td><td>&quot;8808129&quot;</td><td>8.770031e6</td><td>8.740315e6</td><td>8.740024e6</td><td>8.741935e6</td><td>8.760778e6</td><td>8.764642e6</td><td>1.465057e6</td><td>5.958067e6</td><td>6.340328e6</td><td>8.327579e6</td><td>5.524031e6</td><td>8.299103e6</td><td>8.295935e6</td><td>8.326449e6</td><td>8.211747e6</td><td>8.271772e6</td><td>8.293823e6</td><td>8.297261e6</td><td>8.31684e6</td><td>8.325363e6</td><td>8.331191e6</td><td>8.310005e6</td><td>7.748676e6</td><td>6.527544e6</td><td>6.526311e6</td><td>6.644951e6</td><td>8.293822e6</td><td>7.194083e6</td><td>7.194083e6</td><td>7.194083e6</td><td>7.194083e6</td><td>7.194083e6</td><td>7.194083e6</td><td>7.194083e6</td><td>8.700945e6</td><td>8.696527e6</td><td>8.698579e6</td><td>8.702212e6</td><td>&quot;8808129&quot;</td><td>8.808129e6</td><td>&quot;8808129&quot;</td></tr><tr><td>&quot;null_count&quot;</td><td>0.0</td><td>&quot;0&quot;</td><td>38098.0</td><td>67814.0</td><td>68105.0</td><td>66194.0</td><td>47351.0</td><td>43487.0</td><td>7.343072e6</td><td>2.850062e6</td><td>2.467801e6</td><td>480550.0</td><td>3.284098e6</td><td>509026.0</td><td>512194.0</td><td>481680.0</td><td>596382.0</td><td>536357.0</td><td>514306.0</td><td>510868.0</td><td>491289.0</td><td>482766.0</td><td>476938.0</td><td>498124.0</td><td>1.059453e6</td><td>2.280585e6</td><td>2.281818e6</td><td>2.163178e6</td><td>514307.0</td><td>1.614046e6</td><td>1.614046e6</td><td>1.614046e6</td><td>1.614046e6</td><td>1.614046e6</td><td>1.614046e6</td><td>1.614046e6</td><td>107184.0</td><td>111602.0</td><td>109550.0</td><td>105917.0</td><td>&quot;0&quot;</td><td>0.0</td><td>&quot;0&quot;</td></tr><tr><td>&quot;mean&quot;</td><td>3.4974e7</td><td>&quot;2153-10-15 03:38:30.535507&quot;</td><td>87.819938</td><td>120.478139</td><td>65.184255</td><td>84.512153</td><td>21.169934</td><td>137.767256</td><td>38.354066</td><td>46.397577</td><td>3.263103</td><td>1.481856</td><td>2.209595</td><td>219.682421</td><td>12.125175</td><td>30.943375</td><td>137.819228</td><td>139.020176</td><td>4.103613</td><td>9.777631</td><td>30.068222</td><td>24.765002</td><td>103.07367</td><td>13.531571</td><td>1.434417</td><td>100.873874</td><td>42.377398</td><td>7.396889</td><td>184.733394</td><td>0.176258</td><td>0.063585</td><td>0.737237</td><td>0.188185</td><td>0.782889</td><td>1.025641</td><td>251.693691</td><td>3.385539</td><td>5.282393</td><td>3.317324</td><td>11.923467</td><td>null</td><td>62.641503</td><td>null</td></tr><tr><td>&quot;std&quot;</td><td>2.8843e6</td><td>null</td><td>3797.403494</td><td>491.228532</td><td>259.152687</td><td>4828.722159</td><td>2407.519375</td><td>19406.703678</td><td>9.811226</td><td>49.24549</td><td>1433.422544</td><td>1.46838</td><td>5.210624</td><td>132.416995</td><td>8.458586</td><td>25.204517</td><td>57.215769</td><td>5.358748</td><td>0.565285</td><td>1.967902</td><td>5.714107</td><td>5.060515</td><td>6.661833</td><td>3.992657</td><td>0.656184</td><td>58.592579</td><td>10.528077</td><td>0.071144</td><td>364.241888</td><td>1.054701</td><td>8.839282</td><td>6.477451</td><td>2.627308</td><td>14.516527</td><td>17.072216</td><td>1572.401753</td><td>1.003951</td><td>1.494538</td><td>1.858582</td><td>3.831584</td><td>null</td><td>16.111648</td><td>null</td></tr><tr><td>&quot;min&quot;</td><td>3.0000153e7</td><td>&quot;2110-01-11 10:00:00&quot;</td><td>-241395.0</td><td>-94.0</td><td>-40.0</td><td>-9806.0</td><td>0.0</td><td>-951234.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>5.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>67.0</td><td>0.8</td><td>0.0</td><td>0.0</td><td>2.0</td><td>39.0</td><td>-24.0</td><td>0.5</td><td>-32.0</td><td>0.0</td><td>0.94</td><td>-3765.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>&quot;F&quot;</td><td>18.0</td><td>&quot;AMBULATORY OBSERVATION&quot;</td></tr><tr><td>&quot;25%&quot;</td><td>3.2477246e7</td><td>&quot;2133-12-07 02:00:00&quot;</td><td>73.0</td><td>104.0</td><td>53.0</td><td>69.0</td><td>16.0</td><td>95.0</td><td>36.6</td><td>40.0</td><td>1.0</td><td>0.7</td><td>0.4</td><td>131.0</td><td>7.7</td><td>14.0</td><td>104.0</td><td>136.0</td><td>3.7</td><td>8.3</td><td>25.8</td><td>22.0</td><td>99.0</td><td>11.0</td><td>1.1</td><td>61.0</td><td>36.0</td><td>7.36</td><td>50.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>47.833335</td><td>3.0</td><td>5.0</td><td>1.0</td><td>10.0</td><td>null</td><td>53.0</td><td>null</td></tr><tr><td>&quot;50%&quot;</td><td>3.4965363e7</td><td>&quot;2153-08-17 00:00:00&quot;</td><td>85.0</td><td>118.0</td><td>62.0</td><td>78.0</td><td>19.25</td><td>97.0</td><td>37.1</td><td>40.0</td><td>1.4</td><td>1.0</td><td>0.6</td><td>196.0</td><td>10.6</td><td>23.0</td><td>125.0</td><td>139.0</td><td>4.0</td><td>9.4</td><td>29.1</td><td>24.0</td><td>103.0</td><td>13.0</td><td>1.2</td><td>92.0</td><td>41.0</td><td>7.4</td><td>120.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>4.0</td><td>6.0</td><td>4.0</td><td>14.0</td><td>null</td><td>64.0</td><td>null</td></tr><tr><td>&quot;75%&quot;</td><td>3.7460082e7</td><td>&quot;2173-11-27 22:00:00&quot;</td><td>98.0</td><td>134.0</td><td>73.0</td><td>89.0</td><td>24.0</td><td>99.0</td><td>37.6</td><td>50.0</td><td>1.9</td><td>1.7</td><td>1.4</td><td>280.0</td><td>14.5</td><td>39.0</td><td>155.0</td><td>142.0</td><td>4.4</td><td>11.0</td><td>33.5</td><td>28.0</td><td>107.0</td><td>16.0</td><td>1.5</td><td>125.0</td><td>47.0</td><td>7.45</td><td>250.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>295.833342</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>null</td><td>75.0</td><td>null</td></tr><tr><td>&quot;max&quot;</td><td>3.9999858e7</td><td>&quot;2214-08-11 05:00:00&quot;</td><td>1e7</td><td>1.00311e6</td><td>114109.0</td><td>8.99909e6</td><td>7.0004e6</td><td>9.9e6</td><td>987.4</td><td>40100.0</td><td>1.276103e6</td><td>80.0</td><td>87.2</td><td>2385.0</td><td>572.5</td><td>305.0</td><td>5840.0</td><td>185.0</td><td>26.5</td><td>24.6</td><td>71.2</td><td>50.0</td><td>155.0</td><td>89.0</td><td>27.4</td><td>4242.0</td><td>243.0</td><td>7.96</td><td>876587.0</td><td>1099.999975</td><td>4740.164044</td><td>1000.00005</td><td>399.999997</td><td>1008.783077</td><td>1023.107846</td><td>1.0004e6</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>91.0</td><td>&quot;URGENT&quot;</td></tr></tbody></table></div>



## 7 · `stay_id` Başına Satır Dağılımı
Her hastanın kaç saatlik verisi var?


```python
stay_counts = (
    df.group_by("stay_id")
    .agg(pl.len().alias("n_hours"))
    .sort("n_hours", descending=True)
)

print(f"Toplam benzersiz stay_id: {stay_counts.shape[0]:,}")
print()
print(stay_counts["n_hours"].describe())
print()
print("En uzun 10 yatış:")
stay_counts.head(10)
```

    Toplam benzersiz stay_id: 94,458
    
    shape: (9, 2)
    ┌────────────┬────────────┐
    │ statistic  ┆ value      │
    │ ---        ┆ ---        │
    │ str        ┆ f64        │
    ╞════════════╪════════════╡
    │ count      ┆ 94458.0    │
    │ null_count ┆ 0.0        │
    │ mean       ┆ 93.249158  │
    │ std        ┆ 130.288585 │
    │ min        ┆ 1.0        │
    │ 25%        ┆ 31.0       │
    │ 50%        ┆ 53.0       │
    │ 75%        ┆ 101.0      │
    │ max        ┆ 5411.0     │
    └────────────┴────────────┘
    
    En uzun 10 yatış:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 2)</small><table border="1" class="dataframe"><thead><tr><th>stay_id</th><th>n_hours</th></tr><tr><td>i64</td><td>u32</td></tr></thead><tbody><tr><td>36032605</td><td>5411</td></tr><tr><td>36307509</td><td>4006</td></tr><tr><td>39510663</td><td>3421</td></tr><tr><td>30359303</td><td>3269</td></tr><tr><td>35629939</td><td>3051</td></tr><tr><td>31492392</td><td>3040</td></tr><tr><td>39245279</td><td>2683</td></tr><tr><td>32380519</td><td>2457</td></tr><tr><td>38018615</td><td>2423</td></tr><tr><td>31879957</td><td>2386</td></tr></tbody></table></div>



## 8 · Son 5 Satır (tail)
Datanın sonuna da bakalım, forward-fill düzgün çalışmış mı kontrol edelim.


```python
df.tail(10)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 43)</small><table border="1" class="dataframe"><thead><tr><th>stay_id</th><th>hour_bin</th><th>heart_rate</th><th>sbp</th><th>dbp</th><th>mbp</th><th>resp_rate</th><th>spo2</th><th>temp_c</th><th>fio2</th><th>lactate</th><th>creatinine</th><th>bilirubin_total</th><th>platelet</th><th>wbc</th><th>bun</th><th>glucose</th><th>sodium</th><th>potassium</th><th>hemoglobin</th><th>hematocrit</th><th>bicarbonate</th><th>chloride</th><th>anion_gap</th><th>inr</th><th>pao2</th><th>paco2</th><th>ph</th><th>urine_output</th><th>norepinephrine_dose</th><th>epinephrine_dose</th><th>phenylephrine_dose</th><th>vasopressin_dose</th><th>dopamine_dose</th><th>dobutamine_dose</th><th>crystalloid_ml</th><th>gcs_eye</th><th>gcs_motor</th><th>gcs_verbal</th><th>gcs_total</th><th>gender</th><th>age</th><th>admission_type</th></tr><tr><td>i64</td><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>str</td><td>i64</td><td>str</td></tr></thead><tbody><tr><td>39999858</td><td>2167-05-01 06:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.7</td><td>0.7</td><td>226.0</td><td>9.6</td><td>16.0</td><td>117.0</td><td>137.0</td><td>4.1</td><td>12.7</td><td>38.9</td><td>28.0</td><td>100.0</td><td>9.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-02 06:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.8</td><td>0.6</td><td>263.0</td><td>9.2</td><td>15.0</td><td>135.0</td><td>137.0</td><td>4.3</td><td>12.2</td><td>38.0</td><td>28.0</td><td>101.0</td><td>8.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-03 06:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.7</td><td>0.6</td><td>263.0</td><td>9.2</td><td>18.0</td><td>132.0</td><td>139.0</td><td>4.5</td><td>12.2</td><td>38.0</td><td>30.0</td><td>101.0</td><td>8.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-04 07:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.8</td><td>0.6</td><td>263.0</td><td>9.2</td><td>16.0</td><td>121.0</td><td>137.0</td><td>4.3</td><td>12.2</td><td>38.0</td><td>32.0</td><td>97.0</td><td>8.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-05 06:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.8</td><td>0.6</td><td>305.0</td><td>12.2</td><td>17.0</td><td>103.0</td><td>135.0</td><td>4.4</td><td>13.4</td><td>41.4</td><td>32.0</td><td>95.0</td><td>8.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-06 06:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.9</td><td>0.6</td><td>269.0</td><td>12.0</td><td>19.0</td><td>125.0</td><td>136.0</td><td>4.4</td><td>13.7</td><td>41.7</td><td>30.0</td><td>97.0</td><td>9.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-07 08:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.9</td><td>0.6</td><td>244.0</td><td>11.4</td><td>18.0</td><td>142.0</td><td>136.0</td><td>4.3</td><td>14.2</td><td>43.3</td><td>28.0</td><td>99.0</td><td>9.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-08 07:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.9</td><td>0.6</td><td>214.0</td><td>13.2</td><td>15.0</td><td>154.0</td><td>137.0</td><td>5.0</td><td>13.6</td><td>42.6</td><td>29.0</td><td>99.0</td><td>9.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-09 07:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.8</td><td>1.4</td><td>208.0</td><td>12.3</td><td>11.0</td><td>164.0</td><td>135.0</td><td>4.8</td><td>13.3</td><td>39.9</td><td>29.0</td><td>98.0</td><td>8.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr><tr><td>39999858</td><td>2167-05-10 08:00:00</td><td>82.0</td><td>107.0</td><td>57.0</td><td>69.0</td><td>28.0</td><td>90.0</td><td>null</td><td>40.0</td><td>null</td><td>0.7</td><td>1.2</td><td>185.0</td><td>11.5</td><td>10.0</td><td>174.0</td><td>132.0</td><td>3.9</td><td>13.7</td><td>42.1</td><td>26.0</td><td>95.0</td><td>11.0</td><td>1.3</td><td>null</td><td>null</td><td>null</td><td>350.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>249.99999</td><td>4.0</td><td>6.0</td><td>5.0</td><td>15.0</td><td>&quot;M&quot;</td><td>62</td><td>&quot;EW EMER.&quot;</td></tr></tbody></table></div>


