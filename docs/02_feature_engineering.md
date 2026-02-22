# 02 — Feature Engineering (MIMIC-IV Sepsis DRL)

> **Input:** `mimic_hourly_binned.parquet` (8.8M satır, 43 sütun)
> **Output:** `mimic_hourly_binned_feature_engineered.parquet` (8.8M satır, 25 sütun — 176 MB)

2024-2025 literatürüne uygun **~23 feature'lık** standart state vektörü oluşturma.

---

## Pipeline Özeti

| # | Adım | Üretilen Feature(lar) |
|---|------|-----------------------|
| 1 | Veri yükleme & kontrol | — |
| 2 | Norepinefrin eşdeğeri | `total_vaso_equiv` |
| 3 | Sıvı dengesi (4h) | `fluid_balance_4h` |
| 4 | SOFA skoru (6 organ) | `sofa_score` |
| 5 | Mek. ventilasyon & Şok indeksi | `mechanical_ventilation`, `shock_index` |
| 6 | Lag features | `prev_fluid_dose`, `prev_vaso_dose` |
| 7 | Final state vector & kayıt | 23 feature → parquet |

---

## 1. Veri Yükleme & Şema Kontrolü

Ham veri `mimic_hourly_binned.parquet` dosyasından yüklenir. 43 sütun mevcut (vitals, labs, vasopressörler, GCS, demografik vb.).

---

## 2. Norepinefrin Eşdeğeri (Vazopressör Standardizasyonu)

Farklı vazopressörleri tek bir skalaya indirger.

### Formül

```
total_vaso_equiv = Σ (ilaç_dozu × dönüşüm_oranı)
```

### Dönüşüm Tablosu

| İlaç | Kolon | Oran |
|------|-------|------|
| Norepinefrin | `norepinephrine_dose` | ×1.0 |
| Epinefrin | `epinephrine_dose` | ×0.1 |
| Phenylefrin | `phenylephrine_dose` | ×0.1 |
| Vasopressin | `vasopressin_dose` | ×0.4 |
| Dopamin | `dopamine_dose` | ×0.01 |
| Dobutamin | `dobutamine_dose` | ×0.0 ⚠️ |

> [!NOTE]
> Dobutamin inotrop etkili olduğundan vazopressör eşdeğerine dahil edilmez (×0.0).

### İşlem

1. Null dozlar `0` ile doldurulur
2. Her ilaç dozu ilgili oranla çarpılır
3. Tüm eşdeğerler `pl.sum_horizontal()` ile toplanarak `total_vaso_equiv` üretilir

---

## 3. Sıvı Dengesi (Net Fluid Balance — 4h)

### Formül

```
fluid_balance_4h = crystalloid_ml − urine_output
```

| Durum | Klinik Anlam |
|-------|-------------|
| **Pozitif** | Sıvı birikimi → Ödem riski |
| **Negatif** | Sıvı kaybı → Hipovolemi |

### İşlem

- `crystalloid_ml` ve `urine_output` null ise `0` ile doldurulur
- İkisinin farkı alınarak `fluid_balance_4h` elde edilir

---

## 4. SOFA Skoru (Sequential Organ Failure Assessment)

Vincent et al. (1996) — 6 organ sistemi, her biri **0-4 puan**, toplam **0-24**.

### 4A. Solunum (PaO2/FiO2)

```
fio2_ratio = fio2 / 100   (eğer fio2 > 1 ise, yani % olarak geliyorsa)
pf_ratio   = pao2 / fio2_ratio
```

| PF Oranı | Ventilasyon | Puan |
|----------|-------------|------|
| > 400 | — | 0 |
| > 300 | — | 1 |
| > 200 | — | 2 |
| > 100 | ✅ MV | 3 |
| ≤ 100 | ✅ MV | 4 |

> [!NOTE]
> PF ≤ 200 ama ventilasyon yoksa → max **2** puan verilir.

### 4B. Kardiyovasküler (MAP + Vazopressör)

| Koşul | Puan |
|-------|------|
| MAP ≥ 70 ve vazo = 0 | 0 |
| MAP < 70 | 1 |
| `total_vaso_equiv` ≤ 0.1 | 2 |
| `total_vaso_equiv` ≤ 0.5 | 3 |
| `total_vaso_equiv` > 0.5 | 4 |

### 4C. Böbrek (Kreatinin)

| Kreatinin (mg/dL) | Puan |
|-------------------|------|
| < 1.2 | 0 |
| 1.2 – 1.9 | 1 |
| 2.0 – 3.4 | 2 |
| 3.5 – 4.9 | 3 |
| ≥ 5.0 | 4 |

> [!NOTE]
> 24 saatlik idrar kriteri saatlik veride uygulanamaz; yalnızca kreatinin kullanılır.

### 4D. Nörolojik (GCS)

| GCS Toplam | Puan |
|------------|------|
| 15 | 0 |
| 13-14 | 1 |
| 10-12 | 2 |
| 6-9 | 3 |
| < 6 | 4 |

### 4E. Koagülasyon (Trombosit)

| Trombosit (×10³/μL) | Puan |
|---------------------|------|
| > 150 | 0 |
| 101-150 | 1 |
| 51-100 | 2 |
| 21-50 | 3 |
| ≤ 20 | 4 |

### 4F. Karaciğer (Bilirubin)

| Bilirubin (mg/dL) | Puan |
|-------------------|------|
| < 1.2 | 0 |
| 1.2 – 1.9 | 1 |
| 2.0 – 5.9 | 2 |
| 6.0 – 11.9 | 3 |
| ≥ 12.0 | 4 |

### 4G. Toplam SOFA

```
sofa_score = sofa_resp + sofa_cardio + sofa_renal + sofa_neuro + sofa_coag + sofa_liver
```

- Null alt skorlar `0` ile doldurulur
- Sonuç aralığı: **[0, 23]** ✅

---

## 5. Mekanik Ventilasyon & Şok İndeksi

### Mekanik Ventilasyon

```
mechanical_ventilation = 1   eğer FiO2 > 21%
                       = 0   aksi halde
```

> SOFA Respiratory hesabı sırasında üretilir.

### Şok İndeksi

```
shock_index = heart_rate / sbp
```

| Aralık | Klinik Anlam |
|--------|-------------|
| 0.5 – 0.7 | Normal |
| > 1.0 | Şok belirtisi ⚠️ |

---

## 6. Lag Features (Önceki Timestep Dozları)

Agent *"şimdi ne yapmalıyım?"* derken *"az önce ne yaptım?"* bilmeli. İlaç kümülasyonu nedeniyle zorunlu.

### Formül

```
prev_fluid_dose(t) = crystalloid_ml(t-1)
prev_vaso_dose(t)  = total_vaso_equiv(t-1)
```

### İşlem

1. Veri `stay_id` + `hour_bin` olarak sıralanır
2. `shift(1).over("stay_id")` ile her hasta içinde 1 timestep geriye kaydırılır
3. İlk timestep → `null` kalır (sonradan impute edilecek)

---

## 7. Final State Vector

### Gender Encoding

| Değer | Kod |
|-------|-----|
| M | 0 |
| F | 1 |

### State Vector (23 Feature)

| Kategori | Feature'lar |
|----------|-------------|
| **Lag** | `prev_fluid_dose`, `prev_vaso_dose` |
| **Vitals** | `heart_rate`, `sbp`, `dbp`, `mbp`, `resp_rate`, `spo2`, `temp_c` |
| **Labs** | `lactate`, `creatinine`, `platelet`, `bun`, `wbc`, `bilirubin_total` |
| **Organ** | `sofa_score`, `gcs_total`, `urine_output` |
| **Hemodinamik** | `shock_index`, `mechanical_ventilation` |
| **Sıvı** | `fluid_balance_4h` |
| **Demografi** | `age`, `gender` |

### Null Oranları (Kayda Değer)

| Feature | Null % | Not |
|---------|--------|-----|
| `temp_c` | 83.4% | ⚠️ En yüksek |
| `bilirubin_total` | 37.3% | |
| `lactate` | 28.0% | |
| `prev_fluid_dose` | 19.2% | İlk timestep null |
| Diğerleri | < 6% | ✅ |

### Çıktı

- **Dosya:** `mimic_hourly_binned_feature_engineered.parquet`
- **Boyut:** 176.1 MB
- **Satır:** 8,808,129
- **Sütun:** 25 (2 meta + 23 feature)
