# RL Sepsis Veri Ön İşlemesi & Özellik Mühendisliği

---

### 我

# RL Sepsis Ajanı: Veri Ön İşleme (Preprocessing) Konsey Değerlendirmesi

**Bağlam:**
Biz tıp alanında, MIMIC-IV veri setini kullanarak Sepsis yönetimi için Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL) modelleri geliştirmeyi amaçlayan bir proje yürütüyoruz. Amacımız, yoğun bakım (ICU) ortamındaki sepsis hastalarının verilerini işleyerek klinik durumlarını (state) modellemek ve bu verilere dayanarak optimum tedavi stratejilerini (örn. vazopressör dozajı ve sıvı tedavisi) öğrenecek yapay zeka ajanları (RL agent) eğitmektir.

Veri biliminde performansı artırmak ve RAM taşmasını engellemek amacıyla Polars kütüphanesini kullanarak, zaman serisi tıbbi verileri saatli periyotlara (hourly binning) çeviren, lazy evaluation + single-pass group-by içeren bellek dostu bir süreç kurguladık.

**Görev:**
Sen üst düzey bir veri bilimi ve medikal yapay zeka konseyinin danışman üyesisin. Bu istem (prompt) ile birlikte sana projemizin durumunu anlatan dosyalar ve kodlar sağlıyorum. Lütfen sana ilettiğim şu dökümanları incele:
1. `project_summary.md`: Projenin genel mimarisi, yapısı ve hedefi.
2. `notebooks/01_data_explore.md` & `notebooks/02_feature_engineering.md`: Veri üzerinde yaptığımız keşifsel analiz (EDA), hücre (cell) çıktıları, eksik veri grafikleri, saatlik gruplamalar ve korelasyonların sonuçları.
3. `src/preprocess/config.py` & `src/preprocess/pipeline.py`: Araştırmalarımız sonucunda elde ettiğimiz bulguları, Polars kullanarak devasa MIMIC-IV verilerine (OOM hatası almadan) uyguladığımız production-ready (canlıya hazır) nihai bellek dostu ön işleme hattımız.

***(Önemli Not: Notebook'larda verinin yapısı/çıktıları ve EDA yöntemlerimiz mevcuttur. Python (.py) dosyalarında ise bu kararların tüm veriye nasıl bir teknik mimariyle aktarıldığı bulunur. Lütfen her iki kaynağı da sentezleyerek yorum yap.)***

Lütfen sadece "veri ön işleme", "feature engineering" ve "RL State (durum) / Action (aksiyon) alanı seçimi" kapsamlarına odaklanarak kodlarımızı ve çıktıları analiz et. Bize şu soruların cevabını ve stratejik öneriler sun:

1. **Bu projenin veri ön işlemesinde eksik olan kritik bilgiler, özellikler (feature) nelerdir?** 
   - İlettiğim notebook çıktılarına (eksik veri oranları vs.) ve Python dosyalarındaki logic'e bakarak, RL ajanının "state" (durum) uzayı için eksik ama başarı için elzem olan laboratuvar, vital veya demografik veriler var mı? (Örn: Sepsis-3 teşhis kriterleri eksiksiz karşılanıyor mu? Saatlik SOFA skoru, SIRS kriterleri, antibiyotik kullanım başlama zamanı, mekanik ventilasyon durumu eklenmeli mi?)

2. **Mevcut Verilerden Neler Üretilebilir / Eklenebilir?**
   - Hangi yeni klinik/laboratuvar değişkenler modele önemli bir içgörü katar? EDA verilerine dayanarak RL ajanı için state tanımını güçlendiren ek "feature engineering" tavsiyelerin nelerdir? (Örn: Değişim hızları (delta değerleri), ağırlık üzerinden standardize edilmiş dozlar vs.)

3. **Mevcut geliştirme aşamasında (kodda veya metodolojide) geliştirilebilecek mühendislik / istatistiksel yöntemler nelerdir?**
   - Özellikle Notebook'lardaki hücre sonuçlarını dikkate alıp, `pipeline.py` içindeki işlemleri (imputasyon vs.) birleştirdiğinde, daha efektif bir işleme veya doldurma metodu görüyor musun? (Örn: Biz forward-fill uyguladık. İstatistiksel ve tıbbi olarak, ekstrem null oranına sahip feature'larda daha iyi bir interpolasyon/imputasyon uygulanabilir mi? Outlier handling veya scaling konusunda eksiklerimiz nelerdir?)

Lütfen sadece genel-geçer bilgiler vermek yerine, sana verdiğim kod mimarisini ve notebook çıktı sonuçlarını *(null oranları, dağılımlar vs.)* baz alarak projemize özel net, uygulanabilir teknik ve tıbbi tavsiyeleri **maddeler halinde ve detaylı olarak** açıkla.

---
*(Lütfen cevaplarınızı direkt çözüm stratejilerimiz üzerine odaklayın, yukarıdaki metni veya kod bloklarını gereksiz yere tekrar kopyalamayın.)*

### Qwen

# 🏥 MIMIC-IV Sepsis DRL — Veri Ön İşleme Konsey Değerlendirmesi

Sayın Proje Ekibi,

Sağladığınız 5 dosyayı (pipeline.py, config.py, 01_data_explore.md, 02_feature_engineering.md, project_summary.md) detaylı inceledim. Polars tabanlı bellek dostu pipeline mimariniz teknik olarak **takdir edilebilir düzeyde** — özellikle single-pass group_by ve streaming sink_parquet yaklaşımı OOM sorununu başarıyla çözmüş. Ancak **tıbbi/RL perspektifinden** kritik eksiklikler ve iyileştirme alanları tespit ettim.

Aşağıda projenize özel, doğrudan uygulanabilir teknik ve tıbbi tavsiyeleri sunuyorum:

---

## 1️⃣ KRİTİK EKSİK ÖZELLİKLER (State Uzayı İçin)

### 🔴 A. Sepsis-3 Teşhis Kriterleri Tam Karşılanmıyor

| Kriter | Mevcut Durum | Eksik | Öneri |
|--------|--------------|-------|-------|
| **SOFA Δ ≥ 2** | ❌ Sadece absolut SOFA var | **SOFA değişim hızı** | `sofa_delta_1h`, `sofa_delta_4h`, `sofa_delta_24h` ekleyin |
| **Enfeksiyon Şüphesi** | ❌ Hiç yok | **Antibiyotik başlangıç zamanı** | `microbiologyevents` + `prescriptions` join ile `antibiotic_start_hour` |
| **Laktat ≥ 2 mmol/L** | ⚠️ Var ama %28 null | **Laktat clearance** | `lactate_delta_4h = (lactate_t - lactate_t-4) / 4` |

**Neden Kritik?**
Sepsis-3 tanısı **SOFA skorundaki akut değişim** gerektirir. Ajanınız sadece "şu an SOFA=7" biliyor ama "1 saat önce SOFA=3'tü" bilgisine sahip değil. Bu, ajanın **hastanın kötüleşme hızını** anlamasını engeller.

```python
# pipeline.py'ye eklenecek:
df = df.with_columns([
    pl.col("sofa_score").diff().over("stay_id").alias("sofa_delta_1h"),
    pl.col("sofa_score").diff(4).over("stay_id").alias("sofa_delta_4h"),
])
```

---

### 🔴 B. SIRS Kriterleri Eksik

Sepsis-2 hala bazı çalışmalarda kullanılıyor ve SIRS skoru prognostik değer taşıyor:

| SIRS Kriteri | Mevcut Feature | Threshold |
|--------------|----------------|-----------|
| Temperatura | `temp_c` (%83 null!) | <36°C veya >38°C |
| Kalp Atışı | `heart_rate` | >90 bpm |
| Solunum | `resp_rate` | >20/min |
| WBC | `wbc` | <4k veya >12k |

**Öneri:**
```python
df = df.with_columns([
    pl.when(pl.col("temp_c").is_between(36, 38)).then(0).otherwise(1).alias("sirs_temp"),
    pl.when(pl.col("heart_rate") > 90).then(1).otherwise(0).alias("sirs_hr"),
    pl.when(pl.col("resp_rate") > 20).then(1).otherwise(0).alias("sirs_rr"),
    pl.when((pl.col("wbc") < 4) | (pl.col("wbc") > 12)).then(1).otherwise(0).alias("sirs_wbc"),
])
df = df.with_columns(
    (pl.col("sirs_temp") + pl.col("sirs_hr") + pl.col("sirs_rr") + pl.col("sirs_wbc"))
    .alias("sirs_score")
)
```

---

### 🔴 C. Mekanik Ventilasyon Tespiti Yetersiz

**Mevcut Logic (`02_feature_engineering.md`):**
```python
pl.when(pl.col("fio2") > 21).then(1).otherwise(0)
```

**Problem:**
- FiO2 > 21% **her zaman** mekanik ventilasyon anlamına gelmez (nazal kanül, non-invaziv destek)
- `chartevents`'te **ventilasyon modu** itemid'leri var (223848, 223849, 229314)
- %32 FiO2 null oranı → yanlış negatifler

**Öneri:**
```python
# config.py'ye eklenecek:
VENTILATION_IDS = [223848, 223849, 229314, 224700]  # Ventilasyon modu itemid'leri

# pipeline.py'de:
pl.when(pl.col("itemid").is_in(VENTILATION_IDS))
  .then(pl.lit(1)).otherwise(pl.lit(0))
  .max().alias("mechanical_ventilation")
```

---

### 🔴 D. Hasta Ağırlığı Eksik (Doz Standardizasyonu İçin Kritik)

**Mevcut Durum:**
- Vazopressör dozları `mcg/kg/min` olarak **standartize edilmemiş**
- `total_vaso_equiv` hesaplanmış ama **kg başına** değil
- Fluid balance mutlak ml, **ml/kg** değil

**Neden Problem?**
60kg hasta ile 100kg hastaya aynı 500ml sıvı **farklı klinik etki** yaratır. RL ajanı bu farkı öğrenemez.

**Öneri:**
```python
# chartevents'ten hasta ağırlığı çek (itemid: 224639)
# pipeline.py'ye:
df = df.with_columns([
    (pl.col("crystalloid_ml") / pl.col("weight_kg")).alias("fluid_ml_per_kg"),
    (pl.col("total_vaso_equiv") / pl.col("weight_kg")).alias("vaso_equiv_per_kg"),
])
```

---

## 2️⃣ YENİ FEATURE ENGINEERING ÖNERİLERİ

### 🟡 A. Dinamik/Trend Feature'ları (RL İçin Kritik)

RL ajanı **sadece anlık durumu** değil, **değişim yönünü** de bilmeli:

| Feature | Hesaplama | RL Değeri |
|---------|-----------|-----------|
| `hr_trend_4h` | HR'nin son 4 saatlik eğimi | Düşüş = şok iyileşmesi |
| `map_trend_4h` | MAP değişim hızı | Vazo ihtiyacı tahmini |
| `lactate_clearance` | (Laktat_t-4 - Laktat_t) / 4 | Tedavi yanıtı |
| `cumulative_fluid_balance` | ICU girişinden beri toplam sıvı | Ödem riski |

```python
# 02_feature_engineering.md'ye eklenecek:
df = df.with_columns([
    pl.col("heart_rate").rolling_mean(window_size=4).over("stay_id").alias("hr_trend_4h"),
    pl.col("mbp").rolling_mean(window_size=4).over("stay_id").alias("map_trend_4h"),
    (pl.col("lactate").shift(4) - pl.col("lactate")).over("stay_id").alias("lactate_clearance_4h"),
    pl.col("fluid_balance_4h").cum_sum().over("stay_id").alias("cumulative_fluid_balance"),
])
```

---

### 🟡 B. Zaman-Bağımlı Feature'lar

| Feature | Açıklama |
|---------|----------|
| `hours_since_icu_admission` | `hour_bin - intime` |
| `icu_day` | Hangi ICU günü (1, 2, 3...) |
| `hour_of_day` | Saat (0-23) — sirkadiyen ritim etkisi |

**Neden?**
- İlk 24 saatteki tedavi kararları ile 5. gündeki kararlar **farklı risk** taşır
- Gece saatlerinde ölçüm sıklığı azalır → null artar

---

### 🟡 C. Komorbidite Skorları

**Mevcut:** Sadece `age`, `gender`, `admission_type`

**Eksik:**
- **Charlson Comorbidity Index** (`diagnoses_icd` tablosundan)
- **SOFA alt skorlarının ayrı ayrı** kullanımı (ajan hangi organ yetmezliğine odaklanmalı?)

```python
# State vector'a eklenecek:
STATE_FEATURES = [
    # ... mevcut feature'lar ...
    "sofa_resp", "sofa_cardio", "sofa_renal",  # Alt skorlar
    "charlson_score",  # Komorbidite yükü
]
```

---

## 3️⃣ MÜHENDİSLİK / İSTATİSTİKSEL İYİLEŞTİRMELER

### 🟠 A. Forward-Fill Stratejisi Riskli

**Mevcut Durum (`pipeline.py`):**
```python
pl.col(col).forward_fill().over("stay_id")
```

**Problem:**
- `temp_c` %83 null → forward-fill **83 saat eski veri** taşıyabilir
- `lactate` %28 null → kritik laboratuvar değeri eskiyor
- RL ajanı **güncel olmayan state** ile karar verir

**Öneri:**
```python
# 1. Zaman-penceresi ile forward-fill sınırla
df = df.with_columns([
    pl.col("lactate").fill_null(strategy="forward").over("stay_id")
      .where(pl.col("hour_bin") - pl.col("last_lactate_time") < pl.duration(hours=6))
])

# 2. Null flag ekle (ajan "bu veri eski" bilsin)
df = df.with_columns([
    pl.col("lactate").is_null().cast(pl.Int8).alias("lactate_is_imputed"),
])
```

---

### 🟠 B. Outlier Handling Eksik

**EDA'dan Tespit Edilen Anomaliler (`01_data_explore.md`):**

| Feature | Min Değer | Problem |
|---------|-----------|---------|
| `sbp` | -94 mmHg | **Negatif kan basıncı imkansız** |
| `heart_rate` | -241,395 bpm | Sensör hatası |
| `spo2` | -951,234 % | Sensör hatası |
| `fluid_balance_4h` | -876,537 ml | Hesaplama hatası |

**Öneri:**
```python
# pipeline.py'ye eklenecek:
df = df.with_columns([
    pl.when(pl.col("sbp").is_between(50, 250)).then(pl.col("sbp")).otherwise(None).alias("sbp"),
    pl.when(pl.col("heart_rate").is_between(30, 250)).then(pl.col("heart_rate")).otherwise(None).alias("heart_rate"),
    pl.when(pl.col("spo2").is_between(70, 100)).then(pl.col("spo2")).otherwise(None).alias("spo2"),
    pl.when(pl.col("fluid_balance_4h").is_between(-2000, 2000)).then(pl.col("fluid_balance_4h")).otherwise(None).alias("fluid_balance_4h"),
])
```

---

### 🟠 C. Feature Scaling Eksik (RL İçin Kritik)

**Mevcut Durum:**
- Feature'lar raw değerlerle (HR: 30-250, SOFA: 0-24, Fluid: -876k ile +1M)
- **Neural network gradient explosion** riski
- JAX/Flax modelleri **scale edilmemiş veri** ile instabil öğrenir

**Öneri:**
```python
# 02_feature_engineering.md sonuna eklenecek:
from sklearn.preprocessing import RobustScaler  # Outlier'a dayanıklı

# Veya Polars ile:
df = df.with_columns([
    (pl.col("heart_rate") - 87.8) / 3797.4,  # (mean, std) ile standardize
    (pl.col("sofa_score") - 4.7) / 3.6,
    (pl.col("fluid_balance_4h") - 31.6) / 1466.6,
])

# VEYA min-max scaling [0, 1]:
df = df.with_columns([
    (pl.col("sofa_score") / 24).alias("sofa_score_norm"),
    ((pl.col("heart_rate") - 30) / 220).alias("heart_rate_norm"),
])
```

**Önemli:** Scaling parametrelerini (mean, std) **train set'ten** hesaplayıp test/val set'e uygulayın (data leakage önleme).

---

### 🟠 D. Action Space Tanımı Net Değil

**Mevcut Durum:**
- `prev_fluid_dose`, `prev_vaso_dose` lag feature olarak var
- Ama **action space** (ajanın seçebileceği doz aralıkları) tanımlanmamış

**Öneri:**
```python
# config.py'ye eklenecek:
ACTION_SPACE = {
    "fluid_bins": [0, 250, 500, 1000, 2000],  # ml
    "vaso_bins": [0, 0.05, 0.1, 0.2, 0.5],  # norepi equiv mcg/kg/min
}

# RL environment'da discrete action space:
# Action 0-4: Fluid dozu
# Action 5-9: Vazo dozu
```

---

## 📊 ÖNCELİKLENDİRİLMİŞ UYGULAMA PLANI

| Öncelik | Görev | Tahmini Efor | RL Etkisi |
|---------|-------|--------------|-----------|
| **P0** | Outlier handling (negatif SBP, HR) | 2 saat | 🔴 Kritik |
| **P0** | Feature scaling (RL training için) | 3 saat | 🔴 Kritik |
| **P1** | SOFA delta feature'ları | 4 saat | 🟠 Yüksek |
| **P1** | Hasta ağırlığı ile doz standardizasyonu | 6 saat | 🟠 Yüksek |
| **P2** | Antibiyotik başlangıç zamanı | 8 saat | 🟡 Orta |
| **P2** | Null flag feature'ları | 2 saat | 🟡 Orta |
| **P3** | SIRS skoru | 3 saat | 🟢 Düşük |
| **P3** | Mekanik ventilasyon itemid iyileştirmesi | 4 saat | 🟢 Düşük |

---

## 🎯 SONUÇ VE STRATEJİK TAVSİYE

**Güçlü Yönleriniz:**
✅ Polars pipeline mimarisi (bellek optimizasyonu)  
✅ SOFA skoru doğru hesaplanmış  
✅ Vazopressör eşdeğer standardizasyonu mevcut  
✅ Lag feature'ları RL için doğru düşünülmüş

**Kritik İyileştirmeler:**
⚠️ **Outlier handling olmadan RL modeli diverge eder** (negatif SBP gradient patlaması)  
⚠️ **Scaling olmadan JAX/Flax stabil öğrenmez**  
⚠️ **SOFA delta olmadan ajan "kötüleşme hızını" öğrenemez**  
⚠️ **Hasta ağırlığı olmadan doz kararları genelleştirilemez**

**Bir Sonraki Adım:**
`pipeline.py`'ye outlier filtering + scaling ekleyin, ardından `src/envs/sepsis_env.py` adında custom Gymnasium environment oluşturun. Action space'i discrete (0-9 arası doz kategorileri) olarak tanımlayın ve state vector'u yukarıdaki P0-P1 feature'larla genişletin.

Başarılar dilerim. Sorularınız olursa detaylandırabilirim.

— **Medikal AI Konsey Danışmanı**
