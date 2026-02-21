# MIMIC-IV sepsis veri ön işleme ve RL state uzayı değerlendirmesi

**Created:** 2/21/2026 14:25:49  
**Updated:** 2/21/2026 14:28:32  
**Exported:** 2/21/2026 14:46:18  
**Link:** [https://claude.ai/chat/a8944e6b-0df6-4a2b-aa58-af02d3218941](https://claude.ai/chat/a8944e6b-0df6-4a2b-aa58-af02d3218941)  

## Prompt:
2/21/2026, 2:25:51 PM

> File: 01_data_explore.md

> File: 02_feature_engineering.md

> File: config.py

> File: pipeline.py

> File: project_summary.md

# RL Sepsis Ajanı: Veri Ön İşleme (Preprocessing) Konsey Değerlendirmesi **Bağlam:** Biz tıp alanında, MIMIC-IV veri setini kullanarak Sepsis yönetimi için Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL) modelleri geliştirmeyi amaçlayan bir proje yürütüyoruz. Amacımız, yoğun bakım (ICU) ortamındaki sepsis hastalarının verilerini işleyerek klinik durumlarını (state) modellemek ve bu verilere dayanarak optimum tedavi stratejilerini (örn. vazopressör dozajı ve sıvı tedavisi) öğrenecek yapay zeka ajanları (RL agent) eğitmektir. Veri biliminde performansı artırmak ve RAM taşmasını engellemek amacıyla Polars kütüphanesini kullanarak, zaman serisi tıbbi verileri saatli periyotlara (hourly binning) çeviren, lazy evaluation + single-pass group-by içeren bellek dostu bir süreç kurguladık. **Görev:** Sen üst düzey bir veri bilimi ve medikal yapay zeka konseyinin danışman üyesisin. Bu istem (prompt) ile birlikte sana projemizin durumunu anlatan dosyalar ve kodlar sağlıyorum. Lütfen sana ilettiğim şu dökümanları incele: 1. `project_summary.md`: Projenin genel mimarisi, yapısı ve hedefi. 2. `notebooks/01_data_explore.md` & `notebooks/02_feature_engineering.md`: Veri üzerinde yaptığımız keşifsel analiz (EDA), hücre (cell) çıktıları, eksik veri grafikleri, saatlik gruplamalar ve korelasyonların sonuçları. 3. `src/preprocess/config.py` & `src/preprocess/pipeline.py`: Araştırmalarımız sonucunda elde ettiğimiz bulguları, Polars kullanarak devasa MIMIC-IV verilerine (OOM hatası almadan) uyguladığımız production-ready (canlıya hazır) nihai bellek dostu ön işleme hattımız. ***(Önemli Not: Notebook'larda verinin yapısı/çıktıları ve EDA yöntemlerimiz mevcuttur. Python (.py) dosyalarında ise bu kararların tüm veriye nasıl bir teknik mimariyle aktarıldığı bulunur. Lütfen her iki kaynağı da sentezleyerek yorum yap.)*** Lütfen sadece "veri ön işleme", "feature engineering" ve "RL State (durum) / Action (aksiyon) alanı seçimi" kapsamlarına odaklanarak kodlarımızı ve çıktıları analiz et. Bize şu soruların cevabını ve stratejik öneriler sun: 1. **Bu projenin veri ön işlemesinde eksik olan kritik bilgiler, özellikler (feature) nelerdir?** - İlettiğim notebook çıktılarına (eksik veri oranları vs.) ve Python dosyalarındaki logic'e bakarak, RL ajanının "state" (durum) uzayı için eksik ama başarı için elzem olan laboratuvar, vital veya demografik veriler var mı? (Örn: Sepsis-3 teşhis kriterleri eksiksiz karşılanıyor mu? Saatlik SOFA skoru, SIRS kriterleri, antibiyotik kullanım başlama zamanı, mekanik ventilasyon durumu eklenmeli mi?) 2. **Mevcut Verilerden Neler Üretilebilir / Eklenebilir?** - Hangi yeni klinik/laboratuvar değişkenler modele önemli bir içgörü katar? EDA verilerine dayanarak RL ajanı için state tanımını güçlendiren ek "feature engineering" tavsiyelerin nelerdir? (Örn: Değişim hızları (delta değerleri), ağırlık üzerinden standardize edilmiş dozlar vs.) 3. **Mevcut geliştirme aşamasında (kodda veya metodolojide) geliştirilebilecek mühendislik / istatistiksel yöntemler nelerdir?** - Özellikle Notebook'lardaki hücre sonuçlarını dikkate alıp, `pipeline.py` içindeki işlemleri (imputasyon vs.) birleştirdiğinde, daha efektif bir işleme veya doldurma metodu görüyor musun? (Örn: Biz forward-fill uyguladık. İstatistiksel ve tıbbi olarak, ekstrem null oranına sahip feature'larda daha iyi bir interpolasyon/imputasyon uygulanabilir mi? Outlier handling veya scaling konusunda eksiklerimiz nelerdir?) Lütfen sadece genel-geçer bilgiler vermek yerine, sana verdiğim kod mimarisini ve notebook çıktı sonuçlarını *(null oranları, dağılımlar vs.)* baz alarak projemize özel net, uygulanabilir teknik ve tıbbi tavsiyeleri **maddeler halinde ve detaylı olarak** açıkla. --- *(Lütfen cevaplarınızı direkt çözüm stratejilerimiz üzerine odaklayın, yukarıdaki metni veya kod bloklarını gereksiz yere tekrar kopyalamayın.)*



## Response:
2/21/2026, 2:28:32 PM



````plaintext
Thought process: Değerlendirdi eksik dosya içeriklerini ve analiz stratejisini planladı.

The user wants a detailed expert council evaluation of their MIMIC-IV Sepsis DRL preprocessing pipeline. They've uploaded 5 files but I can only see the content of 3 in the context (config.py, pipeline.py, project_summary.md). The notebook files (01_data_explore.md, 02_feature_engineering.md) are uploaded but their content isn't shown in the context window - I need to read them.

Let me check the uploaded files.
````

````plaintext
View: Reading EDA notebook outputs
````

````plaintext
Thought process: Veri ön işleme ve özellik mühendisliği dokümantasyonunu sentezlemeyi planladı.

Let me see the truncated part and the feature engineering notebook.
````

````plaintext
View: Reading truncated null rates section
````

````plaintext
Thought process: Hazırlandı, feature engineering notebook'unu incelemeye başladı.

Now let me read the feature engineering notebook.
````

````plaintext
View: Reading feature engineering notebook
````

````plaintext
Thought process: Incelenmek üzere özellik mühendisliği not defterinin eksik bölümlerini sentezledi.

Now let me also check the truncated middle part of the feature engineering notebook to understand how SOFA, mechanical ventilation, etc. are computed.
````

````plaintext
View: Reading the middle section of feature engineering notebook
````

````plaintext
Thought process: Tanımladı kritik veri boşluklarını ve iyileştirme stratejilerini.

Now I have a comprehensive understanding of the project. Let me analyze everything and provide a detailed expert council evaluation covering:

1. Missing critical features for the RL state space
2. Feature engineering recommendations
3. Engineering/statistical improvements

Key findings from the data:
- 8.8M rows, 94,458 unique stay_ids, median ~53 hours per stay
- temp_c: 83.4% null - extremely high missing rate
- lactate: 28% null
- bilirubin_total: 37.3% null
- fio2: 32.4% null (→ pao2/paco2/ph ~25-26% null)


- vasopressor-related inputs: 18.3% null with forward-fill applied, but significant data quality issues remain

The outlier problem is catastrophic—the describe() statistics reveal impossible values like heart_rate hitting 10 million, systolic BP over 1 million, and respiratory rate at 7 million, which suggests serious data corruption or encoding errors that need immediate investigation and cleaning before any modeling can proceed.

Beyond the extreme outliers, I'm seeing additional red flags: urine output goes negative, shock_index contains NaN and negative values from division by zero, vasopressor doses seem implausibly high, and the forward-fill strategy is masking problems by propagating identical values across multiple days for the same patient, which could artificially inflate temporal patterns in the data.

For the Sepsis-3 criteria specifically, I'm missing critical clinical inputs like antibiotic timing, blood culture orders, and infection suspicion timestamps. The SIRS criteria are compromised by 83% missing temperature data, the mechanical ventilation proxy using FiO2 thresholds is too noisy, and the vasopressor normalization factors I'm using don't align well with clinical literature—plus I'd need patient weights to properly index the dosing.

Beyond that, there are significant gaps in the state representation: no temporal features like time since ICU admission or rate-of-change metrics for key labs, no cumulative intervention tracking, missing ventilator parameters like PEEP and tidal volume, and sparse central venous pressure or ScvO2 data that could help assess tissue perfusion.

The methodological approach has several problems too—forward-filling vitals across thousands of hours creates clinically implausible data, the mechanical ventilation proxy based on FiO2 thresholds is unreliable since room air is 21% and low-flow oxygen patients blur that boundary, and vasopressor dosing units are inconsistently recorded across the database, making normalization by body weight necessary but difficult without reliable weight data.

For imputation, temperature has such severe missingness (83.4%) that it needs special handling beyond standard forward-fill with time caps, while lactate and bilirubin can use forward-fill with time limits followed by median imputation or interpolation for remaining gaps.

Now I need to think through the reward signal—mortality outcomes and clinical trajectories like SOFA and lactate clearance are essential but haven't been addressed yet, so I should organize all of this into a structured answer.
````

Tüm dosyalar okundu. Şimdi detaylı konsey değerlendirmesini yazıyorum.

---

# Konsey Değerlendirmesi: MIMIC-IV Sepsis DRL Veri Ön İşleme

---

## SORU 1 — Eksik Kritik Özellikler (Missing Features)

### 1.1 Outlier Felaketi: Acil Öncelik

Notebook 1'deki `describe()` çıktısı, pipeline'ınızın henüz **outlier temizliği yapılmamış ham veriyle** çalıştığını net biçimde ortaya koyuyor. Bu durum özellikle feature engineering'i zehirliyor:

| Feature | Gözlenen Max | Klinik Gerçek Max | Sorun |
|---|---|---|---|
| `heart_rate` | 10,000,000 | ~250 bpm | Data entry hatası |
| `sbp` | 1,003,110 | ~300 mmHg | Birim hatası |
| `mbp` std = 4828 | — | — | Tüm türev hesapları bozuk |
| `resp_rate` | 7,000,400 | ~60/dk | Çığır açan gürültü |
| `spo2` | 9,900,000 | 100% | — |
| `lactate` | 1,276,103 | ~30 mmol/L | SOFA hesabını bozuyor |
| `urine_output` | min=-3765 | ≥0 | Negatif değer mantıksız |
| `fluid_balance_4h` | min=-876,537 | klinik sınır ±5000 | Imputation'ı bozuyor |
| `shock_index` | mean=NaN, min=-3094 | 0.3–3.0 | SBP=0 → division by zero |

`shock_index`'in ortalaması NaN çıkıyor: bu, sıfır `sbp` değerlerinden kaynaklanan `inf` değerlerin vektöre sızdığını gösteriyor. RL ajanı bu state'lerle hiçbir şey öğrenemez.

### 1.2 Sepsis-3 Tanı Kriterleri Eksik

Projenin "sepsis yönetimi" iddiasına karşın **sepsis onset zamanı pipeline'da tanımsız.** Bunun doğrudan RL tasarımına etkisi var:

**Antibiyotikler (`prescriptions` + `pharmacy` tabloları):** MIMIC-IV'te `antibiotics_flag` ve `antibiotic_onset_hour` türetmek mümkün. Sepsis-3'ün "şüpheli enfeksiyon" bileşeni için kan kültürü alımı (`microbiologyevents`) ve ilk antibiyotik zamanının `intime`'a göre farkı olan `antibiotic_delay_h` feature'ı, ajan için hem state hem de kritik bağlamsal değişken.

**Mekanik Ventilasyon Proxy'si Yanlış:** `fio2 > 21%` mantığı, düşük akış nazal kanül kullanan hiç de intübe olmayan binlerce hastayı "mekanik ventilasyonda" olarak etiketliyor. Gerçek MV için `chartevents`'teki ventilasyon modu item ID'leri kullanılmalı (220339 = Ventilation Mode, 224688 = Ventilator Mode). Mevcut proxy, SOFA respiratory skorunu ciddi ölçüde yanıltıyor.

**Ağırlık Eksikliği:** Tüm vazopressör dozları `mcg/kg/min` standardında raporlanır; ancak pipeline'da hasta ağırlığı yok. `inputevents`'te `amount` yerine `rate` ve `rateuom` kullanılmalı, ardından `chartevents`'ten daily weight (item 224639 veya 226512) ile normalize edilmeli. Mevcut `total_vaso_equiv` hesabınız kg-bağımsız, dolayısıyla 50 kg hasta ile 120 kg hasta için aynı dozu aynı şekilde kodluyor.

**Kümülatif Sıvı Dengesi Eksik:** `fluid_balance_4h` noktasal bir pencere. Literatür (Marik et al., Semler et al.) kümülatif toplam sıvı yükünün (`cumulative_fluid_balance`) tek başına bağımsız öngörücü olduğunu gösteriyor. Şu anki 4h balance, RL ajanının genel klinik tablodaki sıvı birikimini görmesini engelliyor.

**Laktat Klirensi Eksik:** Laktat'ın tek başına değil, trend'i (clearance oranı) kritik. Sepsis-3 "Surviving Sepsis Campaign" kılavuzu laktat klirensini (<2 mmol/L veya ≥10% azalış) tedavi hedefi olarak tanımlıyor. Bu hem bir reward shaping komponenti hem de state feature adayı.

---

## SORU 2 — Üretilebilecek / Eklenmesi Gereken Yeni Feature'lar

### 2.1 Delta (Δ) Özellikler — Muhtemelen En Yüksek Değerli Ekleme

Mevcut state vector anlık snapshot içeriyor. RL ajanının "durum kötüleşiyor mu, iyileşiyor mu?" sorusunu yanıtlaması için değişim hızlarına ihtiyacı var. Polars'ta bu çok ucuz:

```python
# Stay içinde saatlik delta
for col in ["lactate", "creatinine", "mbp", "sofa_score"]:
    df = df.with_columns(
        (pl.col(col) - pl.col(col).shift(1).over("stay_id"))
        .alias(f"delta_{col}")
    )
```

Özellikle `delta_lactate` (laktat klirensi için proxy), `delta_sofa_score` ve `delta_mbp` ajanın prognostik sinyal almasını sağlayacak en önemli feature'lardır.

### 2.2 Zaman Bağlamı — Time Since ICU Admission

Stay'in hangi saatinde olunduğu bilgisi hem state hem de reward shaping açısından kritik:

```python
# Yatıştan geçen saat — erken ve geç fazı birbirinden ayırır
df = df.with_columns(
    (
        (pl.col("hour_bin") - pl.col("hour_bin").first().over("stay_id"))
        .dt.total_hours()
        .alias("hours_since_icu_admission")
    )
)
```

Sepsis yönetimi, ilk 6 saatte (erken hedef odaklı resüsitasyon) çok farklı bir strateji gerektirir. Bu feature olmadan ajan, olayların kronolojik bağlamını göremez. Bu aynı zamanda max stay 5411 saatlik aykırı değerleri belirlemenizi kolaylaştırır.

### 2.3 Vazopressör Rate vs. Amount Sorunu

`build_inputs_hourly()` içinde `pl.col("amount")` kullanıyorsunuz. Bu, `starttime`–`endtime` aralığına yayılan infüzyonlar için yanıltıcı: **amount = toplam verilen miktar, rate = saatlik doz.** Sürekli infüzyon olan vazopressörler için `rate` kolonunu kullanmak ve hem `rateuom` (ünite) hem de `ordercategorydescription` filtresi eklemek gerekiyor. Aksi hâlde bir hastanın 1 saat içinde verilen total miktarıyla 12 saatlik bir infüzyonun miktarı aynı şekilde kodlanıyor.

### 2.4 Kümülatif Müdahale Özellikleri

Klinisyen kararlarında "bu hastaya toplam ne kadar sıvı verdik?" sorusu standart. RL ajanı için de çok değerli:

```python
df = df.with_columns([
    pl.col("crystalloid_ml").fill_null(0).cum_sum().over("stay_id").alias("cumulative_fluid_ml"),
    pl.col("total_vaso_equiv").fill_null(0).cum_sum().over("stay_id").alias("cumulative_vaso_equiv"),
])
```

### 2.5 Saatlik İdrar Çıkışı → 24h Moving Average

`urine_output` tek saatlik değer olarak çok gürültülü. SOFA renal skoru 24 saatlik toplama bakar. Hem SOFA hesabınız hem de ajan için 6h ve 24h rolling sum değerleri daha anlamlı sinyal taşır:

```python
df = df.with_columns(
    pl.col("urine_output").fill_null(0)
      .rolling_sum(window_size=24, min_periods=1)
      .over("stay_id")
      .alias("urine_24h")
)
```

---

## SORU 3 — Mühendislik ve İstatistiksel Metodoloji Geliştirmeleri

### 3.1 Forward-Fill: Zaman Sınırı Olmadan Uygulamak Tıbbi Olarak Yanıltıcı

Notebook 1'in `tail()` çıktısına bakıldığında, `stay_id=39999858` için `heart_rate=82.0`, `sbp=107.0` vb. değerler **aylar boyunca** aynı kalıyor. Bu forward-fill'ın sınırsız uygulandığını doğruluyor.

Tıbbi gerçek şu: 8 saatten eski forward-filled vital sign klinik karar verme açısından anlamsız; 48 saatten eski lab değeri ise aktif olarak yanıltıcı. Önerilen katmanlı strateji:

| Feature Grubu | Max Forward-Fill Süresi | Sonraki Adım |
|---|---|---|
| Vitals (HR, BP, RR, SpO2) | 4 saat | Median imputation |
| Sıcaklık (temp_c, %83.4 null) | 8 saat | Sabit median (37.0°C) |
| Sık çekilen lablar (creatinine, BUN, WBC) | 24 saat | Last-observation-carried-forward |
| Nadir lablar (bilirubin, INR, %37 null) | 48 saat | LOCF + missingness flag kolonu |
| ABG (PaO2, PaCO2, pH, %25-26 null) | 8 saat | Sadece MV hastalarında geçerli |
| Vazopressörler | 2 saat | 0 ile doldur (verilmiyorsa 0) |

Polars'ta `forward_fill(limit=N)` ile bu kolayca uygulanabilir:

```python
pl.col("creatinine").forward_fill(limit=24).over("stay_id")
```

### 3.2 Missingness Indicator (Eksiklik Bayrağı) Kolonları

%37 null olan `bilirubin_total` ve %83 null olan `temp_c` için forward-fill sonrası kaç saat eskimiş değer kullandığınızı izlemek tıbbi olarak kritik. Bir flag kolonu hem ajanın imputed değere ne kadar güvenmesi gerektiğini öğrenmesine yardımcı olur, hem de model yorumlanabilirliğini artırır:

```python
# Ölçüm yapılmış mı yoksa imputed mu?
df = df.with_columns(
    pl.col("bilirubin_total").is_null().cast(pl.Int8).alias("bilirubin_missing_flag")
)
# Forward fill ÖNCESİ bu flag'ı oluşturun
```

### 3.3 Outlier Temizliği — IQR Fence / Klinik Aralık Kırpma

Veriler pipeline'a girerken değil, feature engineering aşamasında kırpılmalı. Klinik aralık tabanlı yaklaşım, IQR'dan daha tutarlı:

```python
CLINICAL_BOUNDS = {
    "heart_rate":   (20, 300),
    "sbp":          (30, 300),
    "dbp":          (10, 200),
    "mbp":          (20, 200),
    "resp_rate":    (4, 80),
    "spo2":         (50, 100),
    "temp_c":       (25, 45),
    "lactate":      (0.1, 30),
    "urine_output": (0, 2000),   # saatlik ml — negatif olamaz
    "crystalloid_ml": (0, 10000),
}

for col, (lo, hi) in CLINICAL_BOUNDS.items():
    if col in df.columns:
        df = df.with_columns(
            pl.col(col).clip(lo, hi)
        )
```

Bu tek adım, `shock_index`'inizdeki NaN/inf sorununu, `fluid_balance_4h`'daki ±876,000 ml saçmalığını ve `sofa_score`'daki bozulmaları doğrudan çözecektir.

### 3.4 Vasopressör `amount` vs. `rate` Hatası

`build_inputs_hourly()` içinde `pl.col("amount")` kullanmak ve bunu `hour_bin`'e göre toplamak, sürekli infüzyonları yanlış temsil ediyor. Düzeltme:

```python
# inputevents'te sürekli infüzyon için rate tercih edilmeli
inputs = pl.scan_csv(INPUTEVENTS_CSV, ...)
    .with_columns(
        # Saatlik normalize doz: rate (mcg/min) → 1 saatlik miktar
        pl.when(pl.col("statusdescription") == "FinishedRunning")
          .then(pl.col("rate"))
          .otherwise(pl.col("amount") / 
            ((pl.col("endtime") - pl.col("starttime")).dt.total_hours() + 1e-6))
          .alias("hourly_rate_approx")
    )
```

### 3.5 SOFA Hesabındaki Yapısal Eksiklik

Mevcut SOFA hesabınız birkaç kritik eksik içeriyor. Bunlar notebook 2'de kısmen ele alınmış ama pratikte:

**Kardiyovasküler SOFA:** MAP < 70 mmHg eşiği doğru kullanılıyor. Ancak vazopressör normalizasyonunuz ağırlıksız olduğu için (kg bilgisi yok) SOFA kardiyovasküler alt skoru sistematik olarak hatalı. Eğer ağırlık verisi eklenmezse, en azından `mbp < 70` bayrağını birincil sinyal olarak kullanın, norepinefrin dozunu ikincil.

**Renal SOFA:** Mevcut implementasyonda kreatinin ve idrar ayrı ayrı skorlanıyor ve maksimum alınıyor mu? Kontrol edin; SOFA renal için kural `max(creatinine_score, urine_score)` olmalı.

**SOFA Skoru Null Davranışı:** `sofa_score` için null_count=0 görmek, eksik feature'larda 0 varsayılan değerinin kullanıldığını düşündürüyor. Eksik bilirubin → bilirubin SOFA = 0 varsayımı kabul edilebilir ama bunu açıkça belgelemeniz gerekiyor; çünkü bu, ciddi karaciğer yetmezliğini gizleyebilir.

### 3.6 Aşırı Uzun Stay'lerin Filtrelenmesi

Median 53 saat, max 5411 saat. 5411 saatlik (225 gün!) bir stay, tanım gereği kronik bakım hastası veya veri kalite sorunudur ve sepsis DRL cohort'unda bulunmamalı. Literatür standardına göre (Komorowski et al. 2018, Nat Med) sepsis DRL çalışmaları için yaygın cohort filtresi:

- ICU yatış süresi ≥ 4 saat, ≤ 30 gün (720 saat)
- Sepsis-3 teşhis kriterlerini karşılayan stay'ler

Bu filtreleme uygulanırsa 8.8M satırın muhtemelen belirgin bir kısmı düşecek ve model kalitesi artacak.

### 3.7 Ödül Sinyali (Reward) Notası

Doğrudan sorulmadı ama RL mimarisi için kritik bir boşluk: `pipeline.py` mortality outcome üretmiyor. MIMIC-IV `admissions.csv.gz`'de `deathtime` var; buna ek olarak `hospital_expire_flag` ve `icustays`'teki `los` (length of stay) ile 28 günlük ölüm etiketi (`mortality_28d`) türetilebilir. Reward şekillendirmesi için laktat klirensi ve SOFA delta'sı da yararlıdır ama terminal sinyal olarak mortality şart.

---

## Özet Öncelik Matrisi

| Öncelik | Aksiyon | Zorluk | Etki |
|---|---|---|---|
| 🔴 Kritik | Klinik aralık clipping (outlier temizliği) | Düşük | Çok Yüksek |
| 🔴 Kritik | Forward-fill zaman limiti ekle | Düşük | Yüksek |
| 🔴 Kritik | `rate` kullan, `amount` değil (vazopressör) | Orta | Yüksek |
| 🟠 Önemli | Delta features (lactate, SOFA, creatinine) | Düşük | Yüksek |
| 🟠 Önemli | Missingness flag kolonları | Düşük | Orta |
| 🟠 Önemli | Cumulative fluid balance | Düşük | Orta |
| 🟠 Önemli | Gerçek MV itemID'leri (FiO2 proxy'den kaçın) | Orta | Yüksek |
| 🟡 Tavsiye | Hasta ağırlığı → doz normalizasyonu | Yüksek | Orta |
| 🟡 Tavsiye | Antibiyotik onset zamanı | Yüksek | Orta |
| 🟡 Tavsiye | Cohort filtresi (≤720 saat, Sepsis-3) | Orta | Orta |
| 🟡 Tavsiye | 28 günlük mortality label (reward için) | Orta | Kritik (RL için) |





---
Powered by [Claude Exporter](https://www.ai-chat-exporter.net)