# MIMIC Sepsis Offline RL

> Not: Bu repoyu çalıştırmadan önce ham MIMIC-IV dosyaları şu dizinde bulunmalıdır:
> `data/raw/physionet.org/files/mimiciv/3.1`
> Bu klasör yoksa veri pipeline'ı çalışmaz.

## TL;DR
MIMIC-IV veri seti üzerinde, klinisyen tedavi yöntemleri ile Offline RL (Çevrimdışı Pekiştirmeli Öğrenme) modellerini (CQL, BCQ, IQL) değerlendiren; veri sızıntısına karşı yalıtılmış ve Sepsis-3 tabanlı şeffaf bir araştırma ve benchmark sistemidir.

## 📌 Proje Hakkında
Bu proje, MIMIC-IV içerisindeki yetişkin yoğun bakım (ICU) sepsis vakalarını 4 saatlik zaman adımlarına (onset -24h ile +48h arası) bölerek bir Markov Karar Sürecine (MDP) dönüştürmektedir.

Ana amaç; veri sızıntısını önleyen (data leakage protected), **klinik olarak makul** ve makale/tez kalitesinde **yeniden üretilebilir (reproducible)** bir çevrimdışı pekiştirmeli öğrenme çalışma alanı sağlamaktır. Sistem online klinik kararlar vermek için değil, retrospektif çevrimdışı politikaları değerlendirmek (Offline Policy Evaluation - OPE) için tasarlanmıştır.

## 🚀 Temel Özellikler
* **Hedef Kohort:** Sepsis-3 kriterlerine uygun yetişkin ICU (Yoğun Bakım) hastaları.
* **MDP Altyapısı (Durum-Eylem):** Sürekli (continuous) hasta durum (state) vektörleri ve tedaviler için (vazopressör ve IV fluid dozlarına bağlı) **25 farklı ayrık eylem (discrete action)**.
* **Katı Sızdırmazlık (Zero Leakage):** Eğitim, doğrulama ve test setleri hasta bazında (patient-level) ayrılarak "scaling / imputation" hesaplamaları tamamen eğitim setine sınırlanır.
* **Güvenli RL Karşılaştırmaları:** En gelişmiş tutucu RL (CQL, BCQ, IQL) yaklaşımlarının aynı veriler ile adil karşılaştırmaları.
* **Donanım Esnekliği:** Hem veri hem de PyTorch eğitim ortamı tek kod tabanından kodlanarak **Apple Silicon (MPS)** ve **NVIDIA GPU (CUDA)** üzerinde çalıştırılabilir.

## 🛠 Teknoloji Yığını

| Bileşen / Kütüphane | Kullanım Amacı | Durum | Notlar |
|-----------------------|-------|---------|--------|
| **Python / uv** | Temel dil, modern ortam yönetimi | ✅ | Projenin çekirdeği |
| **PyTorch & d3rlpy**  | Ağırlıklı ML eğitimi, offline RL opsiyonları | ✅ | Hem MPS hem CUDA performansı |
| **Polars & PyArrow**  | Yüksek hızlı veri transformasyonu | ✅ | Parquet artifaktları üretebilme |
| **scikit-learn**      | Veri imputasyonu, scaling, ayırma | ✅ | Baseline performans algoritmaları |
| **Hydra & MLflow**    | Deney takibi ve konfigürasyon (config) | ✅ | Yeniden üretilebilirlik güvencesi |

## 🏁 Çalıştırma Sırası

Bu proje için temel kural şudur:

- Ham veri önce `data/raw/physionet.org/files/mimiciv/3.1` altında hazır olmalı.
- Sonra veri pipeline'ı sırayla çalıştırılmalı.
- En sonda seçilen yöntem (`cql`, `bcq`, `iql`) eğitilmelidir.

### 1. Ortamı hazırla

```bash
uv sync
```

### 2. Kohortu üret

```bash
uv run python -m mimic_sepsis_rl.cli.build_cohort \
  --config configs/cohort/default.yaml \
  --emit-audit
```

Beklenen çıktı:

- `data/processed/cohort/cohort.parquet`
- `data/processed/cohort/excluded.parquet`
- `data/processed/cohort/audit.json`

### 3. Sepsis onset üret

```bash
uv run python -m mimic_sepsis_rl.data.onset \
  --config configs/onset/default.yaml
```

Beklenen çıktı:

- `data/processed/onset/onset_assignments.parquet`
- `data/processed/onset/onset_candidates.parquet`
- `data/processed/onset/unusable_episodes.parquet`
- `data/processed/onset/onset_audit.json`

### 4. Episode grid üret

```bash
uv run python -m mimic_sepsis_rl.cli.build_episode_grid
```

Beklenen çıktı:

- `data/processed/episodes/episodes.parquet`
- `data/processed/episodes/episode_steps.parquet`
- `data/processed/episodes/grid_audit.json`

### 5. Train / validation / test split üret

```bash
uv run python -m mimic_sepsis_rl.data.splits \
  --config configs/splits/default.yaml \
  --source-episode-set data/processed/episodes/episodes.parquet
```

Beklenen çıktı:

- `data/splits/train_manifest.parquet`
- `data/splits/validation_manifest.parquet`
- `data/splits/test_manifest.parquet`
- `data/splits/split_summary.json`

### 6. State / action / reward / replay dataset üret

```bash
uv run python -m mimic_sepsis_rl.cli.build_transitions
```

Beklenen ana çıktılar:

- `data/processed/features/state_vectors/state_table_raw.parquet`
- `data/processed/features/state_vectors/state_table_normalized.parquet`
- `data/processed/features/train_medians.json`
- `data/processed/features/state_vectors/preprocessing_artifacts.json`
- `data/processed/actions/action_bins.json`
- `data/processed/actions/step_actions.parquet`
- `data/processed/rewards/reward_config.json`
- `data/processed/rewards/step_rewards.parquet`
- `data/replay/replay_train.parquet`
- `data/replay/replay_train_meta.json`
- `data/replay/replay_validation.parquet`
- `data/replay/replay_validation_meta.json`
- `data/replay/replay_test.parquet`
- `data/replay/replay_test_meta.json`

### 7. Runtime doğrulaması yap

```bash
uv run python -m mimic_sepsis_rl.training.device --self-check
```

### 8. Eğitilecek yöntemi doğrula

Burada sadece algoritma adı değişir:

- `cql`
- `bcq`
- `iql`

Örnek:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner \
  --algorithm cql \
  --describe

uv run python -m mimic_sepsis_rl.training.experiment_runner \
  --algorithm cql \
  --dry-run
```

### 9. Eğitimi başlat

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql
```

BCQ veya IQL çalıştırmak için sadece algoritma parametresini değiştir:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm bcq
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql
```

### 10. Daha önce veri aşamalarını çalıştırdıysan

Eğer sende şu dosyalar zaten varsa:

- `data/processed/cohort/*`
- `data/processed/onset/*`
- `data/processed/episodes/*`
- `data/splits/*`

o zaman artık kalan minimum komutlar bunlar:

```bash
uv run python -m mimic_sepsis_rl.cli.build_transitions

uv run python -m mimic_sepsis_rl.training.device --self-check

uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql --describe
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql --dry-run
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql
```

BCQ veya IQL için son üç komutta sadece `--algorithm` değeri değişir.

⚠️ **MIMIC-IV Kullanımı Hakkında:** Orijinal hasta kayıtları üzerinde analiz apmak için **PhysioNet** kapsamında CITI sertifikası ve onaylı bir erişim yetkinliğine sahip olmanız gerekmektedir. Proje açık kaynaklı veri analitiği altyapısını içerir, hasta verisi barındırmaz.

## 📚 Atıflar (Citations)

Proje kapsamında MIMIC-IV veri setini kullanırken referans vermeniz gereken yayınlar:

**MIMIC-IV Dataset:**
> Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

**MIMIC-IV Publication:**
> Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

**PhysioNet Standard Citation:**
> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
