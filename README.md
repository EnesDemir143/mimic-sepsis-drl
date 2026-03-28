# MIMIC Sepsis Offline RL

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

## 🏁 Hızlı Başlangıç (Quick Reference)

Projeyi ayağa kaldırmak ve bağımlılıkları yönetmek için `uv` ekosistemini kullanabilirsiniz:

```bash
# 1. Repoyu bilgisayarınıza alın
git clone <repo-url>
cd mimic-sepsis

# 2. Sanal ortam (venv) yaratın ve aktif edin
uv venv
source .venv/bin/activate

# 3. Temel bağımlılıkları ekleyin
uv sync
```

⚠️ **MIMIC-IV Kullanımı Hakkında:** Orijinal hasta kayıtları üzerinde analiz apmak için **PhysioNet** kapsamında CITI sertifikası ve onaylı bir erişim yetkinliğine sahip olmanız gerekmektedir. Proje açık kaynaklı veri analitiği altyapısını içerir, hasta verisi barındırmaz.

## 📚 Atıflar (Citations)

Proje kapsamında MIMIC-IV veri setini kullanırken referans vermeniz gereken yayınlar:

**MIMIC-IV Dataset:**
> Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

**MIMIC-IV Publication:**
> Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

**PhysioNet Standard Citation:**
> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
