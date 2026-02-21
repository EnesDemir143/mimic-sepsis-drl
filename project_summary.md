# Proje Özeti: MIMIC Sepsis DRL

Bu proje, **MIMIC-IV** klinik veri setini kullanarak Sepsis yönetimi için **Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL)** modelleri geliştirmeyi amaçlayan bir altyapı sunmaktadır. 

## 1. Proje Amacı ve Kapsamı
Projenin temel hedefi, yoğun bakım (ICU) ortamındaki sepsis hastalarının verilerini işleyerek klinik durumlarını (state) modellemek ve bu verilere dayanarak optimum tedavi stratejilerini (örn. vazopressör dozajı ve sıvı tedavisi) öğrenecek yapay zeka ajanları (RL agent) eğitmektir.

## 2. Kullanılan Teknolojiler
- **Derin Öğrenme / RL:** JAX, Flax, Optax (Özellikle Mac GPU ve Linux CUDA üzerinde yüksek performanslı JIT derlemeli eğitim için seçilmiş).
- **Veri İşleme (Data Engineering):** Polars (Çok büyük boyutlu klinik verileri, bellek dostu ve hızlı bir şekilde işleyebilmek için lazy evaluation destekli pipeline).
- **Takviyeli Öğrenme Ortamı:** Gymnasium.
- **Veri Versiyonlama:** DVC (Data Version Control).

## 3. Proje Mimarisi

### A. Veri Hazırlama ve Ön İşleme (`src/preprocess/`)
Projenin en ağır kısımlarından biri devasa boyuttaki MIMIC-IV veri setinin işlenmesidir. `pipeline.py` ve `config.py` dosyaları aracılığıyla veriler şu operasyonlardan geçmektedir:
- **Veri Kaynakları:** `chartevents`, `labevents`, `inputevents`, `outputevents`, `icustays`, `patients`.
- **Saatlik Gruplama (Hourly Binning):** Her hastanın yoğun bakım yatış süreci (stay_id) saat dilimlerine (hour_bin) dönüştürülür.
- **Özellik Çıkarımı (Feature Engineering):** 
  - *Hayati Bulgular (Vitals):* Kalp atış hızı, kan basıncı, solunum hızı, sıcaklık, SpO2 vb.
  - *Laboratuvar Sonuçları:* Laktat, kreatinin, glikoz, vb.
  - *Tedaviler ve Müdahaleler:* Vazopressör (Norepinefrin vb.) müdahaleleri, IV kristalloid sıvı miktarları ve idrar çıkışı.
  - *Klinik Skorlar:* Koma durumunu ifade eden GCS (Glasgow Coma Scale).
- **Performanslı Polars Pipeline:** Veriler, RAM taşırma (OOM) hatalarını engelleyecek şekilde, ardışık full-join işlemleri yerine tek geçişli bir gruplama (single-pass group_by) operasyonuyla birleştirilir. Ardından eksik veriler ileriye dönük (forward-fill) metoduyla doldurularak zaman serisi veri seti kullanıma hazır formata (`.parquet`) dönüştürülür.

### B. Takviyeli Öğrenme Altyapısı (`main.py`)
`main.py` dosyası şu an için Gymnasium kütüphanesine ait `CartPole-v1` ortamında bir **DQN (Deep Q-Network)** algoritmasını çalıştıran bir sandbox/iskelet yapıdır.
- **Bileşenler:** Experience Replay Buffer (Geçmiş tecrübeleri depolama) ve Epsilon-Greedy tabanlı hareket seçimi mekanizması mevcuttur.
- **Ağ Yapısı:** Ajanın (Agent) Q-değerlerini tahmin etmesi için Flax kütüphanesiyle yazılmış basit bir MLP (Çok Katmanlı Algılayıcı) ağı kullanılmaktadır.
- Projede RL'nin yüksek performansla uygulanabilmesi için bu iskelet JAX.jit derlemesiyle optimize edilmiştir. Sepsis ortamı (Custom Gymnasium Environment) bağlandığında eğitim bu iskelet üzerinden gerçekleşecektir.

### C. Keşifsel Veri Analizi (`notebooks/`)
- `01_data_explore.ipynb` ve `02_feature_engineering.ipynb` defterleri, veri setinin yapısını tanımak, istatistiksel analizler çıkarmak ve `pipeline.py` içinde uygulanan işlemleri geliştirmeden önce prototiplemek (EDA) amacıyla kullanılmıştır.

## Genel Değerlendirme
Proje an itibarıyla; devasa boyutlara ulaşabilen klinik veri tablolarını pekiştirmeli öğrenme modellerinin anlayacağı saatlik veri dizilerine çevirebilen oldukça sağlam ve performanlı bir **veri hattı (pipeline)** kurmuştur. Ayrıca modern teknoloji yığınıyla (JAX) DRL eğitmek için gerekli iskeleti hazırlamıştır. Projede atılacak bir sonraki doğal adım, elde edilen bu `.parquet` formatındaki veriyi baz alarak Sepsis tedavisini bir Gymnasium ortamı (Custom Environment) haline getirmek ve `main.py` içindeki yapay zeka ajanını bu ortam üzerinde eğitmektir.
