# Sepsis RL Veri Ön İşleme (Preprocessing) - Todo Listesi

Bu liste, LLM Konseyi Çapraz Analiz Raporu'ndaki (`LLM_Konseyi_Karsilastirmali_Rapor.md`) stratejik mühendislik ve tıbbi önerilerin eyleme dönüştürülebilir adım adım kontrol listesidir.

## 1. Konfigürasyon ve Kaynak Tabloların Güncellenmesi (`config.py`)
- [ ] `prescriptions.csv.gz` ve `microbiologyevents.csv.gz` yollarını `config.py` içine tanımla.
- [ ] Hastanın kilosunu çekebilmek için `omr` veya `chartevents` tablosundan Ağırlık (`weight` / `BMI`) item ID'lerini ekle (Örn: 224639, 226512).
- [ ] Gerçek mekanik ventilasyon bilgilerini çekebilmek için `ventilator_mode` ve `PEEP` item ID'lerini ekle (Örn: 220339, 224688).
- [ ] Charlson Komorbidite İndeksi veya temel eşlik eden hastalıkları çekmek için `diagnoses_icd` tablosunun yolunu tanımla.

## 2. Eksik Kritik Verilerin Boru Hattına (Pipeline) Eklenmesi (`pipeline.py`)
- [ ] Sepsis başlangıç zamanını (Onset Time / $T_0$) hesaplayan fonksiyonu yaz (İlk kültür alımı + İlk antibiyotik zamanı).
- [ ] Vazopressör verilerinde `amount` yerine **`rate` (infüzyon hızı)** kullanımına geç.
- [ ] Hastanın ağırlık (kilogram) verisini kullanarak tüm vazopressör dozlarını **mcg/kg/min** formatına normalize et.
- [ ] Oksijenasyon durumunu netleştirmek için `PaO2 / FiO2` (P/F Oranı) hesaplamasını `pipeline.py` içerisine ekle.
- [ ] ICD-10 kodlarından geçmiş hastalık (Komorbidite) indeksini hesaplayıp demografik veri tablosuna (`demo_lf`) entegre et.

## 3. Yeni Özellik Mühendisliği (Feature Engineering) Adımları
- [ ] Tüm yaşamsal bulgular (HR, SBP, MAP, Temp vb.) ve kritik lablar (Laktat, Kreatinin) için **Fizyolojik İvmelenme (Delta)** kolonları oluştur (Örn: `lactate_delta_4h` veya `sofa_delta_24h`).
- [ ] Gürültülü olan FiO2-tabanlı MV bayrağı (flag) yerine, doğrudan `ventilator_mode` değerlerine bağlı sağlam bir **Mekanik Ventilasyon Flag'i (Binary)** türet.
- [ ] Sınırlı forward-fill (Bkz. Madde 4) işleminden etkilenecek kritik lablar için **"Son Ölçümden İtibaren Geçen Süre" (TSLM - Time Since Last Measurement)** özelliğini ekle.
- [ ] Verilen sıvıların tek bir `hour_bin` yerine, yoğun bakıma yattığından beri toplam yığılımını gösteren **Kümülatif Sıvı Dengesi (`cumulative_fluid_balance`)** kolonunu oluştur.
- [ ] Şok tablosunu algılamak için Modifiye Şok İndeksini (`HR / MAP`) veya Vazopressör Bağımlılık İndeksini (`total_vaso_equiv / mbp`) türet.
- [ ] Ajanın zaman algısını güçlendirmek adına her bir saat diliminin yatıştan itibaren kaçıncı saat olduğunu gösteren (`hours_since_icu_admission`) özelliğini ekle.

## 4. Pipeline Refaktoring (İstatistiksel Savunma)
- [ ] Tıbbi Outlier Kırpılması (Clipping): Çok absürt değerlerin girmesini engellemek için `group_by` adımından hemen önce `pl.col().clip(lower_bound, upper_bound)` uygulayan bir bariyer kur.
- [ ] **Güvenli (Sınırlandırılmış) Forward-Fill:** Vitals için max 4 saat, Laboratuvar için max 24 saat olacak şekilde `pl.forward_fill(limit=X)` argümanını kullan.
- [ ] Çok fazla eksik veriye (null) sahip kolonlar için (örn. Laktat, Bilirubin, Temp) forward-fill adımından hemen önce bir `missingness flag` (`is_null()`) kolonu oluştur.
- [ ] Modeli eğitmeden önceki son aşama olarak tüm sürekli değişkenlere `Min-Max` veya `Z-Score` standartlaştırması (Scaling) ekle.

## 5. DRL Ajanı İçin Aksiyon (Action) Uzayının Düzenlenmesi
- [ ] Sıvı eylemleri (Action Space) için ayrıklaştırılmış (discrete) kümeleri (bin) tanımla (Örn: Sınıf 0: 0ml, Sınıf 1: 1-50ml... Sınıf 4: >500ml).
- [ ] Vazopressör eylemleri için ayrıklaştırılmış kümeleri tanımla (Örn: Sınıf 0: 0, Sınıf 1: 0.01-0.05... Sınıf 4: >0.3 mcg/kg/min).
- [ ] RL ortamındaki (Environment) Ödül (Reward) fonksiyonuna "Ani/Tehlikeli" aksiyon değişimlerini cezalandıran kural tabanlı bir ceza (Penalty) mantığı entegre et.
