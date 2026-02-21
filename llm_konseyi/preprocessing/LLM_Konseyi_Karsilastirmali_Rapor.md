# Sepsis RL Veri Ön İşleme (Preprocessing) LLM Konseyi Çapraz Analiz Raporu

Araştırmada kullanılan çeşitli Yüksek Lisans ve Doktora seviyesindeki tıp/yapay zeka odaklı Büyük Dil Modellerinden (Claude, ChatGPT, Gemini, Kimi) gelen veri ön işleme, özellik mühendisliği (feature engineering) ve RL durum (state) mimarisine dair yapılan teknik değerlendirmelerin ortak bir vizyonda birleştirilmiş stratejik sentezidir. 

Tüm konsey üyeleri (LLM'ler) MIMIC-IV verisinin bellek-dostu (lazy evaluation + single-pass) olarak işlenmesi metodolojisini **üretim kalitesinde (production-ready)** bularak tebrik etmiş, ancak derin pekiştirmeli öğrenme ajanının (DRL Agent) başarılı olabilmesi için tıbbi gerçeklik ve istatistiksel modellerde bazı ciddi "kör noktalar" olduğunu oy birliğiyle vurgulamışlardır.

---

## 1. Kritik Eksiklikler (RL State Uzayının Zenginleştirilmesi)

Konseyin tamamı modelde "Sepsis tanısının başlangıcının (Sepsis-3)" ve "Fiziksel Kısıtların" bulunmadığına dikkat çekti.

* **Sepsis-3 Teşhisi ve Antibiyotik Başlangıcı (Onset Time):** 
  Hastanın yoğun bakıma alınması sepsis olduğu anlamına gelmez. Ajanın, tedavide hangi aşamada olduğunu (altın saatler) anlayabilmesi ve ödül sinyallerini (reward shaping) ayarlayabilmesi için `prescriptions` ve `microbiologyevents` tablolarından **ilk antibiyotik uygulanma zamanı** ile **kültür alınma zamanları** çekilerek Sepsis'in sıfır noktası ($T_0$) hesaplanmalıdır.
* **Vazopressör Doz Metriklerinin Düzeltilmesi (Ağırlık Validasyonu):** 
  Vazopressör dozu olarak `amount` (toplam miktar) değil, infüzyon hızı olan `rate` kullanılmalıdır (Örn: *Norepinefrin eşdeğeri - mcg/kg/min*). Bu, `chartevents` / `omr` tablolarından hastanın ağırlığının (`weight` / `BMI`) çekilerek dozun doğrudan vücut kitle indeksine göre normalize edilmesi gerekliliğini doğurur. 50 kg hasta ile 120 kg hasta aynı miktarı aldığında aynı fizyolojik etkiyi göstermez.
* **Münferit Mekanik Ventilasyon ve Solunum Metrikleri:** 
  FiO2'ye bağlı MV tanımı çok gürültülüdür. `ventilator_mode`, `PEEP` (Pozitif End-Ekspiratuvar Basınç) ve özellikle ARDS tablosunu belirleyen `PaO2/FiO2` (P/F Oranı) eklenerek solunum durum uzayı zenginleştirilmelidir.
* **Komorbidite ve Kırılganlık İndeksleri:**
  Geçmiş kalp ya da böbrek rahatsızlıklarını belirtmek amacıyla Charlson Komorbidite İndeksi veya ICD-10 tablosundan temel eşlik eden hastalıklar çekilmelidir.

## 2. Üretilmesi Gereken Yeni Özellikler (Feature Engineering Hileleri)

Algoritmanın yalnızca anlık ölçümleri değil, patolojik "gidişatı (trajectory ve momentum)" kavrayabilmesini sağlayacak özelliklerin eksik olduğu vurgulanmıştır.

* **Fizyolojik İvmelenme (Delta Değişim Hızları):** 
  En değerli ekleme önerisi budur. `lactate_delta_4h`, `sofa_delta_24h` ve yaşamsal bulgular (MAP, HR) için saatlik veya çok saatlik penceredeki ivmelenmeler/farklar. Örneğin, laktat değeri RL ödül mekanizması için de bir klirens (temizlenme) proxy'si görevi görecektir.
* **Son Ölçümden İtibaren Geçen Süre (Time Since Last Measurement - TSLM):** 
  Markov Karar Sürecinin ihlallerini (Partial Observability) aşmak adına; modele laboratuvar sonucunun "ne kadar taze (eski)" olduğu da kodlanmalıdır.
* **Sıvı Trendi ve Kümülatif Veriler:** 
  4 saatlik pencereden ziyade yatıştan itibaren verilen toplam kristaloid/vazopressör yükü (`cumulative_fluid_balance`), hastanın hipervolemik olup olmadığını RL ajanına öğretecektir.
* **Vazopressör Bağımlılık ve Modifiye Şok İndeksleri:**
  `total_vaso_equiv / mbp` veya Modifiye Şok İndeksi (HR/MAP) gibi kompozit özellikler, şok gelişimini erkenden algılamak için saf Shock Index'ten daha stabildir.
* **Yoğun Bakıma Yatıştan İtibaren Geçen Süre (Stay Length):**
  Zaman bağlamını modele eklemek için her `hour_bin`'in yatış başlangıcına olan mesafesi tutulmalıdır.

## 3. Mühendislik ve Veri Ön İşleme (Preprocessing) Refaktoringleri

En kritik uyarılardan biri EDA dosyalarındaki olağandışı dağılımların temizlenmediği ve Forward-Fill yönteminin tıbbi olarak problemli bir şekilde (sonsuza kadar) uygulandığı yönündedir.

* **Güvenli (Sınırlandırılmış) İleri Doldurma (Capped Forward-Fill):**
  `temp_c` (%83 null) veya `lactate` gibi verilerde sınırsız `forward_fill` uygulamak felakettir. 24 saat önceki laktat değeri o saatin fizyolojisini öngöremez.
  * **Çözüm:** Yaşamsal belirtiler (`vitals`) için tolerans limiti 4 saat, sık tekrarlanan laboratuvar testleri için 24 saat ile sınırlı `pl.forward_fill(limit=X)` çalıştırılıp, sonrasında "missingness flag" (eksik veri maskesi `is_null`) eklenmesi önerilmiştir.
* **Tıbbi Outlier Kırpılması (Clipping):**
  Notebook'ta yer alan HR = 10.000.000 veya Negatif İdrar Çıkışı gibi verilerin boru hattına girmesini önlemek için `group_by` adımından hemen önce `pl.col().clip(lower_bound, upper_bound)` uygulanmalı veya %1 - %99 persentiller kullanılarak Winsorization (Gürültü törpülenmesi) yapılmalıdır.
* **Sürekli Vazopressörler için Rate Doğrulaması:**
  Polars içindeki `pl.when(statusdescription == 'FinishedRunning').then(rate)` benzeri bir mantık kurgulanıp, çok saatlik infüzyon işlemlerinde ilacın sadece verilen saatte kalıp total `amount` olarak alınması hatası engellenmelidir. Sürekli verilen ilaçların "rate" değerlerine ulaşılmalıdır.
* **Ölçeklendirme (Min-Max/Z-Score) Çerçevesi:**
  Sıvı miktarının binleri bulmasıyla `GCS` veya `lactate`'in küçük ondalık düzeyleri arasındaki farkların DRL ajanının Q-değerlerinde (gradient norm) yıkım yaratmaması adına son aşamada Z-Score/Min-Max standartlaştırması uygulanmalıdır.
* **Aksiyon (Action) Uzayının Güvenli Ayrıklaştırılması (Safe RL Constraints):**
  Dozların tamamen "sürekli (continuous)" bir uzayda tahmin edilmesinin keşif/sömürü krizleri yaratabileceği için ilaç ve sıvı yönetiminin (Örn: Vazopressör için 5-bin, Sıvı için 5-bin => 25 Action Space) gibi kısıtlı bir kümeleme sistemine geçirilmesi ve modelin ardışık iki adımda ekstrem doz değişimleri yapmasına kural-tabanlı `-100 Penalty (Ceza)` uygulanması savunulmuştur.

## 📝 Sonraki Adımlar İçin Genel Roadmap

1. **Config.py Değişiklikleri:** OMR (weight) ve Prescription (antibiyotik) item numaralarını ekle. Gerçek solunum verisi (`ventilator_mode`) ekle.
2. **Pipeline Tweakleri:** Kırpma (clip) fonksiyonlarını sisteme dahil et. Forward-fill `limit` atamalarını yap. `Time_since_icu_admission` ve delta değişken operasyonlarını oluştur.
3. **RL Modeli Hazırlığı:** Seçilen eylemler arasındaki ani dalgalanmalara karşı Reward Shaping ve güvenli politika kurallarını geliştir.
