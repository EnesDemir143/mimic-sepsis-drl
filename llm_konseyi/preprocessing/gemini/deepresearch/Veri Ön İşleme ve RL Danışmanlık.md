# RL Sepsis Ajanı: MIMIC-IV Üzerinde DRL Tabanlı Veri Ön İşleme, Durum Uzayı ve Özellik Mühendisliği Stratejik Konsey Raporu

Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL) algoritmalarının klinik karar destek sistemlerine, özellikle yoğun bakım ünitelerindeki (YBÜ) sepsis yönetimine entegrasyonu, modern yapay zeka ve tıp biliminin en umut verici kesişim noktalarından birini temsil etmektedir. Sepsis, enfeksiyona karşı konağın verdiği düzensiz yanıtın neden olduğu, hızla ilerleyen ve yaşamı tehdit eden bir organ yetmezliği tablosudur. Bu tablonun heterojen doğası, sıvı resüsitasyonu ve vazopressör titrasyonu gibi kritik tedavilerin hastaya ve zamana özel olarak kişiselleştirilmesini zorunlu kılmaktadır. Geleneksel klinik kılavuzlar statik kurallar sunarken, DRL ajanları devasa elektronik sağlık kayıtları (EHR) üzerinden hastaların fizyolojik gidişatlarını (trajectories) analiz ederek, mortaliteyi minimize eden dinamik ve optimum tedavi politikaları (optimal policies) öğrenebilmektedir.

Bu rapor, Multi-parameter Intelligent Monitoring in Intensive Care (MIMIC-IV) veri setini kullanarak bir DRL ajanı eğitmek amacıyla kurgulanan veri ön işleme boru hattının (pipeline), özellik mühendisliği (feature engineering) stratejilerinin ve durum/aksiyon (state/action) uzaylarının kapsamlı bir analizi ve konsey değerlendirmesidir. Sağlanan proje özeti, keşifsel veri analizi (EDA) çıktıları, yapılandırma dosyaları ve Polars tabanlı ön işleme kodları derinlemesine incelenmiştir. Modelin klinik geçerliliğini, veri bütünlüğünü ve hesaplamalı öğrenme teorisi standartlarını en üst düzeye çıkarmak amacıyla eksik verilerin tespiti, yeni değişkenlerin üretilmesi ve istatistiksel tamamlama (imputation) metodolojilerinin yeniden yapılandırılmasına dair stratejik mühendislik ve tıp tavsiyeleri detaylandırılmıştır.

## 1\. Mevcut Veri Ön İşleme Mimarisi ve Bellek Optimizasyonunun Analizi

MIMIC-IV gibi yüksek çözünürlüklü ve devasa boyutlu sağlık veritabanları üzerinde çalışırken karşılaşılan en büyük engel, milyonlarca satırlık zaman serisi verilerinin birleştirilmesi (join) sırasında yaşanan bellek (RAM) taşmalarıdır (Out-of-Memory - OOM). İletilen `pipeline.py` ve `config.py` dosyaları incelendiğinde, bu problemi çözmek için Polars kütüphanesinin tembel değerlendirme (lazy evaluation) ve veri akışı (streaming) özelliklerinin ustalıkla kullanıldığı görülmektedir. Geleneksel Pandas veri işleme süreçlerinde, her bir laboratuvar veya yaşamsal bulgu için ayrı veri çerçeveleri oluşturulup ardışık "Full Outer Join" işlemleri yapılması, kartezyen çarpımlar nedeniyle bellekte 60 GB'ı aşan ani ve yıkıcı artışlara yol açmaktadır.  

Bunun yerine kurgulanan "Pivot-Free" (Pivotsuz) strateji, mühendislik açısından son derece başarılıdır. Ham verilerin önce süzülmesi, ardından zamanın `dt.truncate("1h")` ile saatlik dilimlere (hourly binning) yuvarlanması ve tek bir `group_by` operasyonu üzerinden çoklu koşullu kümeleme ifadelerinin (conditional aggregation expressions) çalıştırılması, bellek maliyetini asimptotik düzeyde düşürmektedir. Özellikle `pl.when().then().otherwise().mean().alias()` mantığıyla oluşturulan ifade listelerinin tek geçişte (single-pass) hesaplanması, veritabanı sorgu optimizasyonlarında kullanılan izdüşüm (projection) ve yüklem itme (predicate pushdown) tekniklerinin Python ortamında kusursuz bir tezahürüdür. Nihai tablonun belleğe yüklenmek yerine `sink_parquet(streaming=True)` komutuyla doğrudan yığınlar (chunks) halinde diske yazılması, uçtan uca veri bütünlüğünü sağlayan üretim kalitesinde (production-ready) bir yaklaşımdır.  

Ancak, bir derin pekiştirmeli öğrenme ajanının klinik başarısı yalnızca veriyi ne kadar hızlı ve verimli işlediğine değil, işlenen verinin hastanın biyolojik gerçekliğini ve klinik karar alma süreçlerindeki belirsizlikleri ne kadar eksiksiz yansıttığına bağlıdır. Kod mimarisinin kusursuzluğu, durum uzayının (state space) klinik olarak eksik kurgulandığı gerçeğini telafi edemez. Ajanın karar alırken dayandığı temellerin güçlendirilmesi, MIMIC-IV veritabanının daha derin tablolarından kritik bilgilerin çıkarılması ile mümkündür.

## 2\. DRL Durum (State) Uzayındaki Kritik Eksiklikler ve Sepsis-3 Entegrasyonu

Bir DRL modeli, ortamı (environment) bir Markov Karar Süreci (MDP) çerçevesinde algılar. Ajanın optimum sıvı ve vazopressör aksiyonlarını seçebilmesi için, ortamın durumunu tanımlayan özellik (feature) vektörünün hastanın anlık şok tablosunu, enfeksiyon şiddetini ve kronik dayanıklılığını doğru ifade etmesi şarttır. İletilen `01_data_explore.md` ve `config.py` dosyalarındaki 43 değişkenlik kurgu yaşamsal bulgular ve temel laboratuvar verilerini kapsasa da, literatürde geçerliliği kanıtlanmış referans AI Clinician modellerine ve tıp dünyasının güncel Sepsis-3 standartlarına kıyasla önemli yapısal eksiklikler barındırmaktadır.  

### 2.1. Sepsis Başlangıç Zamanı (Onset Time) ve Sepsis-3 Kriterleri

Mevcut veri boru hattı, hastaların yalnızca yoğun bakım yatış zamanı (`intime`) ile başlayan verilerini birleştirmektedir. Ancak klinik gerçeklikte, bir hastanın yoğun bakıma alınması onun sepsis olduğu anlamına gelmez. Sepsis-3 konsensüsüne göre, sepsisin tanımı "enfeksiyona karşı düzensiz konak yanıtı nedeniyle gelişen ve yaşamı tehdit eden organ yetmezliği" olarak revize edilmiştir. Bu tanımın algoritmik karşılığı, şüpheli enfeksiyon (suspected infection) durumu ile birlikte ardışık SOFA (Sequential Organ Failure Assessment) skorunda temel değere göre en az 2 puanlık akut bir artışın eşzamanlı olarak tespit edilmesidir.  

Ajanın başarılı bir ödül fonksiyonuna (reward shaping) sahip olabilmesi için, tedavinin tam olarak hangi anında (altın saatlerde) bulunduğunu bilmesi gerekir. Şüpheli enfeksiyon zamanı, MIMIC-IV veri setindeki `microbiologyevents` (kültür alımı) ve `prescriptions` (antibiyotik uygulanması) tablolarının sentezlenmesiyle hesaplanır. Antibiyotiğin başlanmasından önceki 72 saat içinde kültür alınmış olması veya kültür alınmasından sonraki 24 saat içinde antibiyotik verilmesi klinik kuralı işletilerek , sepsisin sıfır noktası (T0​) tayin edilmelidir. Ajanın durum uzayına, "sepsis teşhisinden bu yana geçen süre" gibi bir değişkenin eklenmesi, modelin hastalığın erken aşamalarındaki agresif sıvı yüklemesi ile geç aşamalarındaki vazopressör bağımlılığını ayırt etmesini sağlayacaktır.  

### 2.2. Vücut Ağırlığı, Boy ve Farmakokinetik Standardizasyon

Sepsis hastalarında uygulanan vazopressörler (Noradrenalin, Adrenalin, Vazopressin vb.) ve intravenöz (IV) kristaloid sıvılar, klinik ortamda hastanın vücut ağırlığına orantılı olarak rezerve edilir ve infüze edilir. Vazopressörlerin infüzyon hızı farmakolojik olarak `mcg/kg/min` (mikrogram/kilogram/dakika) standart biriminde ayarlanırken , sıvı resüsitasyonu kılavuzları "ilk üç saat içinde 30 ml/kg IV kristaloid" uygulanmasını emreder.  

İletilen `pipeline.py` dosyasındaki `build_inputs_hourly` fonksiyonu, ilaçların ve sıvıların ham `amount` (miktar) değerlerini doğrudan saatlik olarak toplamaktadır. Demografik özellikler arasında ise yalnızca cinsiyet (`gender`), yaş (`age`) ve yatış tipi (`admission_type`) yer almaktadır. Vücut ağırlığının modele dahil edilmemesi, DRL ajanı için ölümcül bir kör noktadır. Zira ajanın, saatte 50 mg noradrenalin alan 45 kilogramlık zayıf bir hasta ile aynı miktarı alan 130 kilogramlık obez bir hasta arasındaki derin fizyolojik farkı algılaması olanaksız hale gelir. Bu durum, modelin Markov Karar Süreci (MDP) içindeki geçiş olasılıklarını (transition probabilities) çarpıtır ve öğrenme sürecini dejenere eder. Hastaların başvuru anındaki kiloları ve boyları, MIMIC-IV `omr` (Outpatient Medical Record) veya `chartevents` tablolarından çıkarılmalı, vücut kitle indeksi (VKİ) hesaplanmalı ve ilaç/sıvı dozları `mcg/kg/min` ile `ml/kg` cinsine dönüştürülerek ajana sunulmalıdır.  

### 2.3. Komorbidite Yükü (Eşlik Eden Hastalıklar) ve ICD-10 Kodları

Yapay zeka klinisyeni, hasta gruplarının kronik hastalık geçmişlerine göre farklı tolerans eşiklerine sahip olduğunu öğrenmek zorundadır. Örneğin, kronik böbrek yetmezliği veya ileri derece konjestif kalp yetmezliği olan bir sepsis hastasına agresif sıvı yüklemesi yapmak, hayat kurtaran bir resüsitasyon hamlesi değil, hastayı ölümcül bir pulmoner ödeme (akciğerde sıvı toplanması) sokacak tıbbi bir hatadır.  

Geliştirilen boru hattında, hastaların yatış öncesi sağlık durumlarını ifade eden statik bir komorbidite skoru bulunmamaktadır. Literatürde, mortalite tahmini ve sepsis pekiştirmeli öğrenme kurgularında Charlson Komorbidite İndeksi veya Elixhauser Komorbidite Skoru mutlaka durum uzayına dahil edilir. Bu skorlar, hastanın tıbbi kırılganlığını sayısal bir değere dönüştürerek ajanın durumu yorumlamasında kritik bir bağlam (context) sağlar. MIMIC-IV `diagnoses_icd` tablosundaki ICD-10 kodları taranarak bu indeksler hesaplanmalı ve statik bir özellik olarak modelin girdi matrisine eklenmelidir.  

### 2.4. Mekanik Ventilasyon ve Kapsamlı Solunum Dinamikleri

Özellik mühendisliği notlarında solunum sayısının (`resp_rate`), oksijen satürasyonunun (`spo2`) ve `pao2`, `paco2` gibi kan gazı parametrelerinin bulunduğu belirtilse de , hastanın mekanik ventilatör (solunum cihazı) desteğinde olup olmadığını gösteren bir ikili (binary) değişken veya solunum makinesinin uyguladığı pozitif basıncı gösteren PEEP (Pozitif End-Ekspiratuar Basınç) değeri eksiktir.  

Mekanik ventilasyon durumu, kardiyovasküler fizyoloji ve sıvı yönetimi ile doğrudan ve derin bir etkileşim içindedir. Yüksek PEEP değerleri, intratorasik (göğüs içi) basıncı artırarak kalbe olan venöz dönüşü azaltır ve hastanın sıvı açığı varmış gibi yanıltıcı bir kan basıncı düşüşüne neden olabilir. Bu fizyolojik etkileşim zincirinin RL ajanı tarafından öğrenilebilmesi ve ajanın gereksiz sıvı yüklemelerinden kaçınabilmesi için `mechanical_ventilation` durumu ve `PEEP` değerleri, `chartevents` veya `procedureevents` tablolarından çekilerek saatlik bazda duruma entegre edilmelidir.  

| Özellik (Feature) Kategorisi | MIMIC-IV Kaynak Tabloları | DRL Ajanı ve Sepsis Yönetimi Açısından Stratejik Önemi |
| --- | --- | --- |
| **Enfeksiyon Başlangıç Zamanı** | `microbiologyevents`, `prescriptions` | Sepsis-3 tanımının karşılanması, erken/geç evre müdahalelerin ayrıştırılması ve altın saatler penceresinin model tarafından algılanması. |
| **Vücut Ağırlığı, Boy ve VKİ** | `chartevents`, `omr` | Farmakokinetik standardizasyon; vazopressör (`mcg/kg/min`) ve sıvı (`ml/kg`) aksiyonlarının hasta kütlesine orantılı normalize edilmesi. |
| **Charlson/Elixhauser Skorları** | `diagnoses_icd`, `admissions` | Kalp veya böbrek yetmezliği geçmişi olan kırılgan hastalarda sıvı aşırı yüklemesinin engellenerek kişiselleştirilmiş politika üretimi. |
| **Mekanik Ventilasyon & PEEP** | `chartevents`, `procedureevents` | İntratorasik basınç artışının venöz dönüşe etkisinin kavranması, ARDS tablosunda doğru sıvı-vazopressör dengesinin kurulması. |

E-Tablolar'a aktar

## 3\. Gelişmiş Özellik Mühendisliği (Feature Engineering) ve Zamansal Dinamiklerin Modellenmesi

Derin nöral ağlar, onlara verilen ham verilerdeki (raw data) ince ve karmaşık örüntüleri (patterns) kendi başlarına çıkarabilme yeteneğine sahip olsalar da, verinin matematiksel veya klinik olarak anlamlı türevsel formlara dönüştürülmesi ağın öğrenme sürecini muazzam ölçüde hızlandırır ve genel geçerliliğini (generalization) artırır. Geliştirilen projede uygulanan vazopressör eşdeğeri standardizasyonu ve SOFA skoru hesaplamaları , doğru atılmış adımlar olmakla birlikte, modelin zaman içindeki değişimi ve belirsizliği anlayabilmesi için aşağıdaki özellik mühendisliği stratejilerinin acilen kod tabanına entegre edilmesi gerekmektedir.  

### 3.1. Kısmi Gözlemlenebilirlik ve Son Ölçümden İtibaren Geçen Süre (Time Since Last Measurement - TSLM)

Sepsis veri setlerindeki en büyük zorluk, verilerin düzenli aralıklarla (regularly sampled) gelmemesidir. Kalp atış hızı cihazlar aracılığıyla her dakika kaydedilirken, laktat, bilirubin veya trombosit sayımı gibi invaziv laboratuvar testleri günde sadece bir veya iki kez yapılır. `01_data_explore.md` dosyasında da laktatın %28, vücut sıcaklığının ise %83.4 gibi muazzam oranlarda eksik (null) olduğu tespit edilmiştir.  

Klinik karar verme sürecinde, bir doktor için sadece ölçülen değer değil, ölçümün "ne zaman" yapıldığı da eşit derecede hayati bir bilgidir. Dört saat önce ölçülen bir yüksek laktat değeri ile on dakika önce ölçülen aynı yüksek laktat değeri, hastanın şu anki fizyolojik durumuna dair farklı düzeylerde kesinlik ve aciliyet ifade eder. Derin pekiştirmeli öğrenmede kullanılan MDP mimarisi, Markov özelliğine, yani "gelecekteki durumun geçmişe değil, yalnızca anlık duruma bağlı olduğu" varsayımına dayanır. Eğer eski bir laboratuvar ölçümü, yeni bir ölçüm yapılana kadar `forward_fill` yöntemiyle saatlerce kopyalanırsa ve modele bu bilginin eski olduğu söylenmezse, ajan hastanın o anki değerinin o olduğunu sanarak Markov varsayımını ihlal eder ve halüsinatif tedaviler uygular.  

Bu körlüğü gidermek için, eksiklik oranı yüksek olan tüm temel değişkenler için `time_since_last_measurement` (Son Ölçümden İtibaren Geçen Süre) değişkeni oluşturulmalıdır. Polars kütüphanesinde bu işlem, ölçümün yapıldığı saatlere ilişkin boolean maskeler oluşturulup, `.dt.epoch()` veya kümülatif toplamlar yardımıyla saatlerin sayılması ve duruma yeni bir boyut olarak eklenmesi yoluyla yüksek performansla gerçekleştirilebilir. Bu ekleme, modeli Kısmi Gözlemlenebilir Markov Karar Sürecine (POMDP) hazırlar ve ajana verinin güvenilirliğine dair zımni bir metrik sunar.  

### 3.2. Fizyolojik İvme: Birinci ve İkinci Türev (Delta ve Delta-Delta) Özellikleri

Sepsis, doğası gereği durağan olmayan (non-stationary) ve hızla bozulan bir süreçtir. Kan basıncının düşük olması şüphesiz risklidir, ancak kan basıncının son iki saat içinde ani bir düşüş trendinde (ivmelenme) olması, hastanın hızla geri dönülmez bir şok tablosuna girdiğinin habercisidir. Ajanın anlık değerler kadar, hastanın gidişatının (trajectory) yönünü ve hızını da öğrenebilmesi için, özelliklerin zaman içindeki türevleri (değişim hızları) hesaplanarak modele dahil edilmelidir.  

-   **Delta (Δ) Özellikleri:** Her saat dilimi için, o saatin değeri ile bir önceki saatin değeri arasındaki fark (Xt​−Xt−1​) hesaplanmalıdır. Özellikle kalp hızı (ΔHR), solunum hızı (ΔRR), ortalama arter basıncı (ΔMAP) ve laktat seviyesindeki saatlik değişimler, mortaliteyi tahminlemede ve DRL ödül mekanizmasını yönlendirmede son derece değerlidir.  
    
-   **Bireyselleştirilmiş Z-Skoru Sapmaları:** Anlık değişimin ötesinde, hastanın mevcut değerinin, kendi son 24 saatlik yatış ortalamasından ne kadar saptığı hesaplanmalıdır. Kimi hastalar için 90 mmHg sistolik kan basıncı normal bir bazal durumken, normotansif bir hasta için bu ciddi bir hipotansiyondur. Polars'ın kayan pencere (sliding window) fonksiyonları olan `.rolling_mean()` ve `.rolling_std()` kullanılarak her saat için hastanın bireysel normallerinden ne kadar uzaklaştığı sayısal olarak ajana sunulabilir.  
    
-   **Delta-Delta Özellikleri:** Hastalığın ivmelenmesini (değişim hızının değişimini) kavramak için ikinci türev değerleri belirli kritik yaşamsal bulgular (örn. kalp atış hızı) için hesaplanabilir.  
    

### 3.3. Kritik Fizyolojik İndeksler ve Klinik Oranlar

Ham laboratuvar ve vital bulguları bir araya getirerek, insan klinisyenlerin de sıkça kullandığı birleşik risk endekslerini modele dahil etmek, boyutluluk lanetini (curse of dimensionality) azaltır ve bilgi yoğunluğunu artırır.  

-   **Şok İndeksi (Shock Index) ve Modifiye Şok İndeksi:** Sadece `shock_index` (Kalp Hızı / Sistolik Kan Basıncı) hesaplanması doğru bir adımdır , ancak literatürde Modifiye Şok İndeksinin (Kalp Hızı / Ortalama Arter Basıncı) mikrovasküler doku perfüzyonunu ve gizli şoku daha erken tespit ettiği kanıtlanmıştır. Sistolik kan basıncı çeşitli fizyolojik kompansasyon mekanizmaları ile son ana kadar normal görünebilirken, Ortalama Arter Basıncı (MAP) perfüzyonun en somut ve stabil göstergesidir.  
    
-   **PaO2 / FiO2 Oranı (P/F Oranı):** Kanda çözünmüş oksijen basıncının, hastaya verilen oksijen yüzdesine oranıdır. Akut Solunum Sıkıntısı Sendromunun (ARDS) tanısında ve Sepsis solunum SOFA skorunda temel alınan bu oran, akciğerin oksijenasyon yeteneğini tek bir sayıda özetleyerek ajana mükemmel bir içgörü sunar.  
    
-   **BUN / Kreatinin Oranı:** Kan Üre Azotu (BUN) ve Kreatinin değerleri böbrek fonksiyonunu gösterir. Ancak bu ikisinin oranı, gelişen akut böbrek yetmezliğinin (AKI) "prerenal" yani sıvı eksikliğinden mi, yoksa doğrudan böbrek dokusunun hasarından mı kaynaklandığını gösterir. Yüksek bir BUN/Kreatinin oranı sıvı açığına işaret eder ve ajanın bu oranı görerek sıvı resüsitasyonu kararı alması, modelin klinik olarak rasyonelleşmesini sağlar.  
    

| Eklenmesi Gereken Gelişmiş Özellik | Hesaplama / Mantık Formülü | RL Politikası Üzerindeki Etkisi |
| --- | --- | --- |
| **TSLM (Son Ölçüm Süresi)** | `t_current - t_last_measurement` | Eksik verinin yaşını göstererek Markov varsayım ihlalini engeller, belirsizliği modele iletir. |
| **Delta Metrikleri (ΔX)** | `X(t) - X(t-1)` ve `X(t) - mean_24h(X)` | Hastalığın o anki yönünü ve kötüleşme hızını belirterek ajanın proaktif davranmasını sağlar. |
| **Modifiye Şok İndeksi (MSI)** | `Heart_Rate / MAP` | Gizli doku hipoperfüzyonunu ve şokun erken evrelerini tespit eder. |
| **P/F Oranı** | `PaO2 / FiO2` | Akciğerlerin oksijenasyon kapasitesini ve ARDS şiddetini anlık olarak yansıtır. |
| **BUN / Kreatinin Oranı** | `BUN / Creatinine` | Dehidratasyon seviyesini saptayarak ajanın vazopressör yerine sıvıyı seçmesini tetikler. |

E-Tablolar'a aktar

## 4\. İstatistiksel Metodoloji, Eksik Veri Tamamlama ve Uç Değer Yönetimi

Araştırma bulgularında paylaşılan Python boru hattı kodlarında `pl.col(col).forward_fill().over("stay_id")` kurgusu ile basit ve doğrusal bir eksik veri doldurma (imputation) yöntemi uygulanmıştır. Hesaplamalı verimlilik ve kod sadeliği açısından bu yöntem çekici görünse de, tıp istatistiği, insan fizyolojisi ve makine öğrenimi doğruluk sınırları bağlamında bu yöntem derin yapısal kusurlar içermektedir.  

### 4.1. Forward-Fill (İleri Doldurma) Yönteminin Kritik Zafiyetleri

Yaşamsal bulgular (Vital Signs) gibi sürekli sensörlerle takip edilen ve eksiklik oranı %1-5 arasında olan değişkenlerde (örneğin Kalp Hızı, Kan Basıncı, SpO2) forward-fill uygulanması son derece mantıklı ve klinik olarak kabul edilebilirdir. Zira bir hastanın kalp atış hızı bir önceki saatte 90 ise, yeni ölçüm gelene kadar geçen kısa sürede 90 civarında kaldığını varsaymak sapmayı minimize eder.  

Ancak, keşifsel analiz sonuçlarında belirtilen laktat (%28 null), total bilirubin (%37.3 null), pH (%24.6 null) veya vücut sıcaklığı (%83.4 null) gibi değerlerde düz mantıkla ileri doldurma uygulamak fizyolojik bir gerçekliğe dayanmaz. İnsan fizyolojisi sürekli bir homeostaz çabası içindedir; hiçbir biyokimyasal bozulma sabit kalmaz, ya tıbbi müdahale ile iyileşir (ortalamaya döner) ya da daha da kötüleşir. Örneğin, laktat değeri 5.0 mmol/L olarak ölçülmüş bir hastaya agresif sıvı tedavisi yapıldığında, laktat değerinin saatler içinde düşmesi, vücudun laktatı temizlemesi (clearance) beklenir. Eğer 16 saat boyunca yeni ölçüm yapılmadıysa ve sistem ilk ölçümü 16 saat boyunca kopyalayarak ajana "laktat hala 5.0" diyorsa, ajan yanılsama içinde kalacak, aslında iyileşmiş hastaya zehirli düzeyde gereksiz müdahalelerde (vazopressör) bulunacaktır.  

### 4.2. Sönümleme (Decay) Fonksiyonları ve Klinik Geçerlilik Pencereleri

Forward-fill işleminin tahribatını önlemek adına, veri boru hattında laboratuvar testleri için "Geçerlilik Pencereleri" (Validity Windows) kurgulanmalıdır. Her bir ölçüm, sadece tıbbi olarak mantıklı olan bir süre boyunca (örneğin laktat için maksimum 6 saat, kreatinin için maksimum 24 saat) ileriye taşınmalıdır. Süre aşıldığında veri tekrar `null` (eksik) statüsüne çekilmeli veya yavaşça popülasyon ortalamasına doğru sönümlenmelidir (Mean Reversion / Decay Function). Polars üzerinde bu işlem, daha önce önerilen TSLM (Son Ölçüm Süresi) sütununa bağlı mantıksal ifadeler kullanılarak kolayca kodlanabilir: `pl.when(tslm < 6).then(forward_fill).otherwise(null)`.  

### 4.3. İleri Düzey İstatistiksel Tamamlama (Advanced Imputation) Stratejileri

Sepsis tahmin ve pekiştirmeli öğrenme modellerinde en yüksek doğruluğa (AUC ve F1 skorları) ve RL konverjansına ulaşan güncel akademik çalışmalar, eksik laboratuvar verilerini gelişmiş makine öğrenmesi algoritmalarıyla tamamlamaktadır.  

-   **K-En Yakın Komşu (KNN) ve Çoklu İstatistiksel İmputasyon (MICE):** Sepsis veri setlerinde rastgele eksik veriler (Missing Completely at Random - MCAR) nadir görülür. Eksikliklerin hastanın diğer klinik göstergeleri ile bir ilişkisi vardır. KNN algoritması veya Rastgele Orman tabanlı MissForest gibi algoritmalar, benzer fizyolojik tablodaki diğer hastaların değerlerine bakarak sentetik ama tutarlı değerler üretir.  
    
-   **Rasgele Örneklem İmputasyonu (Random Sampling Imputation):** İlgili özelliğin eksik olmayan (non-missing) popülasyon dağılımından rastgele değer çekilmesi, veri setindeki genel varyansı koruyarak modellerin ezberleme (overfitting) ihtimalini azaltır.  
    
-   Polars kütüphanesinin akışkan (streaming) doğası gereği, bu karmaşık imputasyon algoritmalarını veri işleme anında çalıştırmak zordur. Stratejik yaklaşım; Polars boru hattının yalnızca TSLM ve decay ile sınırlandırılmış güvenli forward-fill yaparak Parquet dosyasını üretmesi, KNN veya MICE işlemlerinin ise ajanı eğitecek olan PyTorch/TensorFlow veri yükleyici (dataloader) mekanizmasının içine bir ön işleme katmanı olarak eklenmesidir.

### 4.4. Aykırı Değer (Outlier) Yönetimi ve Fizyolojik Sınır Filtrelemesi

MIMIC-IV gibi devasa ölçekli gerçek dünya EHR veritabanları kaçınılmaz olarak insan kaynaklı veri giriş hataları ve cihaz kalibrasyon sorunları (artefaktlar) barındırır. İletilen kurguda saatlik gruplama ve ortalama alma (`mean`) fonksiyonları kullanıldığı için , saatlik dilim içindeki tek bir hatalı ölçüm, tüm saatin karakteristiğini mahveder. Örneğin vücut sıcaklığının 37.0 yerine virgül unutularak 370.0 olarak sisteme kaydedilmesi, saatin ortalamasını absürt seviyelere çıkarır.  

Bunu engellemek için, `group_by` işleminden hemen önce verilerin biyolojik gerçeklik sınırlarına kırpılması (clipping) veya filtrelenmesi hayati önem taşır. Polars expressions yapısına `pl.col("valuenum").clip(lower_bound, upper_bound)` fonksiyonları entegre edilmelidir.

-   Örnek sınırlar: Kalp Hızı \[0 - 300 bpm\], Sistolik Kan Basıncı \[0 - 350 mmHg\], Vücut Sıcaklığı \[25.0 - 45.0 °C\], pH \[6.5 - 8.0\]. Ayrıca, genel özellik matrisi oluşturulduktan sonra, klinik sınırlara uymayan ancak istatistiksel olarak uzun kuyruklu dağılıma sahip uç değerler için, %1 ve %99 persentillerinden Winsorizasyon (Winsorization) uygulanarak değerlerin kırpılması, yapay sinir ağının gradyan patlaması (gradient explosion) yaşamasını engelleyecektir.  
    

### 4.5. Özellik Ölçeklendirme (Feature Scaling) ve Nöral Ağ Stabilitesi

Durum uzayındaki değişkenlerin farklı ölçeklere sahip olması, örneğin trombosit (`platelet`) sayısının 250,000 gibi büyük rakamlar, laktat seviyesinin 2.5 gibi küçük rakamlar, yaşın ise 65 gibi orta ölçekli rakamlar içermesi , DQN veya PPO tabanlı ajanların öğrenme katsayılarının (learning rates) dengesizleşmesine ve optimal değerlere yakınsayamamasına neden olur. Standartlaştırma (Z-score standardization, mean=0, std=1) veya Min-Max ölçeklendirmesi mutlak suretle tüm sürekli değişkenlere uygulanmalıdır. Polars üzerinde tüm eğitim seti üzerinden hesaplanan ortalama ve standart sapma değerleri sabit sözlüklerde saklanarak (data leakage önlenerek) veriler işleme anında standardize edilmelidir.  

## 5\. Aksiyon (Action) Uzayının Tasarımı ve Güvenli Pekiştirmeli Öğrenme (Safe RL) Kısıtlamaları

Sepsis tedavisindeki sıvı ve ilaç uygulamaları sürekli (continuous) değişkenler olmasına rağmen, bu kararların sonsuz ihtimalli uzayda hesaplanması derin pekiştirmeli öğrenme modellerinde keşif (exploration) aşamasını içinden çıkılamaz bir kaosa sürükler. DRL ajanlarının karar mekanizmasını hızlandırmak ve optimal politikayı bulmasını kolaylaştırmak için aksiyon uzayının klinik olarak anlamlı ve ayrık (discrete) parçalara bölünmesi sektör standardıdır.  

### 5.1. Dozların Ayrıklaştırılması ve 25-Bin Aksiyon Matrisi

Referans alınan AI Clinician metodolojileri ve klinik tedavi kılavuzlarına paralel olarak, ajanın seçebileceği IV sıvı resüsitasyonu ve vazopressör uygulama miktarları 5 farklı seviyeye (bin) ayrılmalıdır.  

-   **Sıvı Dozajı Sınıfları (ml/4h veya ml/h):** Örn; Sınıf 0 (0 ml - Yok), Sınıf 1 (1-50 ml), Sınıf 2 (50-250 ml), Sınıf 3 (250-500 ml), Sınıf 4 (>500 ml).
-   **Vazopressör Dozajı Sınıfları (mcg/kg/min):** Örn; Sınıf 0 (0 - Yok), Sınıf 1 (0.001 - 0.05), Sınıf 2 (0.05 - 0.1), Sınıf 3 (0.1 - 0.3), Sınıf 4 (>0.3). Bu iki boyuttaki kararların kesişimi, ajanın seçebileceği toplam 25 ayrık aksiyondan (5x5 = 25) oluşan net ve sınırları belirlenmiş bir eylem matrisi oluşturur.  
    

Ayrıca, özellik mühendisliği adımlarında `total_vaso_equiv` adıyla oluşturulan vazopressör eşdeğeri , doğru bir standardizasyondur. Norepinefrin, Epinefrin, Fenilefrin ve Vazopressin'in klinik güç oranlarına göre çarpılarak tek bir değere indirilmesi, dozlama karmaşasını sadeleştirir. Dobutamin'in (kalp kasılma gücünü artıran inotropik ajan) ve Dopamin'in bu eşitliğin dışında bırakılması veya farklı katsayılarla değerlendirilmesi kararı, kardiyovasküler dinamiklerin farklılığı (kalp hızını ve atım hacmini artırmaları) nedeniyle fevkalade isabetli bir farmakolojik yaklaşımdır.  

### 5.2. Aksiyon Dalgalanmaları ve Kural Tabanlı Ceza (Penalty) Mekanizmaları ile Safe-RL

DRL ajanları, yalnızca kurgulanan ödül (reward) fonksiyonunu en üst düzeye çıkarmak amacıyla çalışırlar. Sepsis modellerinde sıklıkla sadece "hastanın 90 günlük sağ kalımı (+100)" ve "hastanın ölümü (-100)" gibi seyrek ödül sinyalleri (sparse rewards) kullanılır. Bu yapı, ajanın anlık ve ölümcül doz değişikliklerini denemesini engellemez. Ajan eğitim aşamasında; ilk saatte Sınıf 4 vazopressör uygulayıp, ikinci saatte Sınıf 0'a (tamamen kesme) geçebilir. Bir insanda bu denli ani vazopressör kesilmesi refrakter hipotansiyona (kardiyovasküler çöküş) neden olur ve pratikte hastayı anında öldürür. Modelin böyle tehlikeli kararlar üretmesini engellemek için Güvenli DRL (Safe-RL) veya Kısıtlı DRL mimarileri kurulmalıdır.  

İletilen özellik mühendisliği notlarında, geçmişteki dozları belirten `prev_vaso_dose` ve `prev_fluid_dose` özelliklerinin veri setinde zaten bulunduğu görülmektedir. Bu mükemmel bir tasarım temelidir. Ajan eğitilirken, seçeceği mevcut aksiyon ile `prev_vaso_dose` arasında iki kademeden fazla (örneğin Bin 1'den Bin 4'e ani zıplama) bir fark varsa, ödül fonksiyonuna büyük bir eksi ceza (negative penalty) maskesi eklenmelidir. Bu hibrit mimari (Safe-Dueling DDQN + Klinik Kural setleri), ajanın hem uzun vadeli sağ kalımı öğrenmesini hem de insan klinisyenler gibi kademeli, güvenli ve rasyonel doz artırımı/azaltımı (titration/weaning) stratejileri geliştirmesini sağlayacaktır.  

## 6\. Polars Optimizasyonları ile Üretime Hazır (Production-Ready) Kod Mimarisi Tavsiyeleri

Projede kullanılan teknoloji yığınının (Polars lazy API, streaming sink) donanım verimliliği açısından sektör standartlarının üzerinde olduğu tespit edilmiştir. Ancak, yukarıda sıralanan eksik değişkenlerin, zaman dinamiklerinin ve istatistiksel filtrelerin projeye entegre edilmesi, hali hazırda yüksek performanslı olan kodu yavaşlatma riski barındırır. Performanstan ödün vermeden tıbbi geçerliliği sağlamak için aşağıdaki mimari eklentilerin Polars sözdizimi (syntax) yapısına yedirilmesi tavsiye edilir:

1.  **Dinamik Zaman Penceresi ve Kaydırma Operatörleri:** Delta özelliklerinin eklenmesi için Polars'ın `over` ve `shift` metodları, devasa groupby operasyonlarına gerek kalmadan pencereleme yapar. `(pl.col("lactate") - pl.col("lactate").shift(1)).over("stay_id").alias("delta_lactate_1h")` ifadesi, saatlik laktat ivmesini milisaniyeler içinde hesaplayarak sisteme kazandırır.
2.  **Veri Tipi Alt-Ölçeklendirmesi (Downcasting):** EDA sürecinde `stay_id`'nin `Int64` ve diğer özelliklerin `float64` tipinde tutulduğu görülmüştür. Tıbbi yaşamsal bulgular ve laboratuvar verileri 64-bit hassasiyet gerektirmez (örneğin kalp hızının küsuratının milyarda bir detaya inmesine gerek yoktur). Veri çerçevesi oluşturulurken `pl.col(pl.Float64).cast(pl.Float32)` ve tamsayılar için `pl.Int32` uygulanması, RAM tüketimini, disk I/O süresini ve en önemlisi DRL ajanının PyTorch/TensorFlow üzerinde GPU'ya yüklenme süresini (VRAM footprint) anında ve sıfır çaba ile %50 oranında düşürecektir.  
    
3.  **Güvenli Forward-Fill:** Yukarıda bahsedilen geçerlilik süresi mantığı, doğrudan Polars fill stratejisine bir argüman olarak yedirilebilir. `pl.col("feature").forward_fill(limit=6)` kullanımı, verinin sadece 6 saat ileri taşınmasını garanti ederek ajanın halüsinasyon görmesini kod seviyesinde engeller.

## 7\. Sonuç ve Stratejik Aksiyon Planı

Değerlendirmeler sonucunda, yürütülen projenin veri işleme mimarisinin bellek optimizasyonu, yazılım mühendisliği pratikleri ve modern veri bilimi araçlarının (Polars) entegrasyonu bağlamında tıp yapay zekası standartlarında son derece başarılı olduğu görülmektedir. Ancak, veri mühendisliğinin mükemmelliği, modelin klinik zekasının yüksek olacağını garanti etmez. Bir derin pekiştirmeli öğrenme ajanının yoğun bakım ortamı gibi yüksek belirsizlik ve risk barındıran bir arenada başarılı bir tedavi stratejisti olabilmesi için, hastanın durumunu tam, doğru ve zamanın akışına duyarlı bir şekilde algılaması şarttır.

Bu bağlamda, modelin eğitim başarısını ve klinik validasyonunu garanti altına almak adına uygulanması gereken öncelikli stratejiler şunlardır: Sepsis-3 kriterleri çerçevesinde enfeksiyon başlangıç zamanının `microbiologyevents` ve `prescriptions` tablolarından çıkarılarak modele altın saatler penceresinin kazandırılması zorunludur. Farmakokinetik gerçekliklerden kopmamak adına hastaların boy ve vücut ağırlıklarının dahil edilip sıvı ve vazopressör aksiyonlarının `mcg/kg/min` ile normalize edilmesi, modelin yanlış politikalar üretmesini önleyecek en hayati adımlardan biridir. Ayrıca, ajanın kısmi gözlemlenebilirliği aşması ve Markov özelliğini koruyabilmesi için her eksik laboratuvar verisi için son ölçümden itibaren geçen sürenin (TSLM) ve fizyolojik ivmeyi gösteren değişim hızı (Delta) özelliklerinin eklenmesi gerekir. Son olarak, kronik laboratuvar verilerindeki saf forward-fill yönteminin kısıtlanması, aykırı değerlerin filtrelenmesi ve aksiyon uzayının güvenli RL prensipleriyle (ani doz değişimlerine ceza sistemi uygulanarak) 25 sınıfa ayrıklaştırılması, projeyi teorik bir yazılım modeli olmaktan çıkarıp devrimsel bir klinik karar destek yapay zekasına dönüştürecektir. Yürütülen bu çalışma, önerilen iyileştirmelerin entegrasyonuyla birlikte literatürdeki emsallerini geride bırakacak bir potansiyele sahiptir.

Raporda kullanılan kaynaklar

![](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

pipeline.py

![](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

config.py

![](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

01\_data\_explore.md

![](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

02\_feature\_engineering.md

[

![](https://t3.gstatic.com/faviconV2?url=https://rlseminar.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

rlseminar.github.io

The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care - RL Seminar

Yeni pencerede açılır](https://rlseminar.github.io/static/files/AI-Clinician-optimal-treatment.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

matthieukomorowski/AI\_Clinician: Reinforcement learning for medical decisions - GitHub

Yeni pencerede açılır](https://github.com/matthieukomorowski/AI_Clinician)[

![](https://t0.gstatic.com/faviconV2?url=https://echorand.me/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

echorand.me

Sepsis onset calculation from MIMIC-III data using sepsis-3 criteria - Exploring Software

Yeni pencerede açılır](https://echorand.me/posts/sepsis-onset-calcualation-mimic-iii/)[

![](https://t3.gstatic.com/faviconV2?url=https://academic.oup.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

academic.oup.com

OpenSep: a generalizable open source pipeline for SOFA score calculation and Sepsis-3 classification - Oxford Academic

Yeni pencerede açılır](https://academic.oup.com/jamiaopen/article/5/4/ooac105/6955623)[

![](https://t3.gstatic.com/faviconV2?url=https://pages.doit.wisc.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pages.doit.wisc.edu

mimic-iv/concepts/sepsis/sepsis3.sql · v3.0.1 - GitLab

Yeni pencerede açılır](https://pages.doit.wisc.edu/JLMARTIN22/mimic-code/-/blob/v3.0.1/mimic-iv/concepts/sepsis/sepsis3.sql)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

The MIMIC Code Repository: enabling reproducibility in critical care research - PMC - NIH

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC6381763/)[

![](https://t2.gstatic.com/faviconV2?url=https://www.medrxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

medrxiv.org

Achieving Expert-Level Clinical Infection Detection with LLMs from Clinical Documents: Validation in Complex Patient Cases with Cirrhosis | medRxiv

Yeni pencerede açılır](https://www.medrxiv.org/content/10.64898/2026.01.13.26344046v1.full)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

arxiv.org

A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU - arXiv.org

Yeni pencerede açılır](https://arxiv.org/html/2510.24500v1)[

![](https://t3.gstatic.com/faviconV2?url=https://www.science.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

science.gov

photometric titration procedure: Topics by Science.gov

Yeni pencerede açılır](https://www.science.gov/topicpages/p/photometric+titration+procedure)[

![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

researchgate.net

(PDF) Association between triglyceride-glucose index and in-hospital mortality in critically ill patients with sepsis: analysis of the MIMIC-IV database - ResearchGate

Yeni pencerede açılır](https://www.researchgate.net/publication/375492571_Association_between_triglyceride-glucose_index_and_in-hospital_mortality_in_critically_ill_patients_with_sepsis_analysis_of_the_MIMIC-IV_database)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

The surviving sepsis campaign: fluid resuscitation and vasopressor therapy research priorities in adult patients - PMC

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC7917035/)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

Aligning prediction models with clinical information needs: infant sepsis case study - PMC

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC11887542/)[

![](https://t0.gstatic.com/faviconV2?url=https://www.preprints.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

preprints.org

Responsible AI for Sepsis Prediction: Bridging the Gap Between Machine Learning Performance and Clinical Trust - Preprints.org

Yeni pencerede açılır](https://www.preprints.org/manuscript/202601.2327/v1/download)[

![](https://t3.gstatic.com/faviconV2?url=https://ai.jmir.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

ai.jmir.org

Predicting Patient Mortality for Earlier Palliative Care Identification in Medicare Advantage Plans: Features of a Machine Learning Model - JMIR AI

Yeni pencerede açılır](https://ai.jmir.org/2023/1/e42253/)[

![](https://t2.gstatic.com/faviconV2?url=https://www.medrxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

medrxiv.org

Learning to Treat Hypotensive Episodes in Sepsis Patients Using a Counterfactual Reasoning Framework | medRxiv

Yeni pencerede açılır](https://www.medrxiv.org/content/10.1101/2021.03.03.21252863.full)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

arxiv.org

Flexible Swarm Learning May Outpace Foundation Models in Essential Tasks - arXiv

Yeni pencerede açılır](https://arxiv.org/pdf/2510.06349)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

arxiv.org

Closing Gaps: An Imputation Analysis of ICU Vital Signs - arXiv

Yeni pencerede açılır](https://arxiv.org/html/2510.24217v1)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

An Optimal Policy for Patient Laboratory Tests in Intensive Care Units - PMC

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417830/)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

A Reinforcement Learning Model for Optimal Treatment Strategies in Intensive Care: Assessment of the Role of Cardiorespiratory Features - PMC

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC11573419/)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

arxiv.org

Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts - arXiv

Yeni pencerede açılır](https://arxiv.org/html/2512.06652v2)[

![](https://t1.gstatic.com/faviconV2?url=https://trace.tennessee.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

trace.tennessee.edu

Improving Reinforcement Learning Techniques for Medical Decision Making - TRACE

Yeni pencerede açılır](https://trace.tennessee.edu/cgi/viewcontent.cgi?article=8000&context=utk_graddiss)[

![](https://t0.gstatic.com/faviconV2?url=https://docs.pola.rs/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

docs.pola.rs

polars.from\_epoch — Polars documentation

Yeni pencerede açılır](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.from_epoch.html)[

![](https://t0.gstatic.com/faviconV2?url=https://docs.pola.rs/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

docs.pola.rs

polars.Expr.dt.epoch — Polars documentation

Yeni pencerede açılır](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.dt.epoch.html)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

stackoverflow.com

Polars DataFrame filter data in a period of time (start and end time) - Stack Overflow

Yeni pencerede açılır](https://stackoverflow.com/questions/73959629/polars-dataframe-filter-data-in-a-period-of-time-start-and-end-time)[

![](https://t2.gstatic.com/faviconV2?url=https://www.medrxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

medrxiv.org

Characterizing and Predicting End-of-Life Patient Trajectories Using Routine Clinical Data - medRxiv

Yeni pencerede açılır](https://www.medrxiv.org/content/10.1101/2025.08.11.25333434v1.full.pdf)[

![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

researchgate.net

Machine Learning-Driven Classification of Sepsis Using Influential Factor - ResearchGate

Yeni pencerede açılır](https://www.researchgate.net/publication/394805214_Machine_Learning-Driven_Classification_of_Sepsis_Using_Influential_Factor)[

![](https://t2.gstatic.com/faviconV2?url=https://par.nsf.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

par.nsf.gov

Exploring a global interpretation mechanism for deep learning networks when predicting sepsis (Journal Article) | NSF PAGES

Yeni pencerede açılır](https://par.nsf.gov/biblio/10399802)[

![](https://t3.gstatic.com/faviconV2?url=https://pdfs.semanticscholar.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pdfs.semanticscholar.org

Imputation-Enhanced Prediction of Septic Shock in ICU Patients - Semantic Scholar

Yeni pencerede açılır](https://pdfs.semanticscholar.org/2d4f/283ed6973502e91146db95578c0fef88c1e9.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

Deep reinforcement learning extracts the optimal sepsis treatment policy from treatment records - PMC

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC11584651/)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

mimic-code/mimic-iv/concepts\_postgres/score/sofa.sql at main · MIT-LCP/mimic-code

Yeni pencerede açılır](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/score/sofa.sql)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

Prediction of 30-day mortality for ICU patients with Sepsis-3 - PMC - NIH

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC11308624/)[

![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

mdpi.com

Machine Learning Models in Sepsis Outcome Prediction for ICU Patients: Integrating Routine Laboratory Tests—A Systematic Review - MDPI

Yeni pencerede açılır](https://www.mdpi.com/2227-9059/12/12/2892)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

An early sepsis prediction model utilizing machine learning and unbalanced data processing in a clinical context - PMC

Yeni pencerede açılır](https://pmc.ncbi.nlm.nih.gov/articles/PMC11345914/)[

![](https://t3.gstatic.com/faviconV2?url=http://d-scholarship.pitt.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

d-scholarship.pitt.edu

Safe Reinforcement Learning for Sepsis Treatment by Liling Lu BS in Biology Engineering, Nanchang University, China, 2011 MS in

Yeni pencerede açılır](http://d-scholarship.pitt.edu/42914/1/Liling%20Lu%20MS%20Thesis%202022.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://www.jmir.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

jmir.org

Reinforcement Learning for Clinical Decision Support in Critical Care: Comprehensive Review - Journal of Medical Internet Research

Yeni pencerede açılır](https://www.jmir.org/2020/7/e18477/)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

arxiv.org

Exploring Time-Step Size in Reinforcement Learning for Sepsis Treatment - arXiv

Yeni pencerede açılır](https://arxiv.org/html/2511.20913v1)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

pmc.ncbi.nlm.nih.gov

Optimizing sepsis treatment strategies via a reinforcement learning





](https://pmc.ncbi.nlm.nih.gov/articles/PMC10874349/)

