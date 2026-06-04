# MIMIC Sepsis CQL Projesi — Türkçe Özet

Bu proje, MIMIC-IV v3.1 yoğun bakım verisi üzerinde Sepsis-3 kohortu için **Conservative Q-Learning (CQL)** tabanlı çevrimdışı pekiştirmeli öğrenme çalışmasıdır. Amaç bir klinik karar destek sistemi dağıtmak değil; retrospektif sağlık verisi üzerinde sızıntısız, doğrulanabilir ve raporlanabilir bir offline RL deney çerçevesi kurmaktır.

## Kısa Özet

- **Veri:** MIMIC-IV v3.1, PhysioNet erişimi gerektirir. Ham hasta verisi repoda yoktur.
- **Kohort:** Sepsis-3 kriterlerine göre seçilmiş yetişkin ICU epizotları.
- **Zamanlama:** 4 saatlik karar adımları.
- **Durum uzayı:** 62 boyutlu hasta durumu.
- **Aksiyon uzayı:** IV sıvı ve vazopressör dozlarına göre 5 × 5 = 25 ayrık aksiyon.
- **Model:** Conservative Q-Learning (CQL).
- **Seçim protokolü:** Test set kullanılmadan iki aşamalı validation seçimi.
- **Final model:** [Hugging Face — EnesDemir143/mimic-sepsis-cql](https://huggingface.co/EnesDemir143/mimic-sepsis-cql)
- **Rapor:** [../report/main.pdf](../report/main.pdf)

## Final Model

Seçilen checkpoint:

```text
checkpoints/cql_sweep/cql_s1024_sparse_lr1e-4_a0p05/cql_epoch0200_step0007000.pt
```

| Özellik | Değer |
|---|---:|
| Reward variant | sparse |
| Learning rate | 1e-4 |
| CQL alpha | 0.05 |
| Seed | 1024 |
| Epoch | 200 |

## Final Test Sonucu

| Metrik | Değer |
|---|---:|
| FQE mean | 15.689874 |
| FQE 95% GA | [15.616595, 15.755585] |
| WIS mean | 10.018438 |
| WIS 95% GA | [4.121083, 12.658275] |
| ESS | 10.408948 |
| Test epizodu | 2585 |

Bu sonuçlar modelin klinik olarak üstün olduğunu kanıtlamaz. Sonuçlar, seçilmiş offline RL politikasının izole test split üzerinde OPE metrikleriyle nasıl raporlandığını gösterir.

## Rapor Özeti

PDF raporda şu başlıklar yer alır:

1. Sepsis-3 kohort tanımı ve hasta düzeyinde split yapısı.
2. MDP formülasyonu: 62 state özelliği, 25 aksiyon, ödül tasarımı.
3. CQL eğitim ve model seçim süreci.
4. Stage 1 hiperparametre taraması ve Stage 2 çoklu seed doğrulama.
5. Final test değerlendirmesi: FQE, WIS, ESS ve bootstrap güven aralıkları.
6. Klinik güvenlik, veri sızıntısı ve OPE sınırlılıkları.

## Doküman Haritası

- [cohort_selection.md](cohort_selection.md): Kohort kuralları
- [feature_dictionary.md](feature_dictionary.md): State değişkenleri
- [action_mapping.md](action_mapping.md): Aksiyon ayrıklaştırma
- [reward_spec.md](reward_spec.md): Ödül fonksiyonu
- [cql_training.md](cql_training.md): CQL eğitim referansı
- [final_model_selection.md](final_model_selection.md): Final checkpoint seçimi
- [evaluation_protocol.md](evaluation_protocol.md): Değerlendirme protokolü
- [reproducibility.md](reproducibility.md): Yeniden üretilebilirlik

## Uyarı

Bu proje araştırma amaçlıdır. Hasta bakımı veya gerçek zamanlı klinik karar desteği için kullanılmamalıdır.
