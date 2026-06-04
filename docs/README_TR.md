# MIMIC Sepsis CQL Projesi — Türkçe Özet

Bu proje, MIMIC-IV v3.1 yoğun bakım verisi üzerinde Sepsis-3 kohortu için **Conservative Q-Learning (CQL)** tabanlı çevrimdışı pekiştirmeli öğrenme çalışmasıdır. MIMIC-IV ve PhysioNet kaynakları, Sepsis-3 tanımı, sağlıkta RL uyarıları, CQL yöntemi ve OPE literatürü aşağıdaki kaynaklarla açıkça referanslanmıştır. Amaç bir klinik karar destek sistemi dağıtmak değil; retrospektif sağlık verisi üzerinde sızıntısız, doğrulanabilir ve raporlanabilir bir offline RL deney çerçevesi kurmaktır.

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

## Kaynaklar

README ve final raporda kullanılan temel kaynaklar:

1. Johnson ve ark., **MIMIC-IV, a freely accessible electronic health record dataset**, *Scientific Data*, 2023. DOI: [10.1038/s41597-022-01899-x](https://doi.org/10.1038/s41597-022-01899-x).
2. Johnson ve ark., **MIMIC-IV (version 3.1)**, PhysioNet, 2024. DOI: [10.13026/kpb9-mt58](https://doi.org/10.13026/kpb9-mt58).
3. Goldberger ve ark., **PhysioBank, PhysioToolkit, and PhysioNet**, *Circulation*, 2000. DOI: [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215).
4. Singer ve ark., **The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)**, *JAMA*, 2016. DOI: [10.1001/jama.2016.0287](https://doi.org/10.1001/jama.2016.0287).
5. Komorowski ve ark., **The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care**, *Nature Medicine*, 2018. DOI: [10.1038/s41591-018-0213-5](https://doi.org/10.1038/s41591-018-0213-5).
6. Gottesman ve ark., **Guidelines for reinforcement learning in healthcare**, *Nature Medicine*, 2019. DOI: [10.1038/s41591-018-0310-5](https://doi.org/10.1038/s41591-018-0310-5).
7. Kumar ve ark., **Conservative Q-Learning for Offline Reinforcement Learning**, NeurIPS, 2020. DOI: [10.48550/arXiv.2006.04779](https://doi.org/10.48550/arXiv.2006.04779).
8. Levine ve ark., **Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems**, arXiv, 2020. DOI: [10.48550/arXiv.2005.01643](https://doi.org/10.48550/arXiv.2005.01643).
9. Thomas ve Brunskill, **Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning**, ICML, 2016. DOI: [10.48550/arXiv.1604.00923](https://doi.org/10.48550/arXiv.1604.00923).

Tam BibTeX listesi: [`../report/references.bib`](../report/references.bib).
