# Referans Doğrulama Raporu

**Proje:** MIMIC-IV Sepsis Kohortunda Conservative Q-Learning ile Çevrimdışı Pekiştirmeli Öğrenme
**Tarih:** 2026-05-25
**TeX Dosyası:** `report/main.tex`
**BibTeX Dosyası:** `report/references.bib`
**Toplam Referans:** 20

---

## 1. Özet

- Doğrulanan: 20/20
- Gerçek (✅): 20
- Şüpheli (⚠️): 0
- Bulunamadı (❌): 0

**Sonuç:** Tüm referanslar gerçek yayınlardır. Hiçbir halüsinasyon tespit edilmemiştir.

---

## 2. Zorunlu Atıf Kontrolü

MIMIC-IV veri seti için gerekli üç zorunlu atıf:

| Atıf | Anahtar | .bib'de? | TeX'te? | TeX Satırları | Durum |
|------|---------|----------|---------|---------------|-------|
| Johnson et al. (2023) — Sci Data | `johnson2023mimiciv` | ✅ | ✅ | 51, 68 | Uygun |
| Goldberger et al. (2000) — PhysioNet | `goldberger2000physionet` | ✅ | ✅ | 51 | Uygun |
| Johnson et al. (2024) — MIMIC-IV v3.1 | `johnson2024mimicivphysionet` | ✅ | ✅ | 68 | Uygun |

- Satır 51: EHR kaynak tanıtımı — `\cite{johnson2023mimiciv,goldberger2000physionet}` doğru bağlamda
- Satır 68: Veri kümesi alt bölümü — `\cite{johnson2023mimiciv,johnson2024mimicivphysionet}` doğru bağlamda

---

## 3. Tam Referans Tablosu

| # | BibTeX Anahtarı | Yazar(lar) | Yıl | Venue | Doğrulama Kaynağı | Durum |
|---|----------------|------------|-----|-------|-------------------|-------|
| 1 | `singer2016sepsis3` | Singer et al. | 2016 | JAMA 315(8):801-810 | refcheck, web | ✅ |
| 2 | `evans2021surviving` | Evans et al. | 2021 | Intensive Care Med 47:1181-1247 | web | ✅ |
| 3 | `johnson2023mimiciv` | Johnson et al. | 2023 | Scientific Data 10:1 | user | ✅ |
| 4 | `goldberger2000physionet` | Goldberger et al. | 2000 | Circulation 101(23):e215-e220 | user | ✅ |
| 5 | `levine2020offline` | Levine et al. | 2020 | arXiv:2005.01643 | web | ✅ |
| 6 | `gottesman2019guidelines` | Gottesman et al. | 2019 | Nature Medicine 25:16-18 | web | ✅ |
| 7 | `komorowski2018ai` | Komorowski et al. | 2018 | Nature Medicine 24:1716-1720 | refcheck | ✅ |
| 8 | `raghu2017continuous` | Raghu et al. | 2017 | MLHC (PMLR v68) | web | ✅ |
| 9 | `tang2026temporal` | Tang et al. | 2026 | npj Digital Medicine | refcheck | ✅ |
| 10 | `fujimoto2019off` | Fujimoto et al. | 2019 | ICML (PMLR v97) | web | ✅ |
| 11 | `kumar2020cql` | Kumar et al. | 2020 | NeurIPS 33:1179-1191 | web | ✅ |
| 12 | `tang2021model` | Tang & Wiens | 2021 | PMLR v149:2-35 | gs | ✅ |
| 13 | `thomas2016data` | Thomas & Brunskill | 2016 | ICML:2139-2148 | web | ✅ |
| 14 | `precup2000eligibility` | Precup et al. | 2000 | ICML:759-766 | web | ✅ |
| 15 | `ernst2005tree` | Ernst et al. | 2005 | JMLR 6:503-556 | web | ✅ |
| 16 | `le2019batch` | Le et al. | 2019 | ICML (PMLR v97) | gs | ✅ |
| 17 | `voloshin2021empirical` | Voloshin et al. | 2021 | arXiv:1911.06854 | web | ✅ |
| 18 | `johnson2024mimicivphysionet` | Johnson et al. | 2024 | PhysioNet | user | ✅ |
| 19 | `hao2021bootstrapping` | Hao et al. | 2021 | ICML (PMLR v139) | gs | ✅ |
| 20 | `thomas2015highconfidence` | Thomas et al. | 2015 | AAAI 29(1) | refcheck | ✅ |

---

## 4. Sorunlu Referanslar

**Sorunlu referans bulunmamaktadır.**

---

## 5. Metodoloji

Doğrulama şu kaynaklar kullanılarak çok aşamalı yapılmıştır:

1. **refcheck MCP** (`verify_reference`) — CrossRef, Semantic Scholar ve arXiv üzerinden birincil doğrulama
2. **Google Scholar** (`paper-search` MCP) — refcheck'in bulamadığı makaleler için ikincil kaynak
3. **arXiv MCP** (`search_papers`, `get_abstract`) — arXiv preprint'leri için
4. **CrossRef MCP** (`searchByTitle`) — başlık bazlı arama
5. **Web araması** — eski ICML/JMLR makaleleri ve son çare olarak

Kaynak kısaltmaları: `refcheck` = refcheck MCP, `gs` = Google Scholar, `web` = Web araması, `user` = Kullanıcı tarafından sağlandı.

---

## 6. Bulgular ve Öneriler

### Güçlü Yönler
- 20 referansın tamamı gerçek yayın — hiç LLM halüsinasyonu yok
- MIMIC-IV zorunlu atıflarının üçü de eksiksiz ve doğru bağlamda kullanılmış
- Referanslar sepsis, offline RL, OPE ve klinik karar desteği alanlarını dengeli kapsıyor
- Hem klasik (Precup 2000, Ernst 2005) hem güncel (Tang 2026) referanslar mevcut

### Küçük Notlar
- `tang2026temporal` Mayıs 2026'da yayınlanmış çok yeni bir makale — güncellik açısından değerli
- `voloshin2021empirical` anahtarı 2021 yılını kullanıyor ancak arXiv ilk sürümü 2019 — anahtarın 2021 olması kabul edilebilir (muhtemelen konferans basım yılı)
- `raghu2017continuous` için DOI olarak arXiv DOI'si kullanılmış — MLHC proceedings DOI'si ile değiştirilebilir ancak arXiv DOI'si de standarttır

### Genel Değerlendirme
Kaynakça eksiksiz, doğru ve güvenilirdir. Herhangi bir düzeltme gerekmemektedir.
