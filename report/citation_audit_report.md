# Kaynakça ve Atıf Denetim Raporu

Denetim tarihi: 2026-05-24 16:50:43 +03

Denetlenen dosyalar:

- `report/main.tex`
- `report/references.bib`
- Derlenmiş kaynak sırası için `report/main.bbl`
- PDF çıktısı için `report/main.pdf`

## 1. Özet Sonuç

Genel durum: Önceki denetimde bulunan kaynakça ve atıf sorunları düzeltilmiştir.

Son durum:

1. Metinde atıf verilen her anahtar `references.bib` içinde mevcuttur.
   - Eksik BibTeX karşılığı: yok.

2. `references.bib` içinde olup metinde hiç atıf verilmeyen kayıt kalmamıştır.
   - Kullanılmayan BibTeX kaydı: yok.

3. `fu2020d4rl` mevcut rapor metninde kullanılmadığı için kaynakçadan çıkarılmıştır.

4. MIMIC-IV v3.1 için doğru dataset sürüm kaydı metne eklenmiştir.
   - Veri kümesi cümlesi artık `johnson2023mimiciv` ve `johnson2024mimicivphysionet` kayıtlarını birlikte atıflamaktadır.

5. Doğrulanabilen DOI bilgileri `.bib` içine eklenmiş ve PDF kaynakçasında görünür hale getirilmiştir.
   - `tang2026temporal`: `10.1038/s41746-026-02625-2`
   - `thomas2015highconfidence`: `10.1609/aaai.v29i1.9541`
   - arXiv DOI karşılığı doğrulanabilen kayıtlar için `10.48550/arXiv...` DOI'leri eklenmiştir.

6. DOI bulunmayan/tespit edilemeyen bazı konferans veya JMLR/PMLR kayıtları için uydurma DOI eklenmemiştir; bunun yerine resmi kayıt URL'si eklenmiştir.
   - `precup2000eligibility`: resmi çevrimiçi kayıt URL'si
   - `ernst2005tree`: resmi JMLR kayıt URL'si
   - `hao2021bootstrapping`: resmi PMLR kayıt URL'si

7. Build alınmıştır.
   - Çıktı PDF: `report/main.pdf`
   - Son LaTeX derlemesinde tanımsız atıf uyarısı yoktur.

## 2. Atıf-Kaynakça Kapsam Kontrolü

### 2.1 Metindeki benzersiz atıf anahtarları

Toplam benzersiz atıf: 20

- `ernst2005tree`
- `evans2021surviving`
- `fujimoto2019off`
- `goldberger2000physionet`
- `gottesman2019guidelines`
- `hao2021bootstrapping`
- `johnson2023mimiciv`
- `johnson2024mimicivphysionet`
- `komorowski2018ai`
- `kumar2020cql`
- `le2019batch`
- `levine2020offline`
- `precup2000eligibility`
- `raghu2017continuous`
- `singer2016sepsis3`
- `tang2021model`
- `tang2026temporal`
- `thomas2015highconfidence`
- `thomas2016data`
- `voloshin2021empirical`

### 2.2 BibTeX içindeki kayıtlar

Toplam BibTeX kaydı: 20

Metinde atıfı olmayan kayıtlar: yok.

Eksik BibTeX kaydı: yok.

Derlenmiş `main.bbl` içinde basılan kaynak sayısı: 20.

## 3. Düzeltilen Ana Problemler

| Önceki problem | Yapılan düzeltme | Son durum |
|---|---|---|
| `johnson2024mimicivphysionet` kaynakçada vardı ama metinde kullanılmıyordu | MIMIC-IV v3.1 cümlesine eklendi | Düzeltildi |
| `fu2020d4rl` kaynakçada vardı ama metinde kullanılmıyordu | Kaynakçadan çıkarıldı | Düzeltildi |
| `tang2026temporal` DOI eksikti | DOI eklendi: `10.1038/s41746-026-02625-2` | Düzeltildi |
| `thomas2015highconfidence` DOI eksikti | DOI eklendi: `10.1609/aaai.v29i1.9541` | Düzeltildi |
| IEEEtran çıktısında DOI görünmüyordu | DOI'ler `note` alanı ile görünür hale getirildi | Düzeltildi |
| arXiv/PMLR/JMLR kayıtlarında DOI/URL eksikleri vardı | Doğrulanabilen arXiv DOI'leri ve resmi URL'ler eklendi | Düzeltildi |

## 4. DOI/URL Durumu

| BibTeX anahtarı | DOI/URL durumu |
|---|---|
| `johnson2023mimiciv` | DOI: `10.1038/s41597-022-01899-x` |
| `johnson2024mimicivphysionet` | DOI: `10.13026/kpb9-mt58` |
| `goldberger2000physionet` | DOI: `10.1161/01.CIR.101.23.e215` |
| `singer2016sepsis3` | DOI: `10.1001/jama.2016.0287` |
| `komorowski2018ai` | DOI: `10.1038/s41591-018-0213-5` |
| `evans2021surviving` | DOI: `10.1007/s00134-021-06506-y` |
| `gottesman2019guidelines` | DOI: `10.1038/s41591-018-0310-5` |
| `kumar2020cql` | arXiv DOI: `10.48550/arXiv.2006.04779` |
| `levine2020offline` | arXiv DOI: `10.48550/arXiv.2005.01643` |
| `precup2000eligibility` | DOI doğrulanamadı; resmi çevrimiçi kayıt URL'si eklendi |
| `thomas2015highconfidence` | DOI: `10.1609/aaai.v29i1.9541` |
| `thomas2016data` | arXiv DOI: `10.48550/arXiv.1604.00923` |
| `ernst2005tree` | DOI doğrulanamadı; resmi JMLR URL'si eklendi |
| `le2019batch` | arXiv DOI: `10.48550/arXiv.1903.08738` |
| `voloshin2021empirical` | arXiv DOI: `10.48550/arXiv.1911.06854` |
| `hao2021bootstrapping` | DOI doğrulanamadı; resmi PMLR URL'si eklendi |
| `raghu2017continuous` | arXiv DOI: `10.48550/arXiv.1705.08422` |
| `fujimoto2019off` | arXiv DOI: `10.48550/arXiv.1812.02900` |
| `tang2021model` | arXiv DOI: `10.48550/arXiv.2107.11003` |
| `tang2026temporal` | DOI: `10.1038/s41746-026-02625-2` |

## 5. Build ve Teknik Doğrulama

Build komutu standart dört aşamalı derleme ile çalıştırılmıştır:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Son otomatik kontrolde:

- Metindeki benzersiz atıf sayısı: 20
- BibTeX kayıt sayısı: 20
- `main.bbl` içinde basılan kaynak sayısı: 20
- Eksik BibTeX karşılığı: yok
- Kullanılmayan BibTeX kaydı: yok
- Son log içinde tanımsız atıf uyarısı: yok
- PDF kaynakçasında DOI/URL bilgileri görünür durumdadır

## 6. Son Karar

Kaynakça ve atıf yapısı teslim kriterleri açısından düzeltilmiştir:

- Uydurma referans tespit edilmedi.
- Atıf verilen her kaynak kaynakçada vardır.
- Kaynakçada bulunan her kayıt metin içinde atıflanmıştır.
- Doğrulanabilen DOI'ler eklenmiştir.
- DOI bulunmayan kayıtlar için uydurma DOI eklenmemiş, resmi URL verilmiştir.
- PDF başarıyla yeniden derlenmiştir.
