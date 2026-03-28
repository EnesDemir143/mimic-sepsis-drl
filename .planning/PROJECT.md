# MIMIC Sepsis Offline RL

## What This Is

MIMIC-IV verisi uzerinde sepsis tedavisi icin offline reinforcement learning arastirma pipeline'i kuran bir proje. Sistem, Sepsis-3 tanimli ICU episode'larini 4 saatlik adimlarla MDP'ye cevirip vazopressor ve IV fluid aksiyonlarini ogrenebilir bir dataset haline getirecek; ardindan CQL, BCQ ve IQL politikalarini egitip offline policy evaluation ile karsilastiracak. Hedef kitle, bu benchmark'i tez, akademik calisma veya retrospektif klinik karar destek arastirmasi icin kullanacak arastirmacilar; egitim ve degerlendirme katmani hem Apple Silicon `Metal/MPS` hem de NVIDIA `CUDA` uzerinde calisabilecek sekilde tasarlanacak.

## Core Value

Klinik olarak makul, veri sizintisina dayanikli ve yeniden uretilebilir bir offline RL benchmark'i olusturmak.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Sepsis-3 temelli eriskin ICU kohortu ve tekil `sepsis_onset_time` uretimi kurulsun.
- [ ] `onset -24h` ile `onset +48h` arasinda 4 saatlik adimlarla episode ve state pipeline'i olusturulsun.
- [ ] Feature dictionary, imputasyon, normalization ve outlier kurallari dokumante edilip uygulanabilsin.
- [ ] Vazopressor ve IV fluid eylemleri train split'ten ogrenilen 25 ayrik action uzayina donusturulsun.
- [ ] Gecis dataset'i, patient-level split'ler ve baseline modeller ayni tanimlar uzerinde tekrar uretilebilir olsun.
- [ ] CQL, BCQ ve IQL ayni MDP ve reward tanimi uzerinde karsilastirilabilsin.
- [ ] Training ve evaluation pipeline'i ayni kod yoluyla hem MacBook M2 Pro `Metal/MPS` hem de NVIDIA RTX 4070 `CUDA` ortaminda calisabilsin.
- [ ] WIS, FQE, ESS, klinik akla uygunluk ve safety constraint analizleri ile modeller degerlendirilsin.
- [ ] Deney ayarlari, arti̇faktlar ve raporlama paketleri tez/paper kalitesinde yeniden uretilebilir olsun.

### Out of Scope

- Canli klinik sistem ya da bedside deployment — bu asamada yalnizca retrospektif offline arastirma hedefleniyor.
- Sepsis disi kohortlar veya coklu dataset genellemesi — ilk surum yalnizca MIMIC-IV icindeki Sepsis-3 ICU episode'larina odaklaniyor.
- Continuous-action veya model-based RL — baslangic tasarimi discrete 5x5 action ve model-free offline RL olarak sabitlendi.
- Prospektif klinik etki iddiasi — OPE sonucunu gercek dunyada klinik basari yerine gecmez.

## Context

Bu repo su anda PDF'den turetilmis `implematation_plan_gpt.md` dosyasindaki arastirma planini operasyonel GSD is akisina cevirmek icin baslatiliyor. Plan; offline, off-policy, model-free RL, continuous state, discrete 25 action, 4 saatlik timestep, 72 saatlik episode penceresi, terminal ve ara reward'lar, CQL/BCQ/IQL adaylari ve WIS + FQE degerlendirmesini sabit kararlar olarak belirliyor. Proje brownfield bir uygulama degil; bu nedenle gereksinimler hipotez olarak baslatilip fazlara bolunecek.

## Constraints

- **Data**: MIMIC-IV erisimi ve Sepsis-3 operationalization gerekir — kohort ve olay zamanlari bu veri kaynaklarina bagli.
- **Methodology**: Offline, off-policy, model-free RL tercihleri sabit — ilk yol haritasi bu kararlarin disina cikmayacak.
- **Clinical Framing**: Yetiskin ICU sepsis episode birimi korunmali — farkli analiz birimleri metodolojik tutarliligi bozar.
- **Leakage Control**: Action bin sinirlari, scaler'lar ve diger ogrenilen donusumler yalniz train split'te fit edilmeli — aksi halde sonuc guvenilmez olur.
- **Platform Support**: Ayni egitim ve inference kodu Apple Silicon `MPS` ve NVIDIA `CUDA` uzerinde calismali — backend'e ozel cikmaz sokaklar ve CUDA-only bagimliliklar ana yolun parcasi olmamali.
- **Evaluation**: Online etkileşimli ortam yok — politika kalitesi yalniz OPE, sanity check ve support analizi ile savunulabilir.
- **Reproducibility**: Split seed'leri, reward formulu ve deney config'leri kaydedilmeli — tez/paper seviyesinde izlenebilirlik zorunlu.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Offline RL kullan | Klinik veride cevrimici deney yapmak riskli ve pahali | — Pending |
| Off-policy ve model-free yaklasim sec | Gozlemsel veriyle davranis politikasindan ogrenmek daha uygulanabilir | — Pending |
| State uzayini continuous tut | Vitals ve lab degerlerindeki bilgi kaybini azaltmak | — Pending |
| Action uzayini 5x5 discrete yap | Vazopressor ve IV fluid dozlarini guvenli ve karsilastirilabilir sekilde ogrenmek | — Pending |
| Timestep'i 4 saat olarak sabitle | Klinik aksiyonlar ile olcum frekansi arasinda pratik denge kurmak | — Pending |
| Episode penceresini `-24h/+48h` ile sinirla | Sepsis onset cevresi klinik kararlarin yogunlastigi pencere | — Pending |
| Ilk algoritma olarak CQL ile basla | Offline setting'de OOD action riskine karsi daha korumaci bir baslangic saglar | — Pending |
| Ana training yolunu device-agnostic kur | MacBook M2 Pro `MPS` ve RTX 4070 `CUDA` uzerinde ayni pipeline'i calistirmak hedefleniyor | — Pending |
| Degerlendirmede WIS + FQE birlikte kullan | Tek bir OPE metrigine guvenmek saglik alaninda zayif kalir | — Pending |

---
*Last updated: 2026-03-28 after platform portability update*
