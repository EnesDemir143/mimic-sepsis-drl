# Project Research Summary

**Project:** MIMIC Sepsis Offline RL
**Domain:** Offline reinforcement learning benchmark for retrospective sepsis treatment optimization
**Researched:** 2026-03-28
**Confidence:** HIGH

## Executive Summary

Bu proje, MIMIC-IV icindeki Sepsis-3 episode'larini sabit bir MDP kontratina donusturup uzerinde offline RL algoritmalarini karsilastiran bir arastirma pipeline'i olarak ele alinmali. Basarinin anahtari model secimi degil; cohort, onset, episode, state, action ve reward tanimlarinin birbirini bozmayacak sekilde dondurulmasi. Bu nedenle yol haritasi once veri ve MDP katmanini sabitlemeli, sonra dataset ve baseline'lari kilitlemeli, daha sonra RL egitimi ve en sonda OPE/safety/regression raporlamasina gecmeli.

Oncelikli teknik tercih, Python + SQL/Parquet + PyTorch ekseninde ilerlemek. d3rlpy gibi offline-RL odakli kutuphaneler hiz kazandirabilir; ancak projenin asil guvencesi fit/transform sinirlarini koruyan artifact-first mimari olacak. Buna ek olarak ana training yolunun hem Apple Silicon `Metal/MPS` hem de NVIDIA `CUDA` uzerinde calisabilmesi plan seviyesinde zorunlu tutulmali; bu yuzden backend'e ozel degil device-agnostic PyTorch operasyonlari tercih edilmeli. En buyuk riskler yanlis sepsis onset tanimi, patient-level leakage, full-dataset action binning ve OPE'nin klinik etkinlik gibi yorumlanmasi.

## Key Findings

### Recommended Stack

Bu alan icin ana omurga Python 3.11, PyTorch 2.x ve SQL/Parquet artifact akisidir. Kaynak veriyi SQL ile denetlenebilir sekilde cekip ara tabloları Parquet olarak dondurmek, hem tekrarlanabilirligi hem de deney hizi arttirir. Polars/scikit-learn tabanli preprocessing ve d3rlpy veya ozel PyTorch trainer'lari bu yapinin ustune sorunsuz oturur.

**Core technologies:**
- Python 3.11 — ETL, training, evaluation icin ortak dil
- PyTorch 2.x — CQL, BCQ, IQL ve ozel loss'lar icin esnek backend; `MPS` ve `CUDA` icin ortak ana yol
- SQL + Parquet artifacts — audit edilebilir cohort/onset/episode pipeline'i

### Expected Features

Bu tip bir benchmark projesinde kullanicilar once savunulabilir cohort/MDP tanimi, sonra adil model karsilastirmasi ve son olarak guvenilir degerlendirme bekler. Safety overlay ve clinician review paketleri fark yaratan unsurlardir; canli deployment veya UI ise erken fazda scope'u bozar.

**Must have (table stakes):**
- Cohort + onset + episode generation — tum zaman eksenini sabitler
- State/action/reward builders — offline RL dataset kontratini olusturur
- Leakage-safe splits and baselines — adil karsilastirma tabani saglar
- OPE + diagnostics — sonucu yorumlanabilir hale getirir

**Should have (competitive):**
- Support-aware safety filters — dusuk destekli state-action bolgelerini isaretler
- Clinician review pack — nicel sonuclari klinik makulluk ile baglar
- Temporal holdout mode — daha guclu genelleme kontrolu saglar

**Defer (v2+):**
- External dataset validation — temel pipeline oturmadan maliyetli
- Clinician-facing UI — once guvenilir analiz cikisi uretmek gerekir

### Architecture Approach

Onerilen mimari, kaynak veri katmani -> data construction -> dataset contract -> modeling/evaluation seklinde ardiskil ama artifact-first bir yapidir. Bu siralama, her fazda bir sonraki katmana gecmeden once cikan arti̇faktlari dondurup denetlemeye izin verir ve leakage risklerini dogal olarak azaltir.

**Major components:**
1. Cohort and onset pipeline — episode dahil edilme ve zaman merkezi
2. MDP builder — state, action, reward, transition arti̇faktlari
3. Training layer — baseline'lar ile CQL/BCQ/IQL
4. Evaluation layer — WIS, ESS, FQE, safety, ablation, figures

### Critical Pitfalls

1. **Wrong sepsis onset operationalization** — tum MDP'yi bozar; ilk fazda kilitlenmeli
2. **Patient leakage across splits or transforms** — sonuclari gecersizlestirir; split once, fit sonra ilkesi zorunlu
3. **Action bins learned from full data** — gizli leakage yaratir; train-only mapping artifact'i gerekir
4. **Reward over-shaping** — klinik olarak anlamsiz politikalar dogurabilir; sparse baseline ile baslanmali
5. **Treating OPE as clinical efficacy** — raporlamada sinirlar acik yazilmali

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Clinical Data and MDP Specification
**Rationale:** Tum sonraki isler tek bir cohort, onset ve episode tanimina bagli.
**Delivers:** Cohort spec, onset table, episode windows, feature dictionary, preprocessing policy.
**Addresses:** Cohort and state requirements.
**Avoids:** Wrong onset operationalization.

### Phase 2: Dataset Contract and Baseline Readiness
**Rationale:** RL egitiminden once state/action/reward/transitions ve leakage-safe split'ler dondurulmali.
**Delivers:** Action bins, reward tables, transition dataset, split manifests, baseline benchmarks.
**Uses:** SQL/Parquet artifacts, train-only fit boundaries.
**Implements:** Dataset contract layer.

### Phase 3: Offline RL Training and Comparison
**Rationale:** CQL, BCQ ve IQL ancak ortak dataset kontrati uzerinde adil sekilde karsilastirilabilir; ayni zamanda ilk training giris noktasi `MPS` ve `CUDA` uyumlu olmalidir.
**Delivers:** Trained checkpoints, experiment configs, training curves, model comparison tables, cross-device runtime notes.

### Phase 4: Evaluation, Safety, and Reproducibility
**Rationale:** Klinik olarak savunulabilir sonuc, sadece egitim degil OPE, sanity check, ablation ve paketlenmis raporlama ile gelir.
**Delivers:** WIS/FQE reports, ESS diagnostics, clinician review pack, safety notes, reproducible bundle.

### Phase Ordering Rationale

- Cohort/onset/episode tanimi action ve reward tanimindan once sabitlenmeli.
- Dataset kontrati dondurulmadan algoritma karsilastirmasi guvenilir olmaz.
- Safety ve reportlama son fazda toplanmali ama onceki fazlarda gerekli arti̇faktlar uretilmeli.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Sepsis-3 operationalization and feature-source mapping may require schema-level deep dives.
- **Phase 3:** Algorithm implementation choice may depend on library support versus custom PyTorch work.
- **Phase 4:** OPE and safety reporting thresholds need careful interpretation rules.

Phases with standard patterns (skip research-phase):
- **Phase 2:** Transition dataset assembly and split manifests follow well-understood data-engineering patterns once definitions are frozen.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Strong defaults are clear, but exact package pins should be verified during implementation |
| Features | HIGH | The implementation plan already constrains scope well |
| Architecture | HIGH | Artifact-first layered architecture fits this domain strongly |
| Pitfalls | HIGH | Main failure modes are explicit in the plan and standard for retrospective clinical ML |

**Overall confidence:** HIGH

### Gaps to Address

- Exact package/version pins should be finalized when the environment is bootstrapped.
- Sepsis-3 onset operationalization needs explicit schema-level definitions, not only conceptual agreement.
- Whether to use d3rlpy or custom trainers should be decided after phase-1/2 data contracts are stable.
- Exact MPS/CUDA package pins and any unsupported-op fallbacks should be validated during training-phase planning.

## Sources

### Primary (HIGH confidence)
- `implematation_plan_gpt.md` — project methodology and locked decisions

### Secondary (MEDIUM confidence)
- Standard offline RL and clinical ML engineering patterns — stack and architecture defaults

---
*Research completed: 2026-03-28*
*Ready for roadmap: yes*
