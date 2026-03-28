# Requirements: MIMIC Sepsis Offline RL

**Defined:** 2026-03-28
**Core Value:** Klinik olarak makul, veri sizintisina dayanikli ve yeniden uretilebilir bir offline RL benchmark'i olusturmak.

## v1 Requirements

### Cohort

- [ ] **COH-01**: Researcher can generate an adult ICU Sepsis-3 cohort from MIMIC-IV using documented inclusion and exclusion rules.
- [ ] **COH-02**: Researcher can assign exactly one `sepsis_onset_time` per ICU episode or flag the episode as unusable.
- [ ] **COH-03**: Researcher can materialize 4-hour episode windows from `onset -24h` to `onset +48h` with deterministic step boundaries.

### State Construction

- [ ] **STAT-01**: Researcher can build one continuous state vector per episode step from a documented feature dictionary.
- [ ] **STAT-02**: Researcher can apply forward-fill, fallback median imputation, and optional missingness flags with a documented policy.
- [ ] **STAT-03**: Researcher can fit clipping and normalization on the train split only and reuse the same transforms on validation and test data.

### Actions and Rewards

- [ ] **ACT-01**: Researcher can convert vasopressor administrations to norepinephrine-equivalent dose bins learned from the train split only.
- [ ] **ACT-02**: Researcher can convert 4-hour IV fluid volumes into bins and combine them with vasopressor bins into 25 discrete actions.
- [ ] **RWD-01**: Researcher can compute terminal 90-day mortality rewards and intermediate clinical shaping rewards for every transition.

### Dataset and Baselines

- [ ] **DATA-01**: Researcher can export `(s_t, a_t, r_t, s_t+1, done)` transitions and episode-based replay buffers.
- [ ] **DATA-02**: Researcher can create patient-level train, validation, and test splits with fixed seeds and leakage checks.
- [ ] **BASE-01**: Researcher can benchmark clinician behavior, no-treatment, and behavior cloning baselines on the same dataset definition.

### Platform Compatibility

- [ ] **PLAT-01**: Researcher can run the same training and evaluation pipeline on Apple Silicon via `Metal/MPS` and on NVIDIA GPUs via `CUDA`, with documented environment setup and device selection.

### Offline RL Training

- [ ] **RL-01**: Researcher can train a discrete-action CQL policy on the prepared offline dataset.
- [ ] **RL-02**: Researcher can train BCQ and IQL on the same action map, reward function, and data splits for fair comparison.
- [ ] **RL-03**: Researcher can store checkpoints, training curves, and experiment configs for every model run.

### Evaluation and Safety

- [ ] **OPE-01**: Researcher can evaluate each policy with WIS, ESS, and FQE on held-out data.
- [ ] **SAFE-01**: Researcher can inspect policy behavior with clinician-sanity review cases, action-frequency heatmaps, and high-risk subgroup analysis.
- [ ] **SAFE-02**: Researcher can apply support-aware warnings or constraints for poorly supported state-action regions.
- [ ] **EXP-01**: Researcher can run ablations over reward shaping, action granularity, timestep choice, missingness flags, and feature subsets.

### Reproducibility

- [ ] **REPR-01**: Researcher can reproduce reported results from saved cohort definitions, feature dictionaries, split manifests, action bins, and experiment summaries.

## v2 Requirements

### External Validation

- **EXT-01**: Researcher can evaluate the pipeline on an external or temporal holdout dataset beyond the primary internal split.
- **EXT-02**: Researcher can compare transfer performance across ICU populations or hospitals.

### Decision Support Surface

- **DSS-01**: Researcher can review policy suggestions in a clinician-facing analysis interface.
- **DSS-02**: Researcher can inspect explanation artifacts for why a recommendation was produced.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live bedside deployment | Project scope is retrospective offline research, not production clinical software |
| Continuous-action control | Initial design is intentionally fixed to discrete 25-action treatment bins |
| Non-sepsis cohorts | First milestone must de-risk one clinically coherent use case |
| Prospective causal claims | OPE and retrospective analysis do not justify real-world efficacy claims |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| COH-01 | Phase 1 | Pending |
| COH-02 | Phase 2 | Pending |
| COH-03 | Phase 2 | Pending |
| DATA-02 | Phase 3 | Pending |
| STAT-01 | Phase 4 | Pending |
| STAT-02 | Phase 4 | Pending |
| STAT-03 | Phase 4 | Pending |
| ACT-01 | Phase 5 | Pending |
| ACT-02 | Phase 5 | Pending |
| RWD-01 | Phase 5 | Pending |
| DATA-01 | Phase 6 | Pending |
| BASE-01 | Phase 6 | Pending |
| PLAT-01 | Phase 7 | Pending |
| RL-01 | Phase 7 | Pending |
| RL-02 | Phase 8 | Pending |
| RL-03 | Phase 8 | Pending |
| OPE-01 | Phase 9 | Pending |
| SAFE-01 | Phase 9 | Pending |
| SAFE-02 | Phase 9 | Pending |
| EXP-01 | Phase 9 | Pending |
| REPR-01 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-03-28*
*Last updated: 2026-03-28 after platform portability update*
