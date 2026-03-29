# Roadmap: MIMIC Sepsis Offline RL

## Overview

This roadmap turns the locked offline-RL design decisions from `implematation_plan_gpt.md`
and the research summary into a clinically grounded delivery path. The sequence follows
the real dependency chain of a retrospective sepsis benchmark: first freeze cohort and
time anchors, then lock leakage boundaries, then build the MDP contract, then train and
compare policies, and only then interpret them through OPE, safety diagnostics, and
reproducible reporting. The core training path must remain portable across Apple Silicon
`Metal/MPS` and NVIDIA `CUDA` laptops rather than becoming CUDA-only.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Cohort Definition** - Lock the eligible adult ICU Sepsis-3 study population.
- [ ] **Phase 2: Onset Anchoring and Episode Grid** - Assign one usable onset time and deterministic 4-hour windows per episode.
- [ ] **Phase 3: Split Manifests and Leakage Boundaries** - Freeze patient-level data partitions before any train-only fitting.
- [ ] **Phase 4: State Representation Pipeline** - Build leakage-safe continuous state vectors from a documented feature contract.
- [ ] **Phase 5: Treatment and Reward Encoding** - Convert therapies and outcomes into the fixed MDP action-reward contract.
- [ ] **Phase 6: Transition Dataset and Baseline Benchmarks** - Export replay-ready transitions and simple comparison baselines.
- [ ] **Phase 7: CQL Reference Training** - Train the first conservative offline RL policy on the finalized dataset.
- [ ] **Phase 8: Comparative Offline RL Experiments** - Add BCQ/IQL runs and experiment artifacts on the same contract.
- [ ] **Phase 9: Evaluation, Safety, and Reproducible Package** - Validate model claims with OPE, safety checks, ablations, and reproducible outputs.

## Phase Details

### Phase 1: Cohort Definition
**Goal**: Researcher can produce the eligible adult ICU Sepsis-3 study population from MIMIC-IV with explicit inclusion and exclusion rules.
**Depends on**: Nothing (first phase)
**Requirements**: COH-01
**Success Criteria** (what must be TRUE):
  1. Researcher can generate the adult ICU Sepsis-3 cohort from MIMIC-IV using documented inclusion and exclusion rules.
  2. Researcher can review why a stay was included or excluded without reverse-engineering the code.
  3. Cohort output contains only adult ICU stays that satisfy the intended study definition.
**Plans**: 2 plans

Plans:
- [ ] 01-01: Encode cohort rules, CLI scaffold, config, and cohort spec docs
- [ ] 01-02: Build auditable cohort extraction and exclusion reporting

### Phase 2: Onset Anchoring and Episode Grid
**Goal**: Each usable ICU episode has one reliable sepsis anchor and a deterministic 4-hour analysis timeline.
**Depends on**: Phase 1
**Requirements**: COH-02, COH-03
**Success Criteria** (what must be TRUE):
  1. Researcher can assign exactly one `sepsis_onset_time` per ICU episode or flag the episode as unusable.
  2. Researcher can materialize 4-hour episode windows from `onset -24h` to `onset +48h` with deterministic step boundaries.
  3. Researcher can inspect unusable or truncated episodes without ambiguous onset or step indexing.
**Plans**: 2 plans

Plans:
- [ ] 02-01: Implement Sepsis-3 onset assignment with unusable-case handling
- [ ] 02-02: Materialize deterministic 4-hour episode windows and truncation metadata

### Phase 3: Split Manifests and Leakage Boundaries
**Goal**: Patient-level split boundaries are fixed early enough to protect every learned transform from leakage.
**Depends on**: Phase 2
**Requirements**: DATA-02
**Success Criteria** (what must be TRUE):
  1. Researcher can generate patient-level train, validation, and test manifests with fixed seeds.
  2. Researcher can verify that no patient appears in more than one split.
  3. Downstream pipelines can reuse the same split manifests as the boundary for fitting scalers, action bins, and other learned transforms.
**Plans**: 1 plan

Plans:
- [ ] 03-01: Generate patient-level split manifests and leakage guardrails

### Phase 4: State Representation Pipeline
**Goal**: Every episode step can be converted into a leakage-safe continuous state vector from a documented feature contract.
**Depends on**: Phase 3
**Requirements**: STAT-01, STAT-02, STAT-03
**Success Criteria** (what must be TRUE):
  1. Researcher can build one continuous state vector per episode step from a documented feature dictionary.
  2. Researcher can apply forward-fill, fallback median imputation, and optional missingness flags with a documented policy.
  3. Researcher can fit clipping and normalization on the train split only and reuse the same transforms on validation and test data.
  4. State generation is deterministic for the same episode windows and split manifests.
**Plans**: 2 plans

Plans:
- [ ] 04-01: Define feature dictionary and extraction contract
- [ ] 04-02: Build deterministic state vectors with train-only preprocessing

### Phase 5: Treatment and Reward Encoding
**Goal**: Clinician interventions and outcomes are represented in a fixed MDP action-reward contract.
**Depends on**: Phase 4
**Requirements**: ACT-01, ACT-02, RWD-01
**Success Criteria** (what must be TRUE):
  1. Researcher can convert vasopressor administrations into norepinephrine-equivalent dose bins learned from the train split only.
  2. Researcher can convert 4-hour IV fluid volumes into bins and combine them with vasopressor bins into 25 discrete actions.
  3. Researcher can compute terminal 90-day mortality rewards and intermediate clinical shaping rewards for every transition.
**Plans**: 2 plans

Plans:
- [ ] 05-01: Standardize treatments and freeze the 25-action map
- [ ] 05-02: Implement reward contract and reward diagnostics

### Phase 6: Transition Dataset and Baseline Benchmarks
**Goal**: Prepared episodes become replay-ready offline RL datasets with simple baselines on the same contract.
**Depends on**: Phase 5
**Requirements**: DATA-01, BASE-01
**Success Criteria** (what must be TRUE):
  1. Researcher can export `(s_t, a_t, r_t, s_t+1, done)` transitions and episode-based replay buffers.
  2. Researcher can benchmark clinician behavior, no-treatment, and behavior cloning baselines on the exact same dataset definition.
  3. Baseline outputs can be compared to later RL runs without changing split, action, or reward contracts.
**Plans**: 2 plans

Plans:
- [ ] 06-01: Export transitions and replay-ready dataset artifacts
- [ ] 06-02: Implement clinician, no-treatment, and behavior-cloning baselines

### Phase 7: CQL Reference Training
**Goal**: A conservative offline RL reference model can be trained end-to-end on the finalized dataset with one device-agnostic runtime path for `MPS` and `CUDA`.
**Depends on**: Phase 6
**Requirements**: PLAT-01, RL-01
**Success Criteria** (what must be TRUE):
  1. Researcher can train a discrete-action CQL policy on the prepared offline dataset.
  2. Researcher can rerun the same CQL training pipeline against the fixed action map and reward definition without changing the dataset contract.
  3. Researcher can select Apple Silicon `MPS` or NVIDIA `CUDA` from documented config or auto-detection without forking the training code.
  4. Researcher can load the trained CQL policy and produce action outputs for held-out states on both supported accelerator targets or a documented CPU fallback.
**Plans**: 2 plans

Plans:
- [x] 07-01: Build shared CPU/MPS/CUDA runtime abstraction and configs
- [x] 07-02: Implement the CQL training pipeline on the shared runtime layer

### Phase 8: Comparative Offline RL Experiments
**Goal**: Multiple offline RL algorithms can be compared fairly on the same MDP contract with complete run artifacts.
**Depends on**: Phase 7
**Requirements**: RL-02, RL-03
**Success Criteria** (what must be TRUE):
  1. Researcher can train BCQ and IQL on the same action map, reward function, and data splits used by CQL.
  2. Researcher can store checkpoints, training curves, and experiment configs for every model run.
  3. Model comparison is algorithm-to-algorithm, not confounded by changing dataset definitions.
  4. BCQ and IQL reuse the same cross-platform device abstraction introduced in Phase 7.
**Plans**: 2 plans

Plans:
- [x] 08-01: Create the shared algorithm registry and experiment runner
- [x] 08-02: Implement BCQ/IQL training and standardized comparison artifacts

### Phase 9: Evaluation, Safety, and Reproducible Package
**Goal**: Model claims are backed by retrospective OPE, safety diagnostics, robustness checks, and reproducible artifacts.
**Depends on**: Phase 8
**Requirements**: OPE-01, SAFE-01, SAFE-02, EXP-01, REPR-01
**Success Criteria** (what must be TRUE):
  1. Researcher can evaluate each policy with WIS, ESS, and FQE on held-out data.
  2. Researcher can inspect clinician-sanity review cases, action-frequency heatmaps, and high-risk subgroup behavior before calling a policy clinically plausible.
  3. Researcher can surface support-aware warnings or constraints for poorly supported state-action regions.
  4. Researcher can run ablations over reward shaping, action granularity, timestep choice, missingness flags, and feature subsets.
  5. Researcher can reproduce reported results from saved cohort definitions, feature dictionaries, split manifests, action bins, experiment summaries, and recorded accelerator backend metadata.
**Plans**: 2 plans

Plans:
- [ ] 09-01: Implement OPE metrics, safety checks, and evaluation protocol
- [ ] 09-02: Add ablation registry and reproducible reporting bundle

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Cohort Definition | 0/2 | Not started | - |
| 2. Onset Anchoring and Episode Grid | 0/2 | Not started | - |
| 3. Split Manifests and Leakage Boundaries | 0/1 | Not started | - |
| 4. State Representation Pipeline | 0/2 | Not started | - |
| 5. Treatment and Reward Encoding | 0/2 | Not started | - |
| 6. Transition Dataset and Baseline Benchmarks | 0/2 | Not started | - |
| 7. CQL Reference Training | 2/2 | ✅ Complete | 2026-03-29 |
| 8. Comparative Offline RL Experiments | 2/2 | ✅ Complete | 2026-03-29 |
| 9. Evaluation, Safety, and Reproducible Package | 0/2 | Not started | - |
