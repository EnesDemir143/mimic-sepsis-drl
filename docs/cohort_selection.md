# Cohort Selection

## Overview

This document describes the patient population used in the MIMIC Sepsis Offline Reinforcement Learning study. The cohort is defined as the set of adult ICU stays in MIMIC-IV that satisfy the **Sepsis-3** consensus criteria. All rules are encoded in `src/mimic_sepsis_rl/data/cohort/spec.py` and parameterised by `configs/cohort/default.yaml`, ensuring full reproducibility.

---

## Study Unit

The unit of analysis is the **ICU stay** (`icustays.stay_id`). A single hospital admission (`hadm_id`) may produce multiple ICU stays; each is evaluated independently against the eligibility rules. After rule application, **only the first eligible ICU stay per patient** is retained to prevent within-patient data leakage across training and evaluation splits.

---

## Source Tables

| Alias | MIMIC-IV Table | Purpose |
|-------|---------------|---------|
| `icustays` | `mimiciv_icu.icustays` | Base unit of analysis; provides `stay_id`, `intime`, `outtime` |
| `patients` | `mimiciv_hosp.patients` | `anchor_age`, `gender` |
| `admissions` | `mimiciv_hosp.admissions` | Hospital-level linkage via `hadm_id` |
| `sepsis3` | `mimiciv_derived.sepsis3` | Pre-computed Sepsis-3 flag, `sofa_score`, `suspected_infection_time` |

> **Note:** `mimiciv_derived.sepsis3` is part of the MIMIC-IV Clinical Database Derived Data repository. If absent, an equivalent query must be pre-computed and registered under the same alias in the config.

---

## Inclusion Criteria

| Rule | Parameter | Value | Rationale |
|------|-----------|-------|-----------|
| Adult patients only | `min_age_years` | 18 | Paediatric physiology differs substantially; excluding under-18s avoids confounding the reward and action spaces. |
| ICU stay required | `require_icu_stay` | `true` | MDP episodes are anchored to ICU stays; non-ICU admissions lack the monitoring density needed for 4-hour state vectors. |
| Sepsis-3 flag required | `require_sepsis3` | `true` | Restricts the cohort to the clinically defined Sepsis-3 population; ensures that vasopressor utility is within the intended treatment context. |
| Minimum ICU length-of-stay | `min_los_hours` | 4.0 h | Stays shorter than one episode step cannot contribute a meaningful MDP transition and are dropped. |

---

## Exclusion Criteria

| Rule | Parameter | Value | Rationale |
|------|-----------|-------|-----------|
| Missing sepsis anchor | `exclude_missing_sepsis_anchor` | `true` | Without a usable `suspected_infection_time` the onset anchor (Phase 2) cannot be computed; the stay must be dropped rather than imputed. |
| Readmissions | `exclude_readmissions` | `true` | Only the **first** ICU stay per `subject_id` is retained. Subsequent stays share physiology with the patient's earlier trajectory, creating temporal leakage. |
| Missing demographics | `exclude_missing_demographics` | `true` | Null `anchor_age` or `gender` break downstream age-gating and demographic stratification. |
| Maximum age | `max_age_years` | `null` (no limit) | No upper age cap is applied in the default configuration; can be set if the study population requires it. |

---

## Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| Included cohort | `data/processed/cohort/cohort.parquet` | One row per eligible ICU stay with stable identifiers (`subject_id`, `hadm_id`, `stay_id`) |
| Excluded stays | `data/processed/cohort/excluded.parquet` | All dropped stays with the **first applicable exclusion reason** |
| Audit summary | `data/processed/cohort/audit.json` | Count-level summary: included N, excluded N, per-rule breakdown |

---

## Reproducibility

The cohort is generated deterministically from the same source data and config file:

```bash
# Validate config without a database
python -m mimic_sepsis_rl.cli.build_cohort \
    --config configs/cohort/default.yaml \
    --dry-run

# Full extraction with row-level audit
python -m mimic_sepsis_rl.cli.build_cohort \
    --config configs/cohort/default.yaml \
    --emit-audit
```

The YAML config version (`spec_version: "1.0.0"`) is recorded in every output artifact so that any downstream result can be traced to the exact rule set that produced the cohort.

---

## Relation to Downstream Phases

```
Phase 1 (Cohort Definition)  ──► Phase 2 (Onset Anchoring)
    cohort.parquet                  assigns sepsis_onset_time per stay_id

Phase 2  ──► Phase 3 (Split Manifests)
    usable_episodes.parquet         patient-level train / val / test split

Phase 3  ──► Phase 4+ (State, Action, Reward, Training)
    split_manifests/*.parquet       leakage boundary for all learned transforms
```

Phase 1 outputs contain only eligibility-level decisions. No onset times, episode windows, or learned transforms are created here.

---

*Last updated: 2026-03-28*
*Spec version: 1.0.0*
*Requirement: COH-01*
