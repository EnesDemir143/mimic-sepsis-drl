# MIMIC Sepsis CQL Project — English Summary

This project is an offline reinforcement learning study using **Conservative Q-Learning (CQL)** on a MIMIC-IV v3.1 Sepsis-3 ICU cohort. The goal is not clinical deployment, but a leakage-aware and reproducible research pipeline for retrospective policy evaluation.

## Summary

- **Dataset:** MIMIC-IV v3.1, requiring credentialed PhysioNet access. Raw patient data is not included.
- **Cohort:** Adult ICU episodes satisfying Sepsis-3 criteria.
- **Decision interval:** 4-hour windows.
- **State space:** 62-dimensional patient representation.
- **Action space:** 25 discrete actions from IV fluid and vasopressor bins.
- **Algorithm:** Conservative Q-Learning (CQL).
- **Selection:** Two-stage validation-only checkpoint/config selection.
- **Final model:** [Hugging Face — EnesDemir143/mimic-sepsis-cql](https://huggingface.co/EnesDemir143/mimic-sepsis-cql)
- **Report:** [../report/main.pdf](../report/main.pdf)

## Final Model

Selected checkpoint:

```text
checkpoints/cql_sweep/cql_s1024_sparse_lr1e-4_a0p05/cql_epoch0200_step0007000.pt
```

| Field | Value |
|---|---:|
| Reward variant | sparse |
| Learning rate | 1e-4 |
| CQL alpha | 0.05 |
| Seed | 1024 |
| Epoch | 200 |

## Final Test Evaluation

| Metric | Value |
|---|---:|
| FQE mean | 15.689874 |
| FQE 95% CI | [15.616595, 15.755585] |
| WIS mean | 10.018438 |
| WIS 95% CI | [4.121083, 12.658275] |
| ESS | 10.408948 |
| Test episodes | 2585 |

These metrics should be interpreted as offline policy evaluation evidence, not proof of clinical benefit.

## Report Overview

The PDF report covers:

1. Sepsis-3 cohort construction and patient-level splits.
2. MDP design: state features, 25 treatment actions, reward specification.
3. CQL training and validation-only model selection.
4. Stage 1 hyperparameter screening and Stage 2 multi-seed validation.
5. Final held-out test evaluation with FQE, WIS, ESS, and bootstrap confidence intervals.
6. Limitations: retrospective EHR bias, support mismatch, OPE uncertainty, and no prospective validation.

## Documentation Map

- [cohort_selection.md](cohort_selection.md): Cohort rules
- [feature_dictionary.md](feature_dictionary.md): State features
- [action_mapping.md](action_mapping.md): Treatment action discretization
- [reward_spec.md](reward_spec.md): Reward definition
- [cql_training.md](cql_training.md): CQL training reference
- [final_model_selection.md](final_model_selection.md): Final checkpoint selection
- [evaluation_protocol.md](evaluation_protocol.md): Evaluation protocol
- [reproducibility.md](reproducibility.md): Reproducibility guide

## Safety Notice

This repository is a retrospective research artifact and must not be used for patient care or real-time clinical decision support.
