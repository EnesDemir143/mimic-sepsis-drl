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
- **Final model:** [Hugging Face — morpeN1/mimic-sepsis-cql](https://huggingface.co/morpeN1/mimic-sepsis-cql)
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

## References

This repository follows the citation set used in the final report. Key references are:

1. Johnson et al., **MIMIC-IV, a freely accessible electronic health record dataset**, *Scientific Data*, 2023. DOI: [10.1038/s41597-022-01899-x](https://doi.org/10.1038/s41597-022-01899-x).
2. Johnson et al., **MIMIC-IV (version 3.1)**, PhysioNet, 2024. DOI: [10.13026/kpb9-mt58](https://doi.org/10.13026/kpb9-mt58).
3. Goldberger et al., **PhysioBank, PhysioToolkit, and PhysioNet**, *Circulation*, 2000. DOI: [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215).
4. Singer et al., **The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)**, *JAMA*, 2016. DOI: [10.1001/jama.2016.0287](https://doi.org/10.1001/jama.2016.0287).
5. Komorowski et al., **The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care**, *Nature Medicine*, 2018. DOI: [10.1038/s41591-018-0213-5](https://doi.org/10.1038/s41591-018-0213-5).
6. Gottesman et al., **Guidelines for reinforcement learning in healthcare**, *Nature Medicine*, 2019. DOI: [10.1038/s41591-018-0310-5](https://doi.org/10.1038/s41591-018-0310-5).
7. Kumar et al., **Conservative Q-Learning for Offline Reinforcement Learning**, NeurIPS, 2020. DOI: [10.48550/arXiv.2006.04779](https://doi.org/10.48550/arXiv.2006.04779).
8. Levine et al., **Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems**, arXiv, 2020. DOI: [10.48550/arXiv.2005.01643](https://doi.org/10.48550/arXiv.2005.01643).
9. Thomas and Brunskill, **Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning**, ICML, 2016. DOI: [10.48550/arXiv.1604.00923](https://doi.org/10.48550/arXiv.1604.00923).

The complete BibTeX file is available at [`../report/references.bib`](../report/references.bib).
