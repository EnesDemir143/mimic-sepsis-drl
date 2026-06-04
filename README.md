# MIMIC Sepsis CQL Offline Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/) [![Model](https://img.shields.io/badge/HuggingFace-mimic--sepsis--cql-yellow)](https://huggingface.co/EnesDemir143/mimic-sepsis-cql) [![Report](https://img.shields.io/badge/Report-PDF-red)](report/main.pdf)

This repository contains an academic offline reinforcement learning study on the MIMIC-IV v3.1 Sepsis-3 ICU cohort. The final reported model is a **Conservative Q-Learning (CQL)** policy for discretized IV fluid and vasopressor treatment decisions. The project emphasizes leakage-safe cohort construction, patient-level splits, validation-only model selection, and conservative off-policy evaluation rather than clinical deployment.

> **Clinical safety note:** This project is a retrospective research artifact. It is **not** a clinical decision support tool and must not be used for patient care.

## Project Summary

- **Dataset:** MIMIC-IV v3.1, requiring PhysioNet credentialed access. Raw patient data is not included in this repository.
- **Task:** Offline RL for sepsis treatment policy evaluation.
- **State space:** 62-dimensional ICU patient state representation.
- **Action space:** 25 discrete treatment actions from 5 × 5 IV fluid and vasopressor bins.
- **Algorithm:** Conservative Q-Learning (CQL).
- **Selection protocol:** Two-stage validation-only model selection; held-out test used exactly once after final checkpoint selection.
- **Final model on Hugging Face:** [EnesDemir143/mimic-sepsis-cql](https://huggingface.co/EnesDemir143/mimic-sepsis-cql)
- **Full report:** [report/main.pdf](report/main.pdf)

## Final CQL Result

Final selected checkpoint:

```text
checkpoints/cql_sweep/cql_s1024_sparse_lr1e-4_a0p05/cql_epoch0200_step0007000.pt
```

Final hyperparameters:

| Field | Value |
|---|---:|
| reward variant | sparse |
| learning rate | 1e-4 |
| CQL alpha | 0.05 |
| seed | 1024 |
| selected epoch | 200 |

Held-out test evaluation:

| Metric | Value |
|---|---:|
| FQE mean | 15.689874 |
| FQE 95% CI | [15.616595, 15.755585] |
| WIS mean | 10.018438 |
| WIS 95% CI | [4.121083, 12.658275] |
| ESS | 10.408948 |
| Test episodes | 2585 |
| Bootstrap resamples | 1000 |

The report interprets these values as evidence for a reproducible offline RL evaluation workflow, not as proof of clinical superiority.

## Report Overview

The PDF report summarizes the complete study in a paper-like format:

1. **Cohort definition:** Sepsis-3-based ICU cohort construction from MIMIC-IV.
2. **MDP formulation:** 4-hour decision windows, patient-level states, 25 treatment actions, sparse and shaped reward definitions.
3. **Model selection:** Stage 1 hyperparameter screening and Stage 2 multi-seed validation without using the test set.
4. **Evaluation:** FQE, WIS, ESS, bootstrap confidence intervals, clinician agreement, support diagnostics.
5. **Limitations:** Retrospective EHR bias, OPE uncertainty, support mismatch, no prospective validation.

See: [report/main.pdf](report/main.pdf)

## Repository Layout

```text
configs/                 Training and runtime configuration files
src/mimic_sepsis_rl/     Source code for cohort, MDP, training, and evaluation
scripts/                 CQL sweep, final evaluation, and figure generation scripts
docs/                    Project documentation organized by topic
report/                  LaTeX source, figures, bibliography, and compiled PDF
tests/                   Unit tests for data, MDP, training, and evaluation modules
```

Large local artifacts are intentionally excluded from GitHub:

- raw MIMIC-IV data (`data/`)
- replay buffers
- training runs (`runs/`)
- checkpoints (`checkpoints/`)
- local academic delivery bundle (`230202066/`)
- planning metadata (`.planning/`)

## Documentation

Start with the docs index:

- [docs/README.md](docs/README.md) — topic-organized documentation index
- [docs/README_TR.md](docs/README_TR.md) — Türkçe proje özeti
- [docs/README_EN.md](docs/README_EN.md) — English project summary
- [docs/final_model_selection.md](docs/final_model_selection.md) — final model selection protocol and metrics
- [docs/reproducibility.md](docs/reproducibility.md) — reproducibility guide
- [docs/evaluation_protocol.md](docs/evaluation_protocol.md) — OPE and validation protocol

## Reproducibility

Install dependencies:

```bash
uv sync
```

Raw MIMIC-IV data must be available locally under:

```text
data/raw/physionet.org/files/mimiciv/3.1
```

Build the replay dataset:

```bash
uv run python -m mimic_sepsis_rl.cli.build_cohort --config configs/cohort/default.yaml --emit-audit
uv run python -m mimic_sepsis_rl.data.onset --config configs/onset/default.yaml
uv run python -m mimic_sepsis_rl.cli.build_episode_grid
uv run python -m mimic_sepsis_rl.data.splits --config configs/splits/default.yaml --source-episode-set data/processed/episodes/episodes.parquet
uv run python -m mimic_sepsis_rl.cli.build_transitions
```

Train CQL:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql
```

Evaluate the selected final policy:

```bash
uv run python scripts/evaluate_final_selected_policy.py   --checkpoint checkpoints/cql_sweep/cql_s1024_sparse_lr1e-4_a0p05/cql_epoch0200_step0007000.pt   --test-data data/replay/replay_test.parquet   --output runs/cql_sweep/final_test_evaluation.json   --bootstrap-resamples 1000
```

## Data Access and Ethics

MIMIC-IV is distributed through PhysioNet and requires credentialed access plus required training/certification. This repository does not include raw patient records or derived replay buffers.

## Author

Enes Demir — 230202066  
Kocaeli University, Department of Computer Engineering

## Citation

If you use this repository, cite MIMIC-IV and PhysioNet according to their official citation requirements and cite CQL/offline RL references as appropriate. See [report/references.bib](report/references.bib).
