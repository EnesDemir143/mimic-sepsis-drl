# Documentation Index

This folder is organized as a topic-oriented reference for the MIMIC Sepsis CQL offline RL project.

## Quick Summaries

| Document | Purpose |
|---|---|
| [README_TR.md](README_TR.md) | Türkçe proje özeti, sonuçlar ve kullanım notları |
| [README_EN.md](README_EN.md) | English project summary and result overview |
| [final_model_selection.md](final_model_selection.md) | Validation-only final CQL checkpoint selection |

## Dataset and Cohort

| Topic | Document |
|---|---|
| Cohort selection | [cohort_selection.md](cohort_selection.md) |
| Feature dictionary | [feature_dictionary.md](feature_dictionary.md) |
| Leakage boundaries | [leakage_boundaries.md](leakage_boundaries.md) |
| Reproducibility | [reproducibility.md](reproducibility.md) |

## MDP Methodology

| Topic | Document |
|---|---|
| Action discretization | [action_mapping.md](action_mapping.md) |
| Reward design | [reward_spec.md](reward_spec.md) |
| Pipeline and RL positioning | [pipeline_rl_positioning.md](pipeline_rl_positioning.md) |

## Model Training and Selection

| Topic | Document |
|---|---|
| CQL training reference | [cql_training.md](cql_training.md) |
| Final CQL model selection | [final_model_selection.md](final_model_selection.md) |
| Model comparison context | [model_comparison.md](model_comparison.md) |
| Baseline notes | [baseline_benchmarks.md](baseline_benchmarks.md) |

## Evaluation and Results

| Topic | Document |
|---|---|
| Evaluation protocol | [evaluation_protocol.md](evaluation_protocol.md) |
| CQL run report | [cql_run_report.md](cql_run_report.md) |
| Project report draft | [cql_project_report.md](cql_project_report.md) |
| Final PDF report | [../report/main.pdf](../report/main.pdf) |
| Report figures and tables | [assets/report](assets/report) |

## Citation and References

The main citation list is available in the root [README](../README.md#references), the Turkish/English summaries, and [`../report/references.bib`](../report/references.bib). MIMIC-IV, PhysioNet, Sepsis-3, healthcare RL, CQL, and OPE references are included explicitly.

## Safety and Limitations

Key safety constraints are documented in:

- [leakage_boundaries.md](leakage_boundaries.md)
- [evaluation_protocol.md](evaluation_protocol.md)
- [final_model_selection.md](final_model_selection.md)

The project is retrospective and research-only. It is not a deployable clinical decision system.
