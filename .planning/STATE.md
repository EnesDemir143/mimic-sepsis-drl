# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-28)

**Core value:** Klinik olarak makul, veri sizintisina dayanikli ve yeniden uretilebilir bir offline RL benchmark'i olusturmak.
**Current focus:** Phase 9 - Evaluation, Safety, and Reproducible Package

## Current Position

Phase: 9 of 9 (Evaluation, Safety, and Reproducible Package)
Plan: 1 of 2 in current phase
Status: 09-01 ready to execute
Last activity: 2026-03-29 - Phase 8 plan 08-02 completed

Progress: [████████░░] 89% (8 of 9 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 4 (Phases 7-8)
- Average duration: ~44 min/plan (estimated)
- Total execution time: ~3.0 hours (estimated)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 0 | 2 | - |
| 2 | 0 | 2 | - |
| 3 | 0 | 1 | - |
| 4 | 0 | 2 | - |
| 5 | 0 | 2 | - |
| 6 | 0 | 2 | - |
| 7 | 2 | 2 | ~60 min |
| 8 | 2 | 2 | ~28 min |
| 9 | 0 | 2 | - |

**Recent Trend:**
- Last 5 plans: 07-01, 07-02, 08-01, 08-02
- Trend: Active

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Phase 1 must lock the adult ICU Sepsis-3 cohort before onset logic or MDP construction.
- Phase 3 owns patient-level split manifests so all later scalers, bins, and transforms stay train-only.
- The main training path must stay device-agnostic so the same code runs on Apple Silicon `MPS` and NVIDIA `CUDA`.
- Evaluation claims remain retrospective research claims; OPE is not evidence of bedside efficacy.
- Custom PyTorch chosen over d3rlpy for trainer implementation — provides full MPS/CUDA portability without hidden fallback risks.
- CQL is the reference algorithm; Phase 8 BCQ/IQL reuse `common.py`, `device.py`, and `TrainingConfig` without modification.
- Synthetic dry-run tensors must be generated on CPU then moved to device — MPS does not accept a CPU-pinned Generator directly.
- BCQ and IQL now emit the same checkpoint-manifest and JSONL curve envelope as CQL, so comparison tooling can stay algorithm-agnostic.
- Comparison reports should normalize run artifacts from disk and flag dataset-contract drift instead of silently merging incompatible runs.

### Phase 7 Completion Summary

**Plan 07-01 — Device Runtime Abstraction (PLAT-01 ✅)**
- `src/mimic_sepsis_rl/training/device.py` — `resolve_device()`, `DeviceMetadata`, `validate_mps_ops()`, `--self-check` CLI
- `src/mimic_sepsis_rl/training/config.py` — `TrainingConfig`, `load_training_config()`, `build_training_config()`
- `configs/training/runtime.mps.yaml` / `runtime.cuda.yaml`
- 41 tests passing (`tests/training/test_device_runtime.py`)

**Plan 07-02 — CQL Training Pipeline (RL-01 ✅)**
- `src/mimic_sepsis_rl/training/common.py` — `ReplayDataset`, `CheckpointManager`, `MetricLogger`, `set_global_seed`
- `src/mimic_sepsis_rl/training/cql.py` — `QNetwork`, `CQLTrainer`, `CQLPolicy`, `load_cql_policy`, `--dry-run` CLI
- `configs/training/cql.yaml` — `algorithm: cql`, α=1.0, 200 epochs, batch=256
- `docs/cql_training.md`
- 64 tests passing (`tests/training/test_cql_pipeline.py`)

### Phase 8 Progress

**Plan 08-01 — Shared Algorithm Registry and Experiment Runner (RL-02 partial ✅)**
- `src/mimic_sepsis_rl/training/registry.py` — `AlgorithmRegistry`, shared config resolution, CQL adapter, BCQ/IQL placeholders
- `src/mimic_sepsis_rl/training/experiment_runner.py` — `resolve_experiment()`, `DatasetContractSummary`, shared CLI entrypoint
- `configs/training/bcq.yaml` / `iql.yaml` — baseline configs on the frozen CQL data contract
- 9 tests passing (`tests/training/test_algorithm_registry.py`)

**Plan 08-02 — BCQ/IQL Training and Comparison Artifacts (RL-02, RL-03 ✅)**
- `src/mimic_sepsis_rl/training/bcq.py` — BCQ trainer, dry-run, and policy surface on the shared replay contract
- `src/mimic_sepsis_rl/training/iql.py` — IQL trainer, expectile/value/actor stack, and shared artifact emission
- `src/mimic_sepsis_rl/training/comparison.py` — normalized run artifacts and cross-algorithm comparison reports
- `docs/model_comparison.md` — comparison artifact contract and interpretation guide
- 13 tests passing across registry/comparison coverage (`tests/training/test_algorithm_registry.py`, `tests/training/test_comparison_runs.py`)

### Pending Todos

- Phase 9: Implement OPE metrics, safety checks, and evaluation packaging on top of the standardized Phase 8 run artifacts.
- Phases 1–6: Planning artifacts exist but implementation summaries are partially missing from STATE tracking.

### Blockers/Concerns

- Sepsis-3 onset operationalization still needs schema-level clarification during Phase 1 and Phase 2 execution.
- OPE, safety overlays, and ablations remain deferred to Phase 9.
- Full BCQ/IQL training still requires the Phase 6 replay dataset artifacts to exist locally; current verification used shared dry-runs and artifact-schema tests.

## Session Continuity

Last session: 2026-03-29 16:45 +03
Stopped at: Phase 8 complete; Phase 9 plan 09-01 ready to execute
Resume file: None
