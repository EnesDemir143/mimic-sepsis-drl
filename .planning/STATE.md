# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-28)

**Core value:** Klinik olarak makul, veri sizintisina dayanikli ve yeniden uretilebilir bir offline RL benchmark'i olusturmak.
**Current focus:** Phase 8 - Comparative Offline RL Experiments

## Current Position

Phase: 8 of 9 (Comparative Offline RL Experiments)
Plan: 0 of 2 in current phase
Status: Ready to execute
Last activity: 2026-03-29 - Phase 7 CQL Reference Training completed (2/2 plans)

Progress: [███████░░░] 78% (7 of 9 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 2 (Phase 7)
- Average duration: ~60 min/plan (estimated)
- Total execution time: ~2.0 hours (Phase 7)

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
| 8 | 0 | 2 | - |
| 9 | 0 | 2 | - |

**Recent Trend:**
- Last 5 plans: 07-01, 07-02
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

### Pending Todos

- Phase 8: BCQ/IQL trainers should reuse `common.py` utilities and `TrainingConfig` without modification.
- Phase 8: Algorithm registry and experiment runner needed before individual BCQ/IQL implementations.
- Phases 1–6: Planning artifacts exist but implementation summaries are partially missing from STATE tracking.

### Blockers/Concerns

- Sepsis-3 onset operationalization still needs schema-level clarification during Phase 1 and Phase 2 execution.
- Phase 8 must reuse the CQL device abstraction and dataset contract — no forking allowed.
- OPE, safety overlays, and ablations remain deferred to Phase 9.

## Session Continuity

Last session: 2026-03-29 15:43 +03
Stopped at: Phase 7 complete (2/2 plans); all commits on main; Phase 8 ready to execute
Resume file: None