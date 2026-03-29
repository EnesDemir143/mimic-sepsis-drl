---
phase: 08-comparative-offline-rl-experiments
plan: 02
subsystem: training
tags: [offline-rl, bcq, iql, comparison, provenance]
requires:
  - phase: 08-comparative-offline-rl-experiments
    provides: "Shared algorithm registry, runner, and frozen dataset-contract surface from 08-01"
provides:
  - "BCQ and IQL trainers on the shared experiment runner contract"
  - "Standardized comparison artifacts spanning checkpoints, manifests, curves, and config provenance"
  - "Documentation and regression coverage for fair multi-algorithm comparisons"
affects: [08-comparative-offline-rl-experiments, 09-evaluation-safety-and-reproducible-package]
tech-stack:
  added: []
  patterns: [shared artifact normalization, registry-backed multi-algorithm trainers, dataset-contract drift detection]
key-files:
  created:
    - src/mimic_sepsis_rl/training/bcq.py
    - src/mimic_sepsis_rl/training/iql.py
    - src/mimic_sepsis_rl/training/comparison.py
    - tests/training/test_comparison_runs.py
    - docs/model_comparison.md
  modified:
    - src/mimic_sepsis_rl/training/registry.py
    - src/mimic_sepsis_rl/training/__init__.py
    - tests/training/test_algorithm_registry.py
key-decisions:
  - "BCQ and IQL keep the same result envelope and artifact contract as CQL while exposing algorithm-specific loss summaries."
  - "Comparison reporting reads manifests and JSONL curves from disk instead of requiring each trainer to implement a custom exporter."
  - "Dataset-contract mismatches are surfaced as explicit comparison drift instead of being merged silently."
patterns-established:
  - "Algorithm trainers integrate through AlgorithmRegistry handlers and return dict payloads compatible with experiment_runner."
  - "Phase 9 should consume comparison.py reports instead of parsing checkpoint and metrics files ad hoc."
requirements-completed: [RL-02, RL-03]
duration: "~28min"
completed: 2026-03-29
---

# Plan 08-02 Summary: BCQ/IQL Training and Standardized Comparison Artifacts

**BCQ and IQL now train on the shared offline RL surface, emit the same artifact envelope as CQL, and roll up into one comparison-ready schema**

## Performance

- **Duration:** ~28 min
- **Started:** 2026-03-29T13:18:00Z
- **Completed:** 2026-03-29T13:46:31Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Added runnable BCQ and IQL trainers that reuse the shared config, replay dataset, device abstraction, checkpointing, and metric logging layers.
- Replaced planned BCQ/IQL registry placeholders with real handlers so both algorithms launch through `experiment_runner`.
- Added comparison utilities that normalize checkpoints, manifests, metric curves, config provenance, and dataset-contract metadata into one schema.
- Documented how to interpret comparison outputs without confusing algorithm changes with replay-contract drift.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement BCQ and IQL trainers on the shared experiment surface** - `b12ef9e` (`feat`)
2. **Task 2: Standardize comparison artifacts, tests, and docs** - `e53d17e` (`docs`)

## Files Created/Modified

- `src/mimic_sepsis_rl/training/bcq.py` - Shared-surface BCQ trainer, dry-run path, and policy wrapper
- `src/mimic_sepsis_rl/training/iql.py` - Shared-surface IQL trainer with critic/value/actor updates and dry-run path
- `src/mimic_sepsis_rl/training/comparison.py` - Run-artifact normalization and comparison report aggregation
- `src/mimic_sepsis_rl/training/registry.py` - Real BCQ/IQL handlers behind the shared experiment runner
- `tests/training/test_algorithm_registry.py` - Dry-run registry coverage for ready BCQ/IQL algorithms
- `tests/training/test_comparison_runs.py` - Artifact-shape and dataset-drift regression tests
- `docs/model_comparison.md` - Phase 8 comparison artifact contract and interpretation guide

## Decisions Made

- Kept BCQ and IQL result payloads aligned with the existing CQL runner contract: shared core fields plus algorithm-specific final losses.
- Stored comparison provenance by loading the resolved YAML config and adjacent manifest/metric files from disk rather than inventing a second artifact source.
- Let comparison reports explicitly flag dataset-contract drift so Phase 9 does not accidentally compare runs trained under different replay assumptions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Replaced Phase 08-01 placeholder availability assumptions**
- **Found during:** Task 1 (shared experiment surface integration)
- **Issue:** The registry and its existing tests still modeled BCQ and IQL as planned-only algorithms, which blocked the required shared-runner dry-runs.
- **Fix:** Swapped in real BCQ/IQL registry handlers, marked both algorithms as ready, and updated registry coverage to assert successful dry-runs.
- **Files modified:** `src/mimic_sepsis_rl/training/registry.py`, `src/mimic_sepsis_rl/training/__init__.py`, `tests/training/test_algorithm_registry.py`
- **Verification:** `./.venv/bin/python -m pytest -q tests/training/test_algorithm_registry.py tests/training/test_comparison_runs.py` and both runner dry-run commands passed
- **Committed in:** `b12ef9e` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The deviation was required to make BCQ and IQL reachable through the already-shipped shared runner. No scope creep beyond the intended Phase 8 integration.

## Issues Encountered

- PyTorch emitted a local CUDA driver warning during config/device resolution, but the shared runtime correctly fell back to CPU and all targeted verifications still passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 9 can consume `comparison.py` reports and normalized run artifacts without per-algorithm parsing.
- BCQ and IQL now satisfy the same dry-run contract as CQL through `experiment_runner`.
- Full end-to-end training beyond dry-run still depends on the Phase 6 replay dataset artifacts being present locally.

---
*Phase: 08-comparative-offline-rl-experiments*
*Completed: 2026-03-29*
