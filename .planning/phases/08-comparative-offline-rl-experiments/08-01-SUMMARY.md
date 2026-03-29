---
phase: 08-comparative-offline-rl-experiments
plan: 01
subsystem: training
tags: [offline-rl, registry, experiment-runner, cql, bcq, iql]
requires:
  - phase: 07-cql-reference-training
    provides: "Shared runtime abstraction, TrainingConfig, and the CQL reference trainer"
provides:
  - "Registry-backed launch surface for offline RL algorithms"
  - "Shared CLI for listing, describing, and launching experiment configs"
  - "BCQ and IQL baseline configs on the frozen replay contract"
  - "Regression coverage for algorithm selection and runtime wiring"
affects: [08-comparative-offline-rl-experiments, 09-evaluation-safety-and-reproducible-package]
tech-stack:
  added: []
  patterns: [registry-driven experiment launch, shared config resolution, dataset-contract introspection]
key-files:
  created:
    - src/mimic_sepsis_rl/training/registry.py
    - src/mimic_sepsis_rl/training/experiment_runner.py
    - configs/training/bcq.yaml
    - configs/training/iql.yaml
    - tests/training/test_algorithm_registry.py
  modified: []
key-decisions:
  - "BCQ and IQL are registered now with explicit planned handlers; full trainers land in 08-02."
  - "The experiment runner resolves config and dataset contract centrally before any algorithm code executes."
patterns-established:
  - "Algorithm modules plug into one AlgorithmRegistry entry instead of custom launch scripts."
  - "Discrete action count defaults to dataset metadata when available."
requirements-completed: [RL-02]
duration: "~20min"
completed: 2026-03-29
---

# Plan 08-01 Summary: Shared Offline RL Experiment Surface

**Registry-backed offline RL launch surface for CQL, BCQ, and IQL with shared config resolution and dataset-contract wiring**

## What Was Built

### `src/mimic_sepsis_rl/training/registry.py`
Introduced the algorithm registry that defines one catalog for all supported offline RL trainers.

- `AlgorithmDefinition` records algorithm name, default config path, module path, and execution adapter.
- `AlgorithmRegistry` centralizes registration, config lookup, deep-merged device overrides, and execution dispatch.
- CQL is wired to a real shared adapter that reuses `load_replay_dataset`, `TrainingConfig`, and the existing `CQLTrainer`.
- BCQ and IQL are registered on the same surface with explicit `Phase 08-02` placeholder handlers so the launch contract is fixed before trainer implementation.

### `src/mimic_sepsis_rl/training/experiment_runner.py`
Added the shared experiment CLI for multi-algorithm comparisons.

- `resolve_experiment()` resolves an algorithm through the registry, loads its config, and rejects mismatched config/algorithm pairs.
- `DatasetContractSummary` reads replay metadata when available so action-count, reward-spec, and split-contract details can flow through one runtime surface.
- `ResolvedExperiment` standardizes algorithm execution requests and action-count resolution.
- CLI supports `--list-algorithms`, `--describe`, `--dry-run`, explicit config overrides, and runtime-device overrides through one entrypoint.

### `configs/training/bcq.yaml` and `configs/training/iql.yaml`
Created baseline configs for BCQ and IQL that mirror the CQL data and runtime contract.

- Both configs reuse `data/replay/replay_train.parquet` and `data/replay/replay_train_meta.json`.
- Runtime keys match the CQL reference surface (`device`, `seed`, `num_workers`).
- Checkpoint and logging paths are algorithm-specific so future comparison artifacts stay separated without changing the dataset contract.

### `tests/training/test_algorithm_registry.py`
Added 9 regression tests covering:

- Registry membership for BCQ, CQL, and IQL
- Shared runtime config loading with device override preservation
- Dataset and runtime parity across the CQL, BCQ, and IQL config files
- Experiment-resolution contract loading from dataset metadata
- Clear pending-algorithm failure semantics for BCQ and IQL before Phase 08-02
- CLI listing for the shared experiment runner

## Verification Results

```text
./.venv/bin/pytest -q tests/training/test_algorithm_registry.py
9 passed, 1 warning in 3.75s
```

```text
./.venv/bin/python -m mimic_sepsis_rl.training.experiment_runner --list-algorithms
Available offline RL algorithms:
- bcq [planned] config=configs/training/bcq.yaml module=mimic_sepsis_rl.training.bcq
- cql [ready] config=configs/training/cql.yaml module=mimic_sepsis_rl.training.cql
- iql [planned] config=configs/training/iql.yaml module=mimic_sepsis_rl.training.iql
```

## Task Commits

1. **Task 1: Implement shared algorithm registry and run launcher** - `0ab991a` (`feat`)
2. **Task 2: Add BCQ and IQL configs plus registry regression tests** - `f43b018` (`test`)

## Decisions Made

- Kept BCQ and IQL on the shared registry now, but marked them as planned until their trainers are implemented in Plan 08-02.
- Loaded dataset-contract metadata in the runner instead of duplicating action-count assumptions inside algorithm-specific code.
- Kept runtime override merging inside the shared registry surface so future algorithms inherit one consistent config path.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The shell environment did not expose global `python` and `pytest` commands. Verification was run through the repo-local virtual environment (`./.venv/bin/python`, `./.venv/bin/pytest`) instead.

## Next Phase Readiness

- Phase 08-02 can plug `bcq.py` and `iql.py` into the existing registry without redefining dataset loading or device handling.
- Comparison artifact work can build on the runner's resolved config and dataset-contract metadata.
- The shared experiment surface already distinguishes between ready and planned algorithms, so BCQ/IQL implementation can focus on trainer behavior rather than CLI design.

---
*Phase: 08-comparative-offline-rl-experiments*
*Completed: 2026-03-29*
