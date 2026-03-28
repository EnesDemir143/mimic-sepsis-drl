---
phase: 04-state-representation-pipeline
plan: 02
status: completed
commit: uncommitted
duration_min: 0
---

# Summary: 04-02 — Deterministic State Building and Train-Only Preprocessing

## What Was Built

Implemented the executable Phase 4 state pipeline beyond the feature-contract scaffold:

- A new split-aware state-table builder now materializes deterministic step-level vectors from `StepWindowData` while honoring the documented missing-data policy.
- A new preprocessing module now fits train-only median and normalization artifacts, persists them, and replays the same transforms on validation and test rows.
- Dedicated regression suites now verify determinism, episode-boundary fallback behavior, missingness flags, serialization round-trips, and train-only leakage boundaries.

## Artifacts Produced

| Path | Description | Lines |
|------|-------------|------:|
| `src/mimic_sepsis_rl/mdp/features/builder.py` | Deterministic split-aware state-table builder | 275 |
| `src/mimic_sepsis_rl/mdp/preprocessing.py` | Train-only preprocessing artifacts, save/load, and transform utilities | 245 |
| `tests/mdp/test_state_builder.py` | Determinism, imputation, manifest, and missingness regressions | 212 |
| `tests/mdp/test_preprocessing.py` | Train-only fit, serialization, clipping, and normalization regressions | 167 |

## Verification Results

### Phase 4 regression targets
```bash
./.venv/bin/python -m pytest -q tests/mdp/test_state_builder.py tests/mdp/test_preprocessing.py
```
Result:
```text
10 passed in 0.11s
```

### Existing MDP contract regression
```bash
./.venv/bin/python -m pytest -q tests/mdp/test_feature_dictionary.py
```
Result:
```text
139 passed in 0.18s
```

### Combined verification
```bash
./.venv/bin/python -m pytest -q tests/mdp/test_state_builder.py tests/mdp/test_preprocessing.py tests/mdp/test_feature_dictionary.py
```
Result:
```text
150 passed in 0.28s
```

## Design Decisions

- **Manifest-aware builder:** `StateTableBuilder` resolves `subject_id` and split labels directly against `SplitManifest`, so train/validation/test membership is explicit at state-build time.
- **Deterministic ordering:** Rows are grouped by `stay_id`, ordered by `step_index`, and emitted in stable sorted order independent of input window order.
- **Imputation contract preserved:** Raw measurements are used first, forward-fill stays episode-local, and train medians are only consulted when no prior exists.
- **Frozen preprocessing artifacts:** `PreprocessingArtifacts` stores per-feature train medians, clip bounds, and normalization stats in a JSON-serializable bundle for deterministic replay.
- **Transform discipline:** Null fill, clipping, and z-score normalization happen in a fixed order and are replayed unchanged across splits.

## Deviations from Plan

**[Rule 1 - Bug] `weight_kg` routed to the wrong extractor**
- Found during: Task 1 while wiring the new full-registry state builder
- Issue: `weight_kg` was declared as `source_table="chartevents"` in the feature contract, but runtime extraction for demographic/static values expected it to come from the static context. That meant a full-registry build could route `weight_kg` through `ChartEventsExtractor` instead of `DemographicsExtractor`.
- Fix: Reclassified `weight_kg` as a demographics/context feature (`source_table="patients"`, empty `item_ids`) and added a regression test proving builder output reads the static `weight_kg` value.
- Files modified: `src/mimic_sepsis_rl/mdp/features/dictionary.py`, `tests/mdp/test_state_builder.py`
- Verification: All targeted and existing MDP tests passed after the fix.

**Total deviations:** 1 auto-fixed (Rule 1 - Bug). **Impact:** Positive — removed a silent contract/runtime mismatch before downstream phases consume the full state registry.

## STAT-02 and STAT-03 Satisfaction

- **STAT-02:** State vectors now encode forward-fill, train-median fallback, and optional missingness flags with deterministic per-episode behavior and explicit regression coverage.
- **STAT-03:** Preprocessing artifacts are fit from train rows only, persisted independently of validation/test data, and replayed deterministically on subsequent transforms.

## Next Phase Readiness

Phase 5 can now consume model-ready continuous state vectors without redefining missing-data or normalization rules:

- `builder.py` produces repeatable step-level wide state tables from episode windows and split manifests.
- `preprocessing.py` freezes the train-only artifact boundary required by action binning and later model training.
- The new regression suites protect against leakage regressions at both the state-build and preprocessing layers.

## Self-Check: PASSED

- [x] `tests/mdp/test_state_builder.py` passes
- [x] `tests/mdp/test_preprocessing.py` passes
- [x] Train-only preprocessing tests prove validation/test rows do not affect fitted artifacts
- [x] State builder output remains deterministic for the same windows and split manifest
