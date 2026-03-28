---
phase: 04-state-representation-pipeline
plan: 01
status: completed
commit: 4a29b72
duration_min: 35
---

# Summary: 04-01 — Feature Contract and Extraction Scaffolding

## What Was Built

Defined the complete v1 state-feature contract and extraction surface for the MIMIC Sepsis Offline RL pipeline. The feature dictionary is now machine-readable, executable, and documented — every state field that will appear in a step-level state vector has a stable, inspectable descriptor outside the code.

## Artifacts Produced

| Path | Description | Lines |
|------|-------------|------:|
| `src/mimic_sepsis_rl/mdp/features/dictionary.py` | Executable feature contract (FeatureSpec, enums, registry, CLI) | ~1 230 |
| `src/mimic_sepsis_rl/mdp/features/extractors.py` | Extraction scaffolding (BaseWindowExtractor, 5 concrete extractors, StateVectorBuilder) | ~776 |
| `src/mimic_sepsis_rl/mdp/features/__init__.py` | Public API surface for the features package | 36 |
| `src/mimic_sepsis_rl/mdp/__init__.py` | MDP package init | 8 |
| `configs/features/default.yaml` | Externalized feature selection and tuning parameters | 157 |
| `docs/feature_dictionary.md` | Research-facing feature specification (clinical rationale, MIMIC-IV item IDs, leakage controls) | ~929 |
| `tests/mdp/test_feature_dictionary.py` | 139-test regression suite | ~1 174 |
| `tests/mdp/__init__.py` | Test package init | — |

## Feature Registry — v1.0.0

**37 features total** across 7 clinical families:

| Family | Features |
|--------|--------:|
| Vitals (chartevents) | 10 |
| Labs — Blood Gas (labevents) | 6 |
| Labs — Chemistry (labevents) | 9 |
| Labs — Haematology (labevents) | 4 |
| Treatments (inputevents / outputevents) | 3 |
| Demographics (patients / chartevents) | 2 |
| Derived (pipeline-computed) | 3 |

Key aggregation choices:
- `lactate` → `max` (worst-case within window)
- `spo2` → `min` (nadir oxygenation)
- `cum_iv_fluid_ml` / `cum_vasopressor_dose_nor_equiv` → `cumulative` (episode start to window end)
- `urine_output_4h` → `sum` (per-window total)
- All other features → `last` (most recent clinical picture)

25 features emit a paired `{feature_id}_missing` binary flag when imputed.

Missing strategy distribution: `forward_fill` (24), `median_train` (9), `zero` (3), `normal_value` (1).

## Verification Results

### pytest
```
139 passed in 0.24s
```

### CLI validation
```
INFO: Loaded config from configs/features/default.yaml; 37 features active.
INFO: Validation PASSED — 37 features in registry.
```

Both verification gates from the plan passed cleanly.

## STAT-01 Satisfaction

The feature dictionary satisfies STAT-01 by providing:
1. **Machine-readable contract** — every field has `source_table`, `item_ids`, `unit`, `aggregation`, `valid_low/high`, `clip_low/high`, `missing_strategy`, and `normal_value` — all inspectable without reading extraction code.
2. **Externalized config** — `configs/features/default.yaml` exposes feature selection, missingness flag overrides, imputation policy, clipping, and output artefact settings without requiring code edits.
3. **Documented spec** — `docs/feature_dictionary.md` provides the full clinical rationale, MIMIC-IV item ID tables, leakage control notes, and state vector layout for thesis/paper appendices.
4. **Extractors wired to contract** — `StateVectorBuilder` reads `FeatureSpec` at runtime; no clinical constants are hardcoded in extraction logic.

## Deviations from Plan

**[Rule 1 – Bug] Empty DataFrame itemid filter crash**
- Found during: Task 2 (test authoring / test run)
- Issue: `BaseWindowExtractor.extract()` called `window_df.filter(pl.col("itemid").is_in(...))` on an empty `pl.DataFrame()` with no columns, raising `ColumnNotFoundError`.
- Fix: Added guard — skip item-ID filter when DataFrame is empty or lacks `itemid` column; fall through to `pl.DataFrame()` so imputation takes over.
- Files modified: `extractors.py` (3 lines)
- Verification: All 139 tests pass.
- Commit: 4a29b72

**[Rule 1 – Bug] ValueError message case mismatch in test**
- Found during: Task 2 (test run)
- Issue: `load_feature_registry()` raised `"Unknown feature IDs …"` (capital U) but the test used `match="unknown"` (lowercase); pytest `re.search` is case-sensitive.
- Fix: Lowercased the error prefix to `"unknown feature IDs …"` in `dictionary.py`.
- Files modified: `dictionary.py` (1 line)
- Verification: All 139 tests pass.
- Commit: 4a29b72

**Total deviations:** 2 auto-fixed (Rule 1 — Bug). **Impact:** None — both were caught during test authoring before any downstream code consumed the API.

## Next Phase Readiness

Phase 4 now has a stable feature surface. Plan 04-02 (imputation, normalization, state-table builder) can proceed immediately:

- `FEATURE_REGISTRY` provides every field name, aggregation rule, and clip bound needed to drive the normalization step.
- `StateVectorBuilder` scaffolding is in place; Plan 04-02 will wire real MIMIC-IV window DataFrames and fit train-only scalers.
- `configs/features/default.yaml` already references `data/processed/features/train_medians.json` — Plan 04-02 should write this file.
- `docs/feature_dictionary.md` leakage section documents the train-only median/scaler constraint for Plan 04-02 implementors.

No blockers identified.

## Self-Check: PASSED

- [x] `pytest -q tests/mdp/test_feature_dictionary.py` → 139 passed
- [x] `python -m mimic_sepsis_rl.mdp.features.dictionary --config configs/features/default.yaml --validate` → PASSED
- [x] `docs/feature_dictionary.md` matches executable contract categories (7 families, same feature IDs)
- [x] Commit `4a29b72` present in `git log`
- [x] All 8 files listed in `files_modified` exist on disk