# Plan 06-02 Summary: Baseline Benchmarks

**Completed:** 2026-03-29
**Duration:** ~5 minutes
**Status:** ✅ All tasks complete

## What Was Built

### `src/mimic_sepsis_rl/baselines/clinician.py`
Clinician-action replay baseline: reports aggregate metrics over the
observed clinician actions without learning any policy. Serves as the
direct comparator for RL policies.

### `src/mimic_sepsis_rl/baselines/no_treatment.py`
Static policy that always selects action 0 (no treatment). Sets the
absolute floor — any learned policy scoring worse is pathologically bad.

### `src/mimic_sepsis_rl/baselines/behavior_cloning.py`
Lightweight multinomial logistic regression (softmax classifier) that
learns to predict clinician actions from state features. No external
ML dependencies — pure Python gradient descent. Includes:
- `SoftmaxClassifier` with predict/train_step/accuracy methods
- `train_behavior_cloning()` convenience function
- CLI with `--dry-run` validation

### `tests/baselines/test_baselines.py`
21 regression tests covering:
- Clinician: episode returns, action distribution, mortality rate
- No-treatment: fixed action, metric computation
- Behavior cloning: training, accuracy, prediction
- Empty input handling for all baselines
- Dataset compatibility: all baselines consume the same transitions
- Result serialisation

### `docs/baseline_benchmarks.md`
Documentation covering:
- Each baseline's purpose and interpretation
- Output schema
- Comparison table template
- Data contract requirements

## Verification

- ✅ `pytest -q tests/baselines/test_baselines.py` → 21 passed
- ✅ `python -m mimic_sepsis_rl.baselines.behavior_cloning --dry-run` → PASSED
- ✅ All baselines consume the shared transition dataset contract
- ✅ Full test suite: 376 tests passed
