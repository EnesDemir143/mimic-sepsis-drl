---
phase: 09-evaluation-safety-and-reproducible-package
plan: 01
subsystem: evaluation
tags: [ope, wis, ess, fqe, safety, clinician-review]
requires:
  - phase: 08-comparative-offline-rl-experiments
    provides: "Standardized run artifacts, dataset contracts, and frozen training outputs"
provides:
  - "Held-out OPE surface for WIS, ESS, and frozen FQE scoring"
  - "Clinical safety review helpers with heatmaps, subgroup summaries, and support warnings"
  - "Evaluation protocol documentation with retrospective-only interpretation guardrails"
affects: [09-evaluation-safety-and-reproducible-package]
tech-stack:
  added: []
  patterns: [artifact-first evaluation, held-out-only scoring, support-aware clinical review]
key-files:
  created:
    - src/mimic_sepsis_rl/evaluation/__init__.py
    - src/mimic_sepsis_rl/evaluation/ope.py
    - src/mimic_sepsis_rl/evaluation/safety.py
    - tests/evaluation/test_ope_pipeline.py
    - tests/evaluation/test_safety_checks.py
    - docs/evaluation_protocol.md
  modified: []
key-decisions:
  - "FQE is represented as frozen initial-state action values and is rejected if labeled as a held-out fit."
  - "OPE compares training and held-out dataset contracts so split drift is explicit before metrics are interpreted."
  - "Safety review focuses on clinician agreement, action heatmaps, subgroup summaries, and support-aware warnings in one reusable report surface."
patterns-established:
  - "Evaluation consumes Phase 8 RunArtifact objects instead of parsing checkpoint directories ad hoc."
  - "Held-out trajectories must carry logged behavior-policy probabilities before WIS/ESS can run."
requirements-completed: [OPE-01, SAFE-01, SAFE-02]
duration: "~20min"
completed: 2026-03-29
---

# Plan 09-01 Summary: Held-Out OPE and Safety Diagnostics

**Phase 9 now has a reusable evaluation layer that scores frozen policies on held-out data and pairs OPE outputs with clinical safety diagnostics**

## What Was Built

### `src/mimic_sepsis_rl/evaluation/ope.py`
- Added held-out trajectory types, WIS/ESS computation, and a normalized `PolicyOPEReport`.
- Wired evaluation to Phase 8 `RunArtifact` objects so algorithm metadata and dataset-contract provenance flow into the report.
- Added `FrozenFQEOutputs`, which only accepts non-held-out fits and scores held-out initial states without refitting.
- Added dataset-contract compatibility checks between train and held-out artifacts before interpreting results.

### `src/mimic_sepsis_rl/evaluation/safety.py`
- Added row-wise clinician vs policy review records plus a combined `SafetyReviewReport`.
- Implemented 5×5 action heatmaps, policy-minus-clinician delta heatmaps, subgroup summaries, and ranked clinician sanity cases.
- Added support-aware warning generation so weakly supported policy actions are surfaced beside OPE scores.
- Added a helper that builds safety rows directly from held-out episodes and a policy surface.

### Tests and Docs
- `tests/evaluation/test_ope_pipeline.py` verifies WIS/ESS/FQE wiring, required behavior-policy probabilities, and the rule that FQE never fits on held-out data.
- `tests/evaluation/test_safety_checks.py` verifies safety-row construction, heatmaps, subgroup summaries, and support-warning severity.
- `docs/evaluation_protocol.md` documents the required inputs, retrospective-only boundary, and the review sequence for OPE plus safety checks.

## Verification Results

```text
./.venv/bin/pytest -q tests/evaluation/test_ope_pipeline.py tests/evaluation/test_safety_checks.py
5 passed in 1.77s
```

## Deviations from Plan

None.

## Issues Encountered

- The shell environment uses the repo-local virtual environment for verification, so tests were run through `./.venv/bin/pytest`.

## Next Phase Readiness

- Phase 09-02 can build ablations and reporting bundles on top of the normalized OPE and safety outputs added here.
- Evaluation reports now carry enough structure for downstream packaging without re-reading raw checkpoint internals.

---
*Phase: 09-evaluation-safety-and-reproducible-package*
*Completed: 2026-03-29*
