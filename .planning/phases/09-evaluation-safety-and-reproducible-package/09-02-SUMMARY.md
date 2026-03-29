---
phase: 09-evaluation-safety-and-reproducible-package
plan: 02
subsystem: evaluation-reporting
tags: [ablations, reproducibility, reporting, benchmark-package]
requires:
  - phase: 09-evaluation-safety-and-reproducible-package
    provides: "Held-out OPE and safety diagnostics from 09-01"
provides:
  - "Stable ablation registry and comparison helpers under one benchmark schema"
  - "Reproducible reporting bundle with upstream artifact metadata, checkpoints, evaluation outputs, and backend provenance"
  - "Research-facing reproducibility documentation for reruns and audits"
affects: [09-evaluation-safety-and-reproducible-package]
tech-stack:
  added: []
  patterns: [artifact-first reproducibility, typed bundle assembly, schema-stable ablation metadata]
key-files:
  created:
    - src/mimic_sepsis_rl/evaluation/ablations.py
    - src/mimic_sepsis_rl/reporting/package.py
    - tests/evaluation/test_ablation_registry.py
    - tests/reporting/test_repro_bundle.py
    - docs/reproducibility.md
  modified:
    - src/mimic_sepsis_rl/evaluation/__init__.py
    - src/mimic_sepsis_rl/reporting/__init__.py
key-decisions:
  - "Ablation variants inherit one explicit experiment-metadata schema rooted in Phase 8 RunArtifact provenance."
  - "Reproducibility bundles require recorded checkpoint-manifest backend metadata rather than inferred backend labels."
  - "Bundle assembly includes ablation reports so robustness outputs flow into the same research package as OPE and safety metrics."
patterns-established:
  - "Ablation configuration stays registry-driven and serializable before any concrete trainer orchestration is added."
  - "Reproducibility validation happens at bundle-build time by checking required metadata keys and cross-artifact backend consistency."
requirements-completed: [EXP-01, REPR-01]
completed: 2026-03-29
---

# Plan 09-02 Summary: Ablation Registry and Reproducible Reporting Bundle

**Phase 9 now ends with a stable ablation surface and a research-facing reproducibility package instead of isolated evaluation objects**

## What Was Built

### `src/mimic_sepsis_rl/evaluation/ablations.py`
- Added a typed ablation registry covering the five required axes: reward shaping, action granularity, timestep choice, missingness flags, and feature subsets.
- Added `AblationExperimentMetadata`, which derives a stable benchmark schema from Phase 8 `RunArtifact` provenance and carries benchmark version, dataset contract, checkpoint path, and backend context across all variants.
- Added controlled comparison helpers so a dimension report refuses partial metric tables and keeps all variants attributable to one benchmark version.

### `src/mimic_sepsis_rl/reporting/package.py`
- Added a reproducibility bundle assembler with typed records for upstream artifacts, checkpoint-manifest backend metadata, evaluation summaries, and per-run reporting entries.
- Added validation that required cohort, feature, split, action, reward, checkpoint, evaluation, and backend metadata exist before a bundle is considered valid.
- Added default rerun instructions so the package can point researchers back to the correct training config and requested device.

### Tests and Docs
- `tests/evaluation/test_ablation_registry.py` verifies required ablation coverage, shared metadata schema, explicit benchmark versioning, and strict comparison completeness.
- `tests/reporting/test_repro_bundle.py` verifies bundle assembly captures required metadata fields, includes recorded backend provenance, and carries ablation reports into the final package.
- `docs/reproducibility.md` documents required bundle contents, rerun order, and an audit checklist for thesis or paper review.

## Verification Results

```text
./.venv/bin/pytest -q tests/evaluation/test_ablation_registry.py tests/reporting/test_repro_bundle.py
6 passed in 1.29s

./.venv/bin/pytest -q tests/evaluation/test_ope_pipeline.py tests/evaluation/test_safety_checks.py
5 passed in 1.25s
```

## Deviations from Plan

- **[Rule 2 - Missing Critical] Package export surfaces** — Added `src/mimic_sepsis_rl/evaluation/__init__.py` exports and `src/mimic_sepsis_rl/reporting/__init__.py` so the new modules are importable through package-level APIs instead of only by direct module paths.

## Issues Encountered

- None. The new ablation and reporting layers fit the existing Phase 8/9 artifact contracts without needing architectural changes in training code.

## Next Phase Readiness

- Phase 9 now has defensible robustness and reproducibility outputs alongside OPE and safety diagnostics.
- Future execution work can wire concrete ablation launches through the shared experiment runner without redesigning the reporting schema.
- The roadmap now ends with a bundle surface suitable for reruns, audits, and research appendices.
