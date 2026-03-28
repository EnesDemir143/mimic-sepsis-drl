# Phase 3: Split Manifests and Leakage Boundaries - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Freeze patient-level train, validation, and test manifests early enough that every learned transform downstream respects those boundaries. This phase is about leakage control and split reproducibility, not model training.

</domain>

<decisions>
## Implementation Decisions

### Split policy
- Splits must be patient-level, not row-level or stay-level across shared patients.
- Fixed seeds and saved manifests are required.
- Duplicate-patient checks across splits are mandatory.

### Leakage boundaries
- Any learned transform downstream must fit on train only.
- Split artifacts should be the canonical boundary reused by state, action, reward, and model stages.
- The phase should include automated checks or assertions for leakage-sensitive assumptions.

### Scope guardrails
- Temporal holdout can be prepared as a future extension, but the required v1 output is the internal train/val/test split contract.

### Claude's Discretion
- Exact manifest file format and helper utilities.
- How to package split diagnostics and balance summaries.

</decisions>

<specifics>
## Specific Ideas

- This phase exists specifically to prevent accidental full-dataset fitting later.
- Make split manifests reusable from CLI/config without ad hoc rewrites.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 2 episode artifacts define the set of eligible episode keys.

### Established Patterns
- Train-only fit rules are locked project constraints.

### Integration Points
- Phase 4 preprocessing, Phase 5 binning, and all model phases must consume these manifests directly.

</code_context>

<deferred>
## Deferred Ideas

- Feature preprocessing belongs to Phase 4.
- Action bin learning belongs to Phase 5.
- Model training belongs to Phases 7 and 8.

</deferred>

---

*Phase: 03-split-manifests-and-leakage-boundaries*
*Context gathered: 2026-03-28*
