# Phase 4: State Representation Pipeline - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Build deterministic continuous state vectors for each 4-hour episode step from a documented feature contract, including imputation, clipping, and normalization policies. This phase does not yet create action bins or rewards.

</domain>

<decisions>
## Implementation Decisions

### Feature contract
- A feature dictionary must describe source, unit, aggregation rule, and acceptable ranges.
- State vectors should reflect information available at the decision point, not future leakage.
- Missingness handling should be explicit and documented per feature family.

### Preprocessing policy
- Forward fill is the first-line time-series imputation method.
- Median fallback is allowed where no prior value exists.
- Optional missingness flags should be supported where clinically useful.
- Clipping and normalization must be fit on the train split only.

### Artifact expectations
- Produce reusable state tables and preprocessing artifacts.
- Deterministic reruns are required for the same cohort, windowing, and split manifests.

### Claude's Discretion
- Exact feature module boundaries and serialization format for preprocessors.
- Whether missingness flags are bundled with the main state table or as parallel metadata.

</decisions>

<specifics>
## Specific Ideas

- Keep feature engineering explicit enough for thesis/paper appendices.
- Wide clinical tables should favor columnar processing over notebook-driven manual joins.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 2 provides episode steps; Phase 3 provides train-only boundaries.

### Established Patterns
- Artifact-first, leakage-safe preprocessing is non-negotiable.

### Integration Points
- Phase 5 action/reward builders and later training phases depend on the frozen state table contract.

</code_context>

<deferred>
## Deferred Ideas

- Treatment action encoding belongs to Phase 5.
- Replay buffer export belongs to Phase 6.
- Model training belongs to later phases.

</deferred>

---

*Phase: 04-state-representation-pipeline*
*Context gathered: 2026-03-28*
