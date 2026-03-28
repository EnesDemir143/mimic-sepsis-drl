# Phase 8: Comparative Offline RL Experiments - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the reference training pipeline to BCQ and IQL, compare them fairly against CQL, and save complete run artifacts without changing the dataset contract. This phase is about controlled algorithm comparison, not redefining the benchmark.

</domain>

<decisions>
## Implementation Decisions

### Comparison policy
- BCQ and IQL must train on the same action map, reward function, and split manifests used by CQL.
- Algorithm comparisons should isolate algorithm behavior, not data-contract drift.
- Every run needs checkpoints, metrics, and config provenance.

### Shared runtime
- BCQ and IQL must reuse the device abstraction introduced in Phase 7.
- Cross-platform support for `MPS` and `CUDA` remains required.
- Common training helpers should be factored to avoid three divergent trainer stacks.

### Reporting expectations
- Comparison tables and curves should be easy to consume in the final evaluation package.
- Any backend-specific caveat should be captured as run metadata, not hidden in notebooks.

### Claude's Discretion
- Exact sweep structure, hyperparameter organization, and experiment naming conventions.

</decisions>

<specifics>
## Specific Ideas

- Favor one shared training framework with algorithm plugs over three disconnected scripts.
- Keep enough metadata to explain if one backend needed a documented fallback.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 7 device abstraction and CQL training utilities should be reused directly.

### Established Patterns
- Frozen dataset contract and reproducible run metadata are already locked.

### Integration Points
- Phase 9 evaluation should consume outputs from this phase with minimal reshaping.

</code_context>

<deferred>
## Deferred Ideas

- OPE, clinician sanity review, safety warnings, and ablations belong to Phase 9.

</deferred>

---

*Phase: 08-comparative-offline-rl-experiments*
*Context gathered: 2026-03-28*
