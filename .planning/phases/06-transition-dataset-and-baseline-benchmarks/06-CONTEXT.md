# Phase 6: Transition Dataset and Baseline Benchmarks - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Convert frozen state, action, and reward artifacts into replay-ready transition datasets and establish comparison baselines on the exact same contract. This phase prepares the modeling surface without yet running the main offline RL experiments.

</domain>

<decisions>
## Implementation Decisions

### Transition contract
- Export `(s_t, a_t, r_t, s_t+1, done)` transitions and episode-based replay structures.
- Transition generation must preserve phase-3 split boundaries and phase-5 action/reward definitions.
- Dataset artifacts should be reusable by multiple algorithms without per-model mutation.

### Baseline scope
- Required baselines are clinician behavior, no-treatment, and behavior cloning.
- Baselines should read the same artifacts later used by CQL, BCQ, and IQL.
- Benchmark outputs should be easy to compare against later policy runs.

### Artifact quality
- Contract validation and shape checks are mandatory before training starts.
- Saved metadata should record which cohort, split, state, action, and reward versions produced the dataset.

### Claude's Discretion
- Exact replay buffer serialization and baseline result table format.

</decisions>

<specifics>
## Specific Ideas

- Treat this phase as the final freeze point before serious model training.
- Make baseline commands cheap enough to rerun after pipeline changes.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 4 state tables and Phase 5 action/reward artifacts are the direct inputs.

### Established Patterns
- No model should be allowed to bypass the shared transition dataset contract.

### Integration Points
- Phase 7 and 8 training code should consume outputs from this phase without redefining preprocessing.

</code_context>

<deferred>
## Deferred Ideas

- CQL training belongs to Phase 7.
- BCQ/IQL comparison belongs to Phase 8.
- OPE and safety reporting belong to Phase 9.

</deferred>

---

*Phase: 06-transition-dataset-and-baseline-benchmarks*
*Context gathered: 2026-03-28*
