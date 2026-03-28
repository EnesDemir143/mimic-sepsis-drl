# Phase 1: Cohort Definition - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Define the eligible adult ICU Sepsis-3 study population from MIMIC-IV and produce an auditable cohort artifact with explicit inclusion and exclusion rules. This phase does not assign onset times, build episode windows, or create RL-ready features.

</domain>

<decisions>
## Implementation Decisions

### Study population
- Unit of analysis is the ICU episode / ICU stay.
- Cohort is restricted to adult ICU patients only.
- v1 scope is limited to Sepsis-3 episodes within MIMIC-IV.

### Output expectations
- The phase must produce a reproducible cohort artifact, not only exploratory SQL.
- Inclusion and exclusion reasons must be inspectable for each dropped stay.
- Cohort generation should be deterministic for the same source data and config.

### Auditability
- SQL or extraction logic must be preserved in versioned project artifacts.
- Manual notebook-only filtering is not acceptable for the final path.

### Claude's Discretion
- Exact module boundaries for SQL extraction versus Python post-processing.
- Naming of intermediate artifacts and manifests.

</decisions>

<specifics>
## Specific Ideas

- Keep this phase focused on who is eligible, not yet when sepsis onset occurs.
- Make exclusion logging easy to review because onset and downstream MDP steps depend on this population being trustworthy.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- No implementation code yet — planning should assume a greenfield data pipeline.

### Established Patterns
- Artifact-first planning is already locked at the project level.

### Integration Points
- Outputs from this phase feed Phase 2 onset anchoring and Phase 3 split generation.

</code_context>

<deferred>
## Deferred Ideas

- Sepsis onset assignment belongs to Phase 2.
- Episode window generation belongs to Phase 2.
- Train/validation/test manifests belong to Phase 3.

</deferred>

---

*Phase: 01-cohort-definition*
*Context gathered: 2026-03-28*
