# Phase 2: Onset Anchoring and Episode Grid - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Assign one reliable `sepsis_onset_time` per usable ICU episode and generate deterministic 4-hour step windows from `onset -24h` to `onset +48h`. This phase stops at time-axis construction and does not yet build state vectors or treatment bins.

</domain>

<decisions>
## Implementation Decisions

### Onset policy
- Each ICU episode must have exactly one onset timestamp or be flagged unusable.
- Ambiguous or conflicting onset candidates should be surfaced explicitly, not silently resolved.
- Onset logic must be operationalized from Sepsis-3 rules in a reproducible way.

### Episode windowing
- Analysis window is `-24h` to `+48h` around onset.
- Timestep size is fixed at 4 hours.
- Maximum episode length is 18 steps, with truncation reasons captured when ICU discharge or death ends the window earlier.

### Data contract
- Step indexing must be deterministic and inspectable.
- Outputs should clearly separate usable, unusable, and truncated episodes.

### Claude's Discretion
- Exact file layout for onset tables versus episode-step tables.
- Whether intermediate audit tables are split by cohort/onset/window stage.

</decisions>

<specifics>
## Specific Ideas

- Preserve enough metadata to debug why an episode was dropped or truncated.
- Avoid pushing feature aggregation into this phase; keep it about time anchoring only.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 1 cohort artifact is the sole upstream dependency.

### Established Patterns
- Deterministic artifact generation and explicit logging are project-wide constraints.

### Integration Points
- Outputs from this phase feed split manifests in Phase 3 and state construction in Phase 4.

</code_context>

<deferred>
## Deferred Ideas

- Patient-level split manifests belong to Phase 3.
- Feature dictionary and imputation belong to Phase 4.
- Action/reward construction belongs to Phase 5.

</deferred>

---

*Phase: 02-onset-anchoring-and-episode-grid*
*Context gathered: 2026-03-28*
