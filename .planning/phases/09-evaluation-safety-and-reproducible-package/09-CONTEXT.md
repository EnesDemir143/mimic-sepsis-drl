# Phase 9: Evaluation, Safety, and Reproducible Package - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Evaluate trained policies with offline policy evaluation, clinician sanity checks, safety diagnostics, ablations, and reproducible reporting artifacts. This phase packages the benchmark into a defensible research output without claiming prospective clinical efficacy.

</domain>

<decisions>
## Implementation Decisions

### Evaluation contract
- Required OPE outputs are WIS, ESS, and FQE on held-out data.
- Evaluation should consume frozen model checkpoints and held-out artifacts only.
- Reported results must include enough metadata to reproduce the exact runs and accelerator backend used.

### Safety and plausibility
- Clinician sanity review, action-frequency heatmaps, and high-risk subgroup analysis are required before calling a policy clinically plausible.
- Support-aware warnings or constraints should surface poorly supported state-action regions.
- OPE is retrospective evidence only and must not be framed as bedside efficacy.

### Robustness and packaging
- Required ablations include reward shaping, action granularity, timestep choice, missingness flags, and feature subsets.
- Final deliverables should be suitable for thesis/paper method and results sections.

### Claude's Discretion
- Exact report directory structure, plot naming, and packaging format.

</decisions>

<specifics>
## Specific Ideas

- Pair top-line metrics with qualitative review and support diagnostics.
- Record whether a result came from `MPS`, `CUDA`, or CPU fallback so cross-device reproducibility is explicit.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 8 experiment artifacts and Phase 7 runtime metadata feed directly into this phase.

### Established Patterns
- Retrospective claims only; no prospective efficacy framing.

### Integration Points
- Final outputs should update project artifacts in a way that later verification and reporting workflows can consume.

</code_context>

<deferred>
## Deferred Ideas

- External dataset validation remains a future requirement outside this first roadmap.
- Clinician-facing UI work remains deferred beyond this roadmap.

</deferred>

---

*Phase: 09-evaluation-safety-and-reproducible-package*
*Context gathered: 2026-03-28*
