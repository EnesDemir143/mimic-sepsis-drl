# Phase 5: Treatment and Reward Encoding - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Encode clinician treatments and outcomes into the fixed MDP action-reward contract: 25 discrete actions from vasopressor and IV fluid bins plus terminal and intermediate rewards. This phase should freeze the action and reward definitions used by all later baselines and RL policies.

</domain>

<decisions>
## Implementation Decisions

### Action encoding
- Vasopressors must be standardized to norepinephrine-equivalent dosing.
- IV fluids are aggregated over the 4-hour decision window.
- Action space is fixed to 5x5 bins for vasopressor and fluid, with train-only bin learning.
- Zero-dose handling should be explicit, not inferred indirectly.

### Reward policy
- Terminal 90-day mortality reward is required.
- Intermediate shaping should start conservative, centered on SOFA delta, with lactate/MAP shaping added in a controlled way.
- Reward tables and distributions should be inspectable before training begins.

### Leakage and reproducibility
- Action thresholds must be fit on train only and frozen for reuse.
- Reward computation should be deterministic and versioned.

### Claude's Discretion
- Exact artifact names and whether reward diagnostics live beside the reward table or in a reporting module.

</decisions>

<specifics>
## Specific Ideas

- Avoid over-shaping the reward too early; preserve a sparse baseline path.
- Make action mapping easy to decode back into `(vaso_bin, fluid_bin)` for analysis and sanity checks.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 3 split manifests constrain train-only bin fitting.
- Phase 4 state tables define the step-level frame that actions and rewards align against.

### Established Patterns
- Action and reward contracts must be frozen before baselines or RL training.

### Integration Points
- Phase 6 transition export and all training phases rely on the artifacts from this phase.

</code_context>

<deferred>
## Deferred Ideas

- Replay buffer export belongs to Phase 6.
- CQL/BCQ/IQL training belongs to Phases 7 and 8.
- OPE and ablation reporting belong to Phase 9.

</deferred>

---

*Phase: 05-treatment-and-reward-encoding*
*Context gathered: 2026-03-28*
