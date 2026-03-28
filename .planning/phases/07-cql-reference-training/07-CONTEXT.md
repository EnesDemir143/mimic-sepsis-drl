# Phase 7: CQL Reference Training - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Train the first conservative offline RL reference policy, CQL, on the finalized dataset contract while proving the same runtime path works on Apple Silicon `MPS` and NVIDIA `CUDA`. This phase sets the standard training abstraction used by later algorithms.

</domain>

<decisions>
## Implementation Decisions

### Training objective
- CQL is the first offline RL algorithm and should act as the reference implementation.
- Training must use the frozen state/action/reward/split contract from earlier phases.
- Checkpoints, training curves, and run configs are required outputs.

### Platform portability
- Core training code must remain device-agnostic via PyTorch device selection.
- The supported accelerator targets are Apple Silicon `Metal/MPS` and NVIDIA `CUDA`, with documented CPU fallback when required.
- CUDA-only kernels or code paths are not allowed to become the main implementation path.

### Stack flexibility
- The exact trainer surface can be d3rlpy or custom PyTorch, but it must support the portability constraint above.
- Unsupported `MPS` operations must be identified early and handled explicitly rather than discovered during final experiments.

### Claude's Discretion
- Exact config layout, launcher CLI shape, and experiment logging integration.

</decisions>

<specifics>
## Specific Ideas

- Treat this phase as both an algorithm milestone and a platform-compatibility milestone.
- Device selection should be config-driven or auto-detected, not hard-coded in scripts.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 6 replay-ready datasets and baselines define the training surface.

### Established Patterns
- Training code must not redefine preprocessing or binning.
- Cross-platform runtime support is now a first-class requirement.

### Integration Points
- Phase 8 BCQ/IQL runs should reuse the device abstraction and training utilities created here.

</code_context>

<deferred>
## Deferred Ideas

- BCQ and IQL comparative experiments belong to Phase 8.
- OPE, safety overlays, and ablations belong to Phase 9.

</deferred>

---

*Phase: 07-cql-reference-training*
*Context gathered: 2026-03-28*
