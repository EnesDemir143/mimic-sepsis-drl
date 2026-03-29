# Evaluation Protocol

**Phase:** 09 – Evaluation, Safety, and Reproducible Package  
**Requirements:** OPE-01, SAFE-01, SAFE-02  
**Version:** 1.0.0

---

## Scope

This protocol defines how trained offline RL policies are reviewed after Phase 8.
The goal is to make retrospective claims measurable, reproducible, and bounded by
clinical-safety diagnostics.

> **Interpretation boundary:** All outputs in this phase are retrospective only.
> WIS, ESS, FQE, clinician sanity checks, and support diagnostics do **not**
> establish prospective bedside efficacy.

---

## Required Inputs

Evaluation must consume frozen artifacts only:

- Phase 8 `RunArtifact` outputs from `mimic_sepsis_rl.training.comparison`
- held-out dataset metadata for the evaluation split
- held-out trajectories with logged behavior-policy probabilities
- frozen FQE action-value outputs fitted on a non-held-out split
- action-bin artifacts from `mimic_sepsis_rl.mdp.actions.bins`

The evaluation layer must not fit or refit any parameter on held-out data.

---

## OPE Metrics

### Weighted Importance Sampling (WIS)

- Uses held-out episode returns and logged behavior-policy probabilities
- For deterministic target policies, a step contributes only when the policy
  matches the logged clinician action
- Report WIS together with ESS so low-support results are visible

### Effective Sample Size (ESS)

- Compute ESS from the same episode weights used by WIS
- Treat low ESS as a warning sign, not as strong evidence for or against a policy

### Fitted Q Evaluation (FQE)

- FQE values must come from a frozen estimator trained on train/validation data
- Phase 9 only scores held-out initial states with those frozen values
- If an FQE artifact is labeled as `test`, `heldout`, or equivalent, evaluation
  should fail immediately

---

## Clinical Safety Review

Before a policy is described as clinically plausible, review all of the following:

1. Clinician agreement summary
2. Ranked sanity-check cases for manual inspection
3. 5×5 action-frequency heatmaps for clinician vs policy behavior
4. Subgroup summaries for high-risk cohorts
5. Support-aware warnings for poorly supported policy actions

Recommended subgroup axes include shock burden, renal dysfunction, vasopressor
exposure, or other clinically meaningful cohorts available in the held-out data.

---

## Support-Aware Warning Rules

Support diagnostics should flag any policy action that falls below configured
behavior-support thresholds such as:

- low estimated clinician probability for the policy action at that state
- low behavior count for the state-action neighborhood
- repeated overrides in low-support regions

Warnings should be carried into reports beside the top-line OPE metrics.

---

## Review Workflow

1. Load the frozen Phase 8 run artifact for one algorithm.
2. Load held-out trajectories and validate their dataset contract.
3. Run held-out OPE with WIS, ESS, and frozen FQE outputs.
4. Build clinician sanity checks, heatmaps, subgroup summaries, and support warnings.
5. Report metrics and warnings together.
6. State explicitly that the result is retrospective evidence only.

---

## Verification

The regression suite for this protocol is:

```bash
pytest -q tests/evaluation/test_ope_pipeline.py tests/evaluation/test_safety_checks.py
```

These tests enforce:

- held-out OPE metric wiring
- required behavior-policy probabilities
- the rule that FQE is never fitted on held-out data
- clinician heatmap, subgroup, and support-warning generation
