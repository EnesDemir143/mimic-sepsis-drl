# Feature Research

**Domain:** Offline reinforcement learning benchmark for sepsis treatment
**Researched:** 2026-03-28
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Cohort extraction with explicit Sepsis-3 rules | Without a defensible cohort, no result is publishable | HIGH | Must be versioned and reproducible |
| Onset-centered episode construction | The MDP timeline is unusable if onset logic drifts | HIGH | Single onset per ICU stay, deterministic windowing |
| State/action/reward builders | Core offline RL inputs must be explicit and replayable | HIGH | Needs clear feature, action, and reward specs |
| Patient-level split management | Leakage invalidates claims immediately | MEDIUM | Split manifests should be saved as artifacts |
| Baseline and offline RL training pipelines | Users need a fair comparison, not only one policy result | HIGH | Same dataset contracts across all models |
| Offline policy evaluation and diagnostics | Research users expect more than reward curves | HIGH | WIS, ESS, FQE, sanity checks, and plots are mandatory |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Support-aware safety filters | Makes recommendations easier to defend clinically | MEDIUM | Can be added without changing the base MDP contract |
| Clinician-oriented qualitative review pack | Bridges pure ML metrics and clinical plausibility | MEDIUM | Case reviews and subgroup slices matter more than a flashy UI |
| Temporal holdout or external validation mode | Strengthens claims beyond one internal split | HIGH | Best added after the core pipeline is stable |
| Structured experiment registry | Makes ablations and thesis reporting faster and cleaner | MEDIUM | Strong payoff once multiple reward/action variants exist |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Real-time clinician recommender | Feels like the end goal | Unsafe and unjustified before retrospective validation | Keep scope to offline research artifacts |
| Aggressive reward shaping for every physiologic signal | Can improve training curves quickly | Encourages reward hacking and weak clinical interpretation | Start with sparse plus limited shaping, then ablate |
| One giant end-to-end notebook | Fast to demo | Impossible to audit, test, and rerun safely | Separate data, training, evaluation, and reporting modules |

## Feature Dependencies

```text
[Cohort extraction]
    └──requires──> [Sepsis onset detection]
                       └──requires──> [Episode construction]
                                            └──requires──> [State/action/reward builders]
                                                                 └──requires──> [Transition dataset + splits]
                                                                                      └──requires──> [Baselines + offline RL]
                                                                                                           └──requires──> [OPE + safety review]
```

### Dependency Notes

- **Cohort extraction requires Sepsis onset detection:** inclusion rules are not meaningful until onset logic is operationalized.
- **Episode construction requires onset detection:** the `-24h/+48h` window is onset-centered.
- **Training requires transition datasets and splits:** algorithms are interchangeable only if the data contract is frozen first.
- **Safety review enhances OPE:** quantitative metrics alone are insufficient for a clinically framed project.

## MVP Definition

### Launch With (v1)

- [ ] Cohort, onset, and episode generation — establishes the research population and MDP timeline
- [ ] State/action/reward pipeline — produces the offline RL dataset contract
- [ ] Baselines plus CQL/BCQ/IQL training — creates comparable policy candidates
- [ ] WIS/FQE/sanity-check evaluation — makes results reviewable and defensible
- [ ] Reproducible reporting bundle — supports thesis or paper writing

### Add After Validation (v1.x)

- [ ] Temporal holdout evaluation — add once internal split results stabilize
- [ ] Support-aware safety overlay — add when action coverage diagnostics are available
- [ ] Experiment registry/dashboard — add when multiple sweeps make manual tracking painful

### Future Consideration (v2+)

- [ ] External dataset validation — defer until the base pipeline is stable
- [ ] Clinician-facing review UI — defer until there is credible evidence worth presenting
- [ ] Explanation tooling for action recommendations — defer until the chosen final model is settled

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Cohort + onset pipeline | HIGH | HIGH | P1 |
| State/action/reward builders | HIGH | HIGH | P1 |
| Leakage-safe splits | HIGH | MEDIUM | P1 |
| Baseline benchmarking | HIGH | MEDIUM | P1 |
| Offline RL training | HIGH | HIGH | P1 |
| OPE + clinical review | HIGH | HIGH | P1 |
| Safety overlay | MEDIUM | MEDIUM | P2 |
| Temporal holdout | MEDIUM | HIGH | P2 |
| Clinician-facing UI | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A | Competitor B | Our Approach |
|---------|--------------|--------------|--------------|
| Cohort + MDP specification | Academic sepsis RL papers often under-document operational details | Generic offline RL repos assume ready-made datasets | Treat cohort/onset/state/action specs as first-class artifacts |
| OPE and diagnostics | Many papers report a narrow metric set | Generic repos optimize benchmark scores, not clinical sanity | Combine WIS, ESS, FQE, qualitative review, and safety checks |
| Reproducibility | Notebook-heavy research artifacts are common | Production ML repos often skip clinical traceability | Freeze every transform, split, config, and summary table |

## Sources

- `implematation_plan_gpt.md`
- Standard needs of retrospective clinical ML and offline RL benchmark projects

---
*Feature research for: offline sepsis RL benchmark*
*Researched: 2026-03-28*
