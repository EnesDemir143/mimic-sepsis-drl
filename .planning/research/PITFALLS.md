# Pitfalls Research

**Domain:** Offline RL for retrospective sepsis treatment analysis
**Researched:** 2026-03-28
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Wrong Sepsis Onset Operationalization

**What goes wrong:**
Episode windows, actions, and rewards are all centered on the wrong clinical timepoint.

**Why it happens:**
Sepsis-3 rules are translated loosely or multiple candidate onset times are left unresolved.

**How to avoid:**
Lock the operational definition early, version the query/transformation logic, and surface ambiguous episodes as explicit exclusions.

**Warning signs:**
Large numbers of negative or zero-length windows, duplicate onset times per stay, or onset timestamps that postdate obvious interventions.

**Phase to address:**
Phase 1

---

### Pitfall 2: Patient Leakage Across Splits or Transforms

**What goes wrong:**
Validation and test results look better than they should because the model sees patient information indirectly during preprocessing.

**Why it happens:**
Splits are done at row level, or scalers/bin thresholds are fit on the full dataset.

**How to avoid:**
Create patient-level manifests first, fit every learned transform on train only, and add tests that fail on duplicate patient ids across splits.

**Warning signs:**
Unusually stable validation metrics, identical patient ids in multiple manifests, or preprocessing stats computed before split creation.

**Phase to address:**
Phase 2

---

### Pitfall 3: Action Bins Learned From the Wrong Population

**What goes wrong:**
The action space encodes future or test-distribution information, invalidating model comparison and OPE.

**Why it happens:**
Quartile thresholds are computed once on all records for convenience.

**How to avoid:**
Derive bin boundaries on train only, freeze the mapping artifact, and log coverage statistics for val/test.

**Warning signs:**
Action bin JSON recreated during evaluation, val/test bins drifting from train, or impossibly even action distributions everywhere.

**Phase to address:**
Phase 2

---

### Pitfall 4: Reward Over-Shaping and Reward Hacking

**What goes wrong:**
Policies optimize proxy signals instead of clinically meaningful outcomes.

**Why it happens:**
Too many small shaping terms are added before a sparse baseline is understood.

**How to avoid:**
Start with terminal mortality plus limited SOFA shaping, add lactate/MAP shaping as explicit ablations, and inspect reward distributions before large training sweeps.

**Warning signs:**
Reward histograms dominated by shaping terms, models preferring clinically odd behavior that scores well numerically, or unstable training when a shaping term is removed.

**Phase to address:**
Phase 3

---

### Pitfall 5: Treating OPE as Proof of Clinical Efficacy

**What goes wrong:**
The project overstates what retrospective results mean.

**Why it happens:**
WIS/FQE scores are easier to report than careful uncertainty and support analysis.

**How to avoid:**
Report ESS, support diagnostics, and clinician sanity checks alongside OPE; state explicitly that retrospective evaluation is not prospective validation.

**Warning signs:**
Claims that one model is "clinically superior" from a single OPE metric, low-ESS runs treated as definitive, or no subgroup review for high-risk patients.

**Phase to address:**
Phase 4

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Manual split files edited by hand | Fast initial setup | Hidden leakage and irreproducible experiments | Never |
| Notebook-defined feature lists | Quick iteration | Drift between runs and missing audit trail | Only during initial exploration before freezing artifacts |
| Recomputing intermediate tables inside training scripts | Fewer scripts | Impossible-to-debug coupling between ETL and modeling | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| MIMIC-IV joins | Mixing ICU stay, admission, and patient identifiers inconsistently | Pick ICU episode as the unit of analysis and validate every join path against it |
| Medication standardization | Failing to convert vasopressors into a common equivalent dose | Normalize to norepinephrine equivalent before binning |
| Outcome labeling | Using inconsistent death windows across artifacts | Freeze one 90-day mortality definition and reuse it everywhere |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Repeated full-table scans | ETL jobs take hours for minor changes | Materialize cohort and episode artifacts once | Full-cohort reruns |
| Row-wise Python transforms | CPU-heavy preprocessing and memory churn | Use vectorized/columnar transformations | State construction on wide feature sets |
| Unbounded experiment sweeps | Many checkpoints, unclear best run | Use configs, naming conventions, and tracked metrics | After the first multi-algorithm comparison |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Exporting artifacts with direct PHI | Compliance and privacy violations | Strip identifiers to project-safe keys before downstream artifacts |
| Logging raw patient records in debug output | Sensitive data leaks into notebooks or trackers | Log aggregate diagnostics only |
| Sharing unredacted qualitative case reviews | Re-identification risk | Redact timelines and identifiers in clinician review packs |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Undocumented artifact names and versions | Researchers cannot tell which dataset a run used | Include cohort, split, reward, and action version in artifact metadata |
| Comparing models trained on drifting data contracts | Results feel arbitrary and untrustworthy | Freeze one dataset contract per benchmark version |
| Reporting only top-line OPE numbers | Clinical reviewers cannot assess plausibility | Pair metrics with cases, heatmaps, and subgroup analyses |

## "Looks Done But Isn't" Checklist

- [ ] **Cohort pipeline:** Often missing exclusion logging — verify why each dropped episode was removed.
- [ ] **Split generation:** Often missing duplicate-patient checks — verify patient ids are unique across splits.
- [ ] **Action binning:** Often missing train-only fit proof — verify artifact metadata records the source split.
- [ ] **Evaluation:** Often missing ESS and support analysis — verify every WIS result has context.
- [ ] **Reporting:** Often missing config provenance — verify each figure/table points to exact run ids.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong onset logic | HIGH | Recompute episode, state, action, reward, and split artifacts from the corrected cohort version |
| Split leakage | HIGH | Regenerate manifests, refit transforms, retrain all models, and invalidate old comparisons |
| Reward over-shaping | MEDIUM | Roll back to sparse reward baseline and rerun ablations incrementally |
| Weak OPE interpretation | MEDIUM | Add ESS, subgroup review, and explicit uncertainty statements to all reports |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Wrong sepsis onset operationalization | Phase 1 | Reviewed onset spec and exclusion report exist |
| Patient leakage across splits or transforms | Phase 2 | Split-manifest and preprocessing tests pass |
| Action bins learned from wrong population | Phase 2 | Bin artifact metadata shows train-only derivation |
| Reward over-shaping and reward hacking | Phase 3 | Sparse vs shaped reward ablations are reported |
| Treating OPE as proof of efficacy | Phase 4 | Final report includes ESS, caveats, and clinician sanity review |

## Sources

- `implematation_plan_gpt.md`
- Known failure modes in retrospective clinical ML and offline RL workflows

---
*Pitfalls research for: offline sepsis RL benchmark*
*Researched: 2026-03-28*
