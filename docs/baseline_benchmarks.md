# Baseline Benchmarks

## Overview

This document describes the three comparison baselines implemented in Phase 6
of the MIMIC Sepsis Offline RL pipeline. Every offline RL policy trained in
later phases must be compared against these baselines using the **exact same**
transition dataset contract, action map, reward definition, and split
boundaries.

## Baselines

### 1. Clinician Replay Baseline

| Property | Value |
|----------|-------|
| Module | `mimic_sepsis_rl.baselines.clinician` |
| Learning | ❌ None — replays observed actions |
| Purpose | The reference point: what the clinician actually did |

**What it does:**
Replays the recorded clinician actions from the transition dataset and
reports aggregate per-episode and per-step statistics. This is not a learned
policy — it simply evaluates the observed behavior.

**Reported metrics:**
- `n_episodes` / `n_transitions`
- `mean_episode_return` / `std_episode_return`
- `mean_reward` (per step)
- `action_distribution` (frequency per action ID)
- `mortality_rate` (fraction of episodes with negative terminal reward)

**Interpretation:**
Any RL policy that does not improve upon the clinician baseline's
mean episode return provides no actionable signal. However, a clinician
baseline is **not** an upper bound — clinicians may follow suboptimal
patterns that a well-trained RL agent can improve.

---

### 2. No-Treatment Baseline

| Property | Value |
|----------|-------|
| Module | `mimic_sepsis_rl.baselines.no_treatment` |
| Learning | ❌ None — static action = 0 |
| Purpose | Lower-bound: is any treatment better than none? |

**What it does:**
A static policy that always selects action 0 (no vasopressor and no IV
fluid). Since the reward function depends on patient outcomes that already
happened in the observational data, this baseline reports the *observed*
rewards under the assumption that the counterfactual impact of doing nothing
is not modeled.

**Reported metrics:**
- `mean_episode_return` / `std_episode_return`
- `mean_reward`
- `policy_action` (always 0)
- `mortality_rate`

**Interpretation:**
Any learned policy that scores **worse** than the no-treatment baseline
is pathologically bad and should be discarded. This baseline sets the
absolute floor.

---

### 3. Behavior Cloning Baseline

| Property | Value |
|----------|-------|
| Module | `mimic_sepsis_rl.baselines.behavior_cloning` |
| Learning | ✅ Supervised — softmax classifier on state → action |
| Purpose | Imitation upper bound: how well can we copy the clinician? |

**What it does:**
Trains a multinomial logistic regression (softmax classifier) to predict
the clinician's action given the state vector. The implementation is
deliberately minimal — no deep learning dependencies are introduced before
Phase 7.

**Training details:**
- Model: linear softmax classifier (no hidden layers)
- Optimizer: vanilla SGD
- Loss: cross-entropy
- Features: the same state vector used by all RL policies
- Actions: 25-class multinomial output

**Reported metrics:**
- `train_accuracy` / `eval_accuracy` (top-1)
- `mean_episode_return` / `std_episode_return`
- `mortality_rate`
- `n_epochs`

**Interpretation:**
The behavior cloning accuracy shows how predictable clinician decisions
are from the state representation. Low accuracy suggests that either the
state space lacks critical decision signals (e.g., clinical notes) or
clinician decisions are highly variable. RL policies should aim to
**beat the clinician return**, not maximize imitation accuracy.

---

## Output Schema

All baselines produce a `to_dict()` method that returns a standardized
dictionary. The `baseline` key identifies the source:

```json
{
  "baseline": "clinician | no_treatment | behavior_cloning",
  "n_episodes": 1000,
  "n_transitions": 6000,
  "mean_episode_return": -2.34,
  "std_episode_return": 14.12,
  "mortality_rate": 0.42,
  ...
}
```

## Comparison Table Template

After running all baselines, populate this table:

| Metric | Clinician | No-Treatment | BC (train) | BC (eval) |
|--------|-----------|-------------|------------|-----------|
| Mean return | — | — | — | — |
| Std return | — | — | — | — |
| Mortality % | — | — | — | — |
| Top-1 accuracy | N/A | N/A | — | — |

## Data Contract

All baselines consume `list[TransitionRow]` from the shared transition
dataset contract (`mimic_sepsis_rl.datasets.transitions`). The contract
ensures:

1. **Split consistency**: Baselines use the same train/val/test splits.
2. **Action map**: The 25-action map is frozen from Phase 5.
3. **Reward definition**: The reward contract is frozen from Phase 5.
4. **State features**: The state vector comes from the Phase 4 pipeline.

No baseline may redefine preprocessing, recompute bins, or bypass the
manifest boundaries.

## Usage

```python
from mimic_sepsis_rl.datasets.transitions import build_transitions
from mimic_sepsis_rl.baselines.clinician import evaluate_clinician_baseline
from mimic_sepsis_rl.baselines.no_treatment import evaluate_no_treatment_baseline
from mimic_sepsis_rl.baselines.behavior_cloning import train_behavior_cloning

# Build transitions from the merged state+action+reward table
transitions = build_transitions(merged_df, feature_columns=feature_cols)

# Run baselines
clin_result = evaluate_clinician_baseline(transitions)
no_treat_result = evaluate_no_treatment_baseline(transitions)
bc_result, bc_model = train_behavior_cloning(
    train_transitions,
    eval_transitions=eval_transitions,
    n_epochs=20,
)
```

---

*Version: 1.0.0 — Phase 6: Transition Dataset and Baseline Benchmarks*
*Created: 2026-03-29*
