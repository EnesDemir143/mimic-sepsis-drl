# Reward Specification — MDP Reward Contract

**Version:** 1.0.0  
**Phase:** 5 — Treatment and Reward Encoding  
**Requirements:** RWD-01

---

## Overview

This document defines the reward function used by all offline RL policies
and baselines in the MIMIC Sepsis pipeline.  The reward contract is
deterministic: the same episode data and configuration always produce the
same reward sequence.

Three reward variants are available for ablation studies:

| Variant        | Terminal | SOFA Δ | Lactate | MAP  |
|----------------|----------|--------|---------|------|
| `sparse`       | ✅        | ❌      | ❌       | ❌    |
| `sofa_shaped`  | ✅        | ✅      | ❌       | ❌    |
| `full_shaped`  | ✅        | ✅      | ✅       | ✅    |

The default variant is **`sofa_shaped`** — a conservative middle ground
between signal density and reward engineering risk.

## Terminal Reward

Applied only at the **last step** of each episode.

| Outcome               | Reward |
|------------------------|--------|
| Survived 90 days       | +15.0  |
| Died within 90 days    | −15.0  |

These values are configurable via `terminal_reward_survived` and
`terminal_reward_died` in `RewardConfig`.

### Clinical Rationale

The 90-day mortality horizon captures delayed sepsis mortality while
remaining within a clinically interpretable timeframe.  The ±15 magnitude
dominates intermediate shaping to ensure that the terminal outcome remains
the primary training signal.

## Intermediate Shaping

Applied at **non-terminal steps** (steps 1 through N−1).

### 1. SOFA-Delta Shaping (`sofa_shaped`, `full_shaped`)

```
sofa_shaping = sofa_delta_weight × (SOFA_current − SOFA_previous)
```

| Parameter           | Default | Range        |
|---------------------|---------|-------------|
| `sofa_delta_weight` | −0.025  | [−0.5, 0]   |

- **Negative weight × positive delta** (= worsening) → negative reward
- **Negative weight × negative delta** (= improvement) → positive reward

SOFA is the standard sepsis organ-dysfunction score.  Using the *delta*
(change) rather than the absolute value rewards clinical trajectory
improvement, not static severity.

### 2. Lactate Clearance Shaping (`full_shaped` only)

```
clearance = (lactate_prev − lactate_curr) / lactate_prev
lactate_shaping = lactate_clearance_weight × clearance
```

| Parameter                   | Default | Range      |
|-----------------------------|---------|-----------|
| `lactate_clearance_weight`  | 0.0     | [0, 1.0]  |

Disabled by default.  Lactate clearance reflects tissue perfusion
improvement — a key resuscitation target in sepsis guidelines.

### 3. MAP Stability Shaping (`full_shaped` only)

```
if MAP < map_threshold:
    deficit = map_threshold − MAP
    map_shaping = −map_stability_weight × deficit
else:
    map_shaping = 0
```

| Parameter              | Default | Range       |
|------------------------|---------|------------|
| `map_stability_weight` | 0.0     | [0, 1.0]  |
| `map_threshold`        | 65.0    | mmHg       |

Disabled by default.  Penalises episodes where MAP drops below the
Surviving Sepsis Campaign target of 65 mmHg.

## Total Reward Formula

```
reward_t = terminal_t + sofa_shaping_t + lactate_shaping_t + map_shaping_t
```

All components are zero when not applicable (wrong variant, missing
values, non-terminal step).

## Edge Cases

| Scenario                     | Behaviour                                |
|------------------------------|------------------------------------------|
| Missing SOFA at step t       | `sofa_shaping = 0.0`                    |
| Missing SOFA at step t−1     | `sofa_shaping = 0.0`                    |
| First step of episode        | `sofa_shaping = 0.0` (no previous)      |
| Missing mortality at terminal| `terminal = 0.0`                         |
| NaN in lactate or MAP        | Treated as missing → component = 0.0    |
| Lactate_prev = 0             | Clearance skipped → `lactate_shaping = 0.0` |

## Configuration

Reward parameters are stored in JSON:

```json
{
  "version": "1.0.0",
  "variant": "sofa_shaped",
  "terminal_reward_survived": 15.0,
  "terminal_reward_died": -15.0,
  "sofa_delta_weight": -0.025,
  "lactate_clearance_weight": 0.0,
  "map_stability_weight": 0.0,
  "map_threshold": 65.0
}
```

## Reward Diagnostics

The `reward_summary()` function produces aggregate statistics:

| Metric              | Description                               |
|---------------------|-------------------------------------------|
| `n_episodes`        | Number of episodes processed              |
| `n_transitions`     | Total step-level transitions              |
| `n_survived`        | Terminal steps with positive reward       |
| `n_died`            | Terminal steps with negative reward       |
| `mean_total`        | Average total reward across transitions   |
| `std_total`         | Standard deviation of total reward        |
| `mean_sofa_shaping` | Average SOFA shaping component            |
| `mean_terminal`     | Average terminal component                |

These diagnostics should be inspected before training to verify that
the reward distribution is clinically sensible.

## Integration Points

| Consumer             | Uses                                         |
|----------------------|----------------------------------------------|
| Phase 6 Transitions  | `reward` column in `(s,a,r,s',d)` tuples    |
| Phase 7 CQL          | Reward signal for conservative Q-learning    |
| Phase 8 BCQ/IQL      | Same reward contract                         |
| Phase 9 Ablations    | Variant switching for reward shaping studies  |

## Design Decisions

1. **Start conservative**: Default `sofa_shaped` avoids over-engineering
   the reward too early.  Full shaping can be enabled for ablation.
2. **Terminal dominance**: ±15 ensures the mortality signal is not drowned
   by intermediate shaping (typical shaping magnitudes are < 0.5).
3. **Versioned config**: The JSON config records exactly which reward
   formula was used for each experiment, enabling reproducibility.
4. **No reward for action cost**: We intentionally do not penalise
   treatment intensity — the policy should learn clinically appropriate
   dosing from outcomes, not be biased towards under-treatment.

---

*Document generated: 2026-03-29 — Phase 5 Plan 05-02*
