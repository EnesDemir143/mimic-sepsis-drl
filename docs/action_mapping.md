# Action Mapping — 25 Discrete Treatment Actions

**Version:** 1.0.0  
**Phase:** 5 — Treatment and Reward Encoding  
**Requirements:** ACT-01, ACT-02

---

## Overview

This document describes the fixed action contract used by all baselines and RL
policies in the MIMIC Sepsis Offline RL pipeline.  The action space encodes
clinician interventions as a **5 × 5 grid** combining vasopressor dose bins and
IV fluid volume bins, producing **25 discrete actions** (IDs 0–24).

```
action_id = vaso_bin × 5 + fluid_bin
```

## Treatment Signals

### Vasopressor Dose

| Source                  | Column               | Aggregation                          |
|------------------------|----------------------|--------------------------------------|
| MIMIC-IV `inputevents` | `ne_equiv_rate`      | Time-weighted average per 4-hour step |

All vasopressor agents are standardised to a common norepinephrine-equivalent
dose rate (µg/kg/min) before binning:

| Agent           | MIMIC-IV itemid | Factor | Source Unit    |
|-----------------|-----------------|--------|---------------|
| Norepinephrine  | 221906          | 1.0    | µg/kg/min     |
| Epinephrine     | 221289          | 1.0    | µg/kg/min     |
| Dopamine        | 221662          | 0.01   | µg/kg/min     |
| Phenylephrine   | 222315          | 0.1    | µg/kg/min     |
| Vasopressin     | 222042          | 2.5    | units/min     |

### IV Fluid Volume

| Source                  | Column             | Aggregation            |
|-------------------------|--------------------|------------------------|
| MIMIC-IV `inputevents` | `fluid_volume_4h`  | Sum per 4-hour step    |

IV fluids include crystalloids and colloids matching the project's item-ID set.

## Binning Strategy

### Zero-Dose Handling

**Bin 0** is always reserved for **zero dose** (no treatment administered).
This is an explicit design decision — the clinical meaning of "no vasopressor"
or "no IV fluid" in a 4-hour window is categorically different from "a small
but non-zero dose" and must not be merged into the lowest quartile bin.

### Non-Zero Quartile Bins

Bins 1–4 split the **non-zero** training-split doses at quartile boundaries:

| Bin | Range                   | Clinical Meaning   |
|-----|-------------------------|--------------------|
| 0   | dose = 0                | No treatment       |
| 1   | 0 < dose ≤ Q25         | Very low dose      |
| 2   | Q25 < dose ≤ Q50       | Low–moderate dose  |
| 3   | Q50 < dose ≤ Q75       | Moderate–high dose |
| 4   | dose > Q75              | High dose          |

### Train-Only Fitting

Bin edges (Q25, Q50, Q75) are learned **exclusively from training-split data**.
Validation and test splits are mapped using the frozen edges.  This prevents
data leakage from future or held-out patients into the action contract.

The fitted bin edges are persisted as a JSON artifact (`action_bins.json`)
and must be reloaded — never recomputed — for non-train splits.

## Action Grid

```
          fluid_bin →   0          1          2          3          4
vaso_bin ↓          no_fluid  fluid_Q1   fluid_Q2   fluid_Q3   fluid_Q4
     0  no_vaso        0          1          2          3          4
     1  vaso_Q1        5          6          7          8          9
     2  vaso_Q2       10         11         12         13         14
     3  vaso_Q3       15         16         17         18         19
     4  vaso_Q4       20         21         22         23         24
```

### Action Label Format

Each action ID decodes to a human-readable label:

```
action_id  →  "{vaso_label}×{fluid_label}"
```

Examples:
- `action_id=0` → `no_vaso×no_fluid`
- `action_id=6` → `vaso_Q1×fluid_Q1`
- `action_id=24` → `vaso_Q4×fluid_Q4`

## Artifact Schema

The `action_bins.json` artifact contains:

```json
{
  "spec_version": "1.0.0",
  "manifest_seed": 42,
  "vaso_edges": [0.05, 0.15, 0.45],
  "fluid_edges": [150.0, 500.0, 1000.0],
  "n_train_vaso_nonzero": 1420,
  "n_train_fluid_nonzero": 2180
}
```

| Field                    | Description                                          |
|--------------------------|------------------------------------------------------|
| `spec_version`           | Schema version for forward compatibility             |
| `manifest_seed`          | Split manifest seed used during fitting              |
| `vaso_edges`             | Q25, Q50, Q75 thresholds for NE-equiv dose           |
| `fluid_edges`            | Q25, Q50, Q75 thresholds for IV fluid volume         |
| `n_train_vaso_nonzero`   | Count of non-zero vasopressor step observations      |
| `n_train_fluid_nonzero`  | Count of non-zero IV fluid step observations         |

## Integration Points

| Consumer           | Uses                             |
|--------------------|----------------------------------|
| Phase 6 Transitions| `action_id` column in `(s,a,r,s',d)` tuples |
| Phase 7 CQL        | 25-action discrete policy head   |
| Phase 8 BCQ/IQL    | Same 25-action map               |
| Phase 9 Evaluation | Action-frequency heatmaps, OPE   |

---

*Document generated: 2026-03-29 — Phase 5 Plan 05-01*
