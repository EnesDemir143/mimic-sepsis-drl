# Model Comparison Artifacts

**Phase:** 08 – Comparative Offline RL Experiments  
**Requirements:** RL-02, RL-03  
**Version:** 1.0.0

---

## Overview

This document describes the shared artifact contract for comparing discrete-action
offline RL algorithms on the frozen MIMIC Sepsis replay dataset.

Phase 8 adds **BCQ** and **IQL** on the same training surface already used by the
**CQL** reference trainer. The goal is fair algorithm-to-algorithm comparison:

- same replay split
- same action map
- same reward definition
- same runtime abstraction
- same artifact envelope for checkpoints, training curves, and config provenance

> **Research scope:** These outputs support retrospective benchmark comparison only.
> They are not evidence of bedside clinical benefit.

---

## What This Produces

Every algorithm run should leave three artifact families behind:

1. **Checkpoint files** in the algorithm checkpoint directory
2. **Checkpoint manifests** adjacent to each checkpoint
3. **Metric curves** in one JSONL log under the algorithm run directory

The shared comparison utilities in
`mimic_sepsis_rl.training.comparison` normalize those files into one
cross-algorithm schema.

---

## Fair-Comparison Contract

Before comparing runs, verify that the following fields match across `cql.yaml`,
`bcq.yaml`, and `iql.yaml`:

| Contract element | Where it comes from | Why it must match |
|---|---|---|
| `dataset_path` | training YAML | Prevents comparing different replay buffers |
| `dataset_meta_path` | training YAML | Preserves split/action/reward provenance |
| `batch_size` | training YAML | Avoids runtime drift in optimizer exposure |
| `gamma` | training YAML | Keeps Bellman targets comparable |
| `action_spec_version` | dataset metadata JSON | Ensures action bins did not change |
| `reward_spec_version` | dataset metadata JSON | Ensures reward function did not change |
| `manifest_seed` | dataset metadata JSON | Confirms the same split boundary |
| `state_dim`, `n_actions` | dataset metadata JSON | Confirms the same MDP contract |

If these values diverge, the output should be treated as **dataset-contract drift**,
not a meaningful model comparison.

---

## Training Workflow

Run each algorithm through the shared experiment runner:

```bash
python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql
python -m mimic_sepsis_rl.training.experiment_runner --algorithm bcq
python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql
```

For environment checks without real data loading:

```bash
python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql --dry-run
python -m mimic_sepsis_rl.training.experiment_runner --algorithm bcq --dry-run
python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql --dry-run
```

The runner resolves:

- the algorithm registry entry
- the YAML config
- dataset-contract metadata from `dataset_meta_path`
- the shared `n_actions` default

---

## Output Layout

Example output after running all three algorithms:

```text
checkpoints/
├── cql/
│   ├── cql_epoch0200_step0015600.pt
│   └── cql_epoch0200_step0015600_manifest.json
├── bcq/
│   ├── bcq_epoch0200_step0015600.pt
│   └── bcq_epoch0200_step0015600_manifest.json
└── iql/
    ├── iql_epoch0200_step0015600.pt
    └── iql_epoch0200_step0015600_manifest.json

runs/
├── cql/cql_reference_metrics.jsonl
├── bcq/bcq_baseline_metrics.jsonl
└── iql/iql_baseline_metrics.jsonl
```

Each checkpoint manifest carries:

- algorithm name
- epoch and global step
- epoch-summary metrics
- full serialized config
- backend metadata
- timestamp and module version

Each metrics file contains JSONL scalar records:

```json
{"step": 50, "epoch": 1, "name": "td_loss", "value": 0.42, "timestamp": 1743168010.0}
{"step": 50, "epoch": 1, "name": "cql_loss", "value": 0.89, "timestamp": 1743168010.0}
```

Metric names differ by algorithm, but the envelope is identical.

---

## Comparison API

Use the shared Python API to normalize runs:

```python
from pathlib import Path

from mimic_sepsis_rl.training.comparison import build_comparison_report

report = build_comparison_report(
    [
        Path("configs/training/cql.yaml"),
        Path("configs/training/bcq.yaml"),
        Path("configs/training/iql.yaml"),
    ]
)

payload = report.to_dict()
print(payload["dataset_contract_consistent"])
print(payload["algorithms"])
```

The comparison loader infers:

- latest checkpoint from `checkpoint_dir`
- checkpoint manifest adjacent to that checkpoint
- metric log from `<log_dir>/<experiment_name>_metrics.jsonl`
- dataset contract from `dataset_meta_path`

---

## Output Schema

Each normalized run entry has this top-level shape:

```json
{
  "algorithm": "bcq",
  "checkpoint": {
    "checkpoint_path": "checkpoints/bcq/bcq_epoch0200_step0015600.pt",
    "manifest_path": "checkpoints/bcq/bcq_epoch0200_step0015600_manifest.json",
    "epoch": 200,
    "global_step": 15600,
    "metrics": {
      "td_loss_mean": 0.08,
      "imitation_loss_mean": 0.21,
      "total_loss_mean": 0.29
    }
  },
  "curves": [
    {
      "name": "td_loss",
      "points": [{"step": 50, "epoch": 1, "value": 0.42, "timestamp": 1743168010.0}],
      "final_value": 0.42
    }
  ],
  "curve_names": ["td_loss", "imitation_loss", "td_loss_mean", "total_loss_mean"],
  "final_metrics": {
    "td_loss_mean": 0.08,
    "imitation_loss_mean": 0.21,
    "total_loss_mean": 0.29
  },
  "config_provenance": {
    "config_path": "configs/training/bcq.yaml",
    "checkpoint_dir": "checkpoints/bcq",
    "log_dir": "runs/bcq",
    "experiment_name": "bcq_baseline",
    "dataset_path": "data/replay/replay_train.parquet",
    "dataset_meta_path": "data/replay/replay_train_meta.json",
    "batch_size": 256,
    "gamma": 0.99,
    "requested_device": "auto",
    "effective_backend": "mps"
  },
  "dataset_contract": {
    "spec_version": "1.0.0",
    "split_label": "train",
    "n_actions": 25,
    "state_dim": 33,
    "action_spec_version": "1.0.0",
    "reward_spec_version": "1.0.0",
    "manifest_seed": 42,
    "n_episodes": 18432,
    "n_transitions": 147456,
    "feature_columns": ["sofa", "lactate", "map"]
  }
}
```

The full report wraps these entries with:

- `algorithms`
- `dataset_contract_consistent`
- `shared_dataset_contract`
- `runs`

---

## Interpretation

### What counts as a fair difference

A performance or loss difference is interpretable when:

- `dataset_contract_consistent == true`
- `dataset_path`, `dataset_meta_path`, `batch_size`, and `gamma` match
- all runs use the same split label and manifest seed

### What counts as drift instead of model behavior

Treat the comparison as invalid when any of these differ:

- `action_spec_version`
- `reward_spec_version`
- `manifest_seed`
- `n_actions`
- `state_dim`
- replay dataset path

If that happens, fix the contract mismatch first and rerun before making
algorithm claims.

### Backend caveats

Backend differences (`cpu`, `mps`, `cuda`) should be read from
`config_provenance.effective_backend` and manifest `device_meta`, not guessed
from directory names or shell history.

Numerical differences can appear across accelerators due to floating-point
ordering. That is expected. It is still a valid comparison if the replay
contract and config provenance match.

---

## Reported Metrics

Typical epoch-summary metrics by algorithm:

| Algorithm | Expected summary metrics |
|---|---|
| CQL | `td_loss_mean`, `cql_loss_mean`, `total_loss_mean` |
| BCQ | `td_loss_mean`, `imitation_loss_mean`, `total_loss_mean` |
| IQL | `critic_loss_mean`, `value_loss_mean`, `actor_loss_mean`, `total_loss_mean` |

These names are algorithm-specific by design. The shared comparison layer
standardizes the container, not the underlying optimization objective.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `dataset_contract_consistent == false` | Different dataset metadata or reward/action versions | Align YAML paths and rerun from the same replay contract |
| Missing `checkpoint` in a run artifact | No checkpoint saved yet | Verify training completed and `checkpoint_dir` is correct |
| Empty `curves` list | Metrics JSONL file missing | Check `log_dir` and `experiment_name` in the config |
| `shared_dataset_contract` is `null` | At least one run drifted from the common contract | Inspect `runs[*].dataset_contract` and fix the mismatch |
| Backends differ across runs | Experiments executed on different accelerators | Decide whether backend variation is acceptable and document it in the report |

---

*Phase: 08-comparative-offline-rl-experiments | Last updated: 2026-03-29*
