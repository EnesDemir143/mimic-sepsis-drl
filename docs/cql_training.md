# CQL Reference Training

**Phase:** 07 – CQL Reference Training  
**Requirement:** RL-01, PLAT-01  
**Version:** 1.0.0

---

## Overview

This document describes how to train the discrete-action Conservative Q-Learning (CQL)
reference policy on the MIMIC Sepsis offline RL dataset produced by Phase 6.

CQL is a conservative offline RL algorithm that penalises out-of-distribution Q-values,
preventing the learned policy from exploiting spurious high Q-value estimates for
state-action pairs that are absent or rare in the clinical dataset.

> **Research scope:** All results are retrospective research claims evaluated on
> historical ICU data. CQL output is not evidence of bedside clinical efficacy.

---

## Algorithm Summary

Discrete CQL adds a conservative regularisation term to the standard DQN Bellman loss:

```
L = L_TD + α · E_s[ logsumexp_a Q(s,a) − Q(s, a_data) ]
```

- **L_TD** – Bellman MSE loss: minimises the error between predicted and target Q-values.
- **logsumexp term** – penalises high Q-values across all actions (soft-max penalty).
- **Q(s, a_data)** – rewards the observed (in-dataset) clinician action.
- **α (cql_alpha)** – controls conservatism; higher α keeps the policy closer to the
  behaviour (clinician) policy.

**Reference:** Kumar et al. (2020) *Conservative Q-Learning for Offline Reinforcement
Learning*, NeurIPS 2020.

---

## Prerequisites

Before running CQL training, the following Phase 6 artifacts must exist:

| Artifact | Default path | Description |
|---|---|---|
| Train replay buffer | `data/replay/replay_train.parquet` | Flat `(s,a,r,s',done)` transition table |
| Dataset metadata | `data/replay/replay_train_meta.json` | Provenance JSON (split, action spec version, …) |

The train split must be used exclusively. Validation and test splits are never seen
during training.

---

## Quick Start

### 1. Verify the runtime environment

```bash
python -m mimic_sepsis_rl.training.device --self-check
```

This reports the resolved device backend and runs a tensor smoke test. On Apple Silicon:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m mimic_sepsis_rl.training.device --self-check
```

### 2. Dry-run (no real data required)

Validates the full forward/backward computation graph with a synthetic batch:

```bash
python -m mimic_sepsis_rl.training.cql \
    --config configs/training/cql.yaml \
    --dry-run
```

The dry-run exits with code `0` on success and prints resolved device metadata.
Use it to confirm the training stack works before submitting a full run.

### 3. Full training run

```bash
python -m mimic_sepsis_rl.training.cql --config configs/training/cql.yaml
```

On completion, a JSON summary is printed to stdout:

```json
{
  "algorithm": "cql",
  "n_epochs": 200,
  "total_steps": 15600,
  "final_td_loss": 0.0821,
  "final_cql_loss": 0.4312,
  "final_total_loss": 0.5133,
  "checkpoint_path": "checkpoints/cql/cql_epoch0200_step0015600.pt",
  "elapsed_seconds": 183.4,
  "state_dim": 33,
  "n_actions": 25,
  "device_backend": "mps"
}
```

### 4. Override device at runtime

```bash
# Force MPS (Apple Silicon)
python -m mimic_sepsis_rl.training.cql --config configs/training/cql.yaml --device mps

# Force CPU fallback
python -m mimic_sepsis_rl.training.cql --config configs/training/cql.yaml --device cpu
```

---

## Configuration Reference

All fields live in `configs/training/cql.yaml`. The file is parsed by
`mimic_sepsis_rl.training.config.load_training_config`.

### Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `algorithm` | `str` | `"cql"` | Algorithm identifier. Must be `"cql"`. |
| `schema_version` | `str` | `"1.0.0"` | Config schema version sentinel. |
| `n_epochs` | `int` | `200` | Total training epochs (one epoch = one full dataset pass). |
| `batch_size` | `int` | `256` | Mini-batch size for gradient updates. |
| `gamma` | `float` | `0.99` | MDP discount factor γ. |
| `dataset_path` | `str` | `data/replay/replay_train.parquet` | Path to the train split replay buffer. |
| `dataset_meta_path` | `str` | `data/replay/replay_train_meta.json` | Optional provenance metadata path. |

### `runtime` block

| Field | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"auto"` | Device backend: `auto`, `mps`, `cuda`, or `cpu`. |
| `seed` | `int` | `42` | Global random seed for reproducibility. |
| `num_workers` | `int` | `0` | DataLoader worker processes. Keep at `0` for MPS. |

### `checkpoint` block

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpoint_dir` | `str` | `"checkpoints/cql"` | Directory for `.pt` files and manifests. |
| `save_every_n_epochs` | `int` | `20` | Checkpoint cadence (0 = final epoch only). |
| `keep_last_n` | `int` | `3` | Oldest checkpoints to prune (0 = keep all). |

### `logging` block

| Field | Type | Default | Description |
|---|---|---|---|
| `log_dir` | `str` | `"runs/cql"` | Directory for JSONL metric logs. |
| `experiment_name` | `str` | `"cql_reference"` | Prefix for log filenames. |
| `log_every_n_steps` | `int` | `50` | Scalar metric flush cadence. |

### CQL hyper-parameters (extra block)

These fields are passed through the `extra` dict and consumed by the CQL trainer.

| Field | Type | Default | Description |
|---|---|---|---|
| `cql_alpha` | `float` | `1.0` | CQL conservatism coefficient α. Range: `[0.1, 10.0]`. |
| `hidden_sizes` | `list[int]` | `[256, 256]` | Q-network hidden layer widths. |
| `lr` | `float` | `0.0003` | Adam optimiser learning rate. |
| `target_update_freq` | `int` | `10` | Hard target network update frequency (epochs). |
| `polyak_tau` | `float` | `0.005` | Soft target update coefficient (ignored when `use_soft_update: false`). |
| `use_soft_update` | `bool` | `false` | Use Polyak averaging instead of hard target copies. |
| `grad_clip` | `float` | `10.0` | Gradient clipping max norm (0 = disabled). |
| `dry_run_state_dim` | `int` | `33` | Synthetic state dimension used in `--dry-run` mode. |

---

## Output Artifacts

After a successful training run the following artifacts are written to disk:

```
checkpoints/cql/
    cql_epoch0020_step0001560.pt          # model weights + optimizer state
    cql_epoch0020_step0001560_manifest.json
    cql_epoch0040_step0003120.pt
    cql_epoch0040_step0003120_manifest.json
    ...
    cql_epoch0200_step0015600.pt          # final checkpoint
    cql_epoch0200_step0015600_manifest.json

runs/cql/
    cql_reference_metrics.jsonl           # step-level scalar metrics
```

### Checkpoint manifest format

Each `.pt` file is accompanied by a JSON manifest with provenance metadata:

```json
{
  "algorithm": "cql",
  "epoch": 200,
  "global_step": 15600,
  "metrics": {
    "td_loss_mean": 0.082,
    "cql_loss_mean": 0.431,
    "total_loss_mean": 0.513
  },
  "config_dict": { "...": "full training config" },
  "device_meta": {
    "backend": "mps",
    "torch_version": "2.x.x",
    "...": "..."
  },
  "timestamp": 1743168000.0,
  "module_version": "1.0.0"
}
```

### Metric log format

`runs/cql/cql_reference_metrics.jsonl` contains one JSON record per line:

```json
{"step": 50, "epoch": 1, "name": "td_loss", "value": 0.42, "timestamp": 1743168010.0}
{"step": 50, "epoch": 1, "name": "cql_loss", "value": 0.89, "timestamp": 1743168010.0}
```

---

## Loading a Saved Policy for Inference

```python
from pathlib import Path
from mimic_sepsis_rl.training.cql import load_cql_policy

policy = load_cql_policy(
    Path("checkpoints/cql/cql_epoch0200_step0015600.pt"),
    state_dim=33,
    n_actions=25,
    device="cpu",   # or "mps" / "cuda"
)

# Greedy action for a single state
state = [0.1] * 33
action = policy.select_action(state)  # int in [0, 24]

# Full Q-value vector
q_vals = policy.q_values(state)       # list[float], length 25
```

`load_cql_policy` maps the checkpoint to the target device so inference can run on a
different machine (e.g. CPU-only) from where training ran.

---

## Platform Portability

The CQL pipeline runs on **Apple Silicon MPS**, **NVIDIA CUDA**, and **CPU** through a
single shared device abstraction. No algorithm code branches on device type.

### Apple Silicon (MPS)

```bash
# Recommended: enable MPS CPU fallback for ops not yet supported by Metal
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m mimic_sepsis_rl.training.cql \
    --config configs/training/cql.yaml \
    --device mps
```

Use `configs/training/runtime.mps.yaml` as a reference for MPS-specific settings.

**Known considerations:**
- Set `num_workers: 0` to avoid Metal/multiprocessing contention.
- MPS op validation runs automatically before training if MPS is detected.
  Warnings are emitted for any unsupported ops; set `PYTORCH_ENABLE_MPS_FALLBACK=1`
  to route them to CPU.
- Mixed precision (`float16`) on MPS requires PyTorch ≥ 2.1.

### NVIDIA CUDA

```bash
python -m mimic_sepsis_rl.training.cql \
    --config configs/training/cql.yaml \
    --device cuda
```

Use `configs/training/runtime.cuda.yaml` as a reference for CUDA-specific settings
(TF32, cuDNN benchmark, pin memory).

### CPU fallback

If neither CUDA nor MPS is available, the trainer automatically falls back to CPU.
The `device_meta.fallback_applied` field in the checkpoint manifest records whether
a fallback occurred so the provenance is auditable.

---

## Reproducibility Contract

CQL training is reproducible when the following conditions hold:

1. **Frozen dataset contract** – The same `replay_train.parquet` (same split seed,
   action spec version, and reward spec version) is used.
2. **Fixed seed** – `runtime.seed` is set identically across runs.
3. **Same device backend** – Results may differ numerically between CPU, MPS, and CUDA
   due to floating-point ordering. The `device_meta` block in each checkpoint manifest
   records the backend used.
4. **Same hyper-parameters** – `cql_alpha`, `hidden_sizes`, `lr`, etc. from `cql.yaml`.

The checkpoint manifest embeds the full training config and device metadata, providing
the information needed to reproduce or audit any saved run.

---

## Extending for Phase 8 (BCQ / IQL)

Phase 8 algorithms reuse the infrastructure introduced here:

- **`mimic_sepsis_rl.training.device`** – device abstraction (no changes needed).
- **`mimic_sepsis_rl.training.common`** – `ReplayDataset`, `CheckpointManager`,
  `MetricLogger`, `set_global_seed` (no changes needed).
- **`mimic_sepsis_rl.training.config`** – `TrainingConfig` and `build_training_config`
  (extend `extra` block for BCQ/IQL hyper-parameters).

New algorithm trainers should:
1. Accept a `TrainingConfig` and call `cfg.device` for all tensor allocation.
2. Use `load_replay_dataset(cfg)` to consume the frozen dataset contract.
3. Use `build_checkpoint_manager(cfg)` and `MetricLogger.from_config(cfg)` for
   consistent artifact paths.
4. Never redefine preprocessing, action bins, or reward functions.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Dataset not found` | Phase 6 artifacts missing | Run `build_transitions` to generate replay buffers |
| `RuntimeError: ... not implemented for 'MPS'` | Unsupported MPS op | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `CUDA requested but not available` | No GPU on machine | Use `--device auto` or `--device cpu` |
| `NaN losses after N steps` | Learning rate too high or reward scale too large | Reduce `lr`; verify reward contract |
| Checkpoint file not found on reload | `keep_last_n` pruned it | Increase `keep_last_n` or set to `0` |
| Very slow training on MPS | `num_workers > 0` causing Metal contention | Set `num_workers: 0` in config |

---

*Phase: 07-cql-reference-training | Last updated: 2026-03-29*