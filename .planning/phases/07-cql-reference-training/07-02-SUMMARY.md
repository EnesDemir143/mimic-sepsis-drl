# Plan 07-02 Summary: CQL Training Pipeline

**Phase:** 07-cql-reference-training
**Plan:** 02
**Status:** Complete
**Completed:** 2026-03-29

---

## What Was Built

### `src/mimic_sepsis_rl/training/common.py`
Shared training utilities reused by every algorithm trainer (CQL, and later BCQ/IQL in Phase 8):

- `set_global_seed(seed)`: Sets Python `random`, NumPy, and PyTorch (CPU + CUDA) RNGs
  from a single seed for reproducible runs.
- `TransitionBatch` (dataclass): Mini-batch of MDP transitions as typed tensors
  (`states`, `actions`, `rewards`, `next_states`, `dones`) with a `.to(device)` method.
- `ReplayDataset`: In-memory replay dataset loaded from a Phase 6 Parquet file.
  Auto-detects `s_*` / `ns_*` state columns. Provides `iter_batches(batch_size, shuffle,
  epoch)` for epoch-based training and `sample_batch(n)` for random sampling. Shuffle is
  deterministic per epoch via `seed + epoch` generator seeding.
- `CheckpointManifest` (dataclass): Serialisable provenance metadata (algorithm, epoch,
  global_step, metrics, full config dict, device metadata, timestamp). Serialises to JSON.
- `CheckpointManager`: Saves `model_state_dict` + optional optimizer state as `.pt` files
  with adjacent `_manifest.json`. Prunes oldest checkpoints when `keep_last_n > 0`.
  `load(path, device)` maps weights to target device. `load_manifest(path)` reads provenance.
- `MetricLogger`: Accumulates scalar metrics and flushes them to a JSONL log file.
  Step-level and epoch-level logging. Auto-flushes every `log_every_n_steps` steps.
  `epoch_mean(name)` returns the mean of accumulated values and clears the accumulator.
- `load_replay_dataset(cfg)`: Loads `ReplayDataset` from `cfg.dataset_path`.
- `build_checkpoint_manager(cfg)`: Constructs `CheckpointManager` from `TrainingConfig`.
- `should_checkpoint(epoch, cfg, is_last)`: Returns True when a checkpoint should be saved.
- `compute_epoch_metrics(losses, prefix)`: Summarises per-step losses into mean/min/max scalars.

### `src/mimic_sepsis_rl/training/cql.py`
Discrete-action CQL reference trainer implementing:

**Algorithm components:**
- `QNetwork(state_dim, n_actions, hidden_sizes)`: Fully-connected Q-network with ReLU
  activations and orthogonal weight initialisation. Maps state vectors to per-action Q-values.
- `td_loss(q_values, actions, rewards, next_q_values, dones, gamma)`: DQN Bellman MSE loss
  with terminal masking via the `done` flag.
- `cql_loss(q_values, actions)`: CQL conservative regularisation term —
  `mean(logsumexp_a Q(s,a) − Q(s, a_data))` — penalises OOD Q-values.

**Trainer surface:**
- `CQLTrainer(cfg, dataset, n_actions)`: Full training loop. Reads hyper-parameters from
  `cfg.extra` (`cql_alpha`, `hidden_sizes`, `lr`, `target_update_freq`, `polyak_tau`,
  `use_soft_update`, `grad_clip`). Hard or soft target network updates. Gradient clipping.
  Calls `CheckpointManager` and `MetricLogger` at configured cadences.
- `CQLTrainer.train()` → `CQLTrainingResult`: Runs all epochs and returns a typed result
  with final losses, total steps, elapsed time, checkpoint path, and device backend.
- `CQLTrainer.get_policy()` → `CQLPolicy`: Returns current Q-network as an inference-ready
  `CQLPolicy` object.

**Inference surface:**
- `CQLPolicy(q_network, device, state_dim, n_actions)`: Accepts `list[float]` or
  `torch.Tensor` state inputs. `select_action(state)` returns the greedy action (argmax Q).
  `q_values(state)` returns the full per-action Q-value vector.
- `load_cql_policy(checkpoint_path, state_dim, n_actions, device)`: Loads a saved
  checkpoint onto any supported device and returns a `CQLPolicy` ready for held-out
  inference. Maps weights to the target device so CPU inference works after MPS/CUDA training.

**Dry-run:**
- `_dry_run(cfg, n_actions)`: One synthetic mini-batch forward/backward pass with MPS op
  validation. Tensors generated on CPU and moved to device to avoid MPS generator
  restriction. Logs resolved losses and verifies `select_action` output range.
- `--dry-run` CLI flag: Runs `_dry_run` against the loaded config and exits `0`.

### `configs/training/cql.yaml`
Production CQL training configuration. Key fields:

| Section | Notable settings |
|---|---|
| Top-level | `algorithm: cql`, `schema_version: 1.0.0`, `n_epochs: 200`, `batch_size: 256`, `gamma: 0.99` |
| `runtime` | `device: auto`, `seed: 42`, `num_workers: 0` |
| `checkpoint` | `checkpoint_dir: checkpoints/cql`, `save_every_n_epochs: 20`, `keep_last_n: 3` |
| `logging` | `log_dir: runs/cql`, `experiment_name: cql_reference`, `log_every_n_steps: 50` |
| Extra (CQL) | `cql_alpha: 1.0`, `hidden_sizes: [256, 256]`, `lr: 0.0003`, `target_update_freq: 10`, `grad_clip: 10.0` |

### `tests/training/test_cql_pipeline.py`
64 tests across 11 test classes:

| Class | Coverage |
|---|---|
| `TestQNetwork` | Output shape, dtype, gradient flow, custom hidden sizes |
| `TestLossFunctions` | `td_loss` zero when targets match, terminal masking, gradient flow; `cql_loss` scalar output, gradient flow; combined backward |
| `TestReplayDataset` | Parquet loading, state dim detection, batch shape/dtype, action range, shuffle determinism, explicit column passing |
| `TestCheckpointManager` | `.pt` + manifest creation, weight round-trip, pruning, `latest_checkpoint`, manifest loading |
| `TestCQLPolicy` | `select_action` range, tensor input, `q_values` length and finiteness, argmax consistency, determinism |
| `TestLoadCQLPolicy` | Checkpoint round-trip, valid actions post-reload, weight equality with original |
| `TestCQLTrainer` | Training completion, epoch count, step count, finite losses, state/action dim reporting, CPU backend, checkpoint saved, policy returned, metric log created, `to_dict` keys, frozen dataset contract |
| `TestDryRun` | Completes on CPU, small state dim, custom alpha |
| `TestCommonUtilities` | `set_global_seed` determinism, `compute_epoch_metrics`, `should_checkpoint`, `MetricLogger` JSONL output, `TransitionBatch.to()` |
| `TestDatasetContract` | Parquet column schema, `s_` prefix, `ns_` prefix, `ReplayDataset` accepts Phase 6 format |

### `docs/cql_training.md`
Operator-facing documentation covering:
- Algorithm summary and CQL loss formula
- Prerequisites (Phase 6 artifacts)
- Quick-start commands (self-check, dry-run, full run, device override)
- Complete configuration reference for all fields
- Output artifact layout (checkpoint files, manifests, JSONL metric log)
- Inference code example using `load_cql_policy`
- Platform portability guide (MPS, CUDA, CPU fallback)
- Reproducibility contract
- Guidance for Phase 8 reuse (BCQ/IQL extension points)
- Troubleshooting table

---

## Verification Results

```
pytest -q tests/training/test_cql_pipeline.py
64 passed in 5.00s
```

```
python -m mimic_sepsis_rl.training.cql --config configs/training/cql.yaml --dry-run
INFO: Runtime resolved: requested=auto effective=mps device=mps fallback=False
INFO: === CQL DRY-RUN ===
INFO: MPS op validation: all probed ops succeeded.
INFO: Dry-run forward/backward: td_loss=1.3558 cql_loss=3.1602 total=4.5160
INFO: Dry-run complete ✅  device=mps backend=mps
```

---

## Key Design Decisions

- **Frozen dataset contract:** `CQLTrainer` consumes the replay Parquet as-is via
  `ReplayDataset`. It never re-computes features, action bins, or rewards — those belong
  to earlier phases.
- **Device routing:** All tensor allocation in `CQLTrainer` and `CQLPolicy` uses
  `cfg.device` from `TrainingConfig`. No `torch.device(...)` string construction appears
  in algorithm code.
- **CPU-first synthetic data:** `_dry_run` generates tensors on CPU and moves them to
  the target device, avoiding the MPS generator restriction that caused a `RuntimeError`
  when `device=mps` was passed directly to `torch.randn(..., generator=g, device=...)`.
- **Checkpoint provenance:** Every `.pt` file ships with a `_manifest.json` that embeds
  the full training config and `DeviceMetadata`. This ensures a loaded policy can be
  traced back to the exact training run, dataset version, and hardware backend.
- **Phase 8 extension surface:** `common.py` utilities (`ReplayDataset`,
  `CheckpointManager`, `MetricLogger`, `set_global_seed`) are intentionally algorithm-
  agnostic. BCQ/IQL trainers in Phase 8 call the same helpers without modification.

---

## Artifacts Produced

| Path | Type | Description |
|---|---|---|
| `src/mimic_sepsis_rl/training/common.py` | Module | Shared dataset loading, checkpointing, and metric logging helpers |
| `src/mimic_sepsis_rl/training/cql.py` | Module | Discrete CQL trainer and inference loading surface |
| `configs/training/cql.yaml` | Config | Reusable CQL training configuration (`algorithm: cql`) |
| `tests/training/test_cql_pipeline.py` | Tests | 64 passing tests for dataset wiring and saved model loading |
| `docs/cql_training.md` | Docs | Operator-facing training documentation |

---

## Requirements Satisfied

- **RL-01:** Researcher can train a discrete-action CQL policy on the prepared offline
  dataset — satisfied via `CQLTrainer.train()` consuming the Phase 6 replay buffer.
- **RL-01 (rerun):** The same CQL training pipeline can be rerun against the fixed action
  map and reward definition without changing the dataset contract — satisfied by the
  frozen `ReplayDataset` contract and versioned checkpoint manifests.
- **PLAT-01 (continued):** CQL trainer resolves device through `cfg.device` from the
  shared `TrainingConfig` / `device.py` abstraction — no CUDA-specific code paths.
- **Phase 8 readiness:** `common.py` utilities and `TrainingConfig` are designed for
  direct reuse by BCQ/IQL trainers without modification.