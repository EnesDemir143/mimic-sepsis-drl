# Plan 07-01 Summary: Shared CPU/MPS/CUDA Runtime Abstraction

**Phase:** 07-cql-reference-training
**Plan:** 01
**Status:** Complete
**Completed:** 2026-03-29

---

## What Was Built

### `src/mimic_sepsis_rl/training/device.py`
Cross-platform device runtime abstraction. Single entry point (`resolve_device`) for all
training code to obtain a `torch.device` without branching on backend type. Capabilities:

- `Backend` enum: `auto`, `cuda`, `mps`, `cpu` — parsed case-insensitively.
- `resolve_device(requested)` → `(torch.device, DeviceMetadata)`: resolves the requested
  backend, falls back to CPU with a warning when unavailable, and returns a full metadata
  snapshot.
- `DeviceMetadata` dataclass: records effective backend, PyTorch version, CUDA/MPS
  availability, device count, platform info, fallback flag, and MPS op-fallback env state.
  Serialisable to dict/JSON for checkpoint manifests.
- `validate_mps_ops(device)`: probes `torch.cdist`, `torch.linalg.norm`, and
  `scatter_add_` on MPS; returns a list of warning strings for unsupported ops so issues
  surface before full training begins.
- `--self-check` CLI: runs availability probe + tensor smoke test + MPS op validation and
  prints a human-readable report with JSON metadata. Exits `0` on success.

### `src/mimic_sepsis_rl/training/config.py`
Training configuration layer that routes device resolution through `device.py`. Provides:

- `TrainingConfig` (frozen dataclass): top-level resolved config with `algorithm`,
  `schema_version`, `RuntimeConfig`, `CheckpointConfig`, `LoggingConfig`,
  `dataset_path`, `n_epochs`, `batch_size`, `gamma`, and `extra` (algorithm-specific
  hyper-parameters).
- `RuntimeConfig`: holds `requested_device`, resolved `torch.device`, `DeviceMetadata`,
  `seed`, and `num_workers`. The `device` and `device_meta` properties on `TrainingConfig`
  are convenience shortcuts into this sub-config.
- `load_training_config(path, overrides)`: parses a YAML file, applies optional
  key-level overrides, and returns a fully resolved `TrainingConfig`.
- `build_training_config(...)`: programmatic builder for tests and scripts (no file I/O).
- `TrainingConfig.to_dict()`: serialises the full config including `device_meta` for
  checkpoint manifests.

### `configs/training/runtime.mps.yaml`
Apple Silicon runtime configuration. Sets `device: mps`, `seed: 42`,
`num_workers: 0`. Includes operator notes on `PYTORCH_ENABLE_MPS_FALLBACK` and
mixed-precision requirements.

### `configs/training/runtime.cuda.yaml`
NVIDIA CUDA runtime configuration. Sets `device: cuda`, `seed: 42`,
`num_workers: 4`. Includes `cuda.allow_tf32`, `cudnn_benchmark`, `cudnn_deterministic`,
and `pin_memory` sub-fields for CUDA-specific tuning.

### `tests/training/test_device_runtime.py`
41 tests across 9 test classes:

| Class | Coverage |
|---|---|
| `TestBackendEnum` | Valid/invalid string parsing, case-insensitivity |
| `TestResolveDeviceCPU` | CPU always resolves without fallback |
| `TestResolveDeviceCUDA` | Available/unavailable paths, fallback recording |
| `TestResolveDeviceMPS` | Available/unavailable paths, fallback recording |
| `TestResolveDeviceAuto` | Priority order CUDA > MPS > CPU, auto requested-backend |
| `TestDeviceMetadata` | Field completeness, JSON serialisation, zero counts when unavailable |
| `TestMPSFallbackEnv` | `PYTORCH_ENABLE_MPS_FALLBACK` env-var reflection |
| `TestValidateMPSOps` | Non-MPS devices return empty list |
| `TestGetDeviceMetadata` | Convenience wrapper consistency |
| `TestSelfCheck` | Exit code 0 on CPU |
| `TestTrainingConfigDeviceResolution` | `build_training_config` resolves device, fallback recorded, `to_dict` embeds meta |

---

## Verification Results

```
pytest -q tests/training/test_device_runtime.py
41 passed in 0.65s
```

```
python -m mimic_sepsis_rl.training.device --self-check
Effective backend  : mps
Requested backend  : auto
PyTorch device str : mps
✅ Smoke test passed (tensor on mps).
✅ MPS op probes all passed.
```

---

## Key Design Decisions

- **Single entry point:** `resolve_device()` is the only place in the codebase where
  `torch.device(...)` is constructed from a string. Algorithm code calls this function
  rather than building devices directly.
- **Graceful fallback with traceability:** When a requested backend is unavailable,
  `fallback_applied=True` is recorded in `DeviceMetadata` so the checkpoint manifest
  reflects the actual runtime rather than the intended config.
- **MPS early validation:** `validate_mps_ops` is designed to be called *before* training
  begins so unsupported Metal ops are surfaced during setup, not after hours of training.
- **Generator portability:** Synthetic batch generation in dry-run creates tensors on CPU
  first and moves them to the target device, avoiding the MPS generator restriction.

---

## Artifacts Produced

| Path | Type | Description |
|---|---|---|
| `src/mimic_sepsis_rl/training/device.py` | Module | Device selection and fallback abstraction |
| `src/mimic_sepsis_rl/training/config.py` | Module | Training config that routes device through abstraction |
| `configs/training/runtime.mps.yaml` | Config | Apple Silicon MPS runtime settings |
| `configs/training/runtime.cuda.yaml` | Config | NVIDIA CUDA runtime settings |
| `tests/training/test_device_runtime.py` | Tests | 41 passing tests for runtime selection and fallback |

---

## Requirements Satisfied

- **PLAT-01:** One shared runtime path for MPS and CUDA — satisfied. The same
  `resolve_device` call works on both backends without forking training code.
- Backend metadata is recorded in `DeviceMetadata` and embedded in `TrainingConfig.to_dict()`,
  satisfying the reproducibility requirement for run records.