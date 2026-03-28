# Stack Research

**Domain:** Offline reinforcement learning pipeline for retrospective sepsis treatment optimization
**Researched:** 2026-03-28
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11.x | Main implementation language for ETL, modeling, evaluation, and reporting | Scientific Python, PyTorch, and healthcare data tooling are most mature here |
| PyTorch | 2.x stable | Neural network and offline RL training backend | Flexible enough for CQL, BCQ, IQL, and custom loss shaping while supporting both Apple Silicon `MPS` and NVIDIA `CUDA` backends |
| SQL + Parquet | PostgreSQL-compatible SQL + Parquet artifacts | Extract source tables from MIMIC-IV and freeze reusable intermediate datasets | Keeps cohort logic auditable while allowing fast offline iteration on materialized artifacts |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Polars | 1.x | Columnar feature engineering and artifact generation | Use for large episode/state/action tables where pandas becomes memory-heavy |
| scikit-learn | 1.5+ | Preprocessing utilities, split helpers, calibration support, baseline models | Use for normalization, imputers, train/val/test utilities, and simple baselines |
| d3rlpy | Latest stable compatible with PyTorch 2.x | Reference implementation surface for offline RL algorithms | Use if its discrete offline RL implementations match the project's action-space and logging needs |
| Hydra | 1.3+ | Structured experiment configuration | Use once multiple datasets, reward variants, and algorithm sweeps need reproducible config composition |
| MLflow | 2.x | Experiment tracking and model artifact registry | Use when runs, metrics, and checkpoints need a single reproducible index |
| Matplotlib / Seaborn | Current stable | Diagnostic plots and paper-ready figures | Use for reward histograms, action heatmaps, training curves, and ablation charts |
| `torch.device` runtime abstraction | Built-in | Unified device selection across CPU, `MPS`, and `CUDA` | Use in the main training entrypoint so algorithm code does not fork per laptop |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `uv` | Environment and dependency management | Prefer a locked environment over ad hoc `pip install` history |
| `pytest` | Automated verification for data transforms and training utilities | Focus on cohort rules, leakage guards, binning, and reward logic |
| `ruff` | Fast linting and import hygiene | Useful once the codebase grows beyond notebooks into packages/scripts |
| Per-backend lockfiles | Keep `mps` and `cuda` environments reproducible | Store exact wheel/source differences without forking the project structure |
| Jupyter | Exploration and one-off inspection | Keep notebooks read-only for analysis; production logic should live in modules |

## Installation

```bash
# Core environment
uv venv
source .venv/bin/activate
uv pip install polars scikit-learn pyarrow

# Apple Silicon / Metal (MPS)
# Pin the official PyTorch build that exposes MPS support in the lockfile used on macOS.
uv pip install torch

# Offline RL and experiment tooling
uv pip install d3rlpy hydra-core mlflow matplotlib seaborn

# Development
uv pip install pytest ruff jupyter

# NVIDIA / CUDA
# Use a separate lockfile or environment definition that pins the official CUDA-enabled
# PyTorch wheel matching the laptop's driver/runtime.
uv pip install torch
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Polars | pandas | Choose pandas if the team already has mature utilities and the dataset size remains manageable in memory |
| d3rlpy | Clean-room PyTorch implementations | Prefer custom implementations if algorithm support or auditability requirements exceed library flexibility |
| Unified `torch.device` code path | Separate `cuda` and `mps` training scripts | Split scripts only if a specific algorithm or kernel is fundamentally unsupported on one backend |
| MLflow | Weights & Biases | Prefer W&B if the team already standardizes on hosted experiment dashboards |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Notebook-only pipelines | Hidden state and manual reruns make cohort logic and preprocessing irreproducible | Module/script-based pipelines with explicit artifact outputs |
| Full-dataset preprocessing fit | Leaks future information into validation and test results | Train-only fit with frozen transforms applied downstream |
| CUDA-only core training logic | Breaks the MacBook `MPS` target and forks the benchmark | Backend-agnostic PyTorch ops with guarded fast paths |
| Generic online-RL-first libraries | They assume an environment loop and often lack robust offline evaluation ergonomics | PyTorch plus offline-RL-specific tooling |

## Stack Patterns by Variant

**If MIMIC access remains in a SQL warehouse:**
- Keep cohort extraction and onset logic in auditable SQL.
- Because schema-level joins and time filters are easier to review there than in notebook code.

**If running on Apple Silicon laptops:**
- Keep the main path on PyTorch ops known to run on `MPS`, and document CPU fallback for unsupported operations.
- Because some kernels and third-party extensions still have weaker `MPS` coverage than `CUDA`.

**If running on NVIDIA laptops:**
- Pin the PyTorch build against the intended CUDA runtime and record driver compatibility with the experiment metadata.
- Because a working CUDA stack is partly an environment problem, not just a Python dependency problem.

**If offline RL library support is incomplete:**
- Keep dataset contracts, replay buffers, and evaluation interfaces stable while implementing algorithms directly in PyTorch.
- Because algorithm swaps should not force a rewrite of the MDP pipeline.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| Python 3.11.x | PyTorch 2.x | Safe baseline for current scientific Python stacks |
| PyTorch 2.x | d3rlpy latest stable | Verify exact package pins before training; keep them locked in the environment file |
| PyTorch 2.x with `MPS` | Apple Silicon macOS environment | Prefer pure PyTorch operators and test unsupported-op fallbacks early |
| PyTorch 2.x with `CUDA` | NVIDIA driver/runtime pinned for the laptop | Record the exact CUDA runtime used in experiment metadata |
| Polars 1.x | PyArrow current stable | Required for efficient Parquet interchange |

## Sources

- `implematation_plan_gpt.md` — locked methodological decisions and algorithm shortlist
- MIMIC-IV schema/query workflow assumptions — cohort extraction and artifact strategy
- Standard scientific Python and offline RL practice — stack selection to be pinned at implementation time

---
*Stack research for: offline sepsis RL benchmark*
*Researched: 2026-03-28*
