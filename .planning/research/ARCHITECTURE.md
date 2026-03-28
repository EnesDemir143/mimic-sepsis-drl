# Architecture Research

**Domain:** Offline RL research pipeline over MIMIC-IV sepsis data
**Researched:** 2026-03-28
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
┌──────────────────────────────────────────────────────────────┐
│                        Source Data Layer                     │
├──────────────────────────────────────────────────────────────┤
│  MIMIC-IV SQL tables   Cohort specs   Outcome definitions    │
├──────────────────────────────────────────────────────────────┤
│                     Data Construction Layer                  │
├──────────────────────────────────────────────────────────────┤
│  Cohort builder -> Onset detector -> Episode generator       │
│  Feature extractor -> Action builder -> Reward builder       │
├──────────────────────────────────────────────────────────────┤
│                     Dataset Contract Layer                   │
├──────────────────────────────────────────────────────────────┤
│  State tables   Transition dataset   Split manifests         │
├──────────────────────────────────────────────────────────────┤
│                    Modeling and Evaluation                   │
├──────────────────────────────────────────────────────────────┤
│  Baselines   CQL/BCQ/IQL trainers   OPE evaluators          │
│  Safety checks   Ablations   Reporting                       │
└──────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Cohort/onset pipeline | Defines usable episodes and timestamps | SQL extraction plus tested Python transformation steps |
| MDP builder | Produces state, action, reward, and transition artifacts | Columnar batch jobs writing Parquet/CSV manifests |
| Training layer | Fits baselines and offline RL policies from frozen artifacts | PyTorch/d3rlpy trainers parameterized by configs |
| Evaluation layer | Computes OPE, diagnostics, safety checks, and plots | Dedicated evaluation modules reading only held-out artifacts |
| Reporting layer | Packages tables, figures, and experiment summaries | Scripted exports for thesis/paper appendices |

## Recommended Project Structure

```text
src/
├── data/                # Cohort queries, onset logic, episode builders
│   ├── sql/             # Versioned SQL extracts against MIMIC-IV
│   └── pipelines/       # Deterministic artifact generation steps
├── mdp/                 # Feature, action, reward, split, and transition builders
│   ├── features/        # State construction and preprocessing
│   ├── actions/         # Dose conversion and binning
│   └── rewards/         # Reward specification and calculators
├── policies/            # Baselines and offline RL algorithms
│   ├── baselines/       # Clinician, zero-treatment, BC
│   └── offline_rl/      # CQL, BCQ, IQL wrappers/trainers
├── evaluation/          # OPE, safety checks, ablations, figures
├── reporting/           # Tables, exports, and publication artifacts
└── utils/               # Shared config, logging, and IO helpers
configs/                 # Hydra or equivalent experiment configs
artifacts/               # Materialized datasets, checkpoints, plots
notebooks/               # Exploratory analysis only
tests/                   # Leakage guards and pipeline regression tests
```

### Structure Rationale

- **`src/data/`:** Keeps cohort and onset logic isolated because these are the most audit-sensitive clinical definitions.
- **`src/mdp/`:** Freezes the dataset contract before training code touches it.
- **`src/policies/`:** Allows algorithm comparisons without entangling ETL logic.
- **`src/evaluation/`:** Prevents training scripts from silently evaluating on the wrong split.
- **`tests/`:** Gives leakage and unit-conversion checks a permanent home outside notebooks.

## Architectural Patterns

### Pattern 1: Immutable Artifact Stages

**What:** Each pipeline stage writes a versioned artifact and never mutates upstream outputs in place.
**When to use:** Always, from cohort extraction onward.
**Trade-offs:** More disk usage, much easier debugging and reproducibility.

### Pattern 2: Fit/Transform Boundary Discipline

**What:** Any learned transform is fit on train only, then reused everywhere else.
**When to use:** Scaling, imputation fallbacks, binning thresholds, behavior-policy estimators.
**Trade-offs:** Slightly more bookkeeping, dramatically lower leakage risk.

### Pattern 3: Evaluation Isolation

**What:** OPE, safety review, and ablation code consume frozen checkpoints and held-out data only.
**When to use:** After baseline training begins.
**Trade-offs:** Some duplicated loading utilities, stronger guarantee that results are reproducible and reviewable.

## Data Flow

### Request Flow

```text
MIMIC-IV tables
    ↓
Cohort extraction
    ↓
Sepsis onset assignment
    ↓
Episode window generation
    ↓
State/action/reward builders
    ↓
Transition dataset + split manifests
    ↓
Baselines and offline RL training
    ↓
OPE + safety checks + reporting
```

### State Management

```text
Raw tables
    ↓
Materialized intermediate artifacts
    ↓
Frozen train/val/test dataset views
    ↓
Run configs + checkpoints + metrics
```

### Key Data Flows

1. **Clinical ETL flow:** source SQL tables become cohort and episode artifacts.
2. **MDP flow:** episode artifacts become state, action, reward, and transition tables.
3. **Experiment flow:** frozen datasets become runs, metrics, figures, and summary reports.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Pilot subsets | Local artifacts and single-machine training are fine |
| Full internal cohort | Prefer columnar artifacts, batched preprocessing, and cached intermediate tables |
| Many ablations and sweeps | Add experiment registry, config composition, and stricter artifact naming |

### Scaling Priorities

1. **First bottleneck:** repeated raw-table scans — fix with materialized cohort and episode artifacts.
2. **Second bottleneck:** experiment sprawl — fix with structured configs and a results registry.

## Anti-Patterns

### Anti-Pattern 1: Monolithic Notebook Workflow

**What people do:** Query data, build features, train models, and evaluate in one notebook.
**Why it's wrong:** Hidden state and manual cell order make results irreproducible.
**Do this instead:** Separate audited pipelines and make notebooks consumers of frozen artifacts.

### Anti-Pattern 2: Training Code Owns Preprocessing

**What people do:** Let each trainer compute its own scaling, binning, and reward artifacts.
**Why it's wrong:** Model comparisons become unfair and leakage is hard to detect.
**Do this instead:** Freeze the MDP contract once and require all trainers to consume it unchanged.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| MIMIC-IV warehouse | Read-only SQL extraction | Preserve query text and cohort manifests for auditability |
| Artifact store | Local filesystem or object storage | Organize by cohort version, split version, and experiment id |
| Experiment tracker | MLflow or equivalent | Optional at first, valuable once sweeps begin |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `data ↔ mdp` | Materialized tables | Do not pass ad hoc in-memory objects across this boundary |
| `mdp ↔ policies` | Frozen dataset contract | All algorithms must consume the same tensors and manifests |
| `policies ↔ evaluation` | Checkpoints + run metadata | Evaluation should not depend on training-time mutable state |

## Sources

- `implematation_plan_gpt.md`
- Standard architecture patterns for clinical ML and offline RL experimentation

---
*Architecture research for: offline sepsis RL benchmark*
*Researched: 2026-03-28*
