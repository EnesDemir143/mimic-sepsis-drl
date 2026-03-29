# Reproducibility Bundle

This benchmark is intended for retrospective research use. Any reported claim
must stay traceable to the exact cohort, feature, split, action, reward,
training, evaluation, and accelerator metadata that produced it.

## Required Bundle Contents

Every Phase 9 result package should contain the following frozen artifacts:

- Cohort definition metadata: cohort spec version, active inclusion or exclusion
  rules, and the source tables used to build the Sepsis-3 cohort.
- Feature dictionary metadata: feature spec version, total feature count, and
  whether missingness flags were emitted.
- Split manifest metadata: split spec version, manifest seed, source episode
  set, and leakage status.
- Action bin metadata: action spec version, split-manifest seed used during
  fitting, and the frozen vasopressor or fluid bin edges.
- Reward metadata: reward spec version, reward variant, and shaping weights.
- Per-run experiment artifacts: training config path, dataset metadata path,
  checkpoint path, checkpoint manifest path, final metrics, and dataset
  contract metadata.
- Evaluation outputs: WIS, ESS, FQE, held-out contract checks, and any paired
  safety review summary.
- Recorded accelerator backend metadata from the checkpoint manifest:
  requested backend, effective backend, `torch_device_str`, `torch_version`,
  and fallback flags.
- Ablation comparisons: reward-shaping, action-granularity, timestep,
  missingness-flag, and feature-subset comparisons under the same benchmark
  version.

## Rerun Workflow

Use the bundle to rerun a reported experiment in this order:

1. Confirm the cohort, feature dictionary, split manifest, action bins, and
   reward config versions match the bundle metadata.
2. Launch the saved training config for each reported algorithm:

```bash
python -m mimic_sepsis_rl.training.experiment_runner \
  --algorithm cql \
  --config configs/training/cql.yaml \
  --device cpu
```

3. Verify the new checkpoint manifest records the same dataset contract and the
   intended accelerator backend.
4. Recompute held-out OPE and safety outputs using the frozen evaluation
   protocol from [docs/evaluation_protocol.md](./evaluation_protocol.md).
5. Rebuild the reproducibility bundle and compare the top-line metrics,
   ablation rows, and backend metadata against the original package.

## Audit Checklist

- The cohort spec version matches the cohort artifact used downstream.
- The feature dictionary spec version and feature count match the dataset
  metadata used in training and evaluation.
- The split manifest seed matches the action-bin artifact and replay dataset
  contract.
- The reward variant in the bundle matches both the replay metadata and the
  ablation metadata.
- Every run includes a checkpoint manifest with recorded accelerator metadata;
  backend values are never inferred from filenames or machine assumptions.
- OPE reports reference held-out data only and do not fit FQE on the held-out
  split.
- Ablation rows keep one benchmark version and one metadata schema across all
  variants so comparison tables remain attributable.

## Notes

- Backend changes across CPU, MPS, and CUDA can change numerical values due to
  floating-point ordering; record the backend, do not hide it.
- Offline policy evaluation and safety review remain retrospective evidence
  only. The bundle improves auditability, not bedside validity.
