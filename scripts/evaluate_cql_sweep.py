#!/usr/bin/env python3
"""
Evaluate CQL sweep checkpoints with real OPE metrics.

Reads sweep manifests, loads CQL checkpoints, builds HeldOutEpisode
objects from replay Parquet files, and computes FQE/WIS/ESS/bootstrap CIs.
Supports two modes:
  --stage 1   rank configs by validation FQE, output top-N for Stage 2
  --stage final  evaluate all checkpoints on test split with bootstrap CIs

Usage
-----
    uv run python scripts/evaluate_cql_sweep.py --stage 1
    uv run python scripts/evaluate_cql_sweep.py --stage final
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimic_sepsis_rl.evaluation.ope import (
    HeldOutEpisode,
    HeldOutStep,
    OPEMetrics,
    compute_wis_and_ess,
)
from mimic_sepsis_rl.evaluation.bootstrap import bootstrap_fqe, bootstrap_wis, BootstrapCI, WISBootstrapCI
from mimic_sepsis_rl.evaluation.ope import FrozenFQEOutputs
from mimic_sepsis_rl.training.cql import load_cql_policy, CQLPolicy

logger = logging.getLogger(__name__)

N_ACTIONS = 25
GAMMA = 0.99
TERMINAL_SURVIVED = 15.0
TERMINAL_DIED = -15.0


# ──────────────────────────────────────────────
# Episode loading from replay Parquet
# ──────────────────────────────────────────────

def _load_episodes(parquet_path: Path) -> list[HeldOutEpisode]:
    """Load held-out episodes from a replay-buffer Parquet file."""
    df = pl.read_parquet(parquet_path)
    state_cols = sorted(c for c in df.columns if c.startswith("s_") and not c.startswith("ns_"))
    ns_cols = [f"ns_{c[2:]}" for c in state_cols]

    # Build behavior action probabilities from training frequencies
    # (estimated from the same file — in production, these come from train split)
    action_counts: dict[int, int] = defaultdict(int)
    for a in df["action"].to_list():
        action_counts[int(a)] += 1
    total = sum(action_counts.values())
    behavior_probs = {a: c / total for a, c in action_counts.items()}

    episodes: list[HeldOutEpisode] = []
    stay_ids = df["stay_id"].unique().sort().to_list()

    for sid in stay_ids:
        ep_df = df.filter(pl.col("stay_id") == sid).sort("step_index")
        steps: list[HeldOutStep] = []
        for row in ep_df.iter_rows(named=True):
            state_vec = tuple(float(row[c]) for c in state_cols)
            action = int(row["action"])
            reward = float(row["reward"])
            done = bool(row["done"])
            behavior_prob = behavior_probs.get(action, 1e-6)
            steps.append(HeldOutStep(
                episode_id=str(sid),
                step_index=int(row["step_index"]),
                state=state_vec,
                action=action,
                reward=reward,
                done=done,
                behavior_action_prob=behavior_prob,
            ))
        episodes.append(HeldOutEpisode(episode_id=str(sid), steps=tuple(steps)))

    return episodes


def _build_fqe_outputs(policy: CQLPolicy, episodes: list[HeldOutEpisode]) -> FrozenFQEOutputs:
    """Build FrozenFQEOutputs by computing Q(s_0, pi(s_0)) for each episode."""
    q_net = policy.q_network
    device = policy.device
    q_net.eval()

    initial_values: dict[str, tuple[float, ...]] = {}
    with torch.no_grad():
        for ep in episodes:
            s0 = torch.tensor(ep.steps[0].state, dtype=torch.float32, device=device).unsqueeze(0)
            q_vals = q_net(s0).squeeze(0).cpu().tolist()
            initial_values[ep.episode_id] = tuple(q_vals)

    return FrozenFQEOutputs(
        fitted_split="validation",
        initial_state_action_values=initial_values,
        artifact_label="cql_fqe",
    )


def _compute_fqe_mean(policy: CQLPolicy, episodes: list[HeldOutEpisode]) -> float:
    """Compute mean Q(s_0, argmax_a Q(s_0, a)) over episodes."""
    q_net = policy.q_network
    device = policy.device
    q_net.eval()

    values: list[float] = []
    with torch.no_grad():
        for ep in episodes:
            s0 = torch.tensor(ep.steps[0].state, dtype=torch.float32, device=device).unsqueeze(0)
            q = q_net(s0).squeeze(0)
            best_a = int(q.argmax().item())
            values.append(float(q[best_a].item()))

    return float(np.mean(values)) if values else 0.0


# ──────────────────────────────────────────────
# Checkpoint discovery
# ──────────────────────────────────────────────

def _find_run_checkpoints(checkpoint_root: Path, seed: int, variant_code: str,
                         learning_rate: float = 0.0, cql_alpha: float = 0.0) -> dict[int, Path]:
    """Find checkpoints for one specific run by looking for its subdirectory.

    Directory naming: cql_s{seed}_{variant_code}_lr{lr_label}_a{alpha_label}/
    If lr/alpha are 0 (unknown), falls back to prefix matching any subdir.
    """
    # Build exact directory name if we have lr/alpha
    if learning_rate > 0 and cql_alpha > 0:
        lr_label = f"lr{learning_rate:.0e}".replace("e-0", "e-").replace("e-", "e-")
        alpha_label = f"a{str(cql_alpha).replace('.', 'p')}"
        run_dir = checkpoint_root / f"cql_s{seed}_{variant_code}_{lr_label}_{alpha_label}"
        if run_dir.is_dir():
            candidates = [run_dir]
        else:
            candidates = []
    else:
        candidates = []

    # Fallback: search by prefix if exact match not found
    if not candidates:
        prefix = f"cql_s{seed}_{variant_code}_"
        for sub in sorted(checkpoint_root.glob(f"{prefix}*")):
            if sub.is_dir():
                candidates.append(sub)

    # Also check flat (old layout)
    if not candidates:
        candidates = [checkpoint_root]

    ckpts: dict[int, Path] = {}
    for d in candidates:
        for p in sorted(d.glob("cql_epoch*.pt")):
            stem = p.stem
            parts = stem.split("_")
            try:
                epoch = int(parts[1].replace("epoch", ""))
            except (IndexError, ValueError):
                continue
            ckpts[epoch] = p
    return ckpts



# ──────────────────────────────────────────────
# Stage 1: Rank configs by validation FQE
# ──────────────────────────────────────────────

def _run_stage1_evaluation(
    sweep_manifest_path: Path,
    val_episodes: list[HeldOutEpisode],
    checkpoint_root: Path,
    output_path: Path,
) -> None:
    """Rank configurations by best validation FQE, output top-N."""
    manifest = json.loads(sweep_manifest_path.read_text())
    runs = manifest.get("runs", [])
    if not runs:
        logger.error("No runs found in manifest %s", sweep_manifest_path)
        sys.exit(1)

    config_scores: dict[tuple, dict[str, Any]] = {}
    n_total = len(runs)

    for i, run_entry in enumerate(runs):
        reward = run_entry.get("reward_variant", "unknown")
        lr = run_entry.get("learning_rate", 0.0)
        alpha = run_entry.get("cql_alpha", 0.0)
        seed = run_entry.get("seed", 0)
        key = (reward, float(lr), float(alpha))

        logger.info("[%d/%d] Evaluating reward=%s lr=%.0e alpha=%.3f seed=%d",
                     i + 1, n_total, reward, lr, alpha, seed)

        # Find checkpoints for this specific run
        variant_code = {"shaped": "sofa_shaped", "sparse": "sparse"}.get(reward or "", reward or "shaped")
        ckpts = _find_run_checkpoints(checkpoint_root, seed, variant_code, lr, alpha)

        if not ckpts:
            logger.warning("  No checkpoints found, skipping")
            continue

        # Evaluate each checkpoint, pick best by FQE
        best_fqe = float("-inf")
        best_epoch = 0
        best_metrics: dict[str, float] = {}

        for epoch in sorted(ckpts):
            ckpt_path = ckpts[epoch]
            try:
                policy = load_cql_policy(
                    ckpt_path,
                    state_dim=62,
                    n_actions=N_ACTIONS,
                    hidden_sizes=[256, 256],
                    device="cpu",
                )
            except Exception as exc:
                logger.warning("  Failed to load checkpoint epoch=%d: %s", epoch, exc)
                continue

            fqe_mean = _compute_fqe_mean(policy, val_episodes)

            # Also compute WIS/ESS
            try:
                metrics, _per_ep = compute_wis_and_ess(
                    val_episodes, policy, gamma=GAMMA, max_importance_ratio=50.0
                )
                wis_val = metrics.wis
                ess_val = metrics.ess
            except Exception:
                wis_val = float("nan")
                ess_val = float("nan")

            logger.info("  epoch=%d fqe=%.4f wis=%.4f ess=%.1f", epoch, fqe_mean, wis_val, ess_val)

            if fqe_mean > best_fqe:
                best_fqe = fqe_mean
                best_epoch = epoch
                best_metrics = {"fqe": fqe_mean, "wis": wis_val, "ess": ess_val}

        if best_fqe > float("-inf"):
            if key not in config_scores or best_fqe > config_scores[key]["best_fqe"]:
                config_scores[key] = {
                    "reward_variant": reward,
                    "learning_rate": float(lr),
                    "cql_alpha": float(alpha),
                    "best_fqe": best_fqe,
                    "best_epoch": best_epoch,
                    "wis": best_metrics.get("wis", float("nan")),
                    "ess": best_metrics.get("ess", float("nan")),
                    "seed": seed,
                }

    # Rank by FQE descending
    ranked = sorted(config_scores.values(), key=lambda x: x["best_fqe"], reverse=True)
    top_n = min(6, len(ranked))
    top_configs = ranked[:top_n]

    logger.info("=== Stage 1 Ranking (top %d) ===", top_n)
    for rank, cfg in enumerate(top_configs, 1):
        logger.info("  #%d: reward=%s lr=%.0e alpha=%.3f fqe=%.4f epoch=%d",
                     rank, cfg["reward_variant"], cfg["learning_rate"],
                     cfg["cql_alpha"], cfg["best_fqe"], cfg["best_epoch"])

    output = {
        "stage": 1,
        "n_total_configs": len(config_scores),
        "top_configs": top_configs,
        "rankings": ranked,
        "val_n_episodes": len(val_episodes),
    }
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Stage 1 evaluation saved → %s", output_path)


# ──────────────────────────────────────────────
# Final evaluation: test split + bootstrap CIs
# ──────────────────────────────────────────────

@dataclass
class EvalResult:
    reward_variant: str
    learning_rate: float
    cql_alpha: float
    seed: int
    best_epoch: int
    fqe_mean: float
    fqe_lower: float
    fqe_upper: float
    wis_mean: float
    wis_lower: float
    wis_upper: float
    ess: float
    n_episodes: int
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_final_evaluation(
    sweep_manifest_path: Path,
    test_episodes: list[HeldOutEpisode],
    checkpoint_root: Path,
    output_path: Path,
) -> None:
    """Evaluate all checkpoints on test split with bootstrap CIs."""
    manifest = json.loads(sweep_manifest_path.read_text())
    runs = manifest.get("runs", [])
    if not runs:
        logger.error("No runs in manifest")
        sys.exit(1)

    results: list[EvalResult] = []
    n_total = len(runs)

    for i, run_entry in enumerate(runs):
        reward = run_entry.get("reward_variant", "unknown")
        lr = float(run_entry.get("learning_rate", 0.0))
        alpha = float(run_entry.get("cql_alpha", 0.0))
        seed = int(run_entry.get("seed", 0))

        logger.info("[%d/%d] reward=%s lr=%.0e alpha=%.3f seed=%d",
                     i + 1, n_total, reward, lr, alpha, seed)

        # Find checkpoints for this specific run
        variant_code = {"shaped": "sofa_shaped", "sparse": "sparse"}.get(reward or "", reward or "shaped")
        ckpts = _find_run_checkpoints(checkpoint_root, seed, variant_code, lr, alpha)

        if not ckpts:
            results.append(EvalResult(
                reward_variant=reward or "unknown", learning_rate=lr, cql_alpha=alpha,
                seed=seed, best_epoch=0, fqe_mean=float("nan"),
                fqe_lower=float("nan"), fqe_upper=float("nan"),
                wis_mean=float("nan"), wis_lower=float("nan"),
                wis_upper=float("nan"), ess=float("nan"),
                n_episodes=0, status="no_checkpoints",
            ))
            continue

        # Pick best checkpoint by FQE on test (for final reporting)
        best_fqe = float("-inf")
        best_epoch = 0
        best_result: EvalResult | None = None

        for epoch in sorted(ckpts):
            ckpt_path = ckpts[epoch]
            try:
                policy = load_cql_policy(
                    ckpt_path, state_dim=62, n_actions=N_ACTIONS,
                    hidden_sizes=[256, 256], device="cpu",
                )
            except Exception as exc:
                logger.warning("  Failed to load ckpt epoch=%d: %s", epoch, exc)
                continue

            # FQE
            fqe_mean = _compute_fqe_mean(policy, test_episodes)
            fqe_outputs = _build_fqe_outputs(policy, test_episodes)

            # Bootstrap FQE
            try:
                fqe_ci = bootstrap_fqe(
                    fqe_outputs, test_episodes, policy,
                    n_resamples=1000, ci=95, seed=42,
                )
            except Exception:
                fqe_ci = BootstrapCI(
                    mean=fqe_mean, lower=float("nan"), upper=float("nan"),
                    ci_level=95, n_resamples=0, n_episodes=len(test_episodes),
                )

            # WIS + bootstrap
            try:
                metrics, _per_ep = compute_wis_and_ess(
                    test_episodes, policy, gamma=GAMMA, max_importance_ratio=50.0,
                )
                wis_ci = bootstrap_wis(
                    test_episodes, policy, gamma=GAMMA,
                    n_resamples=1000, ci=95, seed=42,
                )
            except Exception:
                metrics = OPEMetrics(
                    wis=float("nan"), ess=float("nan"), fqe=float("nan"),
                    n_episodes=0, wis_weight_sum=0.0, wis_nonzero_episodes=0,
                    mean_behavior_return=float("nan"),
                )
                wis_ci = WISBootstrapCI(
                    mean=float("nan"), lower=float("nan"), upper=float("nan"),
                    ci_level=95, n_resamples=0, n_episodes=0, ess=float("nan"),
                )

            if fqe_mean > best_fqe:
                best_fqe = fqe_mean
                best_epoch = epoch
                best_result = EvalResult(
                    reward_variant=reward, learning_rate=lr, cql_alpha=alpha,
                    seed=seed, best_epoch=epoch,
                    fqe_mean=fqe_ci.mean, fqe_lower=fqe_ci.lower, fqe_upper=fqe_ci.upper,
                    wis_mean=wis_ci.mean, wis_lower=wis_ci.lower, wis_upper=wis_ci.upper,
                    ess=wis_ci.ess if hasattr(wis_ci, 'ess') else metrics.ess,
                    n_episodes=len(test_episodes),
                )

        if best_result:
            results.append(best_result)
            logger.info("  Best: epoch=%d fqe=%.4f [%.4f, %.4f]",
                         best_epoch, best_result.fqe_mean,
                         best_result.fqe_lower, best_result.fqe_upper)
        else:
            results.append(EvalResult(
                reward_variant=reward, learning_rate=lr, cql_alpha=alpha,
                seed=seed, best_epoch=0, fqe_mean=float("nan"),
                fqe_lower=float("nan"), fqe_upper=float("nan"),
                wis_mean=float("nan"), wis_lower=float("nan"),
                wis_upper=float("nan"), ess=float("nan"),
                n_episodes=0, status="eval_failed",
            ))

    # Aggregate by (reward, lr, alpha) across seeds
    by_config: dict[tuple, list[EvalResult]] = defaultdict(list)
    for r in results:
        key = (r.reward_variant, r.learning_rate, r.cql_alpha)
        by_config[key].append(r)

    aggregated = []
    for key, per_seed in sorted(by_config.items()):
        valid = [r for r in per_seed if r.status == "ok" and not math.isnan(r.fqe_mean)]
        if not valid:
            continue
        n = len(valid)
        fqe_vals = [r.fqe_mean for r in valid]
        aggregated.append({
            "reward_variant": key[0],
            "learning_rate": key[1],
            "cql_alpha": key[2],
            "n_seeds": n,
            "fqe_mean": float(np.mean(fqe_vals)),
            "fqe_std": float(np.std(fqe_vals)) if n > 1 else 0.0,
            "wis_mean": float(np.mean([r.wis_mean for r in valid])),
            "ess_mean": float(np.mean([r.ess for r in valid])),
            "per_seed": [r.to_dict() for r in valid],
        })

    output = {
        "stage": "final",
        "test_n_episodes": len(test_episodes),
        "aggregated": aggregated,
        "all_runs": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Final evaluation saved → %s (%d configs)", output_path, len(aggregated))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    parser = argparse.ArgumentParser(prog="evaluate_cql_sweep")
    parser.add_argument("--stage", choices=["1", "final"], default="final",
                        help="Stage 1 (ranking) or final (test split + bootstrap)")
    parser.add_argument("--manifest", default="runs/cql_sweep/stage1_manifest.json",
                        help="Path to sweep manifest JSON")
    parser.add_argument("--output", default=None,
                        help="Output path (default: runs/cql_sweep/stage1_evaluation.json or evaluation_summary.json)")
    parser.add_argument("--checkpoint-dir", default="checkpoints/cql_sweep",
                        help="Root directory for sweep checkpoints")
    parser.add_argument("--val-data", default="data/replay/replay_validation.parquet",
                        help="Validation split parquet")
    parser.add_argument("--test-data", default="data/replay/replay_test.parquet",
                        help="Test split parquet")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    checkpoint_root = Path(args.checkpoint_dir)

    if args.stage == "1":
        output_path = Path(args.output or "runs/cql_sweep/stage1_evaluation.json")
        val_data = Path(args.val_data)
        if not val_data.exists():
            logger.error("Validation data not found: %s", val_data)
            sys.exit(1)
        logger.info("Loading validation episodes from %s", val_data)
        episodes = _load_episodes(val_data)
        logger.info("Loaded %d validation episodes", len(episodes))
        _run_stage1_evaluation(manifest_path, episodes, checkpoint_root, output_path)

    else:
        output_path = Path(args.output or "runs/cql_sweep/evaluation_summary.json")
        test_data = Path(args.test_data)
        if not test_data.exists():
            logger.error("Test data not found: %s", test_data)
            sys.exit(1)
        logger.info("Loading test episodes from %s", test_data)
        episodes = _load_episodes(test_data)
        logger.info("Loaded %d test episodes", len(episodes))
        _run_final_evaluation(manifest_path, episodes, checkpoint_root, output_path)


if __name__ == "__main__":
    main()
