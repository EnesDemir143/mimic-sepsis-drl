#!/usr/bin/env python3
"""Compute Stage 2 multi-seed validation FQE mean±std from checkpoints."""

import json, sys, logging
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mimic_sepsis_rl.training.cql import load_cql_policy
from mimic_sepsis_rl.evaluation.ope import HeldOutEpisode, HeldOutStep

logger = logging.getLogger(__name__)
N_ACTIONS = 25

def _load_episodes(parquet_path: Path) -> list[HeldOutEpisode]:
    import polars as pl
    df = pl.read_parquet(parquet_path)
    state_cols = sorted(c for c in df.columns if c.startswith("s_") and not c.startswith("ns_"))
    episodes: list[HeldOutEpisode] = []
    for sid in df["stay_id"].unique().sort().to_list():
        ep_df = df.filter(pl.col("stay_id") == sid).sort("step_index")
        steps = []
        for row in ep_df.iter_rows(named=True):
            state_vec = tuple(float(row[c]) for c in state_cols)
            steps.append(HeldOutStep(
                episode_id=str(sid), step_index=int(row["step_index"]),
                state=state_vec, action=int(row["action"]),
                reward=float(row["reward"]), done=bool(row["done"]),
                behavior_action_prob=1.0,
            ))
        episodes.append(HeldOutEpisode(episode_id=str(sid), steps=tuple(steps)))
    return episodes

def _compute_fqe_mean(policy, episodes):
    q_net = policy.q_network
    device = policy.device
    q_net.eval()
    values = []
    with torch.no_grad():
        for ep in episodes:
            s0 = torch.tensor(ep.steps[0].state, dtype=torch.float32, device=device).unsqueeze(0)
            q = q_net(s0).squeeze(0)
            best_a = int(q.argmax().item())
            values.append(float(q[best_a].item()))
    return float(np.mean(values)) if values else 0.0

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    basedir = Path("/Users/enesdemir/Documents/mimic-sepsis")
    val_path = basedir / "data" / "replay" / "replay_validation.parquet"
    ckpt_root = basedir / "checkpoints" / "cql_sweep"
    manifest_path = basedir / "runs" / "cql_sweep" / "stage2_manifest.json"
    
    # Validation episodes
    logger.info("Loading validation episodes...")
    episodes = _load_episodes(val_path)
    logger.info("Loaded %d episodes", len(episodes))
    
    # Stage 2 configs from manifest
    manifest = json.loads(manifest_path.read_text())
    top_configs = manifest["top_configs"]
    
    # Build config list: (reward_variant, lr, alpha)
    configs = []
    for tc in top_configs:
        configs.append((
            tc["reward_variant"],
            float(tc["learning_rate"]),
            float(tc["cql_alpha"]),
        ))
    
    # All seeds: 42 (from Stage 1) + 123, 456, 789, 1024
    seeds = [42, 123, 456, 789, 1024]
    
    # Load Stage 1 seed 42 FQE values
    stage1_path = basedir / "runs" / "cql_sweep" / "stage1_evaluation.json"
    stage1 = json.loads(stage1_path.read_text())
    seed42_fqes = {}
    for r in stage1["rankings"]:
        key = (r["reward_variant"], float(r["learning_rate"]), float(r["cql_alpha"]))
        if key in configs:
            seed42_fqes[key] = r["best_fqe"]
    
    variant_map = {"sparse": "sparse", "shaped": "sofa_shaped"}
    results_by_config = {}
    
    for rvar, lr, alpha in configs:
        key = (rvar, lr, alpha)
        fqes = []
        
        # Seed 42 from Stage 1
        if key in seed42_fqes:
            fqes.append(seed42_fqes[key])
            logger.info("%s lr=%.0e alpha=%.2f seed=42 fqe=%.4f", rvar, lr, alpha, seed42_fqes[key])
        
        # Other seeds from checkpoints
        vcode = variant_map.get(rvar, rvar)
        for seed in seeds:
            if seed == 42:
                continue
            lr_label = f"lr{lr:.0e}".replace("e-0", "e-").replace("e-", "e-")
            alpha_label = f"a{str(alpha).replace('.', 'p')}"
            run_dir = ckpt_root / f"cql_s{seed}_{vcode}_{lr_label}_{alpha_label}"
            ckpt = run_dir / "cql_epoch0200_step0007000.pt"
            if not ckpt.exists():
                logger.warning("  Missing: %s", ckpt)
                continue
            try:
                policy = load_cql_policy(ckpt, state_dim=62, n_actions=N_ACTIONS, hidden_sizes=[256, 256], device="cpu")
                fqe = _compute_fqe_mean(policy, episodes)
                fqes.append(fqe)
                logger.info("%s lr=%.0e alpha=%.2f seed=%d fqe=%.4f", rvar, lr, alpha, seed, fqe)
            except Exception as e:
                logger.warning("  Failed seed=%d: %s", seed, e)
        
        if fqes:
            fqe_arr = np.array(fqes)
            results_by_config[key] = {
                "reward_variant": rvar, "learning_rate": lr, "cql_alpha": alpha,
                "n_seeds": len(fqes),
                "fqe_mean": float(np.mean(fqe_arr)),
                "fqe_std": float(np.std(fqe_arr, ddof=1)) if len(fqes) > 1 else 0.0,
                "fqe_values": fqes,
            }
    
    # Print results sorted by mean FQE descending
    print("\n=== Stage 2 Multi-Seed Validation FQE ===\n")
    print(f"{'Rank':<5} {'Reward':<8} {'LR':<10} {'Alpha':<8} {'Seeds':<6} {'FQE Mean':<10} {'FQE Std':<10} {'Values'}")
    print("-" * 80)
    sorted_results = sorted(results_by_config.values(), key=lambda x: x["fqe_mean"], reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        vals_str = ", ".join(f"{v:.4f}" for v in r["fqe_values"])
        print(f"{rank:<5} {r['reward_variant']:<8} {r['learning_rate']:<10.0e} {r['cql_alpha']:<8.2f} {r['n_seeds']:<6} {r['fqe_mean']:<10.4f} {r['fqe_std']:<10.4f} [{vals_str}]")
    
    # Save
    out_path = basedir / "runs" / "cql_sweep" / "stage2_multiseed_validation.json"
    out_path.write_text(json.dumps({
        "stage": 2,
        "split": "validation",
        "n_episodes": len(episodes),
        "seeds": seeds,
        "results": sorted_results,
    }, indent=2))
    logger.info("Saved to %s", out_path)

if __name__ == "__main__":
    main()
