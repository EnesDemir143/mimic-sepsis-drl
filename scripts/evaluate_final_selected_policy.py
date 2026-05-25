#!/usr/bin/env python3
"""Evaluate one pre-selected CQL policy on the held-out test split.

This script is intentionally narrow: it evaluates exactly one checkpoint passed
via --checkpoint. It must be used only after the final policy/checkpoint has
already been selected from validation results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimic_sepsis_rl.evaluation.bootstrap import BootstrapCI, WISBootstrapCI, bootstrap_fqe, bootstrap_wis
from mimic_sepsis_rl.evaluation.ope import OPEMetrics, compute_wis_and_ess
from mimic_sepsis_rl.training.cql import load_cql_policy
from scripts.evaluate_cql_sweep import (
    GAMMA,
    N_ACTIONS,
    _build_fqe_outputs,
    _compute_fqe_mean,
    _load_episodes,
)

logger = logging.getLogger(__name__)


@dataclass
class FinalSelectedPolicyResult:
    checkpoint: str
    split: str
    fqe_mean: float
    fqe_lower: float
    fqe_upper: float
    wis_mean: float
    wis_lower: float
    wis_upper: float
    ess: float
    n_episodes: int
    bootstrap_resamples: int
    ci_level: int
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate_selected_checkpoint(
    checkpoint: Path,
    test_data: Path,
    output: Path,
    bootstrap_resamples: int,
    seed: int,
) -> None:
    episodes = _load_episodes(test_data)
    logger.info("Loaded %d test episodes from %s", len(episodes), test_data)

    policy = cast(Any, load_cql_policy(
        checkpoint,
        state_dim=62,
        n_actions=N_ACTIONS,
        hidden_sizes=[256, 256],
        device="cpu",
    ))

    fqe_mean = _compute_fqe_mean(policy, episodes)
    fqe_outputs = _build_fqe_outputs(policy, episodes)

    try:
        fqe_ci = bootstrap_fqe(
            fqe_outputs,
            episodes,
            policy,
            n_resamples=bootstrap_resamples,
            ci=95,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI fallback
        logger.warning("FQE bootstrap failed: %s", exc)
        fqe_ci = BootstrapCI(
            mean=fqe_mean,
            lower=float("nan"),
            upper=float("nan"),
            ci_level=95,
            n_resamples=0,
            n_episodes=len(episodes),
        )

    try:
        metrics, _per_ep = compute_wis_and_ess(
            episodes,
            policy,
            gamma=GAMMA,
            max_importance_ratio=50.0,
        )
        wis_ci = bootstrap_wis(
            episodes,
            policy,
            gamma=GAMMA,
            n_resamples=bootstrap_resamples,
            ci=95,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI fallback
        logger.warning("WIS/ESS computation failed: %s", exc)
        metrics = OPEMetrics(
            wis=float("nan"),
            ess=float("nan"),
            fqe=float("nan"),
            n_episodes=0,
            wis_weight_sum=0.0,
            wis_nonzero_episodes=0,
            mean_behavior_return=float("nan"),
        )
        wis_ci = WISBootstrapCI(
            mean=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            ci_level=95,
            n_resamples=0,
            n_episodes=0,
            ess=float("nan"),
        )

    result = FinalSelectedPolicyResult(
        checkpoint=str(checkpoint),
        split="test",
        fqe_mean=fqe_ci.mean,
        fqe_lower=fqe_ci.lower,
        fqe_upper=fqe_ci.upper,
        wis_mean=wis_ci.mean,
        wis_lower=wis_ci.lower,
        wis_upper=wis_ci.upper,
        ess=wis_ci.ess if hasattr(wis_ci, "ess") else metrics.ess,
        n_episodes=len(episodes),
        bootstrap_resamples=bootstrap_resamples,
        ci_level=95,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    logger.info("Final selected-policy test evaluation saved to %s", output)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    parser = argparse.ArgumentParser(prog="evaluate_final_selected_policy")
    parser.add_argument("--checkpoint", required=True, help="Pre-selected final CQL checkpoint path")
    parser.add_argument("--test-data", default="data/replay/replay_test.parquet", help="Held-out test split parquet")
    parser.add_argument("--output", default="runs/cql_sweep/final_test_evaluation.json", help="Output JSON path")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000, help="Patient-level bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    test_data = Path(args.test_data)
    output = Path(args.output)

    if not checkpoint.exists():
        logger.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)
    if not test_data.exists():
        logger.error("Test data not found: %s", test_data)
        sys.exit(1)
    if args.bootstrap_resamples <= 0:
        logger.error("--bootstrap-resamples must be positive")
        sys.exit(1)

    _evaluate_selected_checkpoint(
        checkpoint=checkpoint,
        test_data=test_data,
        output=output,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
