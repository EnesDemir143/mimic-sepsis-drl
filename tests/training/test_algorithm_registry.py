"""
Regression tests for the shared offline RL algorithm registry.

Covers:
- Registry membership for CQL, BCQ, and IQL
- Config resolution through the shared runtime abstraction
- Contract parity between the shipped Phase 8 config files
- Experiment-resolution metadata and action-contract loading
- Friendly pending-algorithm failures before Phase 08-02 lands
- CLI listing for the shared experiment runner
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mimic_sepsis_rl.datasets.transitions import TransitionDatasetMeta
from mimic_sepsis_rl.training.experiment_runner import main, resolve_experiment
from mimic_sepsis_rl.training.registry import get_default_registry

CONFIG_DIR = Path("configs/training")
SUPPORTED_ALGORITHMS = ("bcq", "cql", "iql")


def _load_config_payload(path: Path) -> dict:
    with path.open("r") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def _write_dataset_meta(tmp_path: Path, *, n_actions: int = 37) -> Path:
    meta = TransitionDatasetMeta(
        spec_version="1.0.0",
        n_episodes=8,
        n_transitions=64,
        state_dim=33,
        n_actions=n_actions,
        split_label="train",
        manifest_seed=42,
        action_spec_version="1.0.0",
        reward_spec_version="1.0.0",
        feature_columns=("sofa", "lactate", "map"),
    )
    path = tmp_path / "replay_train_meta.json"
    path.write_text(json.dumps(meta.to_dict(), indent=2))
    return path


def _write_temp_config(
    tmp_path: Path,
    *,
    algorithm: str,
    dataset_meta_path: Path,
) -> Path:
    payload = {
        "algorithm": algorithm,
        "schema_version": "1.0.0",
        "runtime": {
            "device": "auto",
            "seed": 123,
            "num_workers": 2,
        },
        "dataset_path": "data/replay/replay_train.parquet",
        "dataset_meta_path": str(dataset_meta_path),
        "n_epochs": 3,
        "batch_size": 32,
        "gamma": 0.99,
        "checkpoint": {
            "checkpoint_dir": str(tmp_path / f"{algorithm}_checkpoints"),
            "save_every_n_epochs": 1,
            "keep_last_n": 2,
        },
        "logging": {
            "log_dir": str(tmp_path / f"{algorithm}_runs"),
            "experiment_name": f"{algorithm}_test",
            "log_every_n_steps": 10,
        },
    }
    path = tmp_path / f"{algorithm}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_default_registry_lists_all_phase8_algorithms() -> None:
    registry = get_default_registry()

    assert registry.names() == SUPPORTED_ALGORITHMS
    for algorithm in SUPPORTED_ALGORITHMS:
        definition = registry.require(algorithm)
        assert definition.default_config_path == CONFIG_DIR / f"{algorithm}.yaml"


@pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
def test_registry_loads_configs_through_shared_runtime_surface(
    algorithm: str,
) -> None:
    registry = get_default_registry()

    cfg = registry.load_config(algorithm, device="cpu")

    assert cfg.algorithm == algorithm
    assert cfg.dataset_path == Path("data/replay/replay_train.parquet")
    assert cfg.dataset_meta_path == Path("data/replay/replay_train_meta.json")
    assert cfg.device.type == "cpu"
    assert cfg.runtime.requested_device == "cpu"
    assert cfg.runtime.seed == 42
    assert cfg.runtime.num_workers == 0


def test_phase8_configs_share_cql_dataset_and_runtime_contract() -> None:
    cql_payload = _load_config_payload(CONFIG_DIR / "cql.yaml")

    for algorithm in ("bcq", "iql"):
        payload = _load_config_payload(CONFIG_DIR / f"{algorithm}.yaml")
        assert payload["runtime"] == cql_payload["runtime"]
        assert payload["dataset_path"] == cql_payload["dataset_path"]
        assert payload["dataset_meta_path"] == cql_payload["dataset_meta_path"]
        assert payload["gamma"] == cql_payload["gamma"]
        assert payload["batch_size"] == cql_payload["batch_size"]


def test_resolve_experiment_rejects_algorithm_mismatch() -> None:
    with pytest.raises(ValueError, match="Config algorithm mismatch"):
        resolve_experiment(
            "bcq",
            config_path=CONFIG_DIR / "cql.yaml",
            device="cpu",
        )


def test_resolve_experiment_loads_shared_dataset_contract(tmp_path: Path) -> None:
    dataset_meta_path = _write_dataset_meta(tmp_path, n_actions=37)
    config_path = _write_temp_config(
        tmp_path,
        algorithm="bcq",
        dataset_meta_path=dataset_meta_path,
    )

    resolved = resolve_experiment("bcq", config_path=config_path, device="cpu")

    assert resolved.dataset_contract is not None
    assert resolved.dataset_contract.n_actions == 37
    assert resolved.dataset_contract.action_spec_version == "1.0.0"
    assert resolved.dataset_contract.reward_spec_version == "1.0.0"
    assert resolved.resolve_n_actions() == 37
    assert resolved.config.runtime.seed == 123
    assert resolved.config.runtime.num_workers == 2
    assert resolved.config.device.type == "cpu"


def test_pending_algorithms_raise_clear_message(tmp_path: Path) -> None:
    dataset_meta_path = _write_dataset_meta(tmp_path)
    config_path = _write_temp_config(
        tmp_path,
        algorithm="iql",
        dataset_meta_path=dataset_meta_path,
    )
    resolved = resolve_experiment("iql", config_path=config_path, device="cpu")

    with pytest.raises(NotImplementedError, match="Phase 08-02"):
        resolved.execute(dry_run=True)


def test_list_algorithms_cli_mentions_all_registry_entries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--list-algorithms"])
    captured = capsys.readouterr()

    for algorithm in SUPPORTED_ALGORITHMS:
        assert algorithm in captured.out
