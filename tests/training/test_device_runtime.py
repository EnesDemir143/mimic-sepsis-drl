"""
Regression tests for the cross-platform device runtime abstraction.

Covers:
- Backend enum parsing (valid and invalid strings)
- resolve_device: auto, explicit cuda/mps/cpu selection
- Fallback to CPU when requested backend is unavailable
- DeviceMetadata completeness and field types
- MPS fallback env-var detection
- validate_mps_ops: non-MPS device returns empty list
- CLI self-check exit-code contract
- TrainingConfig device resolution via build_training_config
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from mimic_sepsis_rl.training.config import build_training_config
from mimic_sepsis_rl.training.device import (
    DEVICE_MODULE_VERSION,
    Backend,
    DeviceMetadata,
    _self_check,
    get_device_metadata,
    resolve_device,
    validate_mps_ops,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_cuda(available: bool):
    """Context manager that patches torch.cuda.is_available."""
    return patch(
        "mimic_sepsis_rl.training.device._cuda_available", return_value=available
    )


def _patch_mps(available: bool, built: bool = True):
    """Context manager that patches MPS availability probes."""
    return (
        patch("mimic_sepsis_rl.training.device._mps_available", return_value=available),
        patch("mimic_sepsis_rl.training.device._mps_built", return_value=built),
    )


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------


class TestBackendEnum:
    """Verify Backend enum parsing."""

    def test_parse_cuda_lowercase(self) -> None:
        assert Backend.from_str("cuda") == Backend.CUDA

    def test_parse_mps_uppercase(self) -> None:
        assert Backend.from_str("MPS") == Backend.MPS

    def test_parse_cpu(self) -> None:
        assert Backend.from_str("cpu") == Backend.CPU

    def test_parse_auto(self) -> None:
        assert Backend.from_str("auto") == Backend.AUTO

    def test_parse_mixed_case(self) -> None:
        assert Backend.from_str("Auto") == Backend.AUTO

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            Backend.from_str("tpu")

    def test_invalid_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            Backend.from_str("")


# ---------------------------------------------------------------------------
# resolve_device: explicit CPU
# ---------------------------------------------------------------------------


class TestResolveDeviceCPU:
    """CPU selection is always available and never falls back."""

    def test_cpu_explicit(self) -> None:
        device, meta = resolve_device("cpu")
        assert device.type == "cpu"
        assert meta.backend == "cpu"

    def test_cpu_no_fallback(self) -> None:
        _, meta = resolve_device("cpu")
        assert meta.fallback_applied is False

    def test_cpu_torch_device_str(self) -> None:
        _, meta = resolve_device("cpu")
        assert meta.torch_device_str == "cpu"

    def test_cpu_requested_backend_recorded(self) -> None:
        _, meta = resolve_device("cpu")
        assert meta.requested_backend == "cpu"


# ---------------------------------------------------------------------------
# resolve_device: CUDA
# ---------------------------------------------------------------------------


class TestResolveDeviceCUDA:
    """CUDA selection with availability mocking."""

    def test_cuda_when_available(self) -> None:
        with _patch_cuda(True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Tesla T4"):
                    device, meta = resolve_device("cuda")
        assert device.type == "cuda"
        assert meta.backend == "cuda"
        assert meta.fallback_applied is False

    def test_cuda_fallback_when_unavailable(self) -> None:
        with _patch_cuda(False):
            device, meta = resolve_device("cuda")
        assert device.type == "cpu"
        assert meta.backend == "cpu"
        assert meta.fallback_applied is True

    def test_cuda_unavailable_requested_backend_still_recorded(self) -> None:
        with _patch_cuda(False):
            _, meta = resolve_device("cuda")
        assert meta.requested_backend == "cuda"


# ---------------------------------------------------------------------------
# resolve_device: MPS
# ---------------------------------------------------------------------------


class TestResolveDeviceMPS:
    """MPS selection with availability mocking."""

    def test_mps_when_available(self) -> None:
        mps_avail, mps_built = _patch_mps(True, True)
        with mps_avail, mps_built:
            with patch("torch.device") as mock_device_cls:
                mock_device_cls.return_value = torch.device("cpu")  # safe
                device, meta = resolve_device("mps")
        # meta.backend should be mps
        assert meta.backend == "mps"
        assert meta.fallback_applied is False

    def test_mps_fallback_when_unavailable(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with mps_avail, mps_built:
            device, meta = resolve_device("mps")
        assert device.type == "cpu"
        assert meta.backend == "cpu"
        assert meta.fallback_applied is True

    def test_mps_unavailable_requested_recorded(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with mps_avail, mps_built:
            _, meta = resolve_device("mps")
        assert meta.requested_backend == "mps"


# ---------------------------------------------------------------------------
# resolve_device: auto
# ---------------------------------------------------------------------------


class TestResolveDeviceAuto:
    """Auto-detection prefers CUDA > MPS > CPU."""

    def test_auto_prefers_cuda_over_mps(self) -> None:
        mps_avail, mps_built = _patch_mps(True, True)
        with _patch_cuda(True), mps_avail, mps_built:
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="A100"):
                    _, meta = resolve_device("auto")
        assert meta.backend == "cuda"
        assert meta.fallback_applied is False

    def test_auto_selects_mps_when_no_cuda(self) -> None:
        mps_avail, mps_built = _patch_mps(True, True)
        with _patch_cuda(False), mps_avail, mps_built:
            with patch("torch.device") as mock_dev:
                mock_dev.return_value = torch.device("cpu")
                _, meta = resolve_device("auto")
        assert meta.backend == "mps"
        assert meta.fallback_applied is False

    def test_auto_falls_back_to_cpu(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            device, meta = resolve_device("auto")
        assert device.type == "cpu"
        assert meta.backend == "cpu"
        assert (
            meta.fallback_applied is False
        )  # CPU is the natural choice, not a fallback

    def test_auto_requested_backend_is_auto(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            _, meta = resolve_device("auto")
        assert meta.requested_backend == "auto"


# ---------------------------------------------------------------------------
# DeviceMetadata completeness
# ---------------------------------------------------------------------------


class TestDeviceMetadata:
    """DeviceMetadata fields are populated correctly."""

    def _get_cpu_meta(self) -> DeviceMetadata:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            return get_device_metadata("cpu")

    def test_metadata_has_torch_version(self) -> None:
        meta = self._get_cpu_meta()
        assert meta.torch_version == torch.__version__

    def test_metadata_has_platform_fields(self) -> None:
        import platform

        meta = self._get_cpu_meta()
        assert meta.platform_system == platform.system()
        assert meta.platform_machine == platform.machine()

    def test_metadata_module_version(self) -> None:
        meta = self._get_cpu_meta()
        assert meta.device_module_version == DEVICE_MODULE_VERSION

    def test_metadata_to_dict_keys(self) -> None:
        meta = self._get_cpu_meta()
        d = meta.to_dict()
        required_keys = {
            "backend",
            "requested_backend",
            "torch_device_str",
            "torch_version",
            "cuda_available",
            "mps_available",
            "mps_built",
            "cuda_device_count",
            "fallback_applied",
            "device_module_version",
            "python_version",
            "platform_system",
            "platform_machine",
        }
        assert required_keys.issubset(d.keys())

    def test_metadata_to_json_is_valid(self) -> None:
        import json

        meta = self._get_cpu_meta()
        parsed = json.loads(meta.to_json())
        assert parsed["backend"] == "cpu"

    def test_metadata_cuda_device_count_zero_when_unavailable(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            _, meta = resolve_device("cpu")
        assert meta.cuda_device_count == 0

    def test_metadata_cuda_device_name_none_when_unavailable(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            _, meta = resolve_device("cpu")
        assert meta.cuda_device_name is None


# ---------------------------------------------------------------------------
# MPS fallback env-var detection
# ---------------------------------------------------------------------------


class TestMPSFallbackEnv:
    """PYTORCH_ENABLE_MPS_FALLBACK env-var is reflected in metadata."""

    def test_mps_fallback_flag_false_by_default(self) -> None:
        import os

        env = {
            k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"
        }
        with patch.dict("os.environ", env, clear=True):
            _, meta = resolve_device("cpu")
        assert meta.mps_fallback_ops_enabled is False

    def test_mps_fallback_flag_true_when_set(self) -> None:
        with patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            _, meta = resolve_device("cpu")
        assert meta.mps_fallback_ops_enabled is True


# ---------------------------------------------------------------------------
# validate_mps_ops
# ---------------------------------------------------------------------------


class TestValidateMPSOps:
    """validate_mps_ops returns empty list for non-MPS devices."""

    def test_cpu_device_returns_empty(self) -> None:
        cpu = torch.device("cpu")
        warnings = validate_mps_ops(cpu)
        assert warnings == []

    def test_cuda_device_returns_empty(self) -> None:
        # We simulate a cuda device type without needing a real GPU.
        fake_device = MagicMock()
        fake_device.type = "cuda"
        warnings = validate_mps_ops(fake_device)
        assert warnings == []


# ---------------------------------------------------------------------------
# get_device_metadata convenience wrapper
# ---------------------------------------------------------------------------


class TestGetDeviceMetadata:
    """get_device_metadata is a thin wrapper around resolve_device."""

    def test_returns_device_metadata_type(self) -> None:
        meta = get_device_metadata("cpu")
        assert isinstance(meta, DeviceMetadata)

    def test_consistent_with_resolve_device(self) -> None:
        _, meta_full = resolve_device("cpu")
        meta_convenience = get_device_metadata("cpu")
        assert meta_full.backend == meta_convenience.backend
        assert meta_full.torch_version == meta_convenience.torch_version


# ---------------------------------------------------------------------------
# CLI self-check exit code
# ---------------------------------------------------------------------------


class TestSelfCheck:
    """_self_check returns 0 on success, 1 on failure."""

    def test_self_check_returns_zero_on_cpu(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            code = _self_check(verbose=False)
        assert code == 0

    def test_self_check_is_int(self) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            code = _self_check(verbose=False)
        assert isinstance(code, int)


# ---------------------------------------------------------------------------
# TrainingConfig device resolution
# ---------------------------------------------------------------------------


class TestTrainingConfigDeviceResolution:
    """build_training_config resolves device through the shared abstraction."""

    def test_config_device_is_torch_device(self, tmp_path) -> None:
        cfg = build_training_config(
            algorithm="cql",
            device="cpu",
            dataset_path=tmp_path / "replay.parquet",
        )
        assert isinstance(cfg.device, torch.device)

    def test_config_device_matches_requested(self, tmp_path) -> None:
        cfg = build_training_config(
            algorithm="cql",
            device="cpu",
            dataset_path=tmp_path / "replay.parquet",
        )
        assert cfg.device.type == "cpu"

    def test_config_device_meta_is_metadata(self, tmp_path) -> None:
        cfg = build_training_config(
            algorithm="cql",
            device="cpu",
            dataset_path=tmp_path / "replay.parquet",
        )
        assert isinstance(cfg.device_meta, DeviceMetadata)

    def test_config_cuda_falls_back_when_unavailable(self, tmp_path) -> None:
        mps_avail, mps_built = _patch_mps(False, False)
        with _patch_cuda(False), mps_avail, mps_built:
            cfg = build_training_config(
                algorithm="cql",
                device="cuda",
                dataset_path=tmp_path / "replay.parquet",
            )
        assert cfg.device.type == "cpu"
        assert cfg.device_meta.fallback_applied is True

    def test_config_to_dict_includes_device_meta(self, tmp_path) -> None:
        cfg = build_training_config(
            algorithm="cql",
            device="cpu",
            dataset_path=tmp_path / "replay.parquet",
        )
        d = cfg.to_dict()
        assert "device_meta" in d["runtime"]
        assert d["runtime"]["effective_device"] == "cpu"
