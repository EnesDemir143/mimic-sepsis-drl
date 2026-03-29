"""
Cross-platform device runtime abstraction for the MIMIC Sepsis training stack.

Supports Apple Silicon ``Metal/MPS``, NVIDIA ``CUDA``, and CPU fallback
through one unified selection surface.  Algorithm code never branches on
device type beyond this module.

Capabilities
------------
- Explicit device selection via config string (``"mps"``, ``"cuda"``,
  ``"cpu"``, ``"auto"``).
- Availability probing with graceful fallback when the requested backend
  is unavailable.
- Backend metadata snapshot for reproducible run records.
- MPS unsupported-op fallback flag so issues are surfaced early rather
  than silently during training.

CLI self-check
--------------
    python -m mimic_sepsis_rl.training.device --self-check

Version history
---------------
v1.0.0  2026-03-29  Initial cross-platform device abstraction.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEVICE_MODULE_VERSION: Final[str] = "1.0.0"

# Environment variable that can override device selection at run-time.
_ENV_DEVICE_OVERRIDE: Final[str] = "MIMIC_RL_DEVICE"


# ---------------------------------------------------------------------------
# Supported backends
# ---------------------------------------------------------------------------


class Backend(str, Enum):
    """Canonical device backend identifiers."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    AUTO = "auto"

    @classmethod
    def from_str(cls, value: str) -> "Backend":
        """Parse a backend name case-insensitively.

        Parameters
        ----------
        value:
            Raw string from config or CLI (e.g. ``"cuda"``, ``"MPS"``).

        Raises
        ------
        ValueError
            If *value* does not map to a known backend.
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Unknown backend '{value}'. Valid choices: {valid}"
            ) from None


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    """Return True if at least one CUDA device is present and CUDA is built."""
    return torch.cuda.is_available()


def _mps_available() -> bool:
    """Return True if MPS (Apple Silicon) is built and available."""
    return (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )  # type: ignore[attr-defined]


def _mps_built() -> bool:
    """Return True if PyTorch was compiled with MPS support."""
    return (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
    )  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Device metadata
# ---------------------------------------------------------------------------


@dataclass
class DeviceMetadata:
    """Snapshot of the resolved runtime device and its capabilities.

    Intended to be serialised into run config / checkpoint manifests so
    training results can be reproduced or audited across machines.

    Attributes
    ----------
    backend : str
        The *effective* backend after availability checks (``"cuda"``,
        ``"mps"``, or ``"cpu"``).
    requested_backend : str
        The backend requested by config before fallback resolution.
    torch_device_str : str
        Full PyTorch device string (e.g. ``"cuda:0"``).
    torch_version : str
        Installed PyTorch version string.
    cuda_available : bool
    mps_available : bool
    mps_built : bool
    cuda_device_count : int
        Number of visible CUDA GPUs (0 if CUDA not available).
    cuda_device_name : str | None
        Name of the first CUDA device, or None.
    python_version : str
    platform_system : str
    platform_machine : str
    device_module_version : str
    fallback_applied : bool
        True when the requested backend was unavailable and CPU was used.
    mps_fallback_ops_enabled : bool
        Whether ``PYTORCH_ENABLE_MPS_FALLBACK`` is active for ops not
        supported on MPS.
    """

    backend: str
    requested_backend: str
    torch_device_str: str
    torch_version: str
    cuda_available: bool
    mps_available: bool
    mps_built: bool
    cuda_device_count: int
    cuda_device_name: str | None
    python_version: str
    platform_system: str
    platform_machine: str
    device_module_version: str
    fallback_applied: bool
    mps_fallback_ops_enabled: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------


def _resolve_effective_backend(requested: Backend) -> tuple[Backend, bool]:
    """Resolve *requested* to an actually-available backend.

    Returns
    -------
    (effective_backend, fallback_applied)
        *fallback_applied* is True when the requested backend was
        unavailable and we fell back to CPU.
    """
    if requested == Backend.AUTO:
        if _cuda_available():
            return Backend.CUDA, False
        if _mps_available():
            return Backend.MPS, False
        return Backend.CPU, False

    if requested == Backend.CUDA:
        if _cuda_available():
            return Backend.CUDA, False
        logger.warning(
            "CUDA requested but not available; falling back to CPU. "
            "Set device to 'auto' to suppress this warning."
        )
        return Backend.CPU, True

    if requested == Backend.MPS:
        if _mps_available():
            return Backend.MPS, False
        logger.warning(
            "MPS requested but not available%s; falling back to CPU.",
            " (MPS not built into this PyTorch)" if not _mps_built() else "",
        )
        return Backend.CPU, True

    # CPU is always available.
    return Backend.CPU, False


def _build_torch_device_str(backend: Backend) -> str:
    """Build the PyTorch device string for *backend*."""
    if backend == Backend.CUDA:
        return "cuda:0"
    return backend.value  # "mps" or "cpu"


def _mps_fallback_enabled() -> bool:
    """Check whether the MPS CPU-fallback env var is set."""
    import os

    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip() == "1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_device(
    requested: str | Backend = Backend.AUTO,
) -> tuple[torch.device, DeviceMetadata]:
    """Resolve a device request into a concrete ``torch.device``.

    This is the *single entry point* that all training code should use
    instead of constructing ``torch.device(...)`` manually.

    Parameters
    ----------
    requested:
        One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"`` (case-insensitive),
        or a :class:`Backend` enum member.  ``"auto"`` prefers CUDA, then MPS,
        then CPU.

    Returns
    -------
    (device, metadata)
        *device* is a ``torch.Tensor``-ready device object.
        *metadata* is a :class:`DeviceMetadata` snapshot suitable for
        logging or checkpoint manifests.

    Examples
    --------
    >>> device, meta = resolve_device("auto")
    >>> tensor = torch.zeros(4, device=device)
    """
    if isinstance(requested, str):
        requested_backend = Backend.from_str(requested)
    else:
        requested_backend = requested

    effective, fallback_applied = _resolve_effective_backend(requested_backend)
    torch_device_str = _build_torch_device_str(effective)
    device = torch.device(torch_device_str)

    cuda_count = torch.cuda.device_count() if _cuda_available() else 0
    cuda_name: str | None = None
    if cuda_count > 0:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = "unknown"

    metadata = DeviceMetadata(
        backend=effective.value,
        requested_backend=requested_backend.value,
        torch_device_str=torch_device_str,
        torch_version=torch.__version__,
        cuda_available=_cuda_available(),
        mps_available=_mps_available(),
        mps_built=_mps_built(),
        cuda_device_count=cuda_count,
        cuda_device_name=cuda_name,
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        device_module_version=DEVICE_MODULE_VERSION,
        fallback_applied=fallback_applied,
        mps_fallback_ops_enabled=_mps_fallback_enabled(),
    )

    logger.info(
        "Runtime resolved: requested=%s effective=%s device=%s fallback=%s",
        requested_backend.value,
        effective.value,
        torch_device_str,
        fallback_applied,
    )

    return device, metadata


def get_device_metadata(requested: str | Backend = Backend.AUTO) -> DeviceMetadata:
    """Convenience wrapper that returns only the :class:`DeviceMetadata`."""
    _, meta = resolve_device(requested)
    return meta


# ---------------------------------------------------------------------------
# MPS unsupported-op validation
# ---------------------------------------------------------------------------


def validate_mps_ops(device: torch.device) -> list[str]:
    """Probe for common ops that may not be supported on MPS.

    Runs a small set of tensor operations known to be problematic on some
    MPS versions.  Returns a list of warning strings for any that failed.
    This should be called *before* full training begins so issues are
    caught early.

    Parameters
    ----------
    device:
        Must be a MPS device; returns empty list for non-MPS devices.

    Returns
    -------
    list[str]
        One warning string per unsupported op (empty if all pass).
    """
    if device.type != "mps":
        return []

    warnings: list[str] = []

    probes: list[tuple[str, callable]] = [
        (
            "torch.cdist",
            lambda: torch.cdist(
                torch.randn(4, 8, device=device),
                torch.randn(4, 8, device=device),
            ),
        ),
        (
            "torch.linalg.norm",
            lambda: torch.linalg.norm(torch.randn(8, device=device)),
        ),
        (
            "scatter_add (used in indexing)",
            lambda: torch.zeros(5, device=device).scatter_add_(
                0,
                torch.tensor([0, 1, 2, 3, 4], device=device),
                torch.ones(5, device=device),
            ),
        ),
    ]

    for name, fn in probes:
        try:
            fn()
        except (RuntimeError, NotImplementedError) as exc:
            msg = f"MPS op '{name}' unsupported: {exc}"
            warnings.append(msg)
            logger.warning(msg)

    if not warnings:
        logger.info("MPS op validation: all probed ops succeeded.")

    return warnings


# ---------------------------------------------------------------------------
# CLI self-check
# ---------------------------------------------------------------------------


def _self_check(verbose: bool = True) -> int:
    """Run a self-check and print device metadata.

    Returns 0 on success, 1 if a critical issue is detected.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    device, meta = resolve_device(Backend.AUTO)

    if verbose:
        print("=== MIMIC Sepsis RL – Runtime Self-Check ===")
        print(f"Effective backend  : {meta.backend}")
        print(f"Requested backend  : {meta.requested_backend}")
        print(f"PyTorch device str : {meta.torch_device_str}")
        print(f"PyTorch version    : {meta.torch_version}")
        print(f"Python version     : {meta.python_version}")
        print(f"Platform           : {meta.platform_system} ({meta.platform_machine})")
        print(f"CUDA available     : {meta.cuda_available}")
        print(f"CUDA device count  : {meta.cuda_device_count}")
        if meta.cuda_device_name:
            print(f"CUDA device name   : {meta.cuda_device_name}")
        print(f"MPS available      : {meta.mps_available}")
        print(f"MPS built          : {meta.mps_built}")
        print(f"MPS fallback env   : {meta.mps_fallback_ops_enabled}")
        print(f"Fallback applied   : {meta.fallback_applied}")
        print(f"Module version     : {meta.device_module_version}")

    # Smoke test: allocate a small tensor on the resolved device.
    try:
        t = torch.zeros(8, device=device)
        _ = t + 1.0
        if verbose:
            print(f"\n✅ Smoke test passed (tensor on {device}).")
    except Exception as exc:
        if verbose:
            print(f"\n❌ Smoke test FAILED: {exc}")
        return 1

    # MPS-specific op validation
    if device.type == "mps":
        mps_issues = validate_mps_ops(device)
        if mps_issues and verbose:
            print(
                "\n⚠️  MPS op warnings (set PYTORCH_ENABLE_MPS_FALLBACK=1 to mitigate):"
            )
            for w in mps_issues:
                print(f"   - {w}")
        elif verbose:
            print("✅ MPS op probes all passed.")

    if verbose:
        print("\nMetadata JSON:")
        print(meta.to_json())

    return 0


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m mimic_sepsis_rl.training.device``."""
    parser = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.training.device",
        description="Resolve and validate the training device runtime.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run device availability probe and smoke test, then exit.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to probe: auto, cuda, mps, cpu (default: auto).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_only",
        help="Print metadata as JSON only (no human-readable summary).",
    )
    args = parser.parse_args(argv)

    if args.self_check or True:  # always run the check when invoked directly
        if args.json_only:
            meta = get_device_metadata(args.device)
            print(meta.to_json())
            sys.exit(0)
        sys.exit(_self_check(verbose=not args.json_only))


if __name__ == "__main__":
    main()


__all__ = [
    "DEVICE_MODULE_VERSION",
    "Backend",
    "DeviceMetadata",
    "resolve_device",
    "get_device_metadata",
    "validate_mps_ops",
]
