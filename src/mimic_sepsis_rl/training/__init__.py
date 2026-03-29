"""
Training package for MIMIC Sepsis Offline RL.

Provides cross-platform runtime abstractions, training utilities,
and algorithm implementations (CQL, and later BCQ/IQL in Phase 8).

Public surface
--------------
- :mod:`mimic_sepsis_rl.training.device`  – CPU/MPS/CUDA runtime abstraction
- :mod:`mimic_sepsis_rl.training.config`  – Training config resolution
- :mod:`mimic_sepsis_rl.training.common`  – Shared training helpers
- :mod:`mimic_sepsis_rl.training.cql`     – Discrete CQL reference trainer
"""

__all__ = [
    "device",
    "config",
    "common",
    "cql",
]
