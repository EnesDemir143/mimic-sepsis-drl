"""
Typed models for patient-level split manifests.

A ``SplitManifest`` records which patient identifiers (``subject_id``) belong
to each partition (train / validation / test) together with the configuration
that produced them.  The manifest is the *canonical* leakage boundary:

  - Downstream transforms (scalers, imputers, bin thresholds, behaviour
    estimators) **must fit on train only** and consume these manifests as the
    boundary authority.

Version history
---------------
v1.0.0  2026-03-28  Initial split manifest contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

SPLIT_SPEC_VERSION: Final[str] = "1.0.0"


# ---------------------------------------------------------------------------
# Split partition labels
# ---------------------------------------------------------------------------


class SplitLabel(str, Enum):
    """Canonical name for each data partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# ---------------------------------------------------------------------------
# Per-patient enrollment record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientRecord:
    """One patient's enrollment in a split.

    Attributes
    ----------
    subject_id:
        MIMIC-IV patient identifier.
    split:
        Which partition this patient belongs to.
    episode_keys:
        Sorted list of ``stay_id`` values that are eligible for this patient
        and fall within the split.
    """

    subject_id: int
    split: SplitLabel
    episode_keys: tuple[int, ...]  # sorted stay_ids; immutable


# ---------------------------------------------------------------------------
# Split statistics (balance summary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitStats:
    """Aggregate statistics for one split partition.

    Attributes
    ----------
    split:
        Partition this summary describes.
    n_patients:
        Number of unique patients assigned to this split.
    n_episodes:
        Total eligible episode keys across all patients in this split.
    mortality_rate:
        Fraction of patients who died during their ICU stay, if known.
        ``None`` when mortality data was not provided.
    """

    split: SplitLabel
    n_patients: int
    n_episodes: int
    mortality_rate: float | None = None


# ---------------------------------------------------------------------------
# Top-level manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitManifest:
    """Complete patient-level split manifest.

    Attributes
    ----------
    spec_version:
        Version of the manifest schema (for forward compatibility checks).
    seed:
        Random seed used to produce this split.  Must be recorded so the
        split can be reproduced exactly.
    source_episode_set:
        Identifier of the episode artifact this manifest was derived from
        (e.g. a Parquet path or a git-commit-stamped label).
    train_ids:
        Frozenset of ``subject_id`` values assigned to the training split.
    validation_ids:
        Frozenset of ``subject_id`` values assigned to the validation split.
    test_ids:
        Frozenset of ``subject_id`` values assigned to the test split.
    records:
        Complete per-patient enrollment records (all three splits combined).
    stats:
        Per-split aggregate statistics (order: train, validation, test).
    """

    spec_version: str
    seed: int
    source_episode_set: str
    train_ids: frozenset[int]
    validation_ids: frozenset[int]
    test_ids: frozenset[int]
    records: tuple[PatientRecord, ...] = field(default_factory=tuple)
    stats: tuple[SplitStats, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def split_for(self, subject_id: int) -> SplitLabel | None:
        """Return the split label for *subject_id*, or ``None`` if unknown."""
        if subject_id in self.train_ids:
            return SplitLabel.TRAIN
        if subject_id in self.validation_ids:
            return SplitLabel.VALIDATION
        if subject_id in self.test_ids:
            return SplitLabel.TEST
        return None

    def all_ids(self) -> frozenset[int]:
        """Return the union of all patient identifiers across every split."""
        return self.train_ids | self.validation_ids | self.test_ids

    def has_leakage(self) -> bool:
        """Return ``True`` when any patient appears in more than one split."""
        splits = [self.train_ids, self.validation_ids, self.test_ids]
        seen: set[int] = set()
        for s in splits:
            if seen & s:
                return True
            seen |= s
        return False
