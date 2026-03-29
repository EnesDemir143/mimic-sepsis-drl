"""
Treatment action encoding for the MIMIC Sepsis Offline RL pipeline.

Public API
----------
VasopressorStandardiser
    Converts raw vasopressor administrations to norepinephrine-equivalent rates.
FluidAggregator
    Sums IV fluid volumes within each 4-hour decision window.
ActionBinner
    Learns 5×5 dose bins from the train split and maps any dose pair to one
    of 25 discrete actions.
ActionBinArtifacts
    Serialisable artifact bundle that freezes the trained bin edges.
"""

from mimic_sepsis_rl.mdp.actions.vasopressors import VasopressorStandardiser
from mimic_sepsis_rl.mdp.actions.fluids import FluidAggregator
from mimic_sepsis_rl.mdp.actions.bins import (
    ActionBinner,
    ActionBinArtifacts,
    ACTION_SPEC_VERSION,
)

__all__ = [
    "VasopressorStandardiser",
    "FluidAggregator",
    "ActionBinner",
    "ActionBinArtifacts",
    "ACTION_SPEC_VERSION",
]
